# File: _normalize.py
# Author: TONE3000 - João Felipe Santos & Woodbury Shortridge

"""
Output-target RMS normalization with head_scale compensation.

The training target ``y`` is scaled to a fixed RMS (with a peak-safety clamp)
before training; the trained WaveNet's ``head_scale`` is then divided by the
same gain so that the *exported* model produces the same output level as if
no normalization had ever been applied. Normalization is therefore strictly a
preconditioning step on the training data -- it does not change the model's
inference-time response.

This module is the single source of truth for the feature. ``nam.train.full``
and ``nam.train.core`` invoke it via :func:`prepare` (before training) and
:func:`compensate_head_scale` (after training).
"""

import warnings as _warnings
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import numpy as _np
import torch as _torch

from ..data import wav_to_np as _wav_to_np

# Default RMS target. -18 dBFS is a conventional broadcast/mixing reference and
# leaves comfortable headroom for transients in guitar/amp captures.
DEFAULT_TARGET_RMS_DBFS = -18.0

# Strict cap on |y * gain|. The existing Dataset validation rejects |y| >= 1.0,
# so we leave a small margin below unity.
_PEAK_LIMIT = 1.0 - 1e-4

# RMS below this is treated as silence; no normalization is applied.
_SILENCE_RMS = 1e-8


class OutputNormalizationSkippedWarning(UserWarning):
    """
    Emitted when output-target normalization was requested but had to be
    skipped (e.g. the model architecture doesn't expose a ``head_scale``
    we can post-hoc compensate).
    """


@_dataclass(frozen=True)
class NormalizationPlan:
    """
    Result of :func:`prepare`.

    :param data_config: A (possibly modified) data config with ``y_scale``
        injected into ``common`` so both train and validation splits apply
        the same gain.
    :param gain: The gain that was applied to ``y``. After training, divide
        the model's ``head_scale`` by this value via
        :func:`compensate_head_scale`. ``1.0`` means a no-op.
    :param target_rms_dbfs: The target dBFS that produced ``gain`` (``None``
        when normalization is disabled / skipped).
    """

    data_config: _Dict
    gain: float
    target_rms_dbfs: _Optional[float]

    @property
    def applied(self) -> bool:
        return self.gain != 1.0 and self.target_rms_dbfs is not None


def _db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)


def compute_y_scale(
    y: _Union[_np.ndarray, _torch.Tensor],
    target_rms_dbfs: float = DEFAULT_TARGET_RMS_DBFS,
) -> float:
    """
    Compute a scalar gain that:

    1. Scales ``y`` to ``target_rms_dbfs`` RMS, then
    2. Attenuates further if needed so that ``|gain * y|`` stays strictly
       below 1.0 (peak-safety clamp).

    Returns ``1.0`` if ``y`` is effectively silent.
    """
    arr = (
        y.detach().to(_torch.float64).cpu().numpy()
        if isinstance(y, _torch.Tensor)
        else _np.asarray(y, dtype=_np.float64)
    )
    rms = float(_np.sqrt(_np.mean(arr * arr)))
    if rms < _SILENCE_RMS:
        return 1.0
    gain = _db_to_amp(target_rms_dbfs) / rms
    peak = float(_np.max(_np.abs(arr))) * gain
    if peak > _PEAK_LIMIT:
        gain *= _PEAK_LIMIT / peak
    return float(gain)


_DATA_CONFIG_KEY = "target_rms_dbfs"


def parse_data_config(data_config: _Dict) -> _Optional[float]:
    """
    Read the top-level ``target_rms_dbfs`` field from a data config.

    :return: Target dBFS to use, or ``None`` if normalization is disabled.

    Conventions (mirrors how ``delay`` / ``nx`` live as flat top-level
    fields elsewhere in this repo):

    * Key missing -> enabled with :data:`DEFAULT_TARGET_RMS_DBFS`.
    * ``null``    -> disabled.
    * ``<float>`` -> enabled with that explicit target.
    """
    if _DATA_CONFIG_KEY not in data_config:
        return DEFAULT_TARGET_RMS_DBFS
    target = data_config[_DATA_CONFIG_KEY]
    if target is None:
        return None
    if isinstance(target, bool) or not isinstance(target, (int, float)):
        raise ValueError(
            f"data_config[{_DATA_CONFIG_KEY!r}] must be null or a number; "
            f"got {target!r}"
        )
    return float(target)


def supports_head_scale_compensation(model_config: _Dict) -> bool:
    """
    True iff the model exposes a ``head_scale`` that we can post-hoc divide
    by the applied gain without altering the model's response.

    Currently:

    * ``WaveNet``       -> supported when the top-level ``head`` is null.
    * ``PackedWaveNet`` -> supported when the top-level and all submodels'
                          ``head`` fields are null.

    Models with a non-trivial top-level head are not supported because the
    head's bias would not commute with a post-hoc output scaling.
    """
    net = model_config.get("net", {})
    name = net.get("name")
    cfg = net.get("config", {}) or {}
    if name == "WaveNet":
        return cfg.get("head") is None
    if name == "PackedWaveNet":
        if cfg.get("head") is not None:
            return False
        return all(
            (sm.get("config", {}) or {}).get("head") is None
            for sm in cfg.get("submodels", [])
        )
    return False


def _resolve_train_y_path(data_config: _Dict) -> _Optional[str]:
    """
    Find the y_path used by the training split. We only normalize when this
    can be resolved unambiguously (single y file feeding the training set).
    """
    common = data_config.get("common", {}) or {}
    if "y_path" in common:
        return common["y_path"]
    train = data_config.get("train")
    if isinstance(train, dict) and "y_path" in train:
        return train["y_path"]
    return None


def prepare(
    data_config: _Dict,
    model_config: _Dict,
    target_rms_dbfs: _Optional[float] = None,
) -> NormalizationPlan:
    """
    Compute the normalization gain and return an updated data config that
    applies it (via the dataset's existing ``y_scale``) uniformly to all
    splits.

    :param target_rms_dbfs: Overrides the value read from ``data_config``.
        Pass ``None`` to use the value from ``data_config`` (or the default).
        Pass an explicit value to override.

    No-ops (returns ``gain == 1.0``) when:

    * Normalization is disabled in the config (silent).
    * The model does not support head_scale compensation (warns).
    * The training ``y_path`` cannot be resolved (warns).
    * The target signal is silent (silent).

    Warnings use :class:`OutputNormalizationSkippedWarning`, a
    :class:`UserWarning` subclass; tests can match it with
    ``pytest.warns(OutputNormalizationSkippedWarning, match=...)``.
    """
    if target_rms_dbfs is None:
        target_rms_dbfs = parse_data_config(data_config)
    if target_rms_dbfs is None:
        return NormalizationPlan(data_config, 1.0, None)
    if not supports_head_scale_compensation(model_config):
        net_name = (model_config.get("net", {}) or {}).get("name", "<unknown>")
        _warnings.warn(
            f"Output normalization (target {target_rms_dbfs:.1f} dBFS) was "
            f"requested but model architecture {net_name!r} does not support "
            "head_scale compensation; skipping. To silence this warning, set "
            f"`{_DATA_CONFIG_KEY}` to null in the data config (or pass "
            "`output_target_rms_dbfs=None`).",
            OutputNormalizationSkippedWarning,
            stacklevel=2,
        )
        return NormalizationPlan(data_config, 1.0, None)
    y_path = _resolve_train_y_path(data_config)
    if y_path is None:
        _warnings.warn(
            f"Output normalization (target {target_rms_dbfs:.1f} dBFS) was "
            "requested but the training `y_path` could not be resolved from "
            "the data config; skipping.",
            OutputNormalizationSkippedWarning,
            stacklevel=2,
        )
        return NormalizationPlan(data_config, 1.0, None)

    y = _wav_to_np(y_path)
    gain = compute_y_scale(y, target_rms_dbfs=target_rms_dbfs)
    if gain == 1.0:
        return NormalizationPlan(data_config, 1.0, target_rms_dbfs)

    updated = _deepcopy(data_config)
    common = updated.setdefault("common", {})
    # Compose with any user-set y_scale so we never silently override it.
    common["y_scale"] = float(common.get("y_scale", 1.0)) * gain
    # Drop the directive from the propagated config so it isn't re-applied
    # downstream by mistake.
    updated.pop(_DATA_CONFIG_KEY, None)
    return NormalizationPlan(updated, gain, target_rms_dbfs)


def compensate_head_scale(net, gain: float) -> None:
    """
    Divide ``net``'s ``head_scale`` by ``gain`` in-place. ``net`` is the
    public ``WaveNet`` / ``PackedWaveNet`` wrapper; both delegate to an
    internal WaveNet that owns the ``_head_scale`` attribute.

    For ``PackedWaveNet`` the per-submodel ``head_scale`` (held on the spec
    and used when extracting submodels for export) is also updated, so the
    exported container reflects the compensation.

    No-op when ``gain == 1.0``.
    """
    if gain == 1.0:
        return
    if gain == 0.0 or not _np.isfinite(gain):
        raise ValueError(f"Cannot compensate for non-finite/zero gain: {gain!r}")
    internal = getattr(net, "_net", None)
    if internal is None or not hasattr(internal, "_head_scale"):
        raise TypeError(
            f"{type(net).__name__} does not expose a head_scale to compensate"
        )
    new_value = float(internal._head_scale) / float(gain)
    internal._head_scale = new_value
    # PackedWaveNet mirrors head_scale in spec submodel configs; keep them in
    # sync so extract_submodel() / _export_config see the new value.
    spec = getattr(net, "_spec", None)
    if spec is not None:
        for sm in getattr(spec, "submodels", ()):
            if "head_scale" in sm.config:
                sm.config["head_scale"] = new_value


__all__ = [
    "DEFAULT_TARGET_RMS_DBFS",
    "NormalizationPlan",
    "OutputNormalizationSkippedWarning",
    "compensate_head_scale",
    "compute_y_scale",
    "parse_data_config",
    "prepare",
    "supports_head_scale_compensation",
]
