"""
Export helpers for :class:`HyperWaveNet`.

The baked export path is intentionally WaveNet-specific: it materializes one fixed control
setting into a plain stock WaveNet export that existing runtimes already understand. Other
parametric architectures still use the generic parametric export path from ``ParametricNet``.
"""

import math as _math
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from copy import deepcopy as _deepcopy
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Optional as _Optional

import torch as _torch

from ...data import ConcatDataset as _ConcatDataset
from ...data import Dataset as _Dataset
from ..wavenet import WaveNet as _WaveNet
from ._base import ParametricNet as _ParametricNet
from ._concat_wavenet import ConcatWaveNet as _ConcatWaveNet
from ._dataset import ParametricDataset as _ParametricDataset
from ._dataset import resolve_named_params as _resolve_named_params
from ._hyperwavenet import HyperWaveNet as _HyperWaveNet

_RawParams = _Mapping[str, _Any] | _Sequence[float] | _torch.Tensor
_Settings = _Mapping[str, _RawParams] | _Sequence[tuple[str, _RawParams]]


class _HyperWaveNetScaleOutputHook(_Dataset._ScaleOutputHook):
    def __init__(self, *, scale: float, base_weight_count: int):
        super().__init__(scale=scale)
        if base_weight_count <= 0:
            raise ValueError(
                f"base_weight_count must be positive; got {base_weight_count}"
            )
        self._base_weight_count = base_weight_count

    def apply(self, model_dict: dict):
        if model_dict["architecture"] != "HyperWaveNet":
            return super().apply(model_dict)
        model_dict["config"]["head_scale"] *= self.scale
        model_dict["weights"][self._base_weight_count - 1] *= self.scale
        self._adjust_metadata_loudness(model_dict)
        return model_dict


class _ConcatWaveNetScaleOutputHook(_Dataset._ScaleOutputHook):
    def apply(self, model_dict: dict):
        if model_dict["architecture"] != "ConcatWaveNet":
            return super().apply(model_dict)
        # Same layout as stock WaveNet: head_scale in config, mirrored as the final weight.
        model_dict["config"]["head_scale"] *= self.scale
        model_dict["weights"][-1] *= self.scale
        self._adjust_metadata_loudness(model_dict)
        return model_dict


def _normalize_output_scale(output_scale: _Optional[float]) -> _Optional[float]:
    if output_scale is None:
        return None
    output_scale = float(output_scale)
    if not _math.isfinite(output_scale) or output_scale == 0.0:
        raise ValueError(
            f"output_scale must be finite and non-zero; got {output_scale}"
        )
    return output_scale


def _resolve_raw_params(model: _HyperWaveNet, params: _RawParams) -> _torch.Tensor:
    nominal_params = model.nominal_params
    if isinstance(params, _Mapping):
        raw = _resolve_named_params(params, model.param_specs)
    else:
        raw = _torch.as_tensor(params)
    raw = raw.to(device=nominal_params.device, dtype=nominal_params.dtype)
    if raw.ndim != 1:
        raise ValueError(
            f"Expected params to resolve to shape ({model.param_dim},); got {tuple(raw.shape)}"
        )
    if raw.shape[0] != model.param_dim:
        raise ValueError(
            f"Expected params length {model.param_dim}; got {raw.shape[0]}"
        )
    if not _torch.isfinite(raw).all():
        raise ValueError("params must be finite")
    return raw


def _iter_settings(settings: _Settings) -> tuple[tuple[str, _RawParams], ...]:
    if isinstance(settings, _Mapping):
        return tuple((basename, params) for basename, params in settings.items())
    return tuple(settings)


def _iter_output_scale_datasets(datasets: _Any) -> _Iterable[_Dataset]:
    if isinstance(datasets, _Dataset):
        yield datasets
        return
    if hasattr(datasets, "_y_scale"):
        yield datasets
        return
    if isinstance(datasets, _ParametricDataset):
        yield from _iter_output_scale_datasets(datasets.dataset)
        return
    if isinstance(datasets, _ConcatDataset):
        for dataset in datasets.datasets:
            yield from _iter_output_scale_datasets(dataset)
        return
    if hasattr(datasets, "dataset"):
        yield from _iter_output_scale_datasets(datasets.dataset)
        return
    if isinstance(datasets, _Iterable) and not isinstance(datasets, (str, bytes)):
        for dataset in datasets:
            yield from _iter_output_scale_datasets(dataset)
        return
    raise TypeError(
        "Expected a Dataset, ParametricDataset, ConcatDataset, or iterable of them; "
        f"got {type(datasets).__name__}"
    )


def bake(
    model: _HyperWaveNet,
    params: _RawParams,
    *,
    output_scale: _Optional[float] = None,
) -> _WaveNet:
    """
    Materialize one HyperWaveNet setting into a standard stock WaveNet export surface.

    The returned wrapper exports with architecture ``"WaveNet"`` and the stock channels_8
    weight layout, so the resulting `.nam` is byte-compatible with the existing runtime.
    ``output_scale`` is the shared dataset ``_y_scale`` used during training; export applies
    its reciprocal so the written `.nam` produces the original unscaled level. Recover it from
    the training datasets with :func:`output_scale_from_datasets` when needed.
    """

    output_scale = _normalize_output_scale(output_scale)
    raw_params = _resolve_raw_params(model, params)
    training = model.training
    try:
        model.eval()
        with _torch.no_grad():
            encoded = model._encode_params(raw_params)
            weight_dict = model._assemble_weight_dict(model._hypernet.generate(encoded))
            # Clone the frozen template (full width: `_slimming_value` stays 1.0 outside a
            # live forward) and overwrite its parameters with the conditioned ones; buffers
            # and head_scale ride along with the copy unchanged.
            baked_inner = _deepcopy(model._template)
            for name, parameter in baked_inner.named_parameters():
                parameter.data.copy_(
                    weight_dict[name].detach().to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                )
        baked = _WaveNet(wavenet=baked_inner, sample_rate=model.sample_rate)
        baked.eval()
    finally:
        model.train(training)
    if output_scale is not None:
        baked.export_model_dict_post_hooks.append(
            _Dataset._ScaleOutputHook(scale=1.0 / output_scale)
        )
    return baked


def bake_to_files(
    model: _HyperWaveNet,
    settings: _Settings,
    outdir: _Path,
    *,
    output_scale: _Optional[float] = None,
    include_snapshot: bool = False,
    user_metadata=None,
    other_metadata=None,
) -> list[_Path]:
    """
    Bake and export a directory of fixed-setting stock WaveNet `.nam` files.
    """

    output_scale = _normalize_output_scale(output_scale)
    paths = []
    seen_basenames = set()
    outdir = _Path(outdir)
    for basename, params in _iter_settings(settings):
        basename = str(basename)
        if basename in seen_basenames:
            raise ValueError(
                f"Duplicate bake basename {basename!r}; each setting must export to a "
                "distinct file."
            )
        seen_basenames.add(basename)
        baked = bake(model, params, output_scale=output_scale)
        baked.export(
            outdir=outdir,
            basename=basename,
            include_snapshot=include_snapshot,
            user_metadata=user_metadata,
            other_metadata=other_metadata,
        )
        paths.append(outdir / f"{basename}{baked.FILE_EXTENSION}")
    return paths


def _make_parametric_scale_hook(
    model: _ParametricNet, scale: float
) -> _Dataset._ScaleOutputHook:
    if isinstance(model, _HyperWaveNet):
        return _HyperWaveNetScaleOutputHook(
            scale=scale,
            base_weight_count=len(model._template.export_weights()),
        )
    if isinstance(model, _ConcatWaveNet):
        return _ConcatWaveNetScaleOutputHook(scale=scale)
    raise NotImplementedError(
        f"Output-scale compensation is not implemented for {type(model).__name__}"
    )


def export_parametric(
    model: _ParametricNet,
    outdir: _Path,
    *,
    output_scale: _Optional[float] = None,
    basename: str = "model",
    include_snapshot: bool = False,
    user_metadata=None,
    other_metadata=None,
) -> None:
    """
    Export the parametric `.nam` for a future parametric-aware runtime.

    ``output_scale`` follows the same contract as :func:`bake`: the caller supplies the
    shared dataset ``_y_scale`` when training normalized the targets, and export writes the
    reciprocal into the base WaveNet head scale slot. Recover it from the training datasets
    with :func:`output_scale_from_datasets` when needed.
    """

    output_scale = _normalize_output_scale(output_scale)

    # A parametric model must never export through a stock `_ScaleOutputHook`: its `apply`
    # dispatches on architecture and raises on parametric names. Strip any (there shouldn't
    # be one) in BOTH branches, then install the architecture-aware hook only when
    # compensation is requested. Doing this unconditionally keeps the no-scale path from
    # detonating on a stray stock hook the way the scaled path is already guarded against.
    original_hooks = list(model.export_model_dict_post_hooks)
    temporary_hooks = [
        hook
        for hook in original_hooks
        if not isinstance(hook, _Dataset._ScaleOutputHook)
    ]
    if output_scale is not None:
        temporary_hooks.append(
            _make_parametric_scale_hook(model, scale=1.0 / output_scale)
        )
    model.export_model_dict_post_hooks = temporary_hooks
    try:
        model.export(
            outdir=outdir,
            basename=basename,
            include_snapshot=include_snapshot,
            user_metadata=user_metadata,
            other_metadata=other_metadata,
        )
    finally:
        model.export_model_dict_post_hooks = original_hooks


def output_scale_from_datasets(datasets: _Any) -> _Optional[float]:
    """
    Recover the single shared output scale applied across stock datasets.

    Accepts a stock :class:`~nam.data.Dataset`, a parametric wrapper, a
    :class:`~nam.data.ConcatDataset`, or an iterable nesting any mix of them. This matches the
    shapes the trainer naturally has on hand at export time, so compensation can be recovered
    from the same dataset objects used during joint normalization.
    """

    scales = []
    for dataset in _iter_output_scale_datasets(datasets):
        scale = dataset._y_scale
        if scale is None:
            scales.append(None)
            continue
        scale = float(scale)
        if not _math.isfinite(scale) or scale == 0.0:
            raise ValueError(
                f"Dataset output scale must be finite and non-zero; got {scale}"
            )
        scales.append(scale)

    if len(scales) == 0 or all(scale is None for scale in scales):
        return None
    if any(scale is None for scale in scales):
        raise ValueError(
            "Datasets must either all be unscaled or share one output scale"
        )

    reference = scales[0]
    assert reference is not None
    for scale in scales[1:]:
        assert scale is not None
        if not _math.isclose(scale, reference):
            raise ValueError(
                f"Datasets must share one output scale; got {reference} and {scale}"
            )
    return reference
