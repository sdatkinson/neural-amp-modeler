"""
Parameter-space bridge for PANAMA-style active-learning capture selection.

Adapted from PANAMA (Parametric Active-learning for Neural Amp Modeling Assistance),
arXiv:2509.26564v1. The sigmoid(latent) continuous control-vector optimization and the
switch/continuous split of the acquisition optimizer variable are due to the PANAMA authors.
"""

import functools as _functools
import itertools as _itertools
import math as _math
from collections.abc import Sequence as _Sequence

import torch as _torch

from ._spec import ParamSpec as _ParamSpec


def _validate_switch_index(spec: _ParamSpec, value: float) -> int:
    """
    Validate that ``value`` is a finite integer in ``[0, K-1]`` for the given switch spec
    and return it as an ``int``. Mirrors the switch contract enforced by
    ``ParametricNet._encode_params`` / ``resolve_named_params``, but in the inverse
    (raw -> name) direction. Switch indices are always integers in this design, so a bad
    value is rejected rather than silently coerced.
    """
    if not _math.isfinite(value):
        raise ValueError(f"Switch parameter {spec.name!r} index must be finite")
    if not float(value).is_integer():
        raise ValueError(f"Switch parameter {spec.name!r} index must be an integer")
    index = int(value)
    if index < 0 or index >= spec.num_inputs:
        raise ValueError(
            f"Switch parameter {spec.name!r} index must be within "
            f"[0, {spec.num_inputs - 1}]"
        )
    return index


def decode_named_params(
    raw: _torch.Tensor | _Sequence[float],
    specs: _Sequence[_ParamSpec],
) -> dict[str, float | str]:
    specs = tuple(specs)
    raw = _torch.as_tensor(raw)
    if raw.ndim != 1:
        raise ValueError(f"Expected raw params to have shape (P,); got {tuple(raw.shape)}")
    if raw.shape[0] != len(specs):
        raise ValueError(
            f"Expected raw params length {len(specs)}; got trailing dimension {raw.shape[0]}"
        )

    decoded: dict[str, float | str] = {}
    for i, spec in enumerate(specs):
        value = float(raw[i])
        if spec.type == "switch":
            if spec.enum_names is None:
                raise RuntimeError(
                    f"Switch ParamSpec {spec.name!r} is missing enum_names after validation"
                )
            decoded[spec.name] = spec.enum_names[_validate_switch_index(spec, value)]
            continue

        # Continuous values are validated and emitted directly (no round-trip through
        # resolve_named_params). Reject non-finite; otherwise pass the value through,
        # rounded to a cosmetic display precision. Out-of-[min, max] values are
        # intentionally preserved: captures may exceed the declared range, and the model
        # linearly rescales them without clamping (see _resolve_continuous_value).
        if not _math.isfinite(value):
            raise ValueError(f"Continuous parameter {spec.name!r} must be finite")
        decoded[spec.name] = round(value, 6)

    return decoded


@_functools.lru_cache(maxsize=None)
def _split_param_indices_cached(
    specs: tuple[_ParamSpec, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    continuous_idx = []
    switch_idx = []
    switch_cardinalities = []
    for i, spec in enumerate(specs):
        if spec.type == "switch":
            switch_idx.append(i)
            switch_cardinalities.append(spec.num_inputs)
        else:
            continuous_idx.append(i)
    return tuple(continuous_idx), tuple(switch_idx), tuple(switch_cardinalities)


def split_param_indices(
    specs: _Sequence[_ParamSpec],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    # Cached on the (hashable, frozen) spec tuple so the g-opt hot loop
    # (assemble_raw_params runs once per Adam step) does not re-walk the specs each call.
    return _split_param_indices_cached(tuple(specs))


def switch_combinations(specs: _Sequence[_ParamSpec]) -> list[tuple[int, ...]]:
    _, _, switch_cardinalities = split_param_indices(specs)
    if len(switch_cardinalities) == 0:
        return [()]
    return list(
        _itertools.product(*(range(cardinality) for cardinality in switch_cardinalities))
    )


def assemble_raw_params(
    z: _torch.Tensor | _Sequence[float],
    switch_combo: tuple[int, ...],
    specs: _Sequence[_ParamSpec],
) -> _torch.Tensor:
    specs = tuple(specs)
    continuous_idx, switch_idx, _ = split_param_indices(specs)
    z = _torch.as_tensor(z)
    if z.ndim == 0:
        raise ValueError(
            f"Expected continuous latents to have shape (C,) or (..., C); got {tuple(z.shape)}"
        )
    if z.shape[-1] != len(continuous_idx):
        raise ValueError(
            f"Expected continuous latents trailing dimension {len(continuous_idx)}; "
            f"got {z.shape[-1]}"
        )
    if len(switch_combo) != len(switch_idx):
        raise ValueError(
            f"Expected switch combination length {len(switch_idx)}; got {len(switch_combo)}"
        )

    batch_shape = tuple(z.shape[:-1])
    continuous_col = 0
    switch_col = 0
    columns = []
    for spec in specs:
        if spec.type == "switch":
            switch_value = switch_combo[switch_col]
            if switch_value < 0 or switch_value >= spec.num_inputs:
                raise ValueError(
                    f"Switch combination index {switch_value} for {spec.name!r} must be within "
                    f"[0, {spec.num_inputs - 1}]"
                )
            # Constant integer column for the swept switch state: no graph edge to z, so
            # autograd leaves the switch dims fixed (D2).
            columns.append(
                _torch.full(
                    batch_shape + (1,),
                    float(switch_value),
                    dtype=z.dtype,
                    device=z.device,
                )
            )
            switch_col += 1
            continue

        # raw_i = min_i + (max_i - min_i) * sigmoid(z_i): a smooth, bounded map keeping
        # suggestions in-range by construction (echoes PANAMA's sigmoid(latent)). NOTE the
        # bounds form an *open* interval, approached only asymptotically as |z| -> inf where
        # sigmoid's gradient vanishes; an Adam ascent from a near-zero init cannot practically
        # reach the extremes. Knob extremes are often high-disagreement regions, so Task 5
        # should compensate (e.g. large-|z| restart inits) and Task 6 may snap near-boundary
        # suggestions to the bound when emitting user-unit proposals.
        width = spec.max - spec.min
        columns.append(spec.min + width * _torch.sigmoid(z[..., continuous_col : continuous_col + 1]))
        continuous_col += 1

    # Concatenate per-spec column blocks (never in-place index assignment) so autograd
    # reaches z through the continuous columns while the switch columns stay constant.
    return _torch.cat(columns, dim=-1)


def abbreviate_param_names(names: _Sequence[str]) -> dict[str, str]:
    """
    Map each param name to the shortest leading slice that is unique among ``names``.

    Comparison is case-insensitive (so ``Treble``/``Tone`` need two letters and
    ``boost``/``Bottom`` need three), but the returned abbreviation preserves the
    original capitalization. A name that is a case-insensitive prefix of another falls
    back to its full spelling.

    Shared by the LHS starter script and the active-learning proposal driver so both
    encode capture settings into filenames the same way.
    """
    names = list(names)
    abbreviations: dict[str, str] = {}
    for index, name in enumerate(names):
        others = [other for i, other in enumerate(names) if i != index]
        length = 1
        while length < len(name) and any(
            other.lower().startswith(name[:length].lower()) for other in others
        ):
            length += 1
        abbreviations[name] = name[:length]
    return abbreviations


def format_param_value(value: float | str) -> str:
    # Continuous params decode to floats (e.g. 4.5 -> "4.5", 1.0 -> "1"); switch params
    # decode to their enum-name string.
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def make_capture_y_path(
    prefix: str,
    params: dict[str, float | str],
    abbreviations: dict[str, str],
) -> str:
    """
    Build a capture filename whose stem encodes the decoded param settings, e.g.
    ``{prefix}G4.5_TOn.wav``. ``abbreviations`` maps each param name to its unique-prefix
    short form (see :func:`abbreviate_param_names`). Shared by the LHS starter script and
    the active-learning proposal driver so both name captures the same way.
    """
    parts = [
        f"{abbreviations[name]}{format_param_value(value)}"
        for name, value in params.items()
    ]
    return f"{prefix}{'_'.join(parts)}.wav"


def quantize_to_capture_grid(
    raw: _torch.Tensor | _Sequence[float],
    specs: _Sequence[_ParamSpec],
    *,
    default_step: float = 0.5,
) -> _torch.Tensor:
    """
    Snap each *continuous* param to the realizable capture grid (nearest ``step``,
    clamped into ``[min, max]``); leave *switch* indices untouched.

    Human knobs can only be dialed to a fixed grid, so the value recorded in
    ``data.json`` must equal the realizable setting that was actually captured (D5). This
    is the single shared helper used by both the LHS starter script (Task 3) and the
    active-learning proposals (Task 6) so they emit values on the *same* grid. It is a
    pure, idempotent, output-time function: never call it inside the g-opt loop, which
    must stay continuous or gradients break (D5).

    A per-``ParamSpec`` ``step`` (realizability metadata only; it does not affect
    normalization or training) overrides ``default_step`` when present. The field itself
    is Task 3.5 plumbing; this helper honors it forward-compatibly via ``getattr`` so it
    keeps working once the field lands without re-touching this code.
    """
    if not _math.isfinite(default_step) or default_step <= 0.0:
        raise ValueError(
            f"default_step must be a positive finite number; got {default_step}"
        )
    specs = tuple(specs)
    raw = _torch.as_tensor(raw)
    if raw.shape[-1] != len(specs):
        raise ValueError(
            f"Expected raw params trailing dimension {len(specs)}; got {raw.shape[-1]}"
        )

    # Output-time only: detach so a stray grad tensor can't leak a graph through the
    # in-place column writes below, and work in float so integer-typed inputs round.
    quantized = raw.detach().to(_torch.float64).clone()
    for i, spec in enumerate(specs):
        if spec.type == "switch":
            continue
        step = getattr(spec, "step", None)
        step = default_step if step is None else float(step)
        if not _math.isfinite(step) or step <= 0.0:
            raise ValueError(
                f"Capture-grid step for {spec.name!r} must be a positive finite number; "
                f"got {step}"
            )
        snapped = _torch.round(quantized[..., i] / step) * step
        quantized[..., i] = _torch.clamp(snapped, spec.min, spec.max)
    return quantized
