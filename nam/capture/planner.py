"""
Latin-Hypercube capture planning.

The sampling core is adapted from PANAMA (Parametric Active-learning for Neural Amp
Modeling Assistance), arXiv:2509.26564v1. The Latin-Hypercube starter-set idea and the
active-learning capture workflow are due to the PANAMA authors. It moved here from
``scripts/make_starter_settings.py`` so the capture app and the starter-settings CLI
share one implementation; the script now imports from this module.

On top of the raw sampling, :func:`plan_captures` produces the app-facing plan: an
ordered list of train and held-out validation settings, quantized to each knob's
capture grid, with filenames that encode the settings.
"""

from __future__ import annotations

from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Literal as _Literal

import numpy as _np

from ..models.parametric import abbreviate_param_names as _abbreviate_param_names
from ..models.parametric import decode_named_params as _decode_named_params
from ..models.parametric import make_capture_y_path as _make_capture_y_path
from ..models.parametric import quantize_to_capture_grid as _quantize_to_capture_grid
from ..models.parametric import switch_combinations as _switch_combinations
from ..models.parametric import ParamSpec as _ParamSpec
from .params import DEFAULT_KNOB_STEP as _DEFAULT_KNOB_STEP
from .params import KnobSpec as _KnobSpec
from .params import validate_knobs as _validate_knobs


# Large, fixed offset so the validation draws are a different LHS stream than the train
# draws (held-out settings), while staying reproducible from the same seed.
VALIDATION_SEED_OFFSET = 2**31 - 1

CAPTURES_DIRNAME = "captures"


def latin_hypercube_unit(
    n: int,
    dim: int,
    *,
    seed: int,
) -> _np.ndarray:
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if dim < 0:
        raise ValueError(f"dim must be non-negative; got {dim}")
    if dim == 0:
        return _np.zeros((n, 0), dtype=_np.float64)

    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim, rng=_np.random.default_rng(seed))
        return _np.asarray(sampler.random(n=n), dtype=_np.float64)
    except (ImportError, TypeError):
        # ImportError: scipy absent. TypeError: scipy < 1.15 predates the ``rng=`` kwarg
        # (it used ``seed=``); rather than special-case the version, fall back to the
        # stratified-numpy sampler, which needs no scipy at all.
        rng = _np.random.default_rng(seed)
        samples = _np.empty((n, dim), dtype=_np.float64)
        for i in range(dim):
            samples[:, i] = (rng.permutation(n) + rng.random(n)) / n
        return samples


def _scale_continuous_samples(
    unit_samples: _np.ndarray,
    continuous_specs: _Sequence[_ParamSpec],
) -> _np.ndarray:
    if unit_samples.shape[1] != len(continuous_specs):
        raise ValueError(
            f"Expected {len(continuous_specs)} continuous columns; got {unit_samples.shape[1]}"
        )
    if len(continuous_specs) == 0:
        return unit_samples

    mins = _np.asarray([spec.min for spec in continuous_specs], dtype=_np.float64)
    widths = _np.asarray(
        [spec.max - spec.min for spec in continuous_specs], dtype=_np.float64
    )
    return mins + unit_samples * widths


def _stratified_switch_assignments(
    specs: _Sequence[_ParamSpec],
    n: int,
    *,
    seed: int,
) -> _np.ndarray:
    combos = _switch_combinations(specs)
    if len(combos) == 1 and len(combos[0]) == 0:
        return _np.zeros((n, 0), dtype=_np.int64)

    rng = _np.random.default_rng(seed)
    combo_array = _np.asarray(combos, dtype=_np.int64)
    num_combos = combo_array.shape[0]
    assignments = _np.empty((n, combo_array.shape[1]), dtype=_np.int64)
    # Deal out shuffled full cycles of every switch combination. Balance is exact only
    # when ``n`` is a multiple of ``num_combos``; otherwise the final partial cycle skews
    # the counts by up to one per combination.
    for start in range(0, n, num_combos):
        stop = min(start + num_combos, n)
        cycle = combo_array[rng.permutation(num_combos)]
        assignments[start:stop] = cycle[: stop - start]
    return assignments


def _assemble_raw_settings(
    continuous_samples: _np.ndarray,
    switch_assignments: _np.ndarray,
    specs: _Sequence[_ParamSpec],
) -> list[_np.ndarray]:
    if continuous_samples.shape[0] != switch_assignments.shape[0]:
        raise ValueError("Continuous and switch samples must have the same row count")

    raw_settings: list[_np.ndarray] = []
    for row_index in range(continuous_samples.shape[0]):
        raw = _np.empty(len(specs), dtype=_np.float64)
        continuous_col = 0
        switch_col = 0
        for spec_index, spec in enumerate(specs):
            if spec.type == "switch":
                raw[spec_index] = float(switch_assignments[row_index, switch_col])
                switch_col += 1
            else:
                raw[spec_index] = float(continuous_samples[row_index, continuous_col])
                continuous_col += 1
        raw_settings.append(raw)
    return raw_settings


def sample_raw_settings(
    specs: _Sequence[_ParamSpec],
    n: int,
    *,
    seed: int = 0,
    full_grid: bool = False,
) -> list[_np.ndarray]:
    specs = tuple(specs)
    continuous_specs = tuple(spec for spec in specs if spec.type == "continuous")
    rng = _np.random.default_rng(seed)
    continuous_seed = int(rng.integers(0, 2**32))
    switch_seed = int(rng.integers(0, 2**32))
    continuous_samples = _scale_continuous_samples(
        latin_hypercube_unit(n, len(continuous_specs), seed=continuous_seed),
        continuous_specs,
    )

    if not full_grid:
        switch_assignments = _stratified_switch_assignments(specs, n, seed=switch_seed)
        return _assemble_raw_settings(continuous_samples, switch_assignments, specs)

    combos = _switch_combinations(specs)
    tiled_continuous = _np.repeat(continuous_samples, len(combos), axis=0)
    repeated_switches = _np.asarray(combos, dtype=_np.int64)
    switch_assignments = _np.tile(repeated_switches, (n, 1))
    return _assemble_raw_settings(tiled_continuous, switch_assignments, specs)


@_dataclass(frozen=True)
class PlannedCapture:
    """
    One reamp to be recorded: a knob setting plus where its output WAV will live
    (relative to the project folder).
    """

    index: int
    split: _Literal["train", "validation"]
    params: dict[str, _Any]
    y_path: str


def _plan_split(
    split: _Literal["train", "validation"],
    specs: _Sequence[_ParamSpec],
    n: int,
    *,
    seed: int,
) -> list[PlannedCapture]:
    if n == 0:
        return []
    raw_settings = sample_raw_settings(specs, n, seed=seed)
    abbreviations = _abbreviate_param_names([spec.name for spec in specs])
    planned = []
    for index, raw in enumerate(raw_settings):
        quantized = _quantize_to_capture_grid(raw, specs, default_step=_DEFAULT_KNOB_STEP)
        params = _decode_named_params(quantized, specs)
        filename = _make_capture_y_path(f"{split}_{index:03d}_", params, abbreviations)
        planned.append(
            PlannedCapture(
                index=index,
                split=split,
                params=params,
                y_path=f"{CAPTURES_DIRNAME}/{filename}",
            )
        )
    return planned


def plan_captures(
    knobs: _Sequence[_KnobSpec],
    *,
    n_train: int,
    n_validation: int,
    seed: int = 0,
) -> tuple[list[PlannedCapture], list[PlannedCapture]]:
    """
    Generate the full capture plan: ``n_train`` Latin-Hypercube training settings and
    ``n_validation`` held-out validation settings (a separate LHS stream derived from
    the same seed), each snapped to its knob's capture grid.
    """
    knobs = _validate_knobs(knobs)
    if n_train <= 0:
        raise ValueError(f"n_train must be positive; got {n_train}")
    if n_validation < 0:
        raise ValueError(f"n_validation must be non-negative; got {n_validation}")
    specs = tuple(knob.to_param_spec() for knob in knobs)
    train = _plan_split("train", specs, n_train, seed=seed)
    validation = _plan_split(
        "validation", specs, n_validation, seed=(seed + VALIDATION_SEED_OFFSET) % 2**32
    )
    return train, validation
