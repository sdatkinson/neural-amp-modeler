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

import itertools as _itertools
import math as _math
from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Literal
from typing import Optional as _Optional

import numpy as _np

from ..models.parametric import abbreviate_param_names as _abbreviate_param_names
from ..models.parametric import decode_named_params as _decode_named_params
from ..models.parametric import make_capture_y_path as _make_capture_y_path
from ..models.parametric import quantize_to_capture_grid as _quantize_to_capture_grid
from ..models.parametric import switch_combinations as _switch_combinations
from ..models.parametric import ParamSpec as _ParamSpec
from .params import DEFAULT_KNOB_STEP as _DEFAULT_KNOB_STEP
from .params import gain_knob_index as _gain_knob_index
from .params import KnobSpec as _KnobSpec
from .params import validate_knobs as _validate_knobs


# Large, fixed offset so the validation draws are a different LHS stream than the train
# draws (held-out settings), while staying reproducible from the same seed.
VALIDATION_SEED_OFFSET = 2**31 - 1

CAPTURES_DIRNAME = "captures"

# Quantizing Latin-Hypercube draws to the realizable knob grid collapses many distinct
# continuous samples onto the same grid point, so raw draws produce duplicate settings
# (and validation draws that coincide with train ones). Every planned capture must be a
# distinct setting, so the split planner deduplicates on the quantized settings. When the
# grid is coarse enough that oversampling keeps colliding, it falls back to enumerating
# the grid directly. These bound that work.
_OVERSAMPLE_BATCHES = 12
# Above this many distinct grid points, never materialize the full grid: a grid this large
# is far finer than any realistic capture count, so oversampling fills without collisions
# and the enumeration fallback is unreachable in practice.
_ENUMERATION_CAP = 5_000_000


def _capture_grid_values(spec: _ParamSpec, default_step: float) -> list[float]:
    """
    The distinct continuous values reachable on ``spec``'s capture grid, i.e. every value
    :func:`quantize_to_capture_grid` can emit for it. Quantization snaps to the nearest
    multiple of ``step`` (measured from zero, matching the shared helper) and clamps into
    ``[min, max]``, so the reachable set is those in-range multiples plus the clamped
    endpoints.
    """
    step = spec.step if spec.step is not None else default_step
    lo = _math.floor(spec.min / step)
    hi = _math.ceil(spec.max / step)
    values = set()
    # ``round(v / step)`` for ``v`` in ``[min, max]`` lands in ``[lo, hi]``; the +/-1 pad
    # covers rounding at the endpoints. Out-of-range multiples clamp onto min/max.
    for k in range(lo - 1, hi + 2):
        clamped = min(max(k * step, spec.min), spec.max)
        values.add(round(clamped, 6))
    if getattr(spec, "avoid_zero", False):
        # Match quantize_to_capture_grid: zero is never a reachable capture setting for
        # an avoid-zero knob, so the enumeration fallback must not offer it either.
        values.discard(0.0)
    return sorted(values)


def _grid_capacity(specs: _Sequence[_ParamSpec], default_step: float) -> int:
    """Total number of distinct on-grid settings (switch combinations x continuous grid)."""
    capacity = 1
    for spec in specs:
        if spec.type == "switch":
            capacity *= spec.num_inputs
        else:
            capacity *= len(_capture_grid_values(spec, default_step))
    return capacity


def _enumerate_grid_params(
    specs: _Sequence[_ParamSpec],
    default_step: float,
    *,
    seed: int,
):
    """
    Yield every distinct on-grid setting as a decoded params dict, in a deterministic
    shuffled order. Only used as the coarse-grid fallback, guarded by ``_ENUMERATION_CAP``.
    """
    value_axes: list[list[float]] = []
    for spec in specs:
        if spec.type == "switch":
            value_axes.append([float(i) for i in range(spec.num_inputs)])
        else:
            value_axes.append(_capture_grid_values(spec, default_step))
    combos = list(_itertools.product(*value_axes))
    _np.random.default_rng(seed).shuffle(combos)
    for combo in combos:
        yield _decode_named_params(_np.asarray(combo, dtype=_np.float64), specs)


def _settings_key(
    params: dict[str, _Any], specs: _Sequence[_ParamSpec]
) -> tuple[_Any, ...]:
    return tuple(params[spec.name] for spec in specs)


def settings_sort_key(
    params: dict[str, _Any], names: _Sequence[str]
) -> tuple[tuple[int, _Any], ...]:
    """
    Ordering key for one capture setting: its values in knob order. Sorting a list of
    settings by this walks the last knob through its range for every value of the knob
    before it, and so on, which is the order that asks the user to move the fewest knobs
    between one capture and the next.

    Each value is paired with a type rank so a numeric knob and a switch's label can never
    be compared against each other. Within one knob every setting has the same type, so the
    rank is constant down that column and the value is what orders it.
    """
    key: list[tuple[int, _Any]] = []
    for name in names:
        value = params[name]
        try:
            key.append((0, float(value)))
        except (TypeError, ValueError):
            key.append((1, str(value)))
    return tuple(key)


def sort_settings(
    param_dicts: _Sequence[dict[str, _Any]], names: _Sequence[str]
) -> list[dict[str, _Any]]:
    """Capture settings in ascending :func:`settings_sort_key` order."""
    return sorted(param_dicts, key=lambda params: settings_sort_key(params, names))


def sample_unique_settings(
    specs: _Sequence[_ParamSpec],
    n: int,
    *,
    seed: int,
    default_step: float,
    exclude: _Optional[set[tuple[_Any, ...]]] = None,
) -> list[dict[str, _Any]]:
    """
    Draw ``n`` distinct on-grid settings as decoded params dicts, none of them in
    ``exclude`` (used to hold one split out of another). Each draw is quantized to the
    capture grid (nearest ``default_step``, per-``ParamSpec`` ``step`` honored) before the
    uniqueness test, so two draws that snap to the same knob positions count as one.

    Shared by the capture app (:func:`plan_captures`) and the starter-settings script so
    both emit unique, held-out settings from one implementation.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    seen = set() if exclude is None else set(exclude)
    ordered: list[dict[str, _Any]] = []
    if n == 0:
        return ordered

    def _consume(raw_settings: list[_np.ndarray]) -> bool:
        for raw in raw_settings:
            quantized = _quantize_to_capture_grid(raw, specs, default_step=default_step)
            params = _decode_named_params(quantized, specs)
            key = _settings_key(params, specs)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(params)
            if len(ordered) == n:
                return True
        return False

    # The first batch reproduces the historical LHS stream exactly, so a plan with no grid
    # collisions is byte-for-byte identical to the pre-deduplication behavior.
    if _consume(sample_raw_settings(specs, n, seed=seed)):
        return ordered

    # Collisions dropped us short. Oversample with fresh, seed-derived LHS batches to fill
    # in the gaps while preserving space-filling coverage.
    rng = _np.random.default_rng(seed)
    batch = max(n, 256)
    for _ in range(_OVERSAMPLE_BATCHES):
        if _consume(sample_raw_settings(specs, batch, seed=int(rng.integers(0, 2**32)))):
            return ordered

    # The grid is coarse enough that random draws keep colliding. Fill the remainder
    # deterministically from the full grid so we never loop indefinitely.
    capacity = _grid_capacity(specs, default_step)
    if capacity > _ENUMERATION_CAP:
        raise ValueError(
            f"Unable to draw {n} unique capture settings after oversampling. The knob "
            "grid is too large to enumerate; widen the sampling or reduce the count."
        )
    for params in _enumerate_grid_params(
        specs, default_step, seed=int(rng.integers(0, 2**32))
    ):
        key = _settings_key(params, specs)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(params)
        if len(ordered) == n:
            return ordered

    raise ValueError(
        f"Cannot plan {n} unique settings: the knob grid yields only {len(ordered)} "
        "distinct settings beyond those already used. Reduce the capture count, widen a "
        "knob's range, or use a finer step."
    )


def plan_unique_splits(
    specs: _Sequence[_ParamSpec],
    *,
    n_train: int,
    n_validation: int,
    seed: int,
    default_step: float,
) -> tuple[list[dict[str, _Any]], list[dict[str, _Any]]]:
    """
    Draw ``n_train`` + ``n_validation`` distinct on-grid settings, with the validation
    settings held out of the train settings. Validation is a separate, reproducible LHS
    stream (see :data:`VALIDATION_SEED_OFFSET`). Returns the two lists of decoded params
    dicts.

    Shared by :func:`plan_captures` (capture app) and ``build_starter_data`` (starter
    script) so both deduplicate and hold out validation the same way.
    """
    if n_train < 0:
        raise ValueError(f"n_train must be non-negative; got {n_train}")
    if n_validation < 0:
        raise ValueError(f"n_validation must be non-negative; got {n_validation}")

    # Every train and validation setting must be distinct, and validation must be held out
    # from train. That is only possible if the knob grid has at least that many distinct
    # settings; fail loudly up front rather than after sampling work.
    required = n_train + n_validation
    capacity = _grid_capacity(specs, default_step)
    if required > capacity:
        raise ValueError(
            f"Cannot plan {n_train} train + {n_validation} validation unique settings: the "
            f"knob grid yields only {capacity} distinct settings. Reduce the capture "
            "counts, widen a knob's range, or use a finer step."
        )

    train = sample_unique_settings(specs, n_train, seed=seed, default_step=default_step)
    train_keys = {_settings_key(params, specs) for params in train}
    validation = sample_unique_settings(
        specs,
        n_validation,
        seed=(seed + VALIDATION_SEED_OFFSET) % 2**32,
        default_step=default_step,
        exclude=train_keys,
    )
    return train, validation


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
    split: Literal["train", "validation"]
    params: dict[str, _Any]
    y_path: str


def _planned_from_params(
    prefix: str,
    split: Literal["train", "validation"],
    specs: _Sequence[_ParamSpec],
    param_dicts: list[dict[str, _Any]],
    *,
    index_offset: int = 0,
    filename_start: int = 0,
) -> list[PlannedCapture]:
    """
    Turn decoded param dicts into planned captures. ``prefix`` is the filename stem prefix
    (``lhs``/``val_lhs``/``corner``); the filename counter starts at ``filename_start`` and
    the model ``index`` starts at ``index_offset``. These are decoupled so appended corners
    can continue the train index space (unique ``index``) while numbering their own files
    from ``corner_000`` (see :func:`plan_corner_captures`).
    """
    abbreviations = _abbreviate_param_names([spec.name for spec in specs])
    planned = []
    for offset, params in enumerate(param_dicts):
        filename = _make_capture_y_path(
            f"{prefix}_{filename_start + offset:03d}_", params, abbreviations
        )
        planned.append(
            PlannedCapture(
                index=index_offset + offset,
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
    the same seed), each snapped to its knob's capture grid. Every planned setting is
    distinct and validation is held out from train (see :func:`plan_unique_splits`).

    Each split is emitted in ascending knob order (see :func:`settings_sort_key`) so the
    user works through one section at a time with the knobs moving as little as possible
    between consecutive captures. Sorting within a split does not touch which settings are
    drawn, only the order they are recorded in.
    """
    knobs = _validate_knobs(knobs)
    if n_train <= 0:
        raise ValueError(f"n_train must be positive; got {n_train}")
    if n_validation < 0:
        raise ValueError(f"n_validation must be non-negative; got {n_validation}")
    specs = tuple(knob.to_param_spec() for knob in knobs)

    train_params, validation_params = plan_unique_splits(
        specs,
        n_train=n_train,
        n_validation=n_validation,
        seed=seed,
        default_step=_DEFAULT_KNOB_STEP,
    )
    names = [spec.name for spec in specs]
    train = _planned_from_params(
        "lhs", "train", specs, sort_settings(train_params, names)
    )
    validation = _planned_from_params(
        "val_lhs", "validation", specs, sort_settings(validation_params, names)
    )
    return train, validation


def _corner_raw_vector(
    specs: _Sequence[_ParamSpec], highs: _Sequence[bool]
) -> _np.ndarray:
    """
    Raw (pre-quantization) param vector for one corner: each entry is the knob's max when
    ``highs[i]`` else its min. Switches use their last index for ``max`` and 0 for ``min``
    (the capture app only produces continuous knobs, but this keeps the vector decodable
    for any spec set).
    """
    raw = _np.empty(len(specs), dtype=_np.float64)
    for i, spec in enumerate(specs):
        if spec.type == "switch":
            raw[i] = float(spec.num_inputs - 1) if highs[i] else 0.0
        else:
            raw[i] = spec.max if highs[i] else spec.min
    return raw


# Below this many knobs the even-parity half fraction stops being a usable design. Its
# defining relation is the product of every factor, so K factors give resolution K: at K=3
# main effects alias two-factor interactions, and at K<=2 they alias each other. The full
# factorial is at most 8 vertices there, so it is cheap enough to take outright.
_MIN_HALF_FRACTION_KNOBS = 4


def _corner_vertices(n: int) -> list[tuple[bool, ...]]:
    """
    The min/max patterns (True = knob at max) to place corners on, over ``n`` knobs.

    For four or more knobs this is the even-parity half fraction -- every pattern with an
    even number of maxes, 2**(n-1) of them. That is the standard 2**(n-1) fractional
    factorial, and it keeps all n knobs mutually orthogonal, so the corner set can tell
    each knob's extreme apart from every other knob's. This is a set, not an order:
    :func:`plan_corner_captures` decides the order the corners are captured in.
    """
    if n < _MIN_HALF_FRACTION_KNOBS:
        return list(_itertools.product((False, True), repeat=n))
    return [
        combination
        for combination in _itertools.product((False, True), repeat=n)
        if sum(combination) % 2 == 0
    ]


def _corner_high_flag_sets(
    n: int, gain_index: _Optional[int]
) -> list[list[bool]]:
    """
    The per-corner max/min pattern (True = knob at max) before dedup: the vertex set from
    :func:`_corner_vertices`, crossed with the gain knob's min and max when one is marked
    so every tone-stack extreme is seen at both ends of the drive range. Duplicates that
    collapse for small knob counts are removed later by :func:`corner_settings`.
    """
    if gain_index is None:
        return [list(vertex) for vertex in _corner_vertices(n)]

    others = [i for i in range(n) if i != gain_index]
    flag_sets: list[list[bool]] = []
    for vertex in _corner_vertices(len(others)):
        for gain_high in (False, True):
            flags = [False] * n
            flags[gain_index] = gain_high
            for position, index in enumerate(others):
                flags[index] = vertex[position]
            flag_sets.append(flags)
    return flag_sets


def corner_settings(
    specs: _Sequence[_ParamSpec],
    *,
    gain_index: _Optional[int] = None,
    default_step: float = _DEFAULT_KNOB_STEP,
) -> list[dict[str, _Any]]:
    """
    Decoded param dicts for the initial "corner" captures: the knob-range extremes that
    bound the amp's behavior (see :func:`_corner_vertices` for the pattern, and
    :func:`_corner_high_flag_sets` for how a marked gain/drive knob is crossed with it).
    Values are quantized to the capture grid (honoring each knob's ``step`` and
    ``avoid_zero``, so an avoid-zero gain knob's min corner nudges off zero), then
    deduplicated so the small knob counts that collapse (a single knob yields only
    min/max) don't repeat a setting.
    """
    specs = tuple(specs)
    seen: set[tuple[_Any, ...]] = set()
    settings: list[dict[str, _Any]] = []
    for highs in _corner_high_flag_sets(len(specs), gain_index):
        raw = _corner_raw_vector(specs, highs)
        quantized = _quantize_to_capture_grid(raw, specs, default_step=default_step)
        params = _decode_named_params(quantized, specs)
        key = _settings_key(params, specs)
        if key in seen:
            continue
        seen.add(key)
        settings.append(params)
    return settings


def corner_capture_count(
    knobs: _Sequence[_KnobSpec], *, default_step: float = _DEFAULT_KNOB_STEP
) -> int:
    """How many distinct corner captures ``knobs`` will produce."""
    knobs = _validate_knobs(knobs)
    specs = tuple(knob.to_param_spec() for knob in knobs)
    return len(
        corner_settings(
            specs, gain_index=_gain_knob_index(knobs), default_step=default_step
        )
    )


def plan_corner_captures(
    knobs: _Sequence[_KnobSpec],
    *,
    exclude: _Optional[set[tuple[_Any, ...]]] = None,
    index_offset: int = 0,
    filename_start: int = 0,
) -> tuple[list[PlannedCapture], int]:
    """
    Plan the corner captures for ``knobs`` as pending ``train`` captures, skipping any whose
    setting already appears in ``exclude`` (the LHS points and any corners already planned).

    The corners are emitted as their own section, in ascending knob order (see
    :func:`settings_sort_key`), so they read as one sorted block wherever they sit in the
    plan. Which corners are captured does not depend on the order.

    Returns the planned corner captures and the number of distinct corners skipped because
    they duplicate an existing setting, so the caller can tell the user (they still get a
    useful boundary set; the overlap just means fewer *new* captures). ``index_offset`` and
    ``filename_start`` continue the train index space and the ``corner_NNN`` file numbering
    when appending to an existing plan.
    """
    knobs = _validate_knobs(knobs)
    specs = tuple(knob.to_param_spec() for knob in knobs)
    gain_index = _gain_knob_index(knobs)
    excluded = set() if exclude is None else set(exclude)

    unique: list[dict[str, _Any]] = []
    skipped = 0
    for params in corner_settings(
        specs, gain_index=gain_index, default_step=_DEFAULT_KNOB_STEP
    ):
        key = _settings_key(params, specs)
        if key in excluded:
            skipped += 1
            continue
        excluded.add(key)
        unique.append(params)

    planned = _planned_from_params(
        "corner",
        "train",
        specs,
        sort_settings(unique, [spec.name for spec in specs]),
        index_offset=index_offset,
        filename_start=filename_start,
    )
    return planned, skipped
