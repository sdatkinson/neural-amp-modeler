"""
Generate a PANAMA-style starter capture set as a parametric ``data.json`` skeleton.

Adapted from PANAMA (Parametric Active-learning for Neural Amp Modeling Assistance),
arXiv:2509.26564v1. The Latin-Hypercube starter-set idea and the active-learning
capture workflow are due to the PANAMA authors.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nam.capture.planner import VALIDATION_SEED_OFFSET
from nam.capture.planner import plan_unique_splits
from nam.capture.planner import sample_raw_settings
from nam.models.parametric import ParamSpec
from nam.models.parametric import abbreviate_param_names
from nam.models.parametric import decode_named_params
from nam.models.parametric import make_capture_y_path
from nam.models.parametric import quantize_to_capture_grid


_DEFAULT_OUTPUT = Path("starter_data.json")
_DEFAULT_ROUND_TO_NEAREST = 0.5
# Default window length (samples) for both train and validation entries. Two constraints
# bound it:
#   * It must exceed the loss ``mask_first`` (8192 in the active-learning ConcatLSTM config);
#     otherwise the whole window is masked out of the loss and there are no loss samples.
#   * It is kept below one ConcatLSTM processing block (65535 samples). A longer window is
#     processed as several sequential LSTM calls, which crashes the LSTM kernel on Apple MPS
#     during the batch-size-1 validation forward.
_DEFAULT_NY = 32768
# Seed a small held-out validation split by default: an empty ``validation`` list is a
# list that ``nam.data.init_dataset`` routes through ``ConcatDataset``, which raises on an
# empty dataset list, so the generated config would not be loadable by the parametric
# training path. A fixed starter holdout also matches the plan's default (see D-plan Open
# Question 1). Pass ``n_validation=0`` to opt out (e.g. when merging into another config).
_DEFAULT_N_VALIDATION = 2
_DEFAULT_VALIDATION_Y_PATH_PREFIX = "starter_val_"
# Validation captures are a tail of unseen audio: start near the end and slice that tail
# into ``_DEFAULT_NY`` windows. A fixed ny (rather than one EOF-length window) keeps each
# validation window to a single ConcatLSTM block; see ``_DEFAULT_NY``.
_DEFAULT_VALIDATION_START_SECONDS = -9.0
_DEFAULT_VALIDATION_STOP_SECONDS: float | None = None
_DEFAULT_VALIDATION_NY: int | None = _DEFAULT_NY
_VALIDATION_SEED_OFFSET = VALIDATION_SEED_OFFSET


def _load_param_specs(model_config_path: Path) -> tuple[ParamSpec, ...]:
    with model_config_path.open() as fp:
        model_config = json.load(fp)
    try:
        raw_specs = model_config["net"]["config"]["params"]
    except KeyError as exc:
        raise ValueError(
            "Model config must define net.config.params for starter-set generation"
        ) from exc
    specs = tuple(ParamSpec.from_dict(spec) for spec in raw_specs)
    if len(specs) == 0:
        raise ValueError("Model config net.config.params must contain at least one ParamSpec")
    return specs


def _load_loss_mask_first(model_config_path: Path) -> int:
    with model_config_path.open() as fp:
        model_config = json.load(fp)
    return int(model_config.get("loss", {}).get("mask_first", 0))


def _decode_capture_params(
    raw: np.ndarray,
    specs: Sequence[ParamSpec],
    *,
    round_to_nearest: float | None,
) -> dict[str, Any]:
    # Quantize continuous values to the realizable knob grid before decoding so the
    # recorded params equal the setting a human can actually dial (D5). The grid logic
    # lives in the shared bridge helper, so the starter (Task 3) and the AL proposals
    # (Task 6) stay on one grid; here we only choose whether to apply it.
    if round_to_nearest is None:
        return decode_named_params(raw, specs)
    return decode_named_params(
        quantize_to_capture_grid(raw, specs, default_step=round_to_nearest),
        specs,
    )


def _entries_from_params(
    param_dicts: Sequence[dict[str, Any]],
    specs: Sequence[ParamSpec],
    *,
    y_path_prefix: str,
    start_seconds: float,
    stop_seconds: float | None,
    ny: int | None,
) -> list[dict[str, Any]]:
    abbreviations = abbreviate_param_names([spec.name for spec in specs])
    entries = []
    for params in param_dicts:
        entries.append(
            {
                "y_path": make_capture_y_path(y_path_prefix, params, abbreviations),
                "params": params,
                "start_seconds": start_seconds,
                "stop_seconds": stop_seconds,
                "ny": ny,
            }
        )
    return entries


def _raw_param_dicts(
    raw_settings: list[np.ndarray],
    specs: Sequence[ParamSpec],
    *,
    round_to_nearest: float | None,
) -> list[dict[str, Any]]:
    return [
        _decode_capture_params(raw, specs, round_to_nearest=round_to_nearest)
        for raw in raw_settings
    ]


def build_starter_data(
    specs: Sequence[ParamSpec],
    *,
    n: int,
    input_wav: str = "input.wav",
    seed: int = 0,
    full_grid: bool = False,
    y_path_prefix: str = "starter_",
    round_to_nearest: float | None = _DEFAULT_ROUND_TO_NEAREST,
    start_seconds: float = 10.0,
    stop_seconds: float | None = -9.0,
    ny: int | None = 32768,
    n_validation: int = _DEFAULT_N_VALIDATION,
    validation_y_path_prefix: str = _DEFAULT_VALIDATION_Y_PATH_PREFIX,
    validation_start_seconds: float = _DEFAULT_VALIDATION_START_SECONDS,
    validation_stop_seconds: float | None = _DEFAULT_VALIDATION_STOP_SECONDS,
    validation_ny: int | None = _DEFAULT_VALIDATION_NY,
) -> dict[str, Any]:
    if n_validation < 0:
        raise ValueError(f"n_validation must be non-negative; got {n_validation}")

    if round_to_nearest is not None and not full_grid:
        # Default path: a capture grid is defined (rounding on) and switches are drawn
        # stratified, so distinct settings are well-defined. Share the app planner so the
        # starter set is deduplicated and the validation split is held out of train, from
        # one implementation.
        train_params, validation_params = plan_unique_splits(
            specs,
            n_train=n,
            n_validation=n_validation,
            seed=seed,
            default_step=round_to_nearest,
        )
    else:
        # Raw modes with no dedup: ``round_to_nearest is None`` has no grid to snap to (so
        # every continuous draw is effectively unique), and ``full_grid`` deliberately
        # crosses each continuous draw with every switch combination (duplicate switch rows
        # are the point). Validation stays a separate, reproducible LHS stream.
        train_params = _raw_param_dicts(
            sample_raw_settings(specs, n, seed=seed, full_grid=full_grid),
            specs,
            round_to_nearest=round_to_nearest,
        )
        validation_params = []
        if n_validation > 0:
            validation_seed = (seed + _VALIDATION_SEED_OFFSET) % 2**32
            validation_params = _raw_param_dicts(
                sample_raw_settings(specs, n_validation, seed=validation_seed, full_grid=False),
                specs,
                round_to_nearest=round_to_nearest,
            )

    train = _entries_from_params(
        train_params,
        specs,
        y_path_prefix=y_path_prefix,
        start_seconds=start_seconds,
        stop_seconds=stop_seconds,
        ny=ny,
    )
    validation: list[dict[str, Any]] = []
    if n_validation > 0:
        validation = _entries_from_params(
            validation_params,
            specs,
            y_path_prefix=validation_y_path_prefix,
            start_seconds=validation_start_seconds,
            stop_seconds=validation_stop_seconds,
            ny=validation_ny,
        )

    return {
        "type": "parametric",
        "common": {"x_path": input_wav, "delay": 0},
        "train": train,
        "validation": validation,
    }


def _format_entry_lines(entries: list[dict[str, Any]]) -> list[str]:
    lines = []
    for index, entry in enumerate(entries, start=1):
        settings = ", ".join(
            f"{name}={value}" for name, value in entry["params"].items()
        )
        lines.append(f"{index}. {entry['y_path']} -> {settings}")
    return lines


def format_capture_checklist(data_config: dict[str, Any]) -> str:
    lines = ["Capture checklist:", "Train:"]
    lines.extend(_format_entry_lines(data_config["train"]))
    if data_config["validation"]:
        lines.append("Validation (held out):")
        lines.extend(_format_entry_lines(data_config["validation"]))
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PANAMA-style starter capture set as a parametric data.json skeleton."
        )
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to a model config JSON with net.config.params.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of Latin-Hypercube continuous draws to generate.",
    )
    parser.add_argument(
        "--input-wav",
        default="input.wav",
        help="Value to write into common.x_path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Where to write the generated parametric data.json skeleton.",
    )
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help=(
            "Cross each continuous draw with every switch combination instead of using "
            "balanced per-switch assignments."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--y-path-prefix",
        default="starter_",
        help="Placeholder output filename prefix, before the param-encoded stem.",
    )
    parser.add_argument(
        "--no-rounding",
        action="store_true",
        help="Disable the default continuous-parameter rounding to the nearest 0.5.",
    )
    parser.add_argument(
        "--n-validation",
        type=int,
        default=_DEFAULT_N_VALIDATION,
        help=(
            "Number of held-out validation settings to seed (a separate LHS stream). "
            "Use 0 to emit an empty validation split (not loadable on its own)."
        ),
    )
    parser.add_argument(
        "--validation-y-path-prefix",
        default=_DEFAULT_VALIDATION_Y_PATH_PREFIX,
        help="Placeholder output filename prefix for validation captures.",
    )
    parser.add_argument(
        "--validation-start-seconds",
        type=float,
        default=_DEFAULT_VALIDATION_START_SECONDS,
        help="Default start_seconds for each emitted validation entry.",
    )
    parser.add_argument(
        "--validation-stop-seconds",
        type=float,
        default=_DEFAULT_VALIDATION_STOP_SECONDS,
        help="Default stop_seconds for each validation entry (omit for end-of-file).",
    )
    parser.add_argument(
        "--validation-ny",
        type=int,
        default=_DEFAULT_VALIDATION_NY,
        help=(
            "Default ny for each validation entry. Keep it below one ConcatLSTM block "
            "(65535) so the validation forward does not crash the LSTM on Apple MPS."
        ),
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=10.0,
        help="Default start_seconds for each emitted training entry.",
    )
    parser.add_argument(
        "--stop-seconds",
        type=float,
        default=-9.0,
        help="Default stop_seconds for each emitted training entry.",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=_DEFAULT_NY,
        help=(
            "Default ny for each emitted training entry. Must exceed the model's loss "
            "mask_first, or the whole window is masked from the loss."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    specs = _load_param_specs(args.model_config)
    mask_first = _load_loss_mask_first(args.model_config)
    for flag, value in (("--ny", args.ny), ("--validation-ny", args.validation_ny)):
        if value is not None and value <= mask_first:
            raise SystemExit(
                f"{flag}={value} must exceed the model's loss mask_first={mask_first}; "
                "otherwise the whole window is masked from the loss and training has no "
                "loss samples."
            )
    data_config = build_starter_data(
        specs,
        n=args.n,
        input_wav=args.input_wav,
        seed=args.seed,
        full_grid=args.full_grid,
        y_path_prefix=args.y_path_prefix,
        round_to_nearest=None if args.no_rounding else _DEFAULT_ROUND_TO_NEAREST,
        start_seconds=args.start_seconds,
        stop_seconds=args.stop_seconds,
        ny=args.ny,
        n_validation=args.n_validation,
        validation_y_path_prefix=args.validation_y_path_prefix,
        validation_start_seconds=args.validation_start_seconds,
        validation_stop_seconds=args.validation_stop_seconds,
        validation_ny=args.validation_ny,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fp:
        json.dump(data_config, fp, indent=4)
        fp.write("\n")

    print(
        f"Wrote {len(data_config['train'])} train + "
        f"{len(data_config['validation'])} validation starter settings to {args.output}"
    )
    print(format_capture_checklist(data_config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
