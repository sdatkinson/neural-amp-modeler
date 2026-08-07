"""
Active-learning capture-selection driver (one invocation == one round).

Adapted from PANAMA (Parametric Active-learning for Neural Amp Modeling Assistance),
arXiv:2509.26564v1. The round-based query-by-committee loop and the disagreement
acquisition objective are due to the PANAMA authors; this CLI ports their multi-GPU
driver to this repo's single-GPU, device-agnostic, parametric-``data.json`` workflow.

Flow per round:
  1. (round 0) load --data-config; (round > 0) load the previous round's
     aggregated_data_config_{i-1}.json from --output-dir.
  2. train a serial ConcatLSTM ensemble (or reuse --ckpts).
  3. search for high-disagreement control settings and select the top proposals.
  4. write proposed_captures_round_{i}.json + aggregated_data_config_{i}.json, then STOP.

The human then reamps the fixed input clip at each proposed setting, fills in the
placeholder y_paths in the aggregated config, and reruns with --round-idx i+1.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from lightning_fabric.utilities.warnings import PossibleUserWarning

from nam.train.active_learning import RoundResult, run_round
from nam.util import filter_warnings


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open() as fp:
        return json.load(fp)


def _nonneg_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {parsed}")
    return parsed


def _resolve_data_config_path(args: argparse.Namespace) -> Path:
    """Round 0 reads --data-config; later rounds read the previous aggregated config."""
    if args.round_idx == 0:
        if args.data_config is None:
            raise SystemExit(
                "--data-config is required for --round-idx 0 (the starter parametric "
                "data.json)"
            )
        return args.data_config
    if args.data_config is not None:
        # Explicit override: the previous round's aggregated config (with its earlier
        # captures) is bypassed, so warn rather than silently dropping that history.
        warnings.warn(
            f"--data-config was given for round {args.round_idx}; the previous round's "
            f"aggregated_data_config_{args.round_idx - 1}.json will be ignored, dropping "
            "any captures it accumulated. Pass it explicitly only if that is intended."
        )
        return args.data_config
    prev = args.output_dir / f"aggregated_data_config_{args.round_idx - 1}.json"
    if not prev.exists():
        raise SystemExit(
            f"Expected the previous round's aggregated config at {prev}; pass "
            "--data-config explicitly or run the previous round first"
        )
    return prev


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one PANAMA-style active-learning round: train a ConcatLSTM ensemble, find "
            "high-disagreement control settings, and propose the next captures."
        )
    )
    parser.add_argument(
        "--round-idx",
        type=_nonneg_int,
        required=True,
        help="Zero-based index of this round.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("active_learning"),
        help="Directory for checkpoints, proposals, and aggregated configs.",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=None,
        help=(
            "Parametric data.json. Required for --round-idx 0; later rounds default to "
            "<output-dir>/aggregated_data_config_{i-1}.json."
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Model config JSON with net.name=ConcatLSTM and net.config.params.",
    )
    parser.add_argument(
        "--learning-config",
        type=Path,
        required=True,
        help="Learning config JSON (trainer / dataloaders) for the ensemble members.",
    )
    parser.add_argument(
        "--g-opt-input-wav",
        type=Path,
        default=None,
        help="Input clip to reamp during g-opt. Defaults to data common.x_path.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=4,
        help="Number of ensemble members.",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=None,
        help=(
            "Also controls how many ensemble members train concurrently, and how many "
            "g-opt (disagreement search) workers run concurrently. Default (unset) is "
            "serial on one device, or one member/worker per GPU on a multi-GPU CUDA box. "
            "For ensemble training, an explicit count may over-subscribe a single GPU "
            "(e.g. --max-workers 4 on a large-VRAM card with a small batch size); the "
            "memory headroom is yours to ensure. g-opt is never over-subscribed: it always "
            "caps at one worker per GPU, staying serial on a single GPU/MPS/CPU regardless "
            "of this value."
        ),
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=8,
        help="Random latent inits per switch combination during g-opt.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Adam ascent steps per restart during g-opt.",
    )
    parser.add_argument(
        "--max-per-round",
        type=int,
        default=10,
        help="Maximum number of proposed captures to emit this round.",
    )
    parser.add_argument(
        "--g-opt-ny",
        type=int,
        default=4096,
        help="g-opt window length (samples per chunk).",
    )
    parser.add_argument(
        "--g-opt-batch-size",
        type=int,
        default=64,
        help="g-opt batch size (chunks per step).",
    )
    parser.add_argument(
        "--g-opt-lr",
        type=float,
        default=0.05,
        help="Adam learning rate for the g-opt latent ascent (PANAMA uses 0.02).",
    )
    parser.add_argument(
        "--use-mel",
        action="store_true",
        help="Add PANAMA's multi-resolution mel-variance term to the disagreement score.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for ensemble member seeds and g-opt latent inits.",
    )
    parser.add_argument(
        "--ckpts",
        type=Path,
        nargs="+",
        default=None,
        help="Reuse these member checkpoints instead of training a new ensemble.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the accepted-capture distribution plot.",
    )
    return parser.parse_args()


def _print_summary(result: RoundResult, data_config_path: Path) -> None:
    print()
    print(f"Round {result.round_idx} complete (input data config: {data_config_path}).")
    print(f"  Ensemble checkpoints ({len(result.checkpoint_paths)}):")
    for path in result.checkpoint_paths:
        print(f"    - {path}")
    print(f"  Candidates searched: {len(result.candidates)}")
    print(f"  Proposed captures:   {len(result.proposals)} -> {result.proposals_path}")
    print(f"  Aggregated config:   {result.aggregated_config_path}")
    print()
    print(
        "Next: record the proposed captures (reamp the fixed input clip at each setting), "
        f"fill in the placeholder y_paths in {result.aggregated_config_path}, then rerun "
        f"with --round-idx {result.round_idx + 1}."
    )


def main() -> int:
    args = _parse_args()
    data_config_path = _resolve_data_config_path(args)

    data_config = _load_json(data_config_path)
    model_config = _load_json(args.model_config)
    learning_config = _load_json(args.learning_config)

    with filter_warnings("ignore", category=PossibleUserWarning):
        result = run_round(
            round_idx=args.round_idx,
            output_dir=args.output_dir,
            data_config=data_config,
            model_config=model_config,
            learning_config=learning_config,
            g_opt_input_wav=args.g_opt_input_wav,
            ensemble_size=args.ensemble_size,
            num_restarts=args.num_restarts,
            num_steps=args.num_steps,
            max_per_round=args.max_per_round,
            g_opt_ny=args.g_opt_ny,
            g_opt_batch_size=args.g_opt_batch_size,
            g_opt_lr=args.g_opt_lr,
            use_mel=args.use_mel,
            seed=args.seed,
            checkpoint_paths=args.ckpts,
            plot=not args.no_plot,
            max_workers=args.max_workers,
        )

    _print_summary(result, data_config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
