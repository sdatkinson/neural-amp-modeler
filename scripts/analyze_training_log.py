#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import tempfile
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from typing import Sequence

from tensorboard.backend.event_processing import event_accumulator


# The parametric trainer logs the diagonal of the seen/unseen grid: the training
# split (audio and control settings the model fit on) and the held-out validation
# split (unseen on both axes). Each metric is namespaced "<METRIC>/<bucket>".
_TRAIN_BUCKET = "ESR/seen_audio_seen_params"
_VALIDATION_BUCKET = "ESR/unseen_audio_unseen_params"
_PARAMETRIC_BUCKETS = (_TRAIN_BUCKET, _VALIDATION_BUCKET)
_DEFAULT_TOP_SPIKES = 3
_DEFAULT_SPIKE_RATIO_THRESHOLD = 10.0
_DEFAULT_HELD_OUT_REGRESSION_THRESHOLD = 1.5


@dataclass(frozen=True)
class ScalarPoint:
    epoch_index: int
    step: int
    value: float


@dataclass(frozen=True)
class ScalarSummary:
    tag: str
    count: int
    nonfinite_count: int
    best_index: int
    best_step: int
    best_value: float
    worst_index: int
    worst_step: int
    worst_value: float
    final_step: int
    final_value: float
    worst_points: tuple[ScalarPoint, ...]

    @property
    def all_nonfinite(self) -> bool:
        return self.nonfinite_count == self.count

    @property
    def min_to_final_ratio(self) -> float:
        if self.best_value == 0.0:
            return math.inf
        return self.final_value / self.best_value

    @property
    def worst_to_best_ratio(self) -> float:
        if self.best_value == 0.0:
            return math.inf
        return self.worst_value / self.best_value


@dataclass(frozen=True)
class RunSummary:
    label: str
    source: Path
    scalar_tags: tuple[str, ...]
    scalar_summaries: tuple[ScalarSummary, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize TensorBoard training logs from a raw event file, a "
            "directory containing one, or a zip archive."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more event files, event directories, or zip archives.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Optional label for each path. Repeat once per input path.",
    )
    parser.add_argument(
        "--top-spikes",
        type=int,
        default=_DEFAULT_TOP_SPIKES,
        help=(
            "Number of worst finite epochs to show for metrics whose worst/best "
            f"ratio exceeds --spike-ratio-threshold. Default: {_DEFAULT_TOP_SPIKES}."
        ),
    )
    parser.add_argument(
        "--spike-ratio-threshold",
        type=float,
        default=_DEFAULT_SPIKE_RATIO_THRESHOLD,
        help=(
            "Print spike details for a metric when worst/best is at least this "
            f"large. Default: {_DEFAULT_SPIKE_RATIO_THRESHOLD:g}."
        ),
    )
    parser.add_argument(
        "--held-out-regression-threshold",
        type=float,
        default=_DEFAULT_HELD_OUT_REGRESSION_THRESHOLD,
        help=(
            "Warn when held-out final/best ESR is at least this large. "
            f"Default: {_DEFAULT_HELD_OUT_REGRESSION_THRESHOLD:g}."
        ),
    )
    return parser.parse_args()


def _iter_event_files(path: Path) -> Iterator[Path]:
    if path.is_file() and path.name.startswith("events.out.tfevents."):
        yield path
        return
    if path.is_dir():
        for candidate in sorted(path.rglob("events.out.tfevents.*")):
            if candidate.is_file():
                yield candidate
        return
    if path.is_file() and path.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    name = Path(member).name
                    if not name.startswith("events.out.tfevents."):
                        continue
                    extracted = tmpdir_path / name
                    extracted.write_bytes(archive.read(member))
                    yield extracted
        return
    raise FileNotFoundError(f"Unsupported log path: {path}")


def _load_event_accumulator(path: Path) -> event_accumulator.EventAccumulator:
    manager = (
        tempfile.TemporaryDirectory()
        if path.is_file() and path.suffix == ".zip"
        else nullcontext()
    )
    with manager as tmpdir:
        if tmpdir is None:
            event_path = next(_iter_event_files(path), None)
        else:
            event_path = None
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    name = Path(member).name
                    if not name.startswith("events.out.tfevents."):
                        continue
                    extracted = Path(tmpdir) / name
                    extracted.write_bytes(archive.read(member))
                    event_path = extracted
                    break
        if event_path is None:
            raise FileNotFoundError(f"No TensorBoard event file found in {path}")
        accumulator = event_accumulator.EventAccumulator(str(event_path))
        accumulator.Reload()
        return accumulator


def _summarize_tag(
    accumulator: event_accumulator.EventAccumulator, tag: str, top_spikes: int
) -> ScalarSummary:
    values = accumulator.Scalars(tag)
    finite_indices = [i for i in range(len(values)) if math.isfinite(values[i].value)]
    # Pick the best over finite points so a single inf/nan epoch can't masquerade
    # as the minimum; fall back to index 0 when the whole series is non-finite.
    best_index = (
        min(finite_indices, key=lambda i: values[i].value)
        if finite_indices
        else 0
    )
    worst_index = (
        max(finite_indices, key=lambda i: values[i].value)
        if finite_indices
        else 0
    )
    worst_points = tuple(
        ScalarPoint(
            epoch_index=i,
            step=values[i].step,
            value=values[i].value,
        )
        for i in sorted(
            finite_indices,
            key=lambda j: values[j].value,
            reverse=True,
        )[:top_spikes]
    )
    best = values[best_index]
    worst = values[worst_index]
    final = values[-1]
    return ScalarSummary(
        tag=tag,
        count=len(values),
        nonfinite_count=len(values) - len(finite_indices),
        best_index=best_index,
        best_step=best.step,
        best_value=best.value,
        worst_index=worst_index,
        worst_step=worst.step,
        worst_value=worst.value,
        final_step=final.step,
        final_value=final.value,
        worst_points=worst_points,
    )


def _preferred_tags(scalar_tags: Sequence[str]) -> list[str]:
    scalar_tag_set = set(scalar_tags)
    if all(tag in scalar_tag_set for tag in _PARAMETRIC_BUCKETS):
        # Lead with the ESR buckets, then any per-bucket MSE/MRSTFT companions.
        bucketed: list[str] = list(_PARAMETRIC_BUCKETS)
        for metric in ("MSE", "MRSTFT"):
            for suffix in ("seen_audio_seen_params", "unseen_audio_unseen_params"):
                tag = f"{metric}/{suffix}"
                if tag in scalar_tag_set:
                    bucketed.append(tag)
        return bucketed

    preferred = [
        "ESR",
        "val_loss",
        "MSE",
        "MRSTFT",
        "ESR_packed_0",
        "ESR_packed_1",
    ]
    return [tag for tag in preferred if tag in scalar_tag_set]


def _summarize_run(
    path: Path, label: str | None = None, top_spikes: int = _DEFAULT_TOP_SPIKES
) -> RunSummary:
    accumulator = _load_event_accumulator(path)
    scalar_tags = tuple(accumulator.Tags()["scalars"])
    tags = _preferred_tags(scalar_tags)
    summaries = tuple(
        _summarize_tag(accumulator, tag, top_spikes=top_spikes) for tag in tags
    )
    return RunSummary(
        label=label or path.stem,
        source=path,
        scalar_tags=scalar_tags,
        scalar_summaries=summaries,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = sum((x - x_mean) ** 2 for x in xs)
    y_denom = sum((y - y_mean) ** 2 for y in ys)
    if x_denom == 0.0 or y_denom == 0.0:
        return math.nan
    return numerator / math.sqrt(x_denom * y_denom)


def _format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.8g}"


def _format_ratio(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.2f}"


def _print_run_summary(
    run: RunSummary,
    path: Path,
    spike_ratio_threshold: float,
    held_out_regression_threshold: float,
) -> None:
    print(f"# {run.label}")
    print(f"source: {path}")
    print(f"scalar_tags: {len(run.scalar_tags)}")
    for summary in run.scalar_summaries:
        if summary.all_nonfinite:
            print(
                f"{summary.tag}: ALL NON-FINITE ({summary.nonfinite_count}/"
                f"{summary.count} epochs inf/nan)"
            )
            continue
        nonfinite_note = (
            f" nonfinite={summary.nonfinite_count}/{summary.count}"
            if summary.nonfinite_count
            else ""
        )
        print(
            f"{summary.tag}: best epoch_idx={summary.best_index} "
            f"step={summary.best_step} value={_format_number(summary.best_value)} "
            f"worst epoch_idx={summary.worst_index} step={summary.worst_step} "
            f"value={_format_number(summary.worst_value)} "
            f"worst/best={_format_ratio(summary.worst_to_best_ratio)} "
            f"final={_format_number(summary.final_value)} "
            f"final/best={_format_ratio(summary.min_to_final_ratio)}"
            f"{nonfinite_note}"
        )
    spike_summaries = [
        summary
        for summary in run.scalar_summaries
        if not summary.all_nonfinite
        and summary.worst_to_best_ratio >= spike_ratio_threshold
    ]
    if spike_summaries:
        print("spikes:")
        for summary in sorted(
            spike_summaries,
            key=lambda item: item.worst_to_best_ratio,
            reverse=True,
        ):
            print(
                f"  {summary.tag}: worst/best="
                f"{_format_ratio(summary.worst_to_best_ratio)}"
            )
            for point in summary.worst_points:
                ratio = (
                    math.inf
                    if summary.best_value == 0.0
                    else point.value / summary.best_value
                )
                print(
                    f"    epoch_idx={point.epoch_index} step={point.step} "
                    f"value={_format_number(point.value)} "
                    f"ratio_to_best={_format_ratio(ratio)}"
                )
    scalar_lookup = {summary.tag: summary for summary in run.scalar_summaries}
    if all(tag in scalar_lookup for tag in _PARAMETRIC_BUCKETS):
        validation = scalar_lookup[_VALIDATION_BUCKET]
        train = scalar_lookup[_TRAIN_BUCKET]
        print("parametric_insight:")
        print(
            "  best deployment (held-out) epoch_idx="
            f"{validation.best_index} ESR={validation.best_value:.6g}"
        )
        if train.all_nonfinite:
            print(
                "  WARNING: training-bucket ESR is non-finite every epoch -- a "
                "(near-)silent batch is dividing the per-batch ESR by ~zero."
            )
        elif train.worst_to_best_ratio >= spike_ratio_threshold:
            print(
                "  WARNING: training-bucket ESR spiked to "
                f"{_format_number(train.worst_value)} at epoch_idx="
                f"{train.worst_index}; worst/best="
                f"{_format_ratio(train.worst_to_best_ratio)}."
            )
        if validation.min_to_final_ratio >= held_out_regression_threshold:
            print(
                "  WARNING: held-out ESR worsened after the best epoch; "
                f"final/best={_format_ratio(validation.min_to_final_ratio)}."
            )


def _print_parametric_relationships(path: Path) -> None:
    accumulator = _load_event_accumulator(path)
    scalar_tags = set(accumulator.Tags()["scalars"])
    if not all(tag in scalar_tags for tag in _PARAMETRIC_BUCKETS):
        return
    values = {
        tag: [entry.value for entry in accumulator.Scalars(tag)]
        for tag in _PARAMETRIC_BUCKETS
    }
    # Correlate train vs held-out ESR over epochs where both are finite; an
    # all-inf training bucket leaves nothing to correlate.
    paired = [
        (train, val)
        for train, val in zip(values[_TRAIN_BUCKET], values[_VALIDATION_BUCKET])
        if math.isfinite(train) and math.isfinite(val)
    ]
    print("parametric_relationships:")
    if len(paired) < 2:
        print("  corr(train, held-out)=n/a (too few finite paired epochs)")
        return
    train_vals = [train for train, _ in paired]
    val_vals = [val for _, val in paired]
    print(
        "  corr(seen_audio_seen_params, unseen_audio_unseen_params)="
        f"{_corr(train_vals, val_vals):.6f} over {len(paired)} epochs"
    )


def main() -> int:
    args = _parse_args()
    if args.label and len(args.label) != len(args.paths):
        raise SystemExit("--label must be repeated exactly once per input path")

    labels = list(args.label) or [None] * len(args.paths)
    runs = [
        _summarize_run(
            Path(path).expanduser(),
            label=label,
            top_spikes=max(args.top_spikes, 0),
        )
        for path, label in zip(args.paths, labels)
    ]

    for index, (path, run) in enumerate(zip(args.paths, runs)):
        if index:
            print()
        resolved = Path(path).expanduser()
        _print_run_summary(
            run,
            resolved,
            spike_ratio_threshold=args.spike_ratio_threshold,
            held_out_regression_threshold=args.held_out_regression_threshold,
        )
        _print_parametric_relationships(resolved)

    if len(runs) >= 2:
        print()
        print("# Comparison")
        for run in runs:
            best_esr = next(
                (
                    summary
                    for summary in run.scalar_summaries
                    if summary.tag in (_VALIDATION_BUCKET, "ESR")
                ),
                None,
            )
            if best_esr is None:
                continue
            print(
                f"{run.label}: best={best_esr.best_value:.8g} "
                f"epoch_idx={best_esr.best_index} "
                f"worst={best_esr.worst_value:.8g} "
                f"worst_epoch_idx={best_esr.worst_index} "
                f"worst/best={_format_ratio(best_esr.worst_to_best_ratio)} "
                f"final={best_esr.final_value:.8g}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
