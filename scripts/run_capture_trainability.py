"""
Trainability check: train each capture on its own for 30 epochs and report the
final TRAINING ESR (ESR/seen_audio_seen_params).

Every capture should overfit its single training file fast. Any that don't reach
a low training ESR are suspect (bad delay, clipping, wrong knob labels, etc.).
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Prefer the console script sitting next to this interpreter (works even when the
# env isn't activated); fall back to PATH lookup.
_cli = Path(sys.executable).parent / "nam-full-parametric"
CLI = str(_cli) if _cli.exists() else "nam-full-parametric"
TRAIN_ESR_TAG = "ESR/seen_audio_seen_params"


def read_train_esr(run_dir: Path) -> float:
    evs = glob.glob(str(run_dir / "lightning_logs" / "*" / "events.out.tfevents.*"))
    if not evs:
        return float("nan")
    ea = EventAccumulator(evs[0])
    ea.Reload()
    if TRAIN_ESR_TAG not in ea.Tags()["scalars"]:
        return float("nan")
    return ea.Scalars(TRAIN_ESR_TAG)[-1].value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train each generated data_XXX.json file under a dataset root's "
            "test/ directory and summarize final training ESR."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset root containing model.json and test/learning.json.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    root = args.root.expanduser().resolve()
    test_dir = root / "test"
    model = root / "model.json"
    learning = test_dir / "learning.json"

    data_files = sorted(test_dir.glob("data_[0-9][0-9][0-9].json"))
    if not data_files:
        sys.exit(f"No data_XXX.json files in {test_dir}")
    if not model.is_file():
        sys.exit(f"Missing model.json at {model}")
    if not learning.is_file():
        sys.exit(f"Missing learning.json at {learning}")

    results = []
    for data_file in data_files:
        name = data_file.stem  # data_000
        before = {p for p in test_dir.iterdir() if p.is_dir()}
        log_path = test_dir / "logs" / f"{name}.log"
        log_path.parent.mkdir(exist_ok=True)

        print(f"[{name}] training...", flush=True)
        with open(log_path, "w") as log:
            proc = subprocess.run(
                [CLI, str(data_file), str(model), str(learning), str(test_dir),
                 "--no-show", "--no-plots"],
                stdout=log, stderr=subprocess.STDOUT,
            )

        # The CLI drops its output in a fresh timestamped subdir of test_dir.
        new_dirs = [p for p in test_dir.iterdir() if p.is_dir() and p not in before]
        if proc.returncode != 0 or not new_dirs:
            print(f"[{name}] FAILED (rc={proc.returncode}); see {log_path}", flush=True)
            results.append((name, float("nan")))
            continue

        run_dir = max(new_dirs, key=lambda p: p.stat().st_mtime)
        esr = read_train_esr(run_dir)
        print(f"[{name}] train ESR = {esr:.4g}", flush=True)
        results.append((name, esr))

    # Summary, worst first. A capture is "suspect" if it fits much worse than its
    # peers (or failed), since every capture should overfit its own file similarly.
    finite = sorted(e for _, e in results if e == e)
    median = finite[len(finite) // 2] if finite else float("nan")
    suspect_cut = 3.0 * median
    print(f"\n==== TRAINING ESR (worst first)  [median={median:.4g}] ====")
    for name, esr in sorted(results, key=lambda r: (r[1] != r[1], r[1]), reverse=True):
        suspect = (esr != esr) or (esr > suspect_cut)
        print(f"{name}: {esr:.4g}{'  <-- suspect' if suspect else ''}")

    summary_path = test_dir / "trainability_esr.csv"
    with open(summary_path, "w") as fp:
        fp.write("data_file,train_esr\n")
        for name, esr in results:
            fp.write(f"{name},{esr}\n")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
