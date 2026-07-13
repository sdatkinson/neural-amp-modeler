"""
Headless active-learning round runner for a capture project.

Wraps ``nam.train.active_learning.run_round`` for a ``CaptureProject``: builds the
ConcatLSTM acquisition-proxy configs from the project's captured entries, runs one
PANAMA-style round (ensemble train + disagreement search + proposal selection), and
writes the proposals for the human to go capture. Round outputs live in
``<project_dir>/active_learning/`` and are never written back into
``capture_project.json`` -- the capture app is the single writer of that file, so a
round can run on a remote box that only ships ``active_learning/`` back.

Runnable directly: ``python -m nam.capture.al_runner --project-dir <dir> [...]``.
Heavy imports (torch, nam.train.active_learning) are function-local so importing this
module stays cheap for the GUI.
"""

from __future__ import annotations

import argparse as _argparse
import json as _json
import os as _os
import re as _re
import wave as _wave
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Optional as _Optional
from typing import TYPE_CHECKING as _TYPE_CHECKING

from .export import AL_NY
from .export import AL_MAX_BATCH_SIZE
from .export import build_al_data_config as _build_al_data_config
from .export import build_al_learning_config as _build_al_learning_config
from .export import build_al_model_config as _build_al_model_config
from .project import CaptureEntryModel
from .project import CaptureProject
from .project import DATA_FILENAME
from .project import LEARNING_CONFIG_FILENAME
from .project import MODEL_CONFIG_FILENAME
from .project import atomic_write_json as _atomic_write_json
from .project import load_project as _load_project

if _TYPE_CHECKING:
    from ..train.active_learning import RoundResult


AL_DIRNAME = "active_learning"
AL_MAX_PER_ROUND_DEFAULT = 10
AL_ENSEMBLE_SIZE_DEFAULT = 4
AL_NUM_RESTARTS_DEFAULT = 16
AL_NUM_STEPS_DEFAULT = 300
RUNNER_SCRIPT_FILENAME = "run_active_learning.sh"

# Target number of optimizer steps per epoch for AL ensemble training. The batch size is
# chosen to hit this (see :func:`compute_al_batch_size`) rather than to fill GPU memory,
# because the acquisition-proxy dataset is tiny (round 0 is ~10 captures) and convergence
# is bottlenecked by the *number* of gradient updates, not throughput. Sizing the batch to
# memory once collapsed round 0 to ~5 steps/epoch (~250 updates over 50 epochs), which is
# far too few for a 3-layer LSTM -- it never left its near-silence initialization. The
# capture set grows by ~10 entries per round, so anchoring on steps-per-epoch (not a fixed
# batch) keeps the per-round update budget ~constant as the dataset grows.
AL_TARGET_STEPS_PER_EPOCH = 30

_RUNNER_SCRIPT = """\
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${{PYTHON:-python}}" -m nam.capture.al_runner --project-dir . {default_args} "$@"
"""

_PROPOSAL_FILENAME_RE = _re.compile(r"^proposed_captures_round_(\d+)\.json$")


def compute_al_batch_size(
    available_bytes: int, ny: int, num_train_windows: int
) -> tuple[int, bool]:
    """
    Pick a batch size (and whether to drop the final partial batch) for one AL ensemble
    member's training. Pure and torch-free so it is cheap to call speculatively from the
    GUI.

    Primary objective -- steps per epoch, not memory. The batch is sized so training does
    ~``AL_TARGET_STEPS_PER_EPOCH`` optimizer steps per epoch
    (``batch ~= num_train_windows / AL_TARGET_STEPS_PER_EPOCH``). Gradient-update count is
    what limits convergence on this tiny acquisition-proxy dataset, and expressing the
    target in *windows* makes it size-adaptive for free: it shrinks the batch for a small
    round-0 set (so it still gets many updates) and grows it as the capture set grows over
    rounds, holding the per-epoch update budget roughly constant. Because window count is
    ``usable_samples // ny``, a change to ``ny`` flows through automatically -- longer
    windows mean fewer of them, hence a smaller batch for the same step target.

    Memory is only a ceiling. Full-BPTT ConcatLSTM training peaks near 5 GiB at batch 32
    and ny 32768 (~160 MiB/window), scaling linearly with ``ny``; 75% of
    ``available_bytes`` is budgeted to leave headroom for the rest of the process. This
    memory cap (and the hard ``AL_MAX_BATCH_SIZE`` clamp) only bites if a large dataset's
    step target would demand a batch that does not fit -- in which case steps-per-epoch
    simply rises above the target, which is harmless.

    When ``num_train_windows`` is 0 the dataset size is unknown (a speculative GUI probe
    before any windows are known); fall back to the memory ceiling.
    """
    ceiling = _al_memory_ceiling(available_bytes, ny)

    if num_train_windows <= 0:
        return ceiling, True

    target_batch = max(round(num_train_windows / AL_TARGET_STEPS_PER_EPOCH), 1)
    batch_size = min(target_batch, ceiling, num_train_windows)

    dropped = num_train_windows % batch_size
    # Keep drop_last True (the shuffled-dataloader default) unless doing so would throw
    # away more than 10% of a small dataset's windows.
    drop_last = dropped <= max(1, num_train_windows // 10)
    return batch_size, drop_last


def _al_memory_ceiling(available_bytes: int, ny: int) -> int:
    """
    Largest batch that fits the memory budget, clamped to ``[1, AL_MAX_BATCH_SIZE]``.
    Full-BPTT ConcatLSTM training peaks near 5 GiB at batch 32 and ny 32768
    (~160 MiB/window, linear in ``ny``); only 75% of ``available_bytes`` is budgeted to
    leave headroom for the rest of the process.
    """
    bytes_per_window = 5120 * ny
    memory_cap = int(available_bytes * 0.75) // bytes_per_window
    return min(max(memory_cap, 1), AL_MAX_BATCH_SIZE)


def compute_al_val_batch_size(available_bytes: int, ny: int) -> int:
    """
    Batch size for AL validation. Unlike training, validation does no backprop, so it has
    no steps-per-epoch concern -- it should run in as *few* batches as memory allows,
    ideally one. ESR is a ratio whose denominator sums target energy over the batch, so a
    per-batch ESR averaged across many small batches is not the whole-set ESR (see the NOTE
    on ``nam.train.lightning_module.ValidationLoss``); fragmenting validation would make
    ``val_loss`` -- and thus best-checkpoint selection -- depend on the batch count. The
    validation set is fixed-size (only the train split grows over rounds), so this is
    simply the memory ceiling, which is a safe bound: a backward-graph-free validation pass
    uses less memory per window than the training calibration it is derived from.
    """
    return _al_memory_ceiling(available_bytes, ny)


def available_accelerator_memory_bytes() -> int:
    """
    Free memory on the device active-learning training will run on, for sizing the batch
    via :func:`compute_al_batch_size`. CUDA reports free VRAM directly; MPS has no such
    query, so it is approximated from Apple's recommended working-set ceiling minus what
    is currently allocated. Falls back to host RAM (psutil, or a conservative 8 GiB guess
    if psutil is unavailable) when neither accelerator is usable.
    """
    import torch

    if torch.cuda.is_available():
        return int(torch.cuda.mem_get_info()[0])

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        available = (
            torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()
        )
        if available > 0:
            return int(available)

    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        return 8 * 1024**3


def _wav_header(path: _Path) -> tuple[int, int]:
    """Return ``(num_frames, sample_rate)`` for a WAV file, reading only its header."""
    try:
        with _wave.open(str(path), "rb") as fp:
            return fp.getnframes(), fp.getframerate()
    except (_wave.Error, EOFError, OSError):
        from ..data import wav_to_np

        array, info = wav_to_np(path, info=True)
        return array.shape[0], info.rate


def count_train_windows(project: CaptureProject, project_dir: _Path) -> int:
    """
    Approximate number of AL train windows (``AL_NY``-length slices) available for
    training, used only to size the batch. Every train capture reamps the same input
    clip, so the input file's header is read once and reused for every captured train
    entry; nx/delay trimming is ignored, which is fine for batch sizing.
    """
    train_entries = [entry for entry in project.captured_entries() if entry.split == "train"]
    if not train_entries:
        return 0
    num_frames, rate = _wav_header(_Path(project_dir) / project.train_input)
    window = project.train_window
    file_seconds = num_frames / rate if rate else 0.0
    stop_seconds = file_seconds if window.stop_seconds is None else window.stop_seconds
    usable_seconds = max(stop_seconds - window.start_seconds, 0.0)
    usable_samples = usable_seconds * rate
    windows_per_entry = max(int(usable_samples // AL_NY), 0)
    return windows_per_entry * len(train_entries)


def _round_indices(project_dir: _Path) -> list[int]:
    al_dir = _Path(project_dir) / AL_DIRNAME
    if not al_dir.is_dir():
        return []
    indices = []
    for path in al_dir.glob("proposed_captures_round_*.json"):
        match = _PROPOSAL_FILENAME_RE.match(path.name)
        if match is None:
            continue
        indices.append(int(match.group(1)))
    return sorted(indices)


def next_round_idx(project_dir: _Path) -> int:
    indices = _round_indices(project_dir)
    return indices[-1] + 1 if indices else 0


def load_round_proposals(project_dir: _Path, round_idx: int) -> list[dict[str, _Any]]:
    path = _Path(project_dir) / AL_DIRNAME / f"proposed_captures_round_{round_idx}.json"
    with path.open() as fp:
        return _json.load(fp)


def unimported_rounds(project: CaptureProject, project_dir: _Path) -> list[int]:
    """
    Round indices with at least one proposal whose ``y_path`` is not (yet) a project
    entry. A round with an empty proposal list is vacuously imported.
    """
    known_y_paths = {entry.y_path for entry in project.entries}
    unimported = []
    for round_idx in _round_indices(project_dir):
        proposals = load_round_proposals(project_dir, round_idx)
        if any(record["y_path"] not in known_y_paths for record in proposals):
            unimported.append(round_idx)
    return unimported


def outstanding_proposal_y_paths(project: CaptureProject, project_dir: _Path) -> list[str]:
    """
    y_paths from every proposal file on disk that still need a capture: either already an
    entry but not yet marked captured, or not imported as an entry at all. Gates starting
    the next round -- every prior round's proposals must be captured first.
    """
    entries_by_y_path = {entry.y_path: entry for entry in project.entries}
    outstanding: dict[str, None] = {}
    for round_idx in _round_indices(project_dir):
        for record in load_round_proposals(project_dir, round_idx):
            y_path = record["y_path"]
            entry = entries_by_y_path.get(y_path)
            if entry is None or entry.status != "captured":
                outstanding[y_path] = None
    return list(outstanding)


def import_round_proposals(
    project: CaptureProject, proposals: list[dict[str, _Any]]
) -> list[CaptureEntryModel]:
    """
    Append one pending ``train`` entry per proposal not already present as an entry
    (matched by ``y_path``), with indices continuing from the project's current highest
    ``train`` index. Mutates ``project.entries`` in place but does not save; the caller
    saves.
    """
    known_y_paths = {entry.y_path for entry in project.entries}
    train_indices = [entry.index for entry in project.entries if entry.split == "train"]
    next_index = max(train_indices) + 1 if train_indices else 0
    appended: list[CaptureEntryModel] = []
    for record in proposals:
        y_path = record["y_path"]
        if y_path in known_y_paths:
            continue
        entry = CaptureEntryModel(
            index=next_index,
            split="train",
            params=dict(record["params"]),
            y_path=y_path,
        )
        project.entries.append(entry)
        known_y_paths.add(y_path)
        appended.append(entry)
        next_index += 1
    return appended


def write_runner_script(
    project_dir: _Path,
    *,
    max_per_round: int = AL_MAX_PER_ROUND_DEFAULT,
    ensemble_size: int = AL_ENSEMBLE_SIZE_DEFAULT,
    num_restarts: int = AL_NUM_RESTARTS_DEFAULT,
    num_steps: int = AL_NUM_STEPS_DEFAULT,
    max_workers: _Optional[int] = None,
) -> _Path:
    """
    Write the remote-runnable ``run_active_learning.sh`` wrapper at the project root.

    The round parameters are baked into the script as default flags so a plain
    ``./run_active_learning.sh`` on the training machine reproduces the settings chosen in
    the GUI (rather than falling back to :func:`main`'s argparse defaults). Because they
    precede ``"$@"``, any flag passed on the command line still overrides its baked-in
    default -- argparse keeps the last occurrence.
    """
    default_args = [
        "--max-per-round",
        str(max_per_round),
        "--ensemble-size",
        str(ensemble_size),
        "--num-restarts",
        str(num_restarts),
        "--num-steps",
        str(num_steps),
    ]
    if max_workers is not None and max_workers > 0:
        default_args += ["--max-workers", str(max_workers)]
    path = _Path(project_dir) / RUNNER_SCRIPT_FILENAME
    path.write_text(_RUNNER_SCRIPT.format(default_args=" ".join(default_args)))
    path.chmod(0o755)
    return path


def write_al_configs(
    project: CaptureProject,
    project_dir: _Path,
    *,
    batch_size: int,
    drop_last: bool,
    val_batch_size: _Optional[int] = None,
    max_per_round: int = AL_MAX_PER_ROUND_DEFAULT,
    ensemble_size: int = AL_ENSEMBLE_SIZE_DEFAULT,
    num_restarts: int = AL_NUM_RESTARTS_DEFAULT,
    num_steps: int = AL_NUM_STEPS_DEFAULT,
    max_workers: _Optional[int] = None,
) -> list[_Path]:
    """
    Write ``active_learning/{model,learning,data}.json`` from the project's current
    captured entries, plus the ``run_active_learning.sh`` wrapper. Meant to be rebuilt
    fresh every round: the AL data config always reflects the project's captured entries
    at call time, so later rounds never need y_path-filling in aggregated configs.

    ``val_batch_size`` sizes the validation dataloader independently of the (small,
    step-sized) training ``batch_size``; see :func:`compute_al_val_batch_size`. When unset
    it falls back to ``batch_size``.

    The round parameters (``max_per_round``, ``ensemble_size``, ``num_restarts``,
    ``num_steps``, ``max_workers``) are baked into the emitted ``run_active_learning.sh``
    so the remote run reproduces the caller's chosen settings; see
    :func:`write_runner_script`.
    """
    al_dir = _Path(project_dir) / AL_DIRNAME
    model_path = al_dir / MODEL_CONFIG_FILENAME
    learning_path = al_dir / LEARNING_CONFIG_FILENAME
    data_path = al_dir / DATA_FILENAME
    _atomic_write_json(model_path, _build_al_model_config(project, batch_size=batch_size))
    _atomic_write_json(
        learning_path,
        _build_al_learning_config(
            batch_size=batch_size, drop_last=drop_last, val_batch_size=val_batch_size
        ),
    )
    _atomic_write_json(data_path, _build_al_data_config(project))
    write_runner_script(
        project_dir,
        max_per_round=max_per_round,
        ensemble_size=ensemble_size,
        num_restarts=num_restarts,
        num_steps=num_steps,
        max_workers=max_workers,
    )
    return [model_path, learning_path, data_path]


def run_active_learning_round(
    project_dir: _Path,
    *,
    round_idx: _Optional[int] = None,
    max_per_round: int = AL_MAX_PER_ROUND_DEFAULT,
    ensemble_size: int = AL_ENSEMBLE_SIZE_DEFAULT,
    num_restarts: int = AL_NUM_RESTARTS_DEFAULT,
    num_steps: int = AL_NUM_STEPS_DEFAULT,
    seed: _Optional[int] = None,
    max_workers: _Optional[int] = None,
    batch_size: _Optional[int] = None,
    force: bool = False,
    plot: bool = True,
) -> "RoundResult":
    """
    Run one active-learning round for the project at ``project_dir``: rebuild the AL
    configs from its captured entries and hand them to
    ``nam.train.active_learning.run_round``. Does not touch ``capture_project.json``.
    """
    from ..train.active_learning import run_round

    project_dir = _Path(project_dir)
    project = _load_project(project_dir)

    if not force:
        captured_train = any(
            entry.split == "train" for entry in project.captured_entries()
        )
        captured_validation = any(
            entry.split == "validation" for entry in project.captured_entries()
        )
        if not captured_train:
            raise RuntimeError(
                "At least one captured train entry is required to run an "
                "active-learning round."
            )
        if not captured_validation:
            raise RuntimeError(
                "At least one captured validation entry is required to run an "
                "active-learning round."
            )
        outstanding = outstanding_proposal_y_paths(project, project_dir)
        if outstanding:
            raise RuntimeError(
                "Previously proposed captures must be recorded before starting another "
                f"round: {', '.join(outstanding)}"
            )

    if round_idx is None:
        round_idx = next_round_idx(project_dir)

    if batch_size is None:
        # Probe memory once and reuse it for both the (step-sized) train batch and the
        # (memory-sized) validation batch.
        available_bytes = available_accelerator_memory_bytes()
        resolved_batch_size, resolved_drop_last = compute_al_batch_size(
            available_bytes,
            AL_NY,
            count_train_windows(project, project_dir),
        )
        resolved_val_batch_size: _Optional[int] = compute_al_val_batch_size(
            available_bytes, AL_NY
        )
    else:
        if not 1 <= batch_size <= AL_MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be in [1, {AL_MAX_BATCH_SIZE}], got {batch_size}"
            )
        # An explicit batch size applies to both splits (val_batch_size None -> batch_size).
        resolved_batch_size, resolved_drop_last = batch_size, True
        resolved_val_batch_size = None

    write_al_configs(
        project,
        project_dir,
        batch_size=resolved_batch_size,
        drop_last=resolved_drop_last,
        val_batch_size=resolved_val_batch_size,
        max_per_round=max_per_round,
        ensemble_size=ensemble_size,
        num_restarts=num_restarts,
        num_steps=num_steps,
        max_workers=max_workers,
    )

    al_dir = project_dir / AL_DIRNAME
    with (al_dir / MODEL_CONFIG_FILENAME).open() as fp:
        model_config = _json.load(fp)
    with (al_dir / LEARNING_CONFIG_FILENAME).open() as fp:
        learning_config = _json.load(fp)
    with (al_dir / DATA_FILENAME).open() as fp:
        data_config = _json.load(fp)

    prior_cwd = _Path.cwd()
    try:
        # data_config/model_config/learning_config x_path/y_path entries are relative to
        # the training process's CWD; the project dir is where captures actually live.
        _os.chdir(project_dir)
        result = run_round(
            round_idx=round_idx,
            output_dir=_Path(AL_DIRNAME),
            data_config=data_config,
            model_config=model_config,
            learning_config=learning_config,
            g_opt_input_wav=project.train_input,
            max_per_round=max_per_round,
            ensemble_size=ensemble_size,
            num_restarts=num_restarts,
            num_steps=num_steps,
            seed=seed if seed is not None else project.seed + round_idx,
            max_workers=max_workers,
            y_path_prefix="captures/r",
            plot=plot,
        )
    finally:
        _os.chdir(prior_cwd)

    return result


def main(argv: _Optional[list[str]] = None) -> int:
    parser = _argparse.ArgumentParser(
        description="Run one active-learning round for a capture project."
    )
    parser.add_argument("--project-dir", default=".", type=_Path)
    parser.add_argument("--round-idx", type=int, default=None)
    parser.add_argument("--max-per-round", type=int, default=AL_MAX_PER_ROUND_DEFAULT)
    parser.add_argument("--ensemble-size", type=int, default=AL_ENSEMBLE_SIZE_DEFAULT)
    parser.add_argument("--num-restarts", type=int, default=AL_NUM_RESTARTS_DEFAULT)
    parser.add_argument("--num-steps", type=int, default=AL_NUM_STEPS_DEFAULT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    result = run_active_learning_round(
        args.project_dir,
        round_idx=args.round_idx,
        max_per_round=args.max_per_round,
        ensemble_size=args.ensemble_size,
        num_restarts=args.num_restarts,
        num_steps=args.num_steps,
        seed=args.seed,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        force=args.force,
        plot=not args.no_plot,
    )

    print(
        f"\nRound {result.round_idx}: {len(result.proposals)} proposal(s) written to "
        f"{result.proposals_path}.\n"
        "Reopen the project in the capture app to import the proposed captures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
