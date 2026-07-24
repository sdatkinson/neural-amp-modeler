"""
Training-facing outputs generated from a capture project.

Users should never have to hand-edit JSON: this module derives everything the
parametric trainer needs from the project file —

* ``data.json``: rewritten after every successful capture (completed entries only)
  so the training config stays valid if the app closes mid-session.
* ``model.json`` / ``learning.json``: a ready-to-train ConcatWaveNet model config
  with the project's knobs as its params, plus a matching learning config.
"""

from __future__ import annotations

from pathlib import Path as _Path
from typing import Any as _Any
from typing import Optional as _Optional

from .project import CaptureProject
from .project import DATA_FILENAME
from .project import LEARNING_CONFIG_FILENAME
from .project import MODEL_CONFIG_FILENAME
from .project import atomic_write_json as _atomic_write_json


# Active-learning defaults, adapted from PANAMA's LSTM recipe for the short single-file
# dataset used here. Batch 37 (matching the best-generalizing prior runs -- the small-batch
# gradient noise is itself a regularizer; batches of 40/64 trained smoother but plateaued at a
# worse unseen-ESR floor) and lr 0.008 over 50 epochs: the hotter rate takes bigger honest
# steps and reached the best unseen ESR of any run (~0.24 min / ~0.30 ensemble mean) in a
# third the epochs of the earlier lr-0.003/150-epoch recipe, with grads staying finite (max
# ~0.9, zero non-finite steps) -- so the generalization edge here comes from the larger step,
# not from the rare gradient blowups seen historically. train_truncate is None: the
# pre-truncation full-BPTT runs at this batch size generalized best of anything tried on this
# capture (unseen ESR ~0.29 vs ~0.35-0.44 for every truncated recipe since). Their tensorboard
# logs additionally showed occasional non-finite gradients (grad norm up to inf), silently
# survived by configure_gradient_clipping's non-finite skip (the step is dropped, not applied;
# Adam's moments are untouched) -- a bounded version of the "catapult" effect (Lewkowycz et al.
# 2020): a step large enough to escape a sharp minimum, discarded only when it overshoots into
# instability. That mechanism plausibly helped those runs, but the current lr-0.008 recipe does
# not depend on it (its grads stay finite). truncate=8192 capped the
# recurrent path specifically to eliminate those events (introduced when a *different*
# regime -- effective batch 512, full BPTT -- destabilized), which bought stability at the cost
# of the mechanism that was likely doing the regularizing. Reverting to None re-enables it,
# at this AL_NY (22050, burn_in 8192 -> ~13.9k scored samples) rather than the ~32768 the
# historical runs used -- a cheaper first test of the same mechanism, not an exact replay.
# Backprop now runs through the whole undetached ~13.9k-sample chain per window instead of two
# ~8192/5666 detached segments, so each step costs more, not less -- this is a fit-quality bet,
# not a speed one. Physical batches are still fitted to each GPU at runtime and accumulated to
# reach the effective batch (accumulation is 1 whenever 40 windows fit).
AL_NY = 22050
AL_MAX_BATCH_SIZE = 512
AL_TARGET_EFFECTIVE_BATCH_SIZE = 40
AL_REFERENCE_BATCH_SIZE = AL_TARGET_EFFECTIVE_BATCH_SIZE
AL_REFERENCE_LEARNING_RATE = 0.008
AL_LSTM_NUM_LAYERS = 3
AL_TRAIN_TRUNCATE = None
AL_MEL_WEIGHT = 6.2e-05
AL_DEFAULT_MAX_EPOCHS = 50
AL_WARMUP_EPOCHS = 16
AL_LR_GAMMA = 0.995

# Retain the stabilized trainer's exceptional-gradient protection and diagnostics around
# the PANAMA recipe. Weight decay zero makes AdamW's update identical to Adam's.
AL_GRADIENT_CLIP_VAL = 1.0
AL_WEIGHT_DECAY = 0.0
AL_LOG_EVERY_N_STEPS = 1
# The ConcatLSTM proxy is tiny and each optimizer step is short, so a synchronous
# (num_workers=0) dataloader stalls the GPU between steps assembling the next batch. Prefetch
# with a few workers, kept alive across epochs. The parallel-ensemble path collapses this back
# to 0 (nested workers under spawned member processes are unsafe); serial training keeps it.
AL_TRAIN_DATALOADER_NUM_WORKERS = 4


def active_learning_learning_rate(batch_size: int) -> float:
    if not 1 <= batch_size <= AL_MAX_BATCH_SIZE:
        raise ValueError(
            f"Active-learning batch_size must be in [1, {AL_MAX_BATCH_SIZE}], got "
            f"{batch_size}"
        )
    return AL_REFERENCE_LEARNING_RATE


def active_learning_steps_per_epoch(
    num_train_windows: int, batch_size: int, drop_last: bool
) -> int:
    """
    Optimizer steps in one epoch for ``num_train_windows`` windows at ``batch_size``, or 0
    when the window count is unknown (a speculative probe). ``drop_last`` mirrors the
    training dataloader: the final partial batch is a step only when it is kept.
    """
    if num_train_windows <= 0 or batch_size <= 0:
        return 0
    if drop_last:
        return num_train_windows // batch_size
    return -(-num_train_windows // batch_size)


def active_learning_max_epochs(steps_per_epoch: int) -> int:
    """Train each fresh ensemble for a fixed number of epochs."""
    return AL_DEFAULT_MAX_EPOCHS


def active_learning_lr_gamma(max_epochs: int) -> float:
    """Return PANAMA's fixed ExponentialLR decay."""
    return AL_LR_GAMMA


def active_learning_warmup_epochs(max_epochs: int) -> int:
    """Warm the learning rate over the first epochs, capped below ``max_epochs``."""
    return min(AL_WARMUP_EPOCHS, max(max_epochs - 1, 0))


def build_data_config(project: CaptureProject) -> dict[str, _Any]:
    """
    Build the parametric ``data.json`` payload from the project's *captured* entries.

    Train and validation entries point at their split's input file, carry their
    measured per-capture delay (entry fields override ``common``), and use the
    project's per-split windowing defaults.
    """
    splits: dict[str, list[dict[str, _Any]]] = {"train": [], "validation": []}
    for entry in project.captured_entries():
        window = project.window_for_split(entry.split)
        splits[entry.split].append(
            {
                "x_path": project.input_for_split(entry.split),
                "y_path": entry.y_path,
                "delay": 0 if entry.delay is None else entry.delay,
                "params": dict(entry.params),
                "start_seconds": window.start_seconds,
                "stop_seconds": window.stop_seconds,
                "ny": window.ny,
            }
        )
    return {
        "type": "parametric",
        "common": {"delay": 0},
        "train": splits["train"],
        "validation": splits["validation"],
    }


def update_data_json(project: CaptureProject, project_dir: _Path) -> _Path:
    path = _Path(project_dir) / DATA_FILENAME
    _atomic_write_json(path, build_data_config(project))
    return path


def build_al_data_config(project: CaptureProject) -> dict[str, _Any]:
    """
    Active-learning counterpart of :func:`build_data_config`: same captured entries and
    per-split paths/delay/params/windowing, but both splits use PANAMA's ``AL_NY`` instead
    of the project's own windows. A finite validation window also sidesteps the Apple-MPS
    LSTM batch-1 validation crash.
    """
    config = build_data_config(project)
    for split in ("train", "validation"):
        for entry in config[split]:
            entry["ny"] = AL_NY
    return config


# ConcatWaveNet counterpart of nam_full_configs/parametric/model.json: the same packed
# channels_8 WaveNet architecture, conditioned by concatenating the encoded params onto
# the audio input (the TwoNotes-style scheme) instead of through a hypernetwork.
# input_size/condition_size are omitted on purpose: ConcatWaveNet derives them from the
# param specs.
_CONCAT_WAVENET_LAYERS: list[dict[str, _Any]] = [
    {
        "channels": 8,
        "kernel_sizes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6],
        "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239],
        "activation": "LeakyReLU",
        "gated": False,
        "head": {"out_channels": 1, "kernel_size": 16, "bias": True},
    }
]


def build_model_config(project: CaptureProject) -> dict[str, _Any]:
    params = []
    for knob in project.knob_specs():
        spec = knob.to_param_spec().to_dict()
        # enum_names is a switch-only field; continuous-only projects don't carry it.
        del spec["enum_names"]
        # avoid_zero is capture-planning metadata only; it has no training meaning.
        del spec["avoid_zero"]
        params.append(spec)
    config: dict[str, _Any] = {
        "layers": [dict(layer) for layer in _CONCAT_WAVENET_LAYERS],
        "head_scale": 0.01,
        "params": params,
    }
    if project.sample_rate is not None:
        config["sample_rate"] = float(project.sample_rate)
    return {
        "net": {"name": "ConcatWaveNet", "config": config},
        "loss": {"val_loss": "esr", "mrstft_weight": 0.0005},
        "optimizer": {"lr": 0.004, "weight_decay": 3.17e-07},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.994}},
    }


# Mirrors nam_full_configs/active_learning/model.json (a PANAMA-style ConcatLSTM
# acquisition proxy, not the shipped ConcatWaveNet), less its _notes.
def build_al_model_config(
    project: CaptureProject,
    *,
    batch_size: int = AL_MAX_BATCH_SIZE,
    max_epochs: int = AL_DEFAULT_MAX_EPOCHS,
) -> dict[str, _Any]:
    params = []
    for knob in project.knob_specs():
        spec = knob.to_param_spec().to_dict()
        # enum_names is a switch-only field; continuous-only projects don't carry it.
        del spec["enum_names"]
        # Unlike build_model_config, step and avoid_zero are kept: the active-learning
        # proposal quantizer (quantize_to_capture_grid) reads them to snap g-opt results
        # onto the realizable capture grid and to honor avoid-zero knobs.
        params.append(spec)
    return {
        "net": {
            "name": "ConcatLSTM",
            "config": {
                "hidden_size": 18,
                "num_layers": AL_LSTM_NUM_LAYERS,
                "train_burn_in": 8192,
                "train_truncate": AL_TRAIN_TRUNCATE,
                "params": params,
            },
        },
        "loss": {
            "val_loss": "esr",
            "mel_weight": AL_MEL_WEIGHT,
        },
        "optimizer": {
            "lr": active_learning_learning_rate(batch_size),
            "weight_decay": AL_WEIGHT_DECAY,
        },
        "lr_scheduler": {
            "class": "ExponentialLR",
            "kwargs": {"gamma": active_learning_lr_gamma(max_epochs)},
            "warmup_epochs": active_learning_warmup_epochs(max_epochs),
        },
    }


def build_learning_config(project: CaptureProject) -> dict[str, _Any]:
    # Mirrors nam_full_configs/parametric/learning.json, but with accelerator "auto"
    # so a generated project trains without edits on CUDA, MPS, or CPU. Keep the
    # lr/batch-size pairing in mind if batch_size changes (0.002 <-> 16, 0.004 <-> 32).
    return {
        "train_dataloader": {
            "batch_size": 16,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {
            "batch_size": 16,
            "pin_memory": True,
            "num_workers": 0,
        },
        "trainer": {
            "accelerator": "auto",
            "devices": 1,
            "precision": "32-true",
            "benchmark": True,
            "max_epochs": 200,
            "gradient_clip_val": 1.0,
            "enable_progress_bar": True,
            "enable_model_summary": True,
        },
        "threshold_esr": None,
        "trainer_fit_kwargs": {},
    }


# Mirrors nam_full_configs/active_learning/learning.json, less its _notes. trainer.accelerator
# is a placeholder: nam.train.active_learning.train_ensemble rewrites it at runtime
# (cuda > mps > cpu).
def build_al_learning_config(
    *,
    batch_size: int,
    drop_last: bool,
    val_batch_size: _Optional[int] = None,
    max_epochs: int = AL_DEFAULT_MAX_EPOCHS,
    accumulate_grad_batches: int = 1,
    auto_batch_size: bool = False,
) -> dict[str, _Any]:
    # val_batch_size is decoupled from the training batch_size on purpose. Validation does
    # no backprop, so it uses a memory-fitting batch to run in as few passes as
    # possible. ESR is a ratio whose denominator sums target energy over the batch, so a
    # per-batch ESR averaged over many small batches is NOT the whole-set ESR (see the NOTE
    # on nam.train.lightning_module.ValidationLoss) -- fragmenting validation would make
    # val_loss, and thus best-checkpoint selection, depend on the batch count. Defaults to
    # batch_size when unset (e.g. an explicit user-forced batch applies to both splits).
    if val_batch_size is None:
        val_batch_size = batch_size
    return {
        "train_dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": drop_last,
            "num_workers": AL_TRAIN_DATALOADER_NUM_WORKERS,
            "persistent_workers": AL_TRAIN_DATALOADER_NUM_WORKERS > 0,
        },
        "val_dataloader": {
            "batch_size": val_batch_size,
            "pin_memory": True,
            "num_workers": 0,
        },
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "benchmark": False,
            "deterministic": "warn",
            "max_epochs": max_epochs,
            "accumulate_grad_batches": accumulate_grad_batches,
            "gradient_clip_val": AL_GRADIENT_CLIP_VAL,
            "log_every_n_steps": AL_LOG_EVERY_N_STEPS,
        },
        "batch_sizing": {
            "auto": auto_batch_size,
            "target_effective_batch_size": AL_TARGET_EFFECTIVE_BATCH_SIZE,
            "max_physical_batch_size": AL_MAX_BATCH_SIZE,
            "memory_fraction": 0.75,
            "bytes_per_timestep": 5120,
            "ny": AL_NY,
        },
        "trainer_fit_kwargs": {},
    }


def write_training_configs(project: CaptureProject, project_dir: _Path) -> list[_Path]:
    """
    Write ``model.json`` and ``learning.json`` into the project folder, overwriting
    previous generated versions.
    """
    project_dir = _Path(project_dir)
    model_path = project_dir / MODEL_CONFIG_FILENAME
    learning_path = project_dir / LEARNING_CONFIG_FILENAME
    _atomic_write_json(model_path, build_model_config(project))
    _atomic_write_json(learning_path, build_learning_config(project))
    return [model_path, learning_path]
