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

from .project import CaptureProject
from .project import DATA_FILENAME
from .project import LEARNING_CONFIG_FILENAME
from .project import MODEL_CONFIG_FILENAME
from .project import atomic_write_json as _atomic_write_json


# Active-learning training window: above the acquisition-proxy loss's mask_first (8192)
# and below one ConcatLSTM processing block (65535). Owned here (not al_runner.py) so
# export.py has no import-time dependency on al_runner; al_runner re-exports it.
AL_NY = 32768
AL_MAX_BATCH_SIZE = 256
AL_REFERENCE_LEARNING_RATE = 0.008


def active_learning_learning_rate(batch_size: int) -> float:
    if not 1 <= batch_size <= AL_MAX_BATCH_SIZE:
        raise ValueError(
            f"Active-learning batch_size must be in [1, {AL_MAX_BATCH_SIZE}], got "
            f"{batch_size}"
        )
    return AL_REFERENCE_LEARNING_RATE * batch_size / AL_MAX_BATCH_SIZE


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
    per-split paths/delay/params/windowing, but both splits use ``AL_NY`` instead of the
    project's own windows. The acquisition-proxy ConcatLSTM needs a window above the loss
    mask_first (8192) and below one LSTM processing block (65535); a finite validation
    window also sidesteps the Apple-MPS LSTM batch-1 validation crash.
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
    project: CaptureProject, *, batch_size: int = AL_MAX_BATCH_SIZE
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
                "num_layers": 3,
                "train_burn_in": 8192,
                "train_truncate": None,
                "params": params,
            },
        },
        "loss": {
            "mask_first": 8192,
            "pre_emph_mrstft_weight": 0.002,
            "pre_emph_mrstft_coef": 0.85,
        },
        "optimizer": {"lr": active_learning_learning_rate(batch_size)},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.995}},
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
def build_al_learning_config(*, batch_size: int, drop_last: bool) -> dict[str, _Any]:
    return {
        "train_dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": drop_last,
            "num_workers": 0,
        },
        "val_dataloader": {
            "batch_size": batch_size,
            "pin_memory": True,
            "num_workers": 0,
        },
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "max_epochs": 50,
            "gradient_clip_val": 1.0,
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
