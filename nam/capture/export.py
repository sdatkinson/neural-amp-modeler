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
