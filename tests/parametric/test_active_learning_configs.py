"""
Tests for the committed active-learning example configs in
``nam_full_configs/active_learning/`` (Task 8).

These assert that the shipped ConcatLSTM ensemble example is internally consistent:
the model config builds and runs, the learning config drives a PyTorch Lightning
Trainer, the starter data config resolves against the model's param specs, and the
three together train a (tiny) ensemble for one epoch the way ``scripts/active_learn.py``
would.

The active-learning workflow is adapted from PANAMA (arXiv 2509.26564v1).
"""

import json as _json
from copy import deepcopy as _deepcopy
from pathlib import Path as _Path

import numpy as _np
import pytest as _pytest
import pytorch_lightning as _pl
import torch as _torch

from nam.models.parametric import data_config_from_model as _data_config_from_model
from nam.models.parametric._dataset import resolve_named_params as _resolve_named_params
from nam.models.parametric._dataset import ParametricDataset as _ParametricDataset
from nam.models.parametric._spec import ParamSpec as _ParamSpec
from nam.data import init_dataset as _init_dataset
from nam.data import np_to_wav as _np_to_wav
from nam.data import Split as _Split
from nam.train.active_learning import train_ensemble as _train_ensemble
from nam.train.parametric import _ParametricLightningModule

_CONFIG_DIR = _Path(__file__).resolve().parents[2] / "nam_full_configs" / "active_learning"


def _load(name: str) -> dict:
    with open(_CONFIG_DIR / name, "r") as f:
        return _json.load(f)


def test_all_active_learning_configs_are_valid_json():
    for name in ("model.json", "learning.json", "data.json"):
        config = _load(name)
        assert isinstance(config, dict)
        assert len(config) >= 1


def test_model_config_is_concat_lstm():
    model_config = _load("model.json")
    assert model_config["net"]["name"] == "ConcatLSTM"
    assert model_config["net"]["config"]["params"], "model must declare param specs"
    assert model_config["net"]["config"]["num_layers"] == 3
    assert model_config["net"]["config"]["train_truncate"] is None
    assert model_config["loss"]["mel_weight"] == _pytest.approx(6.2e-05)


def test_model_config_builds_and_runs_both_shapes():
    model_config = _load("model.json")
    module = _ParametricLightningModule.init_from_config(model_config)
    net = module.net

    specs = tuple(
        _ParamSpec.from_dict(spec) for spec in model_config["net"]["config"]["params"]
    )
    raw = _resolve_named_params({"Gain": 7.0, "Tone": 3.0, "Boost": "On"}, specs)

    # (L,) + (P,)
    x = _torch.randn(8192 + 16, requires_grad=True)
    y = net(x, raw)
    assert y.ndim == 1
    y.sum().backward()
    assert x.grad is not None and _torch.isfinite(x.grad).all()

    # (B, L) + (B, P)
    net.eval()
    x_b = _torch.randn(3, 256)
    raw_b = _torch.stack([raw, raw, raw])
    y_b = net(x_b, raw_b)
    assert y_b.shape[0] == 3


def test_learning_config_has_required_keys_and_builds_trainer():
    learning_config = _load("learning.json")
    for key in ("train_dataloader", "val_dataloader", "trainer"):
        assert key in learning_config, f"learning.json missing {key}"
    assert learning_config["trainer"]["deterministic"] == "warn"
    assert learning_config["trainer"]["benchmark"] is False

    trainer_kw = _deepcopy(learning_config["trainer"])
    # Force CPU + a single minimal step so this is fast and device-agnostic.
    trainer_kw["accelerator"] = "cpu"
    trainer_kw["devices"] = 1
    trainer_kw["max_epochs"] = 1
    trainer_kw["enable_progress_bar"] = False
    trainer = _pl.Trainer(
        limit_train_batches=1,
        limit_val_batches=0,
        **trainer_kw,
    )
    assert trainer is not None


def test_data_config_params_resolve_against_model_specs():
    model_config = _load("model.json")
    data_config = _load("data.json")
    specs = tuple(
        _ParamSpec.from_dict(spec) for spec in model_config["net"]["config"]["params"]
    )

    assert data_config["type"] == "parametric"
    assert "x_path" in data_config["common"]

    entries = list(data_config["train"]) + list(data_config["validation"])
    assert entries, "starter data config should contain captures"
    for entry in entries:
        raw = _resolve_named_params(entry["params"], specs)
        assert raw.shape == (len(specs),)
        assert _torch.isfinite(raw).all()


def _render(x: _np.ndarray, *, gain: float, tone: float, boost: int) -> _np.ndarray:
    drive = 0.3 + 0.06 * gain + 0.1 * boost
    tilt = 0.8 + 0.04 * tone
    return (tilt * _np.tanh(drive * x)).astype(_np.float32)


def _smoke_data_config(tmp_path, *, ny: int) -> dict:
    # ny must clear the committed model.json loss's mask_first (8192); the MRSTFT max FFT
    # window (2048) then fits comfortably in the post-mask region.
    sample_rate = 48_000
    n = ny + 1024
    t = _np.arange(n) / sample_rate
    x = (0.08 * _np.sin(2.0 * _np.pi * 220.0 * t)).astype(_np.float32)
    x_path = tmp_path / "input.wav"
    _np_to_wav(x, x_path, rate=sample_rate)

    settings = (
        ("g2_t8_off.wav", 2.0, 8.0, 0),
        ("g9_t3_on.wav", 9.0, 3.0, 1),
        ("g5_t6_on.wav", 5.0, 6.0, 1),
    )
    for basename, gain, tone, boost in settings:
        _np_to_wav(
            _render(x, gain=gain, tone=tone, boost=boost),
            tmp_path / basename,
            rate=sample_rate,
        )

    def _entry(basename, gain, tone, boost):
        return {
            "y_path": str(tmp_path / basename),
            "params": {"Gain": gain, "Tone": tone, "Boost": "On" if boost else "Off"},
            "start_samples": 0,
            "stop_samples": ny,
            "ny": ny,
        }

    return {
        "type": "parametric",
        "common": {
            "x_path": str(x_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
        "train": [
            _entry("g2_t8_off.wav", 2.0, 8.0, 0),
            _entry("g9_t3_on.wav", 9.0, 3.0, 1),
        ],
        "validation": [
            {
                "y_path": str(tmp_path / "g5_t6_on.wav"),
                "params": {"Gain": 5.0, "Tone": 6.0, "Boost": "On"},
                "start_samples": 0,
                "stop_samples": None,
                "ny": None,
            }
        ],
    }


def test_data_config_loads_via_init_dataset(tmp_path):
    model_config = _load("model.json")
    data_config = _smoke_data_config(tmp_path, ny=2048)
    data_config = _data_config_from_model(data_config, model_config)
    data_config["common"]["nx"] = 1

    train = _init_dataset(data_config, _Split.TRAIN)
    validation = _init_dataset(data_config, _Split.VALIDATION)
    assert len(train) > 0
    assert len(validation) > 0


def test_example_configs_train_one_epoch_ensemble(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    monkeypatch.setattr(
        "nam.train.parametric._ParametricLightningModule._mel_mrstft_loss",
        lambda self, preds, targets: _torch.nn.functional.l1_loss(preds, targets),
    )

    model_config = _load("model.json")
    learning_config = _load("learning.json")
    # Trim the committed learning config to a single fast CPU step.
    learning_config["trainer"].update(
        {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "benchmark": False,
            "logger": False,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
        }
    )
    learning_config["train_dataloader"]["batch_size"] = 2
    learning_config["train_dataloader"]["drop_last"] = False
    learning_config["val_dataloader"]["batch_size"] = 1
    learning_config["trainer"]["accumulate_grad_batches"] = 1
    learning_config["batch_sizing"]["auto"] = False

    # ny must exceed the model's train_burn_in (8192), or the burn-in window consumes the
    # whole sequence and there is nothing left to score; 12288 leaves a 4096-sample region.
    data_config = _smoke_data_config(tmp_path, ny=12288)

    checkpoint_paths = _train_ensemble(
        data_config,
        model_config,
        learning_config,
        tmp_path / "run",
        ensemble_size=2,
    )

    assert len(checkpoint_paths) == 2
    for path in checkpoint_paths:
        assert _Path(path).exists()
