"""
Tests that all committed nam_full_configs parse and are usable the same way as in nam-full.

Nam-full loads three configs (data, model, learning) via JSON and passes them to
nam.train.full.main(). These tests assert that:
- Every JSON file in nam_full_configs/ loads as valid JSON.
- Data configs work with init_dataset() when given valid paths (using temp wav files).
- Model configs work with LightningModule.init_from_config().
- Learning configs have the required keys and work with PyTorch Lightning Trainer.
"""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl

from nam.data import Split, init_dataset, np_to_wav
from nam.train import lightning_module

_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "nam_full_configs"


def _config_paths(subdir: str):
    """Return paths to all JSON config files in nam_full_configs/<subdir>/."""
    d = _CONFIGS_DIR / subdir
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


# --- JSON load (all configs) ---


@pytest.mark.parametrize(
    "path", _config_paths("data") + _config_paths("learning") + _config_paths("models")
)
def test_config_loads_as_json(path):
    """Every committed config file is valid JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) >= 1


# --- Data configs: same usage as full.main -> init_dataset ---


def _make_wav_pair(tmp_path, num_seconds=10, rate=48_000, prefix=""):
    """Create two same-length mono wav files; return (x_path, y_path)."""
    n = num_seconds * rate
    x = (np.random.rand(n).astype(np.float64) - 0.5) * 0.5
    y = (np.random.rand(n).astype(np.float64) - 0.5) * 0.5
    x_path = tmp_path / f"{prefix}x.wav"
    y_path = tmp_path / f"{prefix}y.wav"
    np_to_wav(x, x_path, rate=rate)
    np_to_wav(y, y_path, rate=rate)
    return str(x_path), str(y_path)


def _data_config_with_paths(config, tmp_path, nx=8):
    """Return a deep copy of the data config with paths set to temp wav files."""
    config = deepcopy(config)
    # Remove comment keys so they don't get passed to dataset init
    config.pop("_notes", None)
    config.pop("_comments", None)

    common = config.setdefault("common", {})
    common["nx"] = nx
    # Temp wavs are random; disable pre-silence check so validation split can load
    common["require_input_pre_silence"] = None

    train = config.get("train", {})
    validation = config.get("validation", {})

    # single_pair: common has x_path, y_path; train/validation add start/stop/ny
    if "x_path" in common or "y_path" in common:
        x_path, y_path = _make_wav_pair(tmp_path)
        common["x_path"] = x_path
        common["y_path"] = y_path
        return config

    # two_pairs: train and validation each have x_path, y_path
    train_x, train_y = _make_wav_pair(tmp_path, prefix="train_")
    val_x, val_y = _make_wav_pair(tmp_path, prefix="val_")
    train["x_path"] = train_x
    train["y_path"] = train_y
    validation["x_path"] = val_x
    validation["y_path"] = val_y
    return config


@pytest.mark.parametrize("path", _config_paths("data"))
def test_data_config_init_dataset(tmp_path, path):
    """Each data config parses and init_dataset runs for both splits (with temp wavs)."""
    with open(path, "r") as f:
        data_config = json.load(f)
    data_config = _data_config_with_paths(data_config, tmp_path)
    init_dataset(data_config, Split.TRAIN)
    init_dataset(data_config, Split.VALIDATION)


# --- Model configs: same usage as full.main -> LightningModule.init_from_config ---


@pytest.mark.parametrize("path", _config_paths("models"))
def test_model_config_init_from_config(path):
    """Each model config builds a LightningModule via init_from_config (same as nam-full)."""
    with open(path, "r") as f:
        model_config = json.load(f)
    model = lightning_module.LightningModule.init_from_config(model_config)
    assert model is not None
    assert model.net is not None


# --- Learning configs: same usage as full.main -> Trainer and dataloaders ---


@pytest.mark.parametrize("path", _config_paths("learning"))
def test_learning_config_has_required_keys(path):
    """Each learning config has keys required by full.main."""
    with open(path, "r") as f:
        learning_config = json.load(f)
    for key in ("train_dataloader", "val_dataloader", "trainer"):
        assert key in learning_config, f"{path.name} missing key: {key}"
    assert isinstance(learning_config["train_dataloader"], dict)
    assert isinstance(learning_config["val_dataloader"], dict)
    assert isinstance(learning_config["trainer"], dict)


@pytest.mark.parametrize("path", _config_paths("learning"))
def test_learning_config_trainer_accepts_pl(path):
    """Trainer can be constructed with learning_config['trainer'] (CPU, 1 step)."""
    with open(path, "r") as f:
        learning_config = json.load(f)
    trainer_kw = deepcopy(learning_config["trainer"])
    # Force CPU and minimal run so this is fast and device-agnostic
    trainer_kw["accelerator"] = "cpu"
    trainer_kw["devices"] = 1
    trainer_kw["max_epochs"] = 1
    trainer = pl.Trainer(
        limit_train_batches=1,
        limit_val_batches=0,
        enable_progress_bar=False,
        **trainer_kw,
    )
    assert trainer is not None
