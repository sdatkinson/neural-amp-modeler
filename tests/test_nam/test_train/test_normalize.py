# File: test_normalize.py
# Tests for nam.train._normalize.

import json
import warnings as _warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from nam.data import np_to_wav, wav_to_np
from nam.models.wavenet import PackedWaveNet, WaveNet
from nam.train import _normalize as _norm
from nam.train import full as _full


_RATE = 48_000


def _db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# compute_y_scale
# ---------------------------------------------------------------------------


class TestComputeYScale:
    def test_rms_branch_hits_target(self):
        """A quiet signal is scaled up to exactly the target RMS."""
        target_dbfs = -18.0
        target_rms = _db_to_amp(target_dbfs)
        # Quiet sine: RMS = 0.1 / sqrt(2) ~ 0.0707; peak << 1 even after scaling.
        t = np.arange(_RATE, dtype=np.float64) / _RATE
        y = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)

        gain = _norm.compute_y_scale(y, target_rms_dbfs=target_dbfs)

        scaled_rms = _rms(y * gain)
        np.testing.assert_allclose(scaled_rms, target_rms, rtol=1e-6)
        assert np.max(np.abs(y * gain)) < 1.0

    def test_peak_clamp_activates_for_hot_signal(self):
        """A signal whose RMS-normalized peak would exceed 1.0 gets clamped."""
        # Pulse: RMS << peak. RMS-normalizing to -18 dBFS pushes peak above 1.
        y = np.zeros(_RATE, dtype=np.float64)
        y[::100] = 0.9  # Sparse, very peaky.

        gain = _norm.compute_y_scale(y, target_rms_dbfs=-6.0)

        scaled = y * gain
        peak = float(np.max(np.abs(scaled)))
        assert peak < 1.0
        # Peak is clamped strictly below 1.0 to satisfy Dataset validation.
        assert peak >= 0.999 * (1.0 - 1e-4)

    def test_silent_input_returns_one(self):
        y = np.zeros(_RATE, dtype=np.float64)
        assert _norm.compute_y_scale(y) == 1.0

    def test_accepts_torch_tensor(self):
        y = 0.1 * torch.randn(_RATE)
        gain = _norm.compute_y_scale(y, target_rms_dbfs=-18.0)
        assert gain > 1.0  # Quiet random -> scaled up.


# ---------------------------------------------------------------------------
# parse_data_config
# ---------------------------------------------------------------------------


class TestParseDataConfig:
    def test_missing_key_defaults_on(self):
        assert _norm.parse_data_config({}) == _norm.DEFAULT_TARGET_RMS_DBFS

    def test_null_disables(self):
        assert _norm.parse_data_config({"target_rms_dbfs": None}) is None

    def test_explicit_target(self):
        assert _norm.parse_data_config({"target_rms_dbfs": -12.0}) == -12.0

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            _norm.parse_data_config({"target_rms_dbfs": "loud"})
        with pytest.raises(ValueError):
            _norm.parse_data_config({"target_rms_dbfs": True})


# ---------------------------------------------------------------------------
# supports_head_scale_compensation
# ---------------------------------------------------------------------------


def _wavenet_model_config(head=None):
    return {
        "net": {
            "name": "WaveNet",
            "config": {
                "layers_configs": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "channels": 2,
                        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                        "kernel_size": 2,
                        "dilations": [1],
                        "activation": "Tanh",
                    }
                ],
                "head": head,
                "head_scale": 0.25,
            },
        }
    }


def _packed_model_config():
    submodel = {
        "name": "only",
        "config": {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 2,
                    "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head": None,
            "head_scale": 0.25,
        },
    }
    return {
        "net": {
            "name": "PackedWaveNet",
            "config": {"submodels": [submodel]},
        }
    }


def _slimmable_model_config():
    """Slimmable WaveNet is just a WaveNet with a per-layer 'slimmable' block."""
    return {
        "net": {
            "name": "WaveNet",
            "config": {
                "layers_configs": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "channels": 4,
                        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                        "kernel_size": 2,
                        "dilations": [1, 2],
                        "activation": "Tanh",
                        "slimmable": {
                            "method": "slice_channels_uniform",
                            "kwargs": {"allowed_channels": [2, 4]},
                        },
                    }
                ],
                "head_scale": 0.25,
            },
        }
    }


class TestSupportsHeadScaleCompensation:
    def test_wavenet_no_head(self):
        assert _norm.supports_head_scale_compensation(_wavenet_model_config())

    def test_wavenet_with_top_head_unsupported(self):
        cfg = _wavenet_model_config(
            head={"channels": 1, "activation": "Tanh", "out_channels": 1, "kernel_sizes": [1]}
        )
        assert not _norm.supports_head_scale_compensation(cfg)

    def test_packed_wavenet_no_head(self):
        assert _norm.supports_head_scale_compensation(_packed_model_config())

    def test_slimmable_wavenet_no_head(self):
        assert _norm.supports_head_scale_compensation(_slimmable_model_config())

    def test_unsupported_architecture(self):
        assert not _norm.supports_head_scale_compensation(
            {"net": {"name": "ConvNet", "config": {}}}
        )


# ---------------------------------------------------------------------------
# compensate_head_scale
# ---------------------------------------------------------------------------


class TestCompensateHeadScale:
    def test_wavenet(self):
        net = WaveNet.init_from_config(_wavenet_model_config()["net"]["config"])
        original = net._net._head_scale
        _norm.compensate_head_scale(net, 2.0)
        assert net._net._head_scale == pytest.approx(original / 2.0)

    def test_packed_wavenet(self):
        net = PackedWaveNet.init_from_config(_packed_model_config()["net"]["config"])
        original = net._net._head_scale
        _norm.compensate_head_scale(net, 4.0)
        assert net._net._head_scale == pytest.approx(original / 4.0)

    def test_gain_of_one_is_noop(self):
        net = WaveNet.init_from_config(_wavenet_model_config()["net"]["config"])
        original = net._net._head_scale
        _norm.compensate_head_scale(net, 1.0)
        assert net._net._head_scale == original

    def test_invalid_gain_raises(self):
        net = WaveNet.init_from_config(_wavenet_model_config()["net"]["config"])
        with pytest.raises(ValueError):
            _norm.compensate_head_scale(net, 0.0)
        with pytest.raises(ValueError):
            _norm.compensate_head_scale(net, float("inf"))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _write_quiet_target(tmp_path: Path, num_seconds: float = 0.5) -> Path:
    n = int(num_seconds * _RATE)
    t = np.arange(n, dtype=np.float64) / _RATE
    # 0.1-amp sine -> RMS ~ 0.0707, needs ~1.8x to reach -18 dBFS RMS.
    y = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    path = tmp_path / "target.wav"
    np_to_wav(y, path, rate=_RATE)
    return path


class TestPrepare:
    def test_default_on_for_wavenet(self, tmp_path):
        y_path = _write_quiet_target(tmp_path)
        data_config = {"common": {"y_path": str(y_path)}, "train": {}, "validation": {}}
        plan = _norm.prepare(data_config, _wavenet_model_config())
        assert plan.applied
        assert plan.gain > 1.0
        assert plan.data_config["common"]["y_scale"] == pytest.approx(plan.gain)
        # Directive consumed so it's not double-applied downstream.
        assert "target_rms_dbfs" not in plan.data_config

    def test_explicit_disable(self, tmp_path):
        y_path = _write_quiet_target(tmp_path)
        data_config = {
            "common": {"y_path": str(y_path)},
            "train": {},
            "validation": {},
            "target_rms_dbfs": None,
        }
        plan = _norm.prepare(data_config, _wavenet_model_config())
        assert not plan.applied
        assert plan.gain == 1.0
        assert "y_scale" not in plan.data_config.get("common", {})

    def test_unsupported_model_warns_and_noop(self, tmp_path):
        y_path = _write_quiet_target(tmp_path)
        data_config = {"common": {"y_path": str(y_path)}, "train": {}, "validation": {}}
        with pytest.warns(
            _norm.OutputNormalizationSkippedWarning, match=r"ConvNet"
        ):
            plan = _norm.prepare(
                data_config, {"net": {"name": "ConvNet", "config": {}}}
            )
        assert not plan.applied
        assert plan.gain == 1.0

    def test_unsupported_model_silent_when_disabled(self):
        """If the user disabled normalization, no warning for unsupported models."""
        data_config = {"common": {}, "train": {}, "target_rms_dbfs": None}
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", _norm.OutputNormalizationSkippedWarning)
            plan = _norm.prepare(
                data_config, {"net": {"name": "ConvNet", "config": {}}}
            )
        assert not plan.applied

    def test_unresolvable_y_path_warns(self):
        data_config = {
            "common": {},
            "train": [],  # list-typed -> can't resolve a single train y_path
            "validation": [],
        }
        with pytest.warns(
            _norm.OutputNormalizationSkippedWarning, match=r"y_path"
        ):
            plan = _norm.prepare(data_config, _wavenet_model_config())
        assert not plan.applied

    def test_two_pairs_uses_train_y_path(self, tmp_path):
        y_path = _write_quiet_target(tmp_path)
        data_config = {
            "common": {},
            "train": {"y_path": str(y_path)},
            "validation": {"y_path": str(y_path)},
        }
        plan = _norm.prepare(data_config, _wavenet_model_config())
        assert plan.applied
        assert plan.data_config["common"]["y_scale"] == pytest.approx(plan.gain)

    def test_composes_with_existing_y_scale(self, tmp_path):
        y_path = _write_quiet_target(tmp_path)
        data_config = {
            "common": {"y_path": str(y_path), "y_scale": 2.0},
            "train": {},
            "validation": {},
        }
        plan = _norm.prepare(data_config, _wavenet_model_config())
        assert plan.data_config["common"]["y_scale"] == pytest.approx(2.0 * plan.gain)

    def test_explicit_override_target(self, tmp_path):
        y_path = _write_quiet_target(tmp_path)
        data_config = {"common": {"y_path": str(y_path)}, "train": {}, "validation": {}}
        plan = _norm.prepare(
            data_config, _wavenet_model_config(), target_rms_dbfs=-30.0
        )
        assert plan.target_rms_dbfs == -30.0
        # Quieter target -> smaller gain than default -18 dBFS.
        default_plan = _norm.prepare(data_config, _wavenet_model_config())
        assert plan.gain < default_plan.gain


# ---------------------------------------------------------------------------
# End-to-end: full.main exports model with compensated head_scale
# ---------------------------------------------------------------------------


_E2E_NUM_SAMPLES = 256
_E2E_NUM_VAL = 64
_E2E_NY = 8


def _e2e_write_wav_pair(tmp_path):
    t = np.arange(_E2E_NUM_SAMPLES, dtype=np.float64) / _RATE
    x = 0.10 * np.sin(2.0 * np.pi * 220.0 * t)
    y = 0.05 * x + 0.005 * np.sin(2.0 * np.pi * 440.0 * t)
    x_path = tmp_path / "input.wav"
    y_path = tmp_path / "output.wav"
    np_to_wav(x, x_path, rate=_RATE)
    np_to_wav(y, y_path, rate=_RATE)
    return x_path, y_path


def _e2e_data_config(x_path, y_path, target_rms_dbfs):
    config = {
        "common": {
            "x_path": str(x_path),
            "y_path": str(y_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
        "train": {"stop_samples": -_E2E_NUM_VAL, "ny": _E2E_NY},
        "validation": {"start_samples": -_E2E_NUM_VAL, "ny": None},
    }
    if target_rms_dbfs is not _UNSET:
        config["target_rms_dbfs"] = target_rms_dbfs
    return config


_UNSET = object()


_E2E_HEAD_SCALE = 0.25


def _e2e_wavenet_model_config():
    return {
        "net": {
            "name": "WaveNet",
            "config": {
                "layers_configs": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "channels": 2,
                        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                        "kernel_size": 2,
                        "dilations": [1],
                        "activation": "Tanh",
                    }
                ],
                "head": None,
                "head_scale": _E2E_HEAD_SCALE,
            },
        },
        "optimizer": {"lr": 0.001},
        "lr_scheduler": None,
        "loss": {"val_loss": "mse"},
    }


def _e2e_packed_model_config():
    sub = _e2e_wavenet_model_config()["net"]["config"]
    return {
        "net": {
            "name": "PackedWaveNet",
            "config": {
                "submodels": [
                    {"name": "small", "config": deepcopy(sub)},
                    {"name": "large", "config": deepcopy(sub)},
                ],
            },
        },
        "optimizer": {"lr": 0.001},
        "lr_scheduler": None,
        "loss": {"val_loss": "mse"},
    }


def _e2e_slimmable_model_config():
    return {
        "net": {
            "name": "WaveNet",
            "config": {
                "layers_configs": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "channels": 4,
                        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                        "kernel_size": 2,
                        "dilations": [1],
                        "activation": "Tanh",
                        "slimmable": {
                            "method": "slice_channels_uniform",
                            "kwargs": {"allowed_channels": [2, 4]},
                        },
                    }
                ],
                "head": None,
                "head_scale": _E2E_HEAD_SCALE,
            },
        },
        "optimizer": {"lr": 0.001},
        "lr_scheduler": None,
        "loss": {"val_loss": "mse"},
    }


def _e2e_learning_config():
    return {
        "train_dataloader": {
            "batch_size": 2,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 0,
        },
        "val_dataloader": {"batch_size": 1, "num_workers": 0},
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
        "trainer_fit_kwargs": {},
    }


def _expected_gain_from_file(y_path: Path) -> float:
    y = wav_to_np(str(y_path))
    return _norm.compute_y_scale(y, target_rms_dbfs=_norm.DEFAULT_TARGET_RMS_DBFS)


class TestFullMainCompensation:
    def test_wavenet_head_scale_is_compensated(self, tmp_path):
        x_path, y_path = _e2e_write_wav_pair(tmp_path)
        outdir = tmp_path / "out"
        outdir.mkdir()

        expected_gain = _expected_gain_from_file(y_path)
        assert expected_gain > 1.0  # Sanity: data is quiet enough to scale up.

        _full.main(
            _e2e_data_config(x_path, y_path, _UNSET),
            _e2e_wavenet_model_config(),
            _e2e_learning_config(),
            outdir,
            no_show=True,
            make_plots=False,
        )

        with open(outdir / "model.nam", "r") as fp:
            exported = json.load(fp)
        actual = float(exported["config"]["head_scale"])
        assert actual == pytest.approx(_E2E_HEAD_SCALE / expected_gain, rel=1e-5)

    def test_disable_via_null_keeps_head_scale(self, tmp_path):
        x_path, y_path = _e2e_write_wav_pair(tmp_path)
        outdir = tmp_path / "out_disabled"
        outdir.mkdir()

        _full.main(
            _e2e_data_config(x_path, y_path, None),
            _e2e_wavenet_model_config(),
            _e2e_learning_config(),
            outdir,
            no_show=True,
            make_plots=False,
        )

        with open(outdir / "model.nam", "r") as fp:
            exported = json.load(fp)
        # No normalization applied -> head_scale persists as configured.
        assert float(exported["config"]["head_scale"]) == pytest.approx(
            _E2E_HEAD_SCALE, rel=1e-5
        )

    def test_slimmable_wavenet_head_scale_is_compensated(self, tmp_path):
        x_path, y_path = _e2e_write_wav_pair(tmp_path)
        outdir = tmp_path / "out_slimmable"
        outdir.mkdir()

        expected_gain = _expected_gain_from_file(y_path)
        assert expected_gain > 1.0

        _full.main(
            _e2e_data_config(x_path, y_path, _UNSET),
            _e2e_slimmable_model_config(),
            _e2e_learning_config(),
            outdir,
            no_show=True,
            make_plots=False,
        )

        with open(outdir / "model.nam", "r") as fp:
            exported = json.load(fp)
        actual = float(exported["config"]["head_scale"])
        assert actual == pytest.approx(_E2E_HEAD_SCALE / expected_gain, rel=1e-5)

    def test_packed_wavenet_head_scale_is_compensated(self, tmp_path):
        x_path, y_path = _e2e_write_wav_pair(tmp_path)
        outdir = tmp_path / "out_packed"
        outdir.mkdir()

        expected_gain = _expected_gain_from_file(y_path)
        assert expected_gain > 1.0

        _full.main(
            _e2e_data_config(x_path, y_path, _UNSET),
            _e2e_packed_model_config(),
            _e2e_learning_config(),
            outdir,
            no_show=True,
            make_plots=False,
        )

        with open(outdir / "model.nam", "r") as fp:
            container = json.load(fp)

        # Every packed submodel shares the same head_scale -> all compensated.
        submodels = container["config"]["submodels"]
        assert len(submodels) == 2
        for entry in submodels:
            assert float(entry["model"]["config"]["head_scale"]) == pytest.approx(
                _E2E_HEAD_SCALE / expected_gain, rel=1e-5
            )
