import json

import numpy as np

from nam.data import np_to_wav as _np_to_wav
from nam.train import full as _full


_NUM_SAMPLES = 128
_NUM_VALIDATION_SAMPLES = 32
_NY = 8
_RATE = 48_000


def _write_wav_pair(tmp_path):
    t = np.arange(_NUM_SAMPLES, dtype=np.float64) / _RATE
    x = 0.10 * np.sin(2.0 * np.pi * 220.0 * t)
    y = 0.50 * x + 0.02 * np.sin(2.0 * np.pi * 440.0 * t)
    x_path = tmp_path / "input.wav"
    y_path = tmp_path / "output.wav"
    _np_to_wav(x, x_path, rate=_RATE)
    _np_to_wav(y, y_path, rate=_RATE)
    return x_path, y_path


def _data_config(x_path, y_path):
    return {
        "common": {
            "x_path": str(x_path),
            "y_path": str(y_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
        "train": {
            "stop_samples": -_NUM_VALIDATION_SAMPLES,
            "ny": _NY,
        },
        "validation": {
            "start_samples": -_NUM_VALIDATION_SAMPLES,
            "ny": None,
        },
    }


def _wavenet_config(channels):
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": channels,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            }
        ],
        "head": None,
        "head_scale": 0.25,
    }


def _model_config():
    return {
        "net": {
            "name": "PackedWaveNet",
            "config": {
                "submodels": [
                    {"name": "small", "config": _wavenet_config(2)},
                    {"name": "large", "config": _wavenet_config(4)},
                ],
            },
        },
        "optimizer": {"lr": 0.001},
        "lr_scheduler": None,
        "loss": {"val_loss": "mse"},
    }


def _learning_config():
    return {
        "train_dataloader": {
            "batch_size": 2,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 0,
        },
        "val_dataloader": {
            "batch_size": 1,
            "num_workers": 0,
        },
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


def test_full_main_exports_packed_slimmable_container(tmp_path):
    x_path, y_path = _write_wav_pair(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()

    _full.main(
        _data_config(x_path, y_path),
        _model_config(),
        _learning_config(),
        outdir,
        no_show=True,
        make_plots=False,
    )

    model_path = outdir / "model.nam"
    assert model_path.exists()
    with open(model_path, "r") as fp:
        model = json.load(fp)

    assert model["architecture"] == "SlimmableContainer"
    assert model["weights"] == []
    submodels = model["config"]["submodels"]
    assert len(submodels) == 2
    assert [entry["model"]["architecture"] for entry in submodels] == [
        "WaveNet",
        "WaveNet",
    ]

    assert (outdir / "packed_best.json").exists()
    for i in range(2):
        assert (outdir / f"packed_best_submodel_{i}.ckpt").exists()
