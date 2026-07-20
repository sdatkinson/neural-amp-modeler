import json as _json
import math as _math
from pathlib import Path as _Path
from typing import Any as _Any
from typing import cast as _cast

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.data import Dataset as _Dataset
from nam.data import np_to_wav as _np_to_wav
from nam.models.parametric import ParametricDataset as _ParametricDataset
from nam.models._from_nam import init_from_nam as _init_from_nam
from nam.models.wavenet import WaveNet as _WaveNet
from nam.train.core import _ValidationStopping as _ValidationStopping
from nam.train.parametric import _CaptureBatchSampler as _CaptureBatchSampler
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _TRAIN_BUCKET as _TRAIN_BUCKET
from nam.train.parametric import _VALIDATION_BUCKET as _VALIDATION_BUCKET
from nam.train.parametric import _create_parametric_callbacks as _create_parametric_callbacks
from nam.train.parametric import _make_parametric_dataloader as _make_parametric_dataloader
from nam.train.parametric import _parametric_plot_label as _parametric_plot_label
from nam.train.parametric import main as _main
from tests.test_nam.test_models.test_base import MockBaseNet as _MockBaseNet


def test_parametric_panama_mel_loss(mocker):
    obj = _ParametricLightningModule(
        _MockBaseNet(1.0),
        loss_config=_ParametricLossConfig(mel_weight=6.2e-05),
    )
    mocked = mocker.patch.object(
        obj,
        "_mel_mrstft_loss",
        return_value=_torch.tensor(2.0),
    )
    preds = _torch.randn((3, 2048))
    targets = _torch.randn(preds.shape)

    loss_dict = obj._get_loss_dict(preds, targets)

    mocked.assert_called_once_with(preds, targets)
    assert loss_dict["Mel"].weight == _pytest.approx(6.2e-05)
    assert loss_dict["Mel"].value == _pytest.approx(_torch.tensor(2.0))


def test_parametric_loss_config_parses_mel_weight():
    parsed = _ParametricLossConfig.parse_config({"mel_weight": 6.2e-05})
    assert parsed["mel_weight"] == _pytest.approx(6.2e-05)


def test_mel_spectrogram_shapes_and_finite_without_torchaudio():
    from nam.train.parametric import _MelSpectrogram

    mel = _MelSpectrogram(
        sample_rate=48000, n_fft=256, win_length=256, hop_length=64, n_mels=40, power=1.0
    )
    batched = mel(_torch.randn(3, 4096))
    assert batched.shape[0] == 3 and batched.shape[1] == 40
    assert _torch.isfinite(batched).all()

    # A single (T,) waveform keeps the batch dim collapsed.
    single = mel(_torch.randn(4096))
    assert single.shape[0] == 40 and _torch.isfinite(single).all()


def test_parametric_mel_loss_runs_and_backprops_without_torchaudio():
    obj = _ParametricLightningModule(
        _MockBaseNet(1.0),
        loss_config=_ParametricLossConfig(mel_weight=6.2e-05),
    )
    # train_ensemble injects the dataset's sample rate onto the net; mirror that here.
    obj.net.sample_rate = 48000
    preds = _torch.randn((2, 8192), requires_grad=True)
    targets = _torch.randn((2, 8192))

    loss = obj._mel_mrstft_loss(preds, targets)

    assert loss.ndim == 0 and _torch.isfinite(loss)
    loss.backward()
    assert preds.grad is not None and _torch.isfinite(preds.grad).all()


def _write_json(path: _Path, payload: dict) -> None:
    with open(path, "w") as fp:
        _json.dump(payload, fp)


def _load_json(path: _Path) -> dict:
    with open(path) as fp:
        return _json.load(fp)


def _tiny_model_config() -> dict:
    return {
        "net": {
            "name": "HyperWaveNet",
            "config": {
                "layers": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "channels": 3,
                        "head": {
                            "out_channels": 2,
                            "kernel_size": 1,
                            "bias": False,
                        },
                        "kernel_size": 3,
                        "dilations": [1, 2],
                        "activation": "Tanh",
                    },
                    {
                        "condition_size": 1,
                        "input_size": 3,
                        "channels": 2,
                        "head": {
                            "out_channels": 1,
                            "kernel_size": 1,
                            "bias": True,
                        },
                        "kernel_sizes": [3, 3],
                        "dilations": [1, 2],
                        "activation": "Tanh",
                    },
                ],
                "head": None,
                "head_scale": 0.02,
                "params": [
                    {
                        "name": "gain",
                        "min": 0.0,
                        "max": 10.0,
                        "default": 5.0,
                        "type": "continuous",
                    },
                    {
                        "name": "mode",
                        "min": 0,
                        "max": 2,
                        "default": 1,
                        "type": "switch",
                        "enum_names": ["clean", "crunch", "lead"],
                    },
                ],
                "hypernet": {},
            },
        },
        "loss": {"val_loss": "esr"},
        "optimizer": {"lr": 0.01},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.99}},
    }


def _learning_config() -> dict:
    return {
        "train_dataloader": {
            "batch_size": 2,
            "shuffle": True,
            "pin_memory": False,
            "drop_last": False,
            "num_workers": 0,
        },
        "val_dataloader": {"batch_size": 1},
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "logger": False,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
        },
        "threshold_esr": None,
        "trainer_fit_kwargs": {},
    }


def _render_output(x: _np.ndarray, *, gain: float, mode_index: int) -> _np.ndarray:
    drive = 0.35 + 0.05 * gain
    voicing = 1.0 + 0.1 * mode_index
    return voicing * _np.tanh(drive * x)


def _build_data_config(tmp_path, *, normalize: bool) -> dict:
    sample_rate = 48_000
    x = (
        0.08 * _np.sin(2.0 * _np.pi * 220.0 * _np.arange(192) / sample_rate)
        + 0.03 * _np.cos(2.0 * _np.pi * 440.0 * _np.arange(192) / sample_rate)
    ).astype(_np.float32)
    captures = {
        "gain2_clean": {"gain": 2.0, "mode": "clean", "mode_index": 0},
        "gain8_crunch": {"gain": 8.0, "mode": "crunch", "mode_index": 1},
        "gain5_lead": {"gain": 5.0, "mode": "lead", "mode_index": 2},
    }

    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    outputs_dir.mkdir()

    x_path = inputs_dir / "input.wav"
    _np_to_wav(x, x_path, rate=sample_rate)
    for basename, capture in captures.items():
        y = _render_output(
            x, gain=capture["gain"], mode_index=capture["mode_index"]
        ).astype(_np.float32)
        _np_to_wav(y, outputs_dir / f"{basename}.wav", rate=sample_rate)

    data_config = {
        "type": "parametric",
        "common": {
            "x_path": str(x_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
        "train": [
            {
                "y_path": str(outputs_dir / "gain2_clean.wav"),
                "params": {"gain": 2.0, "mode": "clean"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": 16,
            },
            {
                "y_path": str(outputs_dir / "gain8_crunch.wav"),
                "params": {"gain": 8.0, "mode": "crunch"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": 16,
            },
        ],
        "validation": [
            {
                "y_path": str(outputs_dir / "gain5_lead.wav"),
                "params": {"gain": 5.0, "mode": "lead"},
                "start_samples": 96,
                "stop_samples": None,
                "ny": None,
            }
        ],
    }
    if normalize:
        data_config["joint"] = [
            {
                "name": "nam.data.normalize_joint_dataset_output",
                "kwargs": {"level_rms_dbfs": -18.0},
            }
        ]
    return data_config


def test_parametric_training_main_exports_baked_and_parametric_models(tmp_path):
    model_config = _tiny_model_config()
    data_config = _build_data_config(tmp_path, normalize=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _main(
        data_config=data_config,
        model_config=model_config,
        learning_config=_learning_config(),
        outdir=run_dir,
        no_show=True,
        make_plots=False,
    )

    baked = _load_json(run_dir / "model.nam")
    parametric = _load_json(run_dir / "model_parametric.nam")

    assert baked["architecture"] == "WaveNet"
    assert parametric["architecture"] == "HyperWaveNet"
    assert "params" in parametric["config"]
    assert (run_dir / "config_data.json").exists()
    assert (run_dir / "config_model.json").exists()
    assert (run_dir / "config_learning.json").exists()
    assert any(run_dir.rglob("*.ckpt"))

    # --- Output-scale compensation wiring (the fragile "approach B" path). ---
    # head_scale is a fixed (untrained) constant, so the exported value moving away from it
    # is direct evidence the joint -18 dBFS normalization was compensated for at export. A
    # silently-dropped scale (e.g. output_scale recovered as None) would leave it untouched.
    raw_head_scale = model_config["net"]["config"]["head_scale"]
    baked_head_scale = baked["config"]["head_scale"]
    assert baked_head_scale != _pytest.approx(raw_head_scale, rel=1e-3)
    # The stock WaveNet stores head_scale in its final weight slot; the scale hook moves both
    # by the same factor, so they must stay equal.
    assert baked["weights"][-1] == _pytest.approx(baked_head_scale)
    # Both export paths must apply the one shared compensation factor.
    assert parametric["config"]["head_scale"] == _pytest.approx(baked_head_scale)

    # The baked file is a plain stock WaveNet that reloads and runs.
    round_tripped = _init_from_nam(_load_json(run_dir / "model.nam"))
    assert isinstance(round_tripped, _WaveNet)
    y = round_tripped(_torch.randn(round_tripped.receptive_field + 64), pad_start=False)
    assert _torch.isfinite(y).all()


def test_parametric_training_without_normalization_skips_compensation(tmp_path):
    model_config = _tiny_model_config()
    data_config = _build_data_config(tmp_path, normalize=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _main(
        data_config=data_config,
        model_config=model_config,
        learning_config=_learning_config(),
        outdir=run_dir,
        no_show=True,
        make_plots=False,
    )

    baked = _load_json(run_dir / "model.nam")
    parametric = _load_json(run_dir / "model_parametric.nam")

    # No joint normalization => no output scaling => export applies no compensation, so the
    # constant head_scale survives untouched in both export paths.
    raw_head_scale = model_config["net"]["config"]["head_scale"]
    assert baked["config"]["head_scale"] == _pytest.approx(raw_head_scale)
    assert baked["weights"][-1] == _pytest.approx(raw_head_scale)
    assert parametric["config"]["head_scale"] == _pytest.approx(raw_head_scale)


def test_parametric_callbacks_include_validation_stopping():
    callbacks = _create_parametric_callbacks(
        {
            "trainer": {},
            "threshold_esr": 0.01,
        }
    )

    validation_stopping = [
        callback
        for callback in callbacks
        if isinstance(callback, _ValidationStopping)
    ]
    assert len(validation_stopping) == 1
    assert validation_stopping[0].monitor == "ESR"


def test_parametric_plot_label_uses_output_filename():
    dataset = _Dataset(
        x=_torch.zeros(3),
        y=_torch.zeros(3),
        nx=1,
        ny=None,
        y_path="/tmp/held_out_capture.wav",
    )
    ds = _ParametricDataset(dataset, _torch.tensor([1.0]))

    assert _parametric_plot_label(ds) == "held_out_capture.wav"


def test_parametric_plot_label_falls_back_to_params():
    dataset = _Dataset(x=_torch.zeros(3), y=_torch.zeros(3), nx=1, ny=None)
    ds = _ParametricDataset(dataset, _torch.tensor([3.0, -3.0]))

    assert _parametric_plot_label(ds) == "params=[3, -3]"


def test_parametric_lightning_training_logs_seen_audio_seen_params_bucket(monkeypatch):
    module = _ParametricLightningModule(_MockBaseNet(1.0))
    captured: dict[str, _Any] = {}

    def capture(dictionary, **kwargs):
        captured.update(dictionary)

    monkeypatch.setattr(module, "log_dict", capture)
    x = _torch.randn(3, 9)
    targets = _torch.randn(3, 9)

    module.on_train_epoch_start()
    loss = module.training_step((x, targets), 0)
    # Metrics are reduced once at epoch end, not logged per step.
    assert captured == {}
    module.on_train_epoch_end()

    esr_key = f"ESR/{_TRAIN_BUCKET}"
    assert esr_key in captured
    # The bucket reduces ESR as a global energy ratio (summed squared error over
    # summed squared target), which is robust to silent windows.
    preds = module(x, pad_start=False)
    expected_esr = float(_torch.sum((preds - targets) ** 2) / _torch.sum(targets ** 2))
    assert captured[esr_key] == _pytest.approx(expected_esr)
    # MSE is tracked per bucket alongside ESR.
    assert f"MSE/{_TRAIN_BUCKET}" in captured
    assert _torch.allclose(
        loss,
        _cast(_torch.Tensor, module._get_loss_dict(x, targets)["MSE"].value),
    )


def test_parametric_lightning_training_esr_survives_a_silent_batch(monkeypatch):
    """A near-silent batch must not poison the epoch's ESR with a divide-by-zero."""
    module = _ParametricLightningModule(_MockBaseNet(1.0))
    captured: dict[str, _Any] = {}
    monkeypatch.setattr(module, "log_dict", lambda d, **k: captured.update(d))

    module.on_train_epoch_start()
    module.training_step((_torch.randn(3, 9), _torch.randn(3, 9)), 0)
    # Silent target -> zero energy; a per-batch ESR mean would go inf here.
    module.training_step((_torch.randn(3, 9), _torch.zeros(3, 9)), 1)
    module.on_train_epoch_end()

    assert _math.isfinite(captured[f"ESR/{_TRAIN_BUCKET}"])


def test_parametric_lightning_validation_logs_unseen_audio_unseen_params_bucket(monkeypatch):
    module = _ParametricLightningModule(_MockBaseNet(1.0))
    captured: dict[str, _Any] = {}

    def capture(dictionary, **kwargs):
        captured.update(dictionary)

    monkeypatch.setattr(module, "log_dict", capture)
    x = _torch.randn(2, 7)
    targets = _torch.randn(2, 7)

    module.on_validation_epoch_start()
    assert module.validation_step((x, targets), 0) is None
    assert captured == {}
    module.on_validation_epoch_end()

    bucket_key = f"ESR/{_VALIDATION_BUCKET}"
    preds = module(x, pad_start=False)
    expected_esr = float(_torch.sum((preds - targets) ** 2) / _torch.sum(targets ** 2))
    assert captured[bucket_key] == _pytest.approx(expected_esr)
    # Bare keys back the checkpoint monitor and filename.
    assert captured["ESR"] == _pytest.approx(expected_esr)
    assert "val_loss" in captured
    assert captured["val_loss"] == _pytest.approx(captured[module._val_loss_key()])
    assert f"MSE/{_VALIDATION_BUCKET}" in captured


def test_parametric_lightning_nonfinite_gradient_preserves_adamw_state(monkeypatch):
    module = _ParametricLightningModule(
        _torch.nn.Linear(1, 1, bias=False),
        optimizer_config={
            "lr": 0.01,
            "weight_decay": 0.1,
        },
    )
    monkeypatch.setattr(module, "clip_gradients", lambda *args, **kwargs: None)
    optimizer = module.configure_optimizers()
    parameter = next(module.parameters())

    parameter.grad = _torch.full_like(parameter, 0.25)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    weight_before = parameter.detach().clone()
    state_before = {
        key: value.detach().clone() if _torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }

    parameter.grad = _torch.full_like(parameter, float("nan"))
    module.configure_gradient_clipping(
        optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
    )

    assert parameter.grad is None
    optimizer.step()
    assert _torch.equal(parameter, weight_before)
    for key, before in state_before.items():
        after = optimizer.state[parameter][key]
        if _torch.is_tensor(before):
            assert _torch.equal(after, before)
        else:
            assert after == before


def test_parametric_lightning_large_finite_gradient_updates(monkeypatch):
    module = _ParametricLightningModule(
        _torch.nn.Linear(1, 1, bias=False),
        optimizer_config={"lr": 0.01},
    )
    monkeypatch.setattr(module, "clip_gradients", lambda *args, **kwargs: None)
    optimizer = module.configure_optimizers()
    parameter = next(module.parameters())
    weight_before = parameter.detach().clone()
    parameter.grad = _torch.full_like(parameter, 2.0)

    module.configure_gradient_clipping(
        optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
    )

    assert parameter.grad is not None
    optimizer.step()
    assert not _torch.equal(parameter, weight_before)
    assert optimizer.state[parameter]["step"] == 1


def test_capture_batch_sampler_keeps_batches_within_one_capture():
    capture_lengths = [5, 3, 4]
    sampler = _CaptureBatchSampler(
        capture_lengths, batch_size=2, shuffle=False, drop_last=False
    )
    offsets = [0, 5, 8]
    ranges = [range(o, o + n) for o, n in zip(offsets, capture_lengths)]

    batches = list(sampler)
    # Every batch's indices must come from a single capture range.
    for batch in batches:
        owners = {
            next(i for i, r in enumerate(ranges) if idx in r) for idx in batch
        }
        assert len(owners) == 1, f"batch {batch} mixed captures {owners}"

    # ceil(5/2)+ceil(3/2)+ceil(4/2) = 3+2+2 = 7 batches, covering every index once.
    assert len(batches) == len(sampler) == 7
    assert sorted(idx for batch in batches for idx in batch) == list(range(12))


def test_capture_batch_sampler_drop_last_drops_partial_per_capture():
    sampler = _CaptureBatchSampler(
        [5, 3], batch_size=2, shuffle=False, drop_last=True
    )
    batches = list(sampler)
    # floor(5/2)+floor(3/2) = 2+1 = 3 full batches; the partial tail of each capture is gone.
    assert len(batches) == len(sampler) == 3
    assert all(len(batch) == 2 for batch in batches)


def test_capture_batch_sampler_shuffle_is_reproducible_and_varies_by_epoch():
    sampler = _CaptureBatchSampler(
        [4, 4], batch_size=2, shuffle=True, drop_last=False
    )
    _torch.manual_seed(0)
    epoch0 = list(sampler)
    epoch1 = list(sampler)
    _torch.manual_seed(0)
    rerun0 = list(sampler)
    assert epoch0 == rerun0  # same global seed -> same stream
    assert epoch0 != epoch1  # advancing the RNG reshuffles each epoch


def test_make_parametric_dataloader_can_opt_out_of_capture_grouping():
    dataset = list(range(6))
    grouped = _make_parametric_dataloader(
        dataset, {"batch_size": 2, "shuffle": False}
    )
    assert isinstance(grouped.batch_sampler, _CaptureBatchSampler)

    ungrouped = _make_parametric_dataloader(
        dataset, {"batch_size": 2, "shuffle": False, "capture_grouped_batches": False}
    )
    assert not isinstance(ungrouped.batch_sampler, _CaptureBatchSampler)
