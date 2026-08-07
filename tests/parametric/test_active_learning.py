import json as _json
import math as _math
from copy import deepcopy as _deepcopy

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.models.parametric import assemble_raw_params as _assemble_raw_params
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric import split_param_indices as _split_param_indices
from nam.models.parametric import switch_combinations as _switch_combinations
from nam.data import init_dataset as _init_dataset
from nam.data import np_to_wav as _np_to_wav
from nam.data import Split as _Split
from nam.train.active_learning import _clear_device_cache as _clear_device_cache
from nam.train.active_learning import (
    DisagreementCandidate as _DisagreementCandidate,
)
from nam.train.active_learning import (
    append_to_data_config as _append_to_data_config,
)
from nam.train.active_learning import (
    cluster_and_select as _cluster_and_select,
)
from nam.train.active_learning import (
    emit_proposals as _emit_proposals,
)
from nam.train.active_learning import (
    find_disagreement_settings as _find_disagreement_settings,
)
from nam.train.active_learning import train_ensemble as _train_ensemble
from nam.train.active_learning import (
    _resolve_ensemble_parallel_plan as _resolve_ensemble_parallel_plan,
)
from nam.train.active_learning import (
    _resolve_g_opt_parallel_plan as _resolve_g_opt_parallel_plan,
)
from nam.train.active_learning import _g_opt_item_seed as _g_opt_item_seed
from nam.train.active_learning import _resolve_member_batching as _resolve_member_batching
from nam.train.active_learning import _g_opt_cuda_train_mode_safe as _g_opt_cuda_train_mode_safe
from nam.train.active_learning import (
    _coerce_parallel_dataloader_workers as _coerce_parallel_dataloader_workers,
)
from nam.train.parametric import _ParametricLightningModule


def _concat_lstm_model_config() -> dict:
    return {
        "net": {
            "name": "ConcatLSTM",
            "config": {
                "hidden_size": 4,
                "num_layers": 1,
                "train_burn_in": 8,
                "train_truncate": 8,
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
        "val_dataloader": {"batch_size": 1, "pin_memory": False, "num_workers": 0},
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


def _selection_model_config() -> dict:
    return {
        "net": {
            "name": "ConcatLSTM",
            "config": {
                "params": [
                    {
                        "name": "gain",
                        "min": 0.0,
                        "max": 10.0,
                        "default": 5.0,
                        "type": "continuous",
                    },
                    {
                        "name": "boost",
                        "min": 0,
                        "max": 1,
                        "default": 0,
                        "type": "switch",
                        "enum_names": ["Off", "On"],
                    },
                ]
            },
        }
    }


def _candidate(gain: float, boost: int, score: float) -> _DisagreementCandidate:
    return _DisagreementCandidate(
        raw_params=_torch.tensor([gain, float(boost)], dtype=_torch.float32),
        switch_combo=(boost,),
        score=score,
    )


def _switch_only_model_config() -> dict:
    return {
        "net": {
            "name": "ConcatLSTM",
            "config": {
                "params": [
                    {
                        "name": "boost",
                        "min": 0,
                        "max": 2,
                        "default": 0,
                        "type": "switch",
                        "enum_names": ["Off", "Low", "High"],
                    }
                ]
            },
        }
    }


def _switch_only_candidate(boost: int, score: float) -> _DisagreementCandidate:
    return _DisagreementCandidate(
        raw_params=_torch.tensor([float(boost)], dtype=_torch.float32),
        switch_combo=(boost,),
        score=score,
    )


def _render_output(x: _np.ndarray, *, gain: float, mode_index: int) -> _np.ndarray:
    drive = 0.35 + 0.05 * gain
    voicing = 1.0 + 0.1 * mode_index
    return (voicing * _np.tanh(drive * x)).astype(_np.float32)


def _build_data_config(tmp_path, *, train_ny: int = 16) -> dict:
    sample_rate = 48_000
    x = (
        0.08 * _np.sin(2.0 * _np.pi * 220.0 * _np.arange(192) / sample_rate)
        + 0.03 * _np.cos(2.0 * _np.pi * 440.0 * _np.arange(192) / sample_rate)
    ).astype(_np.float32)

    x_path = tmp_path / "input.wav"
    _np_to_wav(x, x_path, rate=sample_rate)

    for basename, gain, mode_index in (
        ("gain2_clean.wav", 2.0, 0),
        ("gain8_crunch.wav", 8.0, 1),
        ("gain5_lead.wav", 5.0, 2),
    ):
        _np_to_wav(
            _render_output(x, gain=gain, mode_index=mode_index),
            tmp_path / basename,
            rate=sample_rate,
        )

    return {
        "type": "parametric",
        "common": {
            "x_path": str(x_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
        "train": [
            {
                "y_path": str(tmp_path / "gain2_clean.wav"),
                "params": {"gain": 2.0, "mode": "clean"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": train_ny,
            },
            {
                "y_path": str(tmp_path / "gain8_crunch.wav"),
                "params": {"gain": 8.0, "mode": "crunch"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": train_ny,
            },
        ],
        "validation": [
            {
                "y_path": str(tmp_path / "gain5_lead.wav"),
                "params": {"gain": 5.0, "mode": "lead"},
                "start_samples": 96,
                "stop_samples": None,
                "ny": None,
            }
        ],
    }


def test_train_ensemble_writes_stable_member_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    outdir = tmp_path / "run"

    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        _concat_lstm_model_config(),
        _learning_config(),
        outdir,
        ensemble_size=2,
    )

    assert len(checkpoint_paths) == 2
    assert [path.name for path in checkpoint_paths] == ["best.ckpt", "best.ckpt"]
    assert all(path.exists() for path in checkpoint_paths)


def test_train_ensemble_checkpoints_reload_via_task5_path(tmp_path, monkeypatch):
    # Task 5 (find_disagreement_settings) reloads each member with
    # load_from_checkpoint(path, **parse_config(model_config)) and runs net forward on
    # raw params. Prove a returned checkpoint honors exactly that contract.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()

    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    module = _ParametricLightningModule.load_from_checkpoint(
        str(checkpoint_paths[0]),
        **_ParametricLightningModule.parse_config(model_config),
    )
    module.eval()
    x = _torch.zeros(32)
    raw_params = _torch.tensor([5.0, 1.0])  # gain (continuous), mode index (switch)
    with _torch.no_grad():
        y = module(x, raw_params)
    assert y.shape[-1] == x.shape[-1]
    assert _torch.isfinite(y).all()


def test_train_ensemble_accepts_ny_in_common(tmp_path, monkeypatch):
    # init_dataset merges common into each entry, so ny declared once in common is valid
    # and must not be rejected as a missing per-entry ny.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    data_config = _build_data_config(tmp_path)
    data_config["common"]["ny"] = 16
    for entry in data_config["train"]:
        del entry["ny"]

    checkpoint_paths = _train_ensemble(
        data_config,
        _concat_lstm_model_config(),
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0].exists()


def test_train_ensemble_allows_default_full_length_windows(tmp_path, monkeypatch):
    # ny=None is the stock "use the full clip length" sentinel: the normal long-window
    # LSTM case, which must not trip the burn-in guard.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    data_config = _build_data_config(tmp_path)
    for entry in data_config["train"]:
        entry["ny"] = None

    checkpoint_paths = _train_ensemble(
        data_config,
        _concat_lstm_model_config(),
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0].exists()


def test_train_ensemble_rejects_train_windows_at_or_below_burn_in(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )

    with _pytest.raises(ValueError, match=r"train\[0\]\.ny \(=8\) must be greater than train_burn_in=8"):
        _train_ensemble(
            _build_data_config(tmp_path, train_ny=8),
            _concat_lstm_model_config(),
            _learning_config(),
            tmp_path / "run",
            ensemble_size=1,
        )


def test_train_ensemble_allows_round_tripped_param_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    data_config = _build_data_config(tmp_path)
    data_config["common"]["param_specs"] = _deepcopy(model_config["net"]["config"]["params"])

    checkpoint_paths = _train_ensemble(
        data_config,
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0].exists()


def test_train_ensemble_rejects_mismatched_round_tripped_param_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    data_config = _build_data_config(tmp_path)
    data_config["common"]["param_specs"] = _deepcopy(model_config["net"]["config"]["params"])
    data_config["common"]["param_specs"][0]["max"] = 11.0

    with _pytest.raises(ValueError, match="common.param_specs does not match"):
        _train_ensemble(
            data_config,
            model_config,
            _learning_config(),
            tmp_path / "run",
            ensemble_size=1,
        )


def test_clear_device_cache_calls_mps_empty_cache(monkeypatch):
    mps_module = getattr(_torch, "mps", None)
    if mps_module is None or not hasattr(mps_module, "empty_cache"):
        _pytest.skip("torch.mps.empty_cache is unavailable in this environment")

    calls = []
    monkeypatch.setattr(mps_module, "empty_cache", lambda: calls.append(True))

    _clear_device_cache(_torch.device("mps"))

    assert calls == [True]


def test_find_disagreement_settings_returns_candidates_per_restart_and_switch_combo(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=2,
        base_seed=7,
    )

    candidates = _find_disagreement_settings(
        checkpoint_paths,
        model_config,
        g_opt_input_wav=tmp_path / "input.wav",
        num_restarts=2,
        num_steps=3,
        g_opt_ny=32,
        g_opt_batch_size=2,
        seed=11,
    )

    specs = tuple(
        _ParamSpec.from_dict(spec)
        for spec in model_config["net"]["config"]["params"]
    )
    continuous_idx, switch_idx, _ = _split_param_indices(specs)
    expected_switch_combos = set(_switch_combinations(specs))

    assert len(candidates) == 2 * len(expected_switch_combos)
    for candidate in candidates:
        assert _math.isfinite(candidate.score)
        assert candidate.switch_combo in expected_switch_combos
        assert candidate.raw_params.device.type == "cpu"
        assert not candidate.raw_params.requires_grad
        assert candidate.raw_params.shape == (len(specs),)
        for raw_index, combo_value in zip(switch_idx, candidate.switch_combo):
            assert candidate.raw_params[raw_index].item() == _pytest.approx(combo_value)
        for raw_index in continuous_idx:
            spec = specs[raw_index]
            value = candidate.raw_params[raw_index].item()
            assert spec.min <= value <= spec.max


def test_find_disagreement_settings_keeps_gradient_flow_on_continuous_latents(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
        base_seed=3,
    )

    specs = tuple(
        _ParamSpec.from_dict(spec)
        for spec in model_config["net"]["config"]["params"]
    )
    module = _ParametricLightningModule.load_from_checkpoint(
        str(checkpoint_paths[0]),
        **_ParametricLightningModule.parse_config(model_config),
    )
    member = module.net
    member.eval()
    member.requires_grad_(False)

    x_batch = _torch.stack(
        [
            _torch.linspace(-0.5, 0.5, 32),
            _torch.linspace(0.5, -0.5, 32),
        ]
    )
    z = _torch.tensor([0.0], dtype=_torch.float32, requires_grad=True)
    raw_params = _assemble_raw_params(z, (1,), specs)
    outputs = member(x_batch, raw_params)
    score = outputs.var(dim=0, unbiased=False).mean()
    score.backward()

    assert z.grad is not None
    assert _torch.isfinite(z.grad).all()
    assert z.grad.shape == z.shape

    z_for_switch = _torch.tensor([0.25], dtype=_torch.float32, requires_grad=True)
    switch_grad = _torch.autograd.grad(
        _assemble_raw_params(z_for_switch, (0,), specs)[1], z_for_switch
    )[0]
    assert _torch.equal(switch_grad, _torch.zeros_like(z_for_switch))


def _all_switch_model_config() -> dict:
    config = _deepcopy(_concat_lstm_model_config())
    config["net"]["config"]["params"] = [
        {
            "name": "mode",
            "min": 0,
            "max": 2,
            "default": 1,
            "type": "switch",
            "enum_names": ["clean", "crunch", "lead"],
        }
    ]
    return config


def _all_switch_data_config(tmp_path) -> dict:
    data_config = _deepcopy(_build_data_config(tmp_path))
    for entry in data_config["train"]:
        entry["params"] = {"mode": entry["params"]["mode"]}
    for entry in data_config["validation"]:
        entry["params"] = {"mode": entry["params"]["mode"]}
    return data_config


def test_find_disagreement_settings_all_switch_evaluates_once_per_combo(
    tmp_path, monkeypatch
):
    # With no continuous latents every restart/step would reproduce the identical
    # candidate, so the search short-circuits to one evaluation per switch combo
    # regardless of num_restarts (no duplicate candidates, no wasted no-op steps).
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _all_switch_model_config()
    checkpoint_paths = _train_ensemble(
        _all_switch_data_config(tmp_path),
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=2,
        base_seed=5,
    )

    specs = tuple(
        _ParamSpec.from_dict(spec)
        for spec in model_config["net"]["config"]["params"]
    )
    expected_switch_combos = set(_switch_combinations(specs))

    candidates = _find_disagreement_settings(
        checkpoint_paths,
        model_config,
        g_opt_input_wav=tmp_path / "input.wav",
        num_restarts=4,
        num_steps=3,
        g_opt_ny=32,
        g_opt_batch_size=2,
        seed=13,
    )

    assert len(candidates) == len(expected_switch_combos)
    assert {c.switch_combo for c in candidates} == expected_switch_combos
    for candidate in candidates:
        assert _math.isfinite(candidate.score)
        assert candidate.raw_params.shape == (len(specs),)


def test_cluster_and_select_collapses_near_duplicates_per_switch_combo():
    selected = _cluster_and_select(
        [
            _candidate(5.00, 0, 0.91),
            _candidate(5.02, 0, 0.84),
            _candidate(8.40, 0, 0.55),
            _candidate(5.01, 1, 0.87),
        ],
        _selection_model_config(),
        max_per_round=10,
        cluster_threshold=0.01,
    )

    assert [(candidate.switch_combo, candidate.score) for candidate in selected] == [
        ((0,), 0.91),
        ((1,), 0.87),
        ((0,), 0.55),
    ]
    assert [candidate.raw_params.tolist() for candidate in selected] == [
        [5.0, 0.0],
        [5.0, 1.0],
        [8.5, 0.0],
    ]


def test_cluster_and_select_dedupes_after_quantization():
    selected = _cluster_and_select(
        [
            _candidate(5.10, 0, 0.72),
            _candidate(5.24, 0, 0.94),
        ],
        _selection_model_config(),
        max_per_round=10,
        cluster_threshold=0.01,
    )

    assert len(selected) == 1
    assert selected[0].score == _pytest.approx(0.94)
    assert selected[0].switch_combo == (0,)
    assert selected[0].raw_params.tolist() == [5.0, 0.0]


def test_cluster_and_select_limits_count_sorts_globally_and_quantizes():
    selected = _cluster_and_select(
        [
            _candidate(1.10, 0, 0.20),
            _candidate(4.26, 0, 0.80),
            _candidate(7.74, 0, 0.50),
            _candidate(2.24, 1, 0.90),
        ],
        _selection_model_config(),
        max_per_round=3,
        cluster_threshold=0.001,
    )

    assert [candidate.score for candidate in selected] == [0.90, 0.80, 0.50]
    assert len(selected) == 3
    for candidate in selected:
        quantized_gain = float(candidate.raw_params[0])
        assert 2.0 * quantized_gain == _pytest.approx(round(2.0 * quantized_gain))


def test_emit_proposals_writes_json_with_decoded_named_params(tmp_path, capsys):
    selected = [
        _candidate(4.5, 1, 1.25),
        _candidate(0.0, 0, 0.50),
    ]

    output_path, proposals = _emit_proposals(
        selected,
        _selection_model_config(),
        round_idx=3,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "proposed_captures_round_3.json"
    with output_path.open() as fp:
        payload = _json.load(fp)

    expected = [
        {
            "params": {"gain": 4.5, "boost": "On"},
            "score": 1.25,
            "y_path": "r3_g4.5_bOn.wav",
        },
        {
            "params": {"gain": 0.0, "boost": "Off"},
            "score": 0.5,
            "y_path": "r3_g0_bOff.wav",
        },
    ]
    assert payload == expected
    assert proposals == expected
    captured = capsys.readouterr()
    assert "Capture checklist:" in captured.out
    assert "1. r3_g4.5_bOn.wav -> gain=4.5, boost=On" in captured.out


def test_append_to_data_config_preserves_splits_and_does_not_mutate_input(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    model_config = _selection_model_config()
    selected = [
        _candidate(4.5, 1, 1.25),
        _candidate(8.0, 0, 0.75),
    ]
    x = _np.linspace(-0.5, 0.5, 64, dtype=_np.float32)
    for filename, scale in (
        ("input.wav", 1.0),
        ("seed.wav", 0.8),
        ("held_out.wav", 0.6),
        ("r4_g4.5_bOn.wav", 0.7),
        ("r4_g8_bOff.wav", 0.9),
    ):
        _np_to_wav(x * scale, tmp_path / filename, rate=48_000)
    prev_data_config = {
        "type": "parametric",
        "common": {
            "x_path": "input.wav",
            "delay": 0,
            "nx": 1,
            "param_specs": _deepcopy(model_config["net"]["config"]["params"]),
        },
        "train": {
            "y_path": "seed.wav",
            "params": {"gain": 1.0, "boost": "Off"},
            "start_seconds": 0.0,
            "stop_seconds": None,
            "ny": 16,
        },
        "validation": [
            {
                "y_path": "held_out.wav",
                "params": {"gain": 2.0, "boost": "On"},
            }
        ],
    }
    original = _deepcopy(prev_data_config)
    proposal_path, proposals = _emit_proposals(
        selected,
        model_config,
        round_idx=4,
        output_dir=tmp_path,
    )

    new_data_config, output_path = _append_to_data_config(
        prev_data_config,
        proposals,
        model_config,
        round_idx=4,
        output_dir=tmp_path,
    )

    assert prev_data_config == original
    assert proposal_path.exists()
    assert output_path == tmp_path / "aggregated_data_config_4.json"
    plot_path = tmp_path / "accepted_capture_distributions_round_4.png"
    assert plot_path.exists()
    assert new_data_config["type"] == "parametric"
    assert new_data_config["common"] == {
        "x_path": "input.wav",
        "delay": 0,
        "nx": 1,
        "param_specs": [
            _ParamSpec.from_dict(spec).to_dict()
            for spec in model_config["net"]["config"]["params"]
        ],
    }
    assert new_data_config["validation"] == original["validation"]
    assert isinstance(new_data_config["train"], list)
    assert new_data_config["train"][0] == original["train"]
    assert new_data_config["train"][1:] == [
        {
            "y_path": "r4_g4.5_bOn.wav",
            "params": {"gain": 4.5, "boost": "On"},
            "start_seconds": 0.0,
            "stop_seconds": None,
            "ny": 16,
        },
        {
            "y_path": "r4_g8_bOff.wav",
            "params": {"gain": 8.0, "boost": "Off"},
            "start_seconds": 0.0,
            "stop_seconds": None,
            "ny": 16,
        },
    ]
    with proposal_path.open() as fp:
        proposals = _json.load(fp)
    assert [entry["y_path"] for entry in proposals] == [
        train_entry["y_path"] for train_entry in new_data_config["train"][1:]
    ]
    with output_path.open() as fp:
        persisted = _json.load(fp)
    assert persisted == new_data_config
    _init_dataset(new_data_config, _Split.TRAIN)


def test_cluster_and_select_rejects_bad_arguments():
    model_config = _selection_model_config()
    with _pytest.raises(ValueError, match="max_per_round"):
        _cluster_and_select([_candidate(5.0, 0, 0.5)], model_config, max_per_round=0)
    with _pytest.raises(ValueError, match="cluster_threshold"):
        _cluster_and_select(
            [_candidate(5.0, 0, 0.5)],
            model_config,
            max_per_round=1,
            cluster_threshold=-0.1,
        )


def test_cluster_and_select_empty_candidates_returns_empty():
    assert _cluster_and_select([], _selection_model_config(), max_per_round=5) == []


def test_cluster_and_select_switch_only_collapses_per_combo():
    selected = _cluster_and_select(
        [
            _switch_only_candidate(0, 0.10),
            _switch_only_candidate(0, 0.90),
            _switch_only_candidate(2, 0.50),
        ],
        _switch_only_model_config(),
        max_per_round=10,
    )

    assert [(candidate.switch_combo, candidate.score) for candidate in selected] == [
        ((0,), 0.90),
        ((2,), 0.50),
    ]
    assert [candidate.raw_params.tolist() for candidate in selected] == [[0.0], [2.0]]


def test_cluster_and_select_rejects_switch_combo_raw_mismatch():
    mismatched = _DisagreementCandidate(
        raw_params=_torch.tensor([5.0, 1.0], dtype=_torch.float32),
        switch_combo=(0,),
        score=0.5,
    )
    with _pytest.raises(ValueError, match="switch column"):
        _cluster_and_select(
            [mismatched], _selection_model_config(), max_per_round=1
        )


def test_emit_proposals_empty_selection_writes_empty_list(tmp_path, capsys):
    output_path, proposals = _emit_proposals(
        [], _selection_model_config(), round_idx=2, output_dir=tmp_path
    )

    assert proposals == []
    with output_path.open() as fp:
        assert _json.load(fp) == []
    assert "Capture checklist:" in capsys.readouterr().out


def test_append_to_data_config_empty_selection_is_noop(tmp_path):
    model_config = _selection_model_config()
    prev_data_config = {
        "type": "parametric",
        "common": {"x_path": "input.wav", "delay": 0, "nx": 1},
        "train": [
            {"y_path": "seed.wav", "params": {"gain": 1.0, "boost": "Off"}, "ny": 16}
        ],
        "validation": [],
    }
    original = _deepcopy(prev_data_config)

    new_data_config, output_path = _append_to_data_config(
        prev_data_config,
        [],
        model_config,
        round_idx=1,
        output_dir=tmp_path,
        plot=False,
    )

    assert prev_data_config == original
    assert output_path == tmp_path / "aggregated_data_config_1.json"
    # train is canonicalized to a list and param_specs injected, but no entries are added.
    assert new_data_config["train"] == original["train"]
    assert "param_specs" in new_data_config["common"]
    assert not (tmp_path / "accepted_capture_distributions_round_1.png").exists()


def _member_state_dicts(checkpoint_paths, model_config):
    states = []
    for path in checkpoint_paths:
        module = _ParametricLightningModule.load_from_checkpoint(
            str(path),
            **_ParametricLightningModule.parse_config(model_config),
        )
        states.append({k: v.detach().clone() for k, v in module.net.state_dict().items()})
    return states


def test_resolve_ensemble_parallel_plan_defaults_to_serial_off_multi_gpu(monkeypatch):
    # CPU / single-GPU / MPS default to one worker so the default never over-subscribes
    # a single device.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    workers, devices = _resolve_ensemble_parallel_plan(4, None)
    assert workers == 1
    assert devices == [_torch.device("cpu")] * 4


def test_resolve_ensemble_parallel_plan_single_gpu_defaults_serial(monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cuda"),
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    workers, devices = _resolve_ensemble_parallel_plan(3, None)
    assert workers == 1
    assert devices == [_torch.device("cuda:0")] * 3


def test_resolve_ensemble_parallel_plan_multi_gpu_one_member_per_gpu(monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cuda"),
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    workers, devices = _resolve_ensemble_parallel_plan(4, None)
    assert workers == 2
    assert devices == [
        _torch.device("cuda:0"),
        _torch.device("cuda:1"),
        _torch.device("cuda:0"),
        _torch.device("cuda:1"),
    ]


def test_resolve_ensemble_parallel_plan_explicit_max_workers_oversubscribes(monkeypatch):
    # An explicit count over-subscribes a single GPU (round-robin collapses to cuda:0),
    # capped at ensemble_size.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cuda"),
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    workers, devices = _resolve_ensemble_parallel_plan(4, 3)
    assert workers == 3
    assert devices == [_torch.device("cuda:0")] * 4

    # Capped at ensemble_size even when max_workers exceeds it.
    workers, _ = _resolve_ensemble_parallel_plan(2, 8)
    assert workers == 2


def test_resolve_ensemble_parallel_plan_rejects_nonpositive_max_workers(monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    with _pytest.raises(ValueError):
        _resolve_ensemble_parallel_plan(4, 0)


def test_resolve_g_opt_parallel_plan_defaults_to_serial(monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    workers, devices = _resolve_g_opt_parallel_plan(10, None)
    assert workers == 1
    assert devices == [_torch.device("cpu")]


def test_resolve_g_opt_parallel_plan_single_gpu_stays_serial_even_with_max_workers(
    monkeypatch,
):
    # Unlike ensemble training, g-opt never over-subscribes a device -- a single GPU
    # always stays serial regardless of --max-workers.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cuda"),
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    workers, devices = _resolve_g_opt_parallel_plan(10, 4)
    assert workers == 1
    assert devices == [_torch.device("cuda")]


def test_resolve_g_opt_parallel_plan_multi_gpu_caps_at_one_per_gpu(monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cuda"),
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    workers, devices = _resolve_g_opt_parallel_plan(10, 4)
    assert workers == 2
    assert devices == [_torch.device("cuda:0"), _torch.device("cuda:1")]


def test_resolve_g_opt_parallel_plan_caps_at_num_items(monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cuda"),
    )
    monkeypatch.setattr("torch.cuda.device_count", lambda: 4)
    workers, _ = _resolve_g_opt_parallel_plan(2, 4)
    assert workers == 2


def test_resolve_g_opt_parallel_plan_rejects_nonpositive_max_workers(monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    with _pytest.raises(ValueError):
        _resolve_g_opt_parallel_plan(10, 0)


def test_g_opt_item_seed_is_deterministic_and_distinct():
    assert _g_opt_item_seed(11, 0, 0) == _g_opt_item_seed(11, 0, 0)
    assert _g_opt_item_seed(11, 0, 0) != _g_opt_item_seed(11, 0, 1)
    assert _g_opt_item_seed(11, 0, 0) != _g_opt_item_seed(11, 1, 0)
    assert _g_opt_item_seed(11, 0, None) != _g_opt_item_seed(11, 0, 0)


def test_find_disagreement_settings_parallel_matches_serial(tmp_path, monkeypatch):
    # Per-item seeding (nam.train.active_learning._g_opt_item_seed) depends only on
    # (seed, combo_idx, restart_idx), not on worker count or completion order, so a
    # forced-parallel run must reproduce the serial candidates exactly.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=2,
        base_seed=7,
    )
    common_kwargs = dict(
        checkpoint_paths=checkpoint_paths,
        model_config=model_config,
        g_opt_input_wav=tmp_path / "input.wav",
        num_restarts=2,
        num_steps=3,
        g_opt_ny=32,
        g_opt_batch_size=2,
        seed=11,
    )

    serial = _find_disagreement_settings(**common_kwargs)

    import nam.train.active_learning as al

    monkeypatch.setattr(
        al,
        "_resolve_g_opt_parallel_plan",
        lambda num_items, max_workers: (
            2,
            [_torch.device("cpu"), _torch.device("cpu")],
        ),
    )
    parallel = _find_disagreement_settings(max_workers=2, **common_kwargs)

    key = lambda c: (c.switch_combo, c.raw_params.tolist())
    serial_sorted = sorted(serial, key=key)
    parallel_sorted = sorted(parallel, key=key)
    assert len(serial_sorted) == len(parallel_sorted)
    for s, p in zip(serial_sorted, parallel_sorted):
        assert s.switch_combo == p.switch_combo
        assert _torch.allclose(s.raw_params, p.raw_params, atol=1e-5)
        assert s.score == _pytest.approx(p.score, abs=1e-5)


def test_find_disagreement_settings_parallel_propagates_worker_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=2,
        base_seed=7,
    )

    import nam.train.active_learning as al

    monkeypatch.setattr(
        al,
        "_resolve_g_opt_parallel_plan",
        lambda num_items, max_workers: (
            2,
            [_torch.device("cpu"), _torch.device("cpu")],
        ),
    )
    with _pytest.raises(RuntimeError, match="failed during parallel search"):
        _find_disagreement_settings(
            [tmp_path / "does_not_exist.ckpt", checkpoint_paths[0]],
            model_config,
            g_opt_input_wav=tmp_path / "input.wav",
            num_restarts=2,
            num_steps=3,
            g_opt_ny=32,
            g_opt_batch_size=2,
            seed=11,
            max_workers=2,
        )


def test_resolve_member_batching_uses_assigned_gpu_memory(monkeypatch):
    ny = 22050
    bytes_per_window = 5120 * ny
    learning_config = {
        "train_dataloader": {"batch_size": 32, "drop_last": True},
        "val_dataloader": {"batch_size": 32},
        "trainer": {"accumulate_grad_batches": 16},
        "batch_sizing": {
            "auto": True,
            "target_effective_batch_size": 512,
            "max_physical_batch_size": 512,
            "memory_fraction": 0.75,
            "bytes_per_timestep": 5120,
            "ny": ny,
        },
    }
    monkeypatch.setattr(
        "nam.train.active_learning._device_memory_bytes",
        lambda device: (4 * bytes_per_window, 8 * bytes_per_window),
    )
    monkeypatch.setattr("torch.cuda.get_device_name", lambda device: "test GPU")

    plan = _resolve_member_batching(
        learning_config,
        device=_torch.device("cuda:2"),
        num_train_windows=2_000,
        num_validation_windows=20,
    )

    assert plan["device"] == "cuda:2"
    assert plan["device_name"] == "test GPU"
    assert plan["physical_batch_size"] == 3
    assert plan["accumulate_grad_batches"] == 171
    assert plan["effective_batch_size"] == 513
    assert learning_config["train_dataloader"]["batch_size"] == 3
    assert learning_config["trainer"]["accumulate_grad_batches"] == 171
    assert learning_config["val_dataloader"]["batch_size"] == 3


def test_train_ensemble_parallel_matches_serial_members(tmp_path, monkeypatch):
    # Parallel training must reproduce the serial per-member results (same seed per
    # member, results keyed by member index) up to floating-point noise.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    data_config = _build_data_config(tmp_path)

    serial = _train_ensemble(
        _deepcopy(data_config),
        model_config,
        _learning_config(),
        tmp_path / "serial",
        ensemble_size=2,
    )
    parallel = _train_ensemble(
        _deepcopy(data_config),
        model_config,
        _learning_config(),
        tmp_path / "parallel",
        ensemble_size=2,
        max_workers=2,
    )

    assert len(parallel) == 2
    assert [p.parent.name for p in parallel] == ["member_00", "member_01"]

    serial_states = _member_state_dicts(serial, model_config)
    parallel_states = _member_state_dicts(parallel, model_config)
    for member_idx, (s_state, p_state) in enumerate(zip(serial_states, parallel_states)):
        assert s_state.keys() == p_state.keys()
        for key in s_state:
            assert _torch.allclose(s_state[key], p_state[key], atol=1e-4), (
                f"member {member_idx} weight {key} diverged between serial and parallel"
            )


def test_train_ensemble_parallel_propagates_member_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    data_config = _build_data_config(tmp_path)
    # Passes the parent-side window validation but fails in the worker when the dataset
    # tries to load a missing capture wav.
    data_config["train"][0]["y_path"] = str(tmp_path / "does_not_exist.wav")

    with _pytest.raises(RuntimeError, match="failed during parallel training"):
        _train_ensemble(
            data_config,
            _concat_lstm_model_config(),
            _learning_config(),
            tmp_path / "run",
            ensemble_size=2,
            max_workers=2,
        )


def test_g_opt_cuda_train_mode_safe_requires_no_truncation():
    # The g-opt cuda fast path (train() mode re-enabling cuDNN RNN backward) is only valid
    # when train() forward equals eval() forward: train_truncate must be off, no dropout/bn.
    cfg = _concat_lstm_model_config()
    members = [_ParametricLightningModule.init_from_config(cfg).net]

    assert cfg["net"]["config"]["train_truncate"] == 8
    assert not _g_opt_cuda_train_mode_safe(members, cfg)

    cfg_none = _deepcopy(cfg)
    cfg_none["net"]["config"]["train_truncate"] = None
    assert _g_opt_cuda_train_mode_safe(members, cfg_none)


def test_find_disagreement_disables_truncation_for_members(monkeypatch):
    # find_disagreement_settings must strip train_truncate before loading members so the
    # cuda fast path is reachable; without it a truncated recipe forces the slow native unroll.
    import nam.train.active_learning as al

    seen = {}

    def _fake_load(checkpoint_paths, model_config, device):
        seen["train_truncate"] = model_config["net"]["config"].get("train_truncate", "MISSING")
        return []  # empty -> find_disagreement_settings returns early

    monkeypatch.setattr(al, "_resolve_device", lambda: _torch.device("cpu"))
    monkeypatch.setattr(al, "_load_disagreement_members", _fake_load)

    cfg = _concat_lstm_model_config()
    assert cfg["net"]["config"]["train_truncate"] == 8
    result = _find_disagreement_settings(["ckpt_a.ckpt"], cfg, g_opt_input_wav="x.wav")

    assert result == []
    assert seen["train_truncate"] is None
    # The caller's config is not mutated.
    assert cfg["net"]["config"]["train_truncate"] == 8


def test_coerce_parallel_dataloader_workers_disables_workers_and_persistence():
    cfg = {
        "train_dataloader": {"batch_size": 4, "num_workers": 4, "persistent_workers": True},
        "val_dataloader": {"batch_size": 4, "num_workers": 0},
    }
    out = _coerce_parallel_dataloader_workers(cfg)

    assert out["train_dataloader"]["num_workers"] == 0
    assert "persistent_workers" not in out["train_dataloader"]
    assert out["val_dataloader"]["num_workers"] == 0
    # Input config is copied, not mutated.
    assert cfg["train_dataloader"]["num_workers"] == 4
    assert cfg["train_dataloader"]["persistent_workers"] is True
