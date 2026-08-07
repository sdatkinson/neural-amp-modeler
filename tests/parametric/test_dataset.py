import json as _json
import subprocess as _subprocess
import sys as _sys
from copy import deepcopy as _deepcopy
from pathlib import Path as _Path
from typing import cast as _cast

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Split as _Split
from nam.data import init_dataset as _init_dataset
from nam.data import np_to_wav as _np_to_wav
from nam.models.parametric import ParametricDataset as _ParametricDataset
from nam.models.parametric import data_config_from_model as _data_config_from_model
from nam.models.parametric import ParamSpec as _ParamSpec


def _make_wav(path: _Path, values: _np.ndarray, rate: int = 48_000) -> str:
    _np_to_wav(values, path, rate=rate)
    return str(path)


def _make_signal(num_samples: int = 256) -> _np.ndarray:
    return _np.linspace(-0.25, 0.25, num_samples, dtype=_np.float64)


def _make_parametric_pair(tmp_path: _Path, prefix: str) -> tuple[str, str]:
    x = _make_signal()
    y = x * 0.5
    x_path = _make_wav(tmp_path / f"{prefix}_x.wav", x)
    y_path = _make_wav(tmp_path / f"{prefix}_y.wav", y)
    return x_path, y_path


def _model_config_with_param_specs() -> dict:
    root = _Path(__file__).resolve().parents[2]
    with open(root / "nam_full_configs" / "parametric" / "model.json", "r") as fp:
        return _json.load(fp)


def test_parametric_dataset_returns_params_x_y(tmp_path):
    x_path, y_path = _make_parametric_pair(tmp_path, "capture")
    dataset = _ParametricDataset.init_from_config(
        {
            "x_path": x_path,
            "y_path": y_path,
            "nx": 8,
            "ny": 4,
            "require_input_pre_silence": None,
            "param_specs": _model_config_with_param_specs()["net"]["config"]["params"],
            "params": {"Gain": 8.0, "Tone": 2.0, "Boost": "On"},
        }
    )

    x, params, y = dataset[0]

    assert _torch.equal(params, _torch.tensor([8.0, 2.0, 1.0]))
    assert x.shape == (11,)
    assert y.shape == (4,)
    assert dataset.nx == 8
    assert dataset.ny == 4


def test_parametric_dataset_accepts_switch_indices(tmp_path):
    x_path, y_path = _make_parametric_pair(tmp_path, "switch_index")
    dataset = _ParametricDataset.init_from_config(
        {
            "x_path": x_path,
            "y_path": y_path,
            "nx": 8,
            "ny": 4,
            "require_input_pre_silence": None,
            "param_specs": [
                _ParamSpec(
                    name="mode",
                    min=0,
                    max=1,
                    default=0,
                    type="switch",
                    enum_names=("clean", "lead"),
                ).to_dict()
            ],
            "params": {"mode": 1},
        }
    )

    assert _torch.equal(dataset.params, _torch.tensor([1.0]))


@_pytest.mark.parametrize(
    "config_updates, match",
    [
        (
            {"params": {"Gain": 8.0, "Tone": 4.0}},
            "params is missing declared parameter name\\(s\\): Boost",
        ),
        (
            {"params": {"Gain": 8.0, "Tone": 4.0, "Boost": "Off", "Depth": 2.0}},
            "params contains unknown parameter name\\(s\\): Depth",
        ),
        (
            {"params": {"Gain": 8.0, "Tone": 4.0, "Boost": "high-gain"}},
            "is not a valid enum name",
        ),
        (
            {"param_specs": []},
            "param_specs must contain at least one parameter definition",
        ),
    ],
)
def test_parametric_dataset_validation_errors(tmp_path, config_updates, match):
    x_path, y_path = _make_parametric_pair(tmp_path, "invalid")
    config = {
        "x_path": x_path,
        "y_path": y_path,
        "nx": 8,
        "ny": 4,
        "require_input_pre_silence": None,
        "param_specs": _model_config_with_param_specs()["net"]["config"]["params"],
        "params": {"Gain": 8.0, "Tone": 4.0, "Boost": "Off"},
    }
    config.update(config_updates)

    with _pytest.raises(ValueError, match=match):
        _ParametricDataset.init_from_config(config)


def test_data_config_from_model_injects_param_specs():
    data_config = {"type": "parametric", "common": {"x_path": "in.wav"}}

    config = _data_config_from_model(data_config, _model_config_with_param_specs())

    assert "param_specs" in config["common"]
    assert tuple(spec["name"] for spec in config["common"]["param_specs"]) == (
        "Gain",
        "Tone",
        "Boost",
    )
    assert "param_specs" not in data_config["common"]


def test_parametric_list_config_builds_concat_dataset(tmp_path):
    x_path, y_path_a = _make_parametric_pair(tmp_path, "capture_a")
    _, y_path_b = _make_parametric_pair(tmp_path, "capture_b")
    config = _data_config_from_model(
        {
            "type": "parametric",
            "common": {
                "x_path": x_path,
                "nx": 8,
                "ny": 4,
                "require_input_pre_silence": None,
            },
            "train": [
                {"y_path": y_path_a, "params": {"Gain": 2.0, "Tone": 9.0, "Boost": "Off"}},
                {"y_path": y_path_b, "params": {"Gain": 6.0, "Tone": 3.0, "Boost": "On"}},
            ],
            "validation": {
                "y_path": y_path_a,
                "params": {"Gain": 2.0, "Tone": 9.0, "Boost": "Off"},
            },
        },
        _model_config_with_param_specs(),
    )

    dataset = _init_dataset(config, _Split.TRAIN)

    assert isinstance(dataset, _ConcatDataset)
    assert all(isinstance(subdataset, _ParametricDataset) for subdataset in dataset.datasets)
    subdataset = _cast(_ParametricDataset, dataset.datasets[0])
    _, params, _ = subdataset[0]
    assert _torch.equal(params, _torch.tensor([2.0, 9.0, 0.0]))


def test_parametric_train_and_validation_lists_use_held_out_settings(tmp_path):
    x_path, y_path_a = _make_parametric_pair(tmp_path, "train_a")
    _, y_path_b = _make_parametric_pair(tmp_path, "train_b")
    _, y_path_val = _make_parametric_pair(tmp_path, "val_held_out")
    config = _data_config_from_model(
        {
            "type": "parametric",
            "common": {
                "x_path": x_path,
                "nx": 8,
                "ny": 4,
                "require_input_pre_silence": None,
            },
            "train": [
                {"y_path": y_path_a, "params": {"Gain": 2.0, "Tone": 8.0, "Boost": "Off"}},
                {"y_path": y_path_b, "params": {"Gain": 8.0, "Tone": 2.0, "Boost": "On"}},
            ],
            # Held-out setting the model never trains on.
            "validation": [
                {"y_path": y_path_val, "params": {"Gain": 6.0, "Tone": 5.0, "Boost": "On"}},
            ],
        },
        _model_config_with_param_specs(),
    )

    train_dataset = _init_dataset(config, _Split.TRAIN)
    validation_dataset = _init_dataset(config, _Split.VALIDATION)

    assert isinstance(train_dataset, _ConcatDataset)
    assert isinstance(validation_dataset, _ConcatDataset)
    assert len(train_dataset.datasets) == 2
    assert len(validation_dataset.datasets) == 1
    _, validation_params, _ = _cast(_ParametricDataset, validation_dataset.datasets[0])[0]
    assert _torch.equal(validation_params, _torch.tensor([6.0, 5.0, 1.0]))


def test_example_parametric_data_config_init_dataset(tmp_path):
    root = _Path(__file__).resolve().parents[2]
    with open(root / "nam_full_configs" / "parametric" / "data.json", "r") as fp:
        config = _json.load(fp)

    config = _data_config_from_model(_deepcopy(config), _model_config_with_param_specs())
    config["common"]["nx"] = 8
    config["common"]["require_input_pre_silence"] = None
    example_signal = _make_signal(30 * 48_000)
    x_path = _make_wav(tmp_path / "shared_x.wav", example_signal)
    config["common"]["x_path"] = x_path
    for split_name in ("train", "validation"):
        for i, capture in enumerate(config[split_name]):
            capture["y_path"] = _make_wav(
                tmp_path / f"{split_name}_{i}.wav",
                example_signal * (0.4 + 0.1 * i),
            )

    train_dataset = _init_dataset(config, _Split.TRAIN)
    validation_dataset = _init_dataset(config, _Split.VALIDATION)

    assert isinstance(train_dataset, _ConcatDataset)
    assert isinstance(validation_dataset, _ConcatDataset)
    assert len(train_dataset.datasets) == 2
    assert len(validation_dataset.datasets) == 2


def test_parametric_dataset_type_registered_on_plain_import():
    # F1 regression: importing nam (here via nam.data, which pulls in nam.models) must
    # register the "parametric" dataset type WITHOUT anyone importing nam.models.parametric
    # explicitly. Run in a fresh interpreter so this test file's own imports can't mask it.
    code = (
        "import nam.data as d;"
        "assert 'parametric' in d._dataset_init_registry, 'parametric not registered';"
        "print('REGISTERED')"
    )
    result = _subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "REGISTERED" in result.stdout
