import json as _json
from pathlib import Path as _Path

import pytest as _pytest

from nam.capture.export import build_concat_learning_config as _build_concat_learning_config
from nam.capture.export import build_concat_model_config as _build_concat_model_config
from nam.capture.export import build_data_config as _build_data_config
from nam.capture.export import build_hyper_learning_config as _build_hyper_learning_config
from nam.capture.export import build_hyper_model_config as _build_hyper_model_config
from nam.capture.export import concat_wavenet_channels as _concat_wavenet_channels
from nam.capture.export import update_data_json as _update_data_json
from nam.capture.export import (
    write_concat_training_configs as _write_concat_training_configs,
)
from nam.capture.export import (
    write_hyper_training_configs as _write_hyper_training_configs,
)
from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.project import (
    CONCAT_LEARNING_CONFIG_FILENAME as _CONCAT_LEARNING_CONFIG_FILENAME,
)
from nam.capture.project import (
    CONCAT_MODEL_CONFIG_FILENAME as _CONCAT_MODEL_CONFIG_FILENAME,
)
from nam.capture.project import DATA_FILENAME as _DATA_FILENAME
from nam.capture.project import (
    HYPER_LEARNING_CONFIG_FILENAME as _HYPER_LEARNING_CONFIG_FILENAME,
)
from nam.capture.project import HYPER_MODEL_CONFIG_FILENAME as _HYPER_MODEL_CONFIG_FILENAME
from nam.capture.project import QAModel as _QAModel
from nam.capture.project import mark_captured as _mark_captured
from nam.capture.project import new_project as _new_project


def _knobs() -> list[_KnobSpec]:
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]


def _project(**kwargs):
    defaults = dict(n_train=3, n_validation=2, seed=0)
    defaults.update(kwargs)
    return _new_project(_knobs(), **defaults)


def test_build_data_config_includes_only_captured_entries():
    project = _project()
    train_entry = project.entries_for_split("train")[0]
    _mark_captured(train_entry, delay=7, qa=_QAModel())

    config = _build_data_config(project)

    assert config["type"] == "parametric"
    assert config["common"] == {"delay": 0}
    assert len(config["train"]) == 1
    assert config["validation"] == []
    assert config["train"][0]["y_path"] == train_entry.y_path

    # Pending entries never leak into either split.
    total_pending = len(project.pending_entries())
    assert total_pending == len(project.entries) - 1


def test_build_data_config_maps_paths_delay_and_window():
    project = _project()
    project.train_window.start_seconds = 1.0
    project.train_window.stop_seconds = 9.0
    project.train_window.ny = 4096
    project.validation_window.start_seconds = 2.0
    project.validation_window.stop_seconds = None
    project.validation_window.ny = 8192

    train_entry = project.entries_for_split("train")[0]
    validation_entry = project.entries_for_split("validation")[0]
    no_delay_entry = project.entries_for_split("train")[1]

    _mark_captured(train_entry, delay=15, qa=_QAModel())
    _mark_captured(validation_entry, delay=3, qa=_QAModel())
    _mark_captured(no_delay_entry, delay=None, qa=_QAModel())

    config = _build_data_config(project)

    train_by_path = {entry["y_path"]: entry for entry in config["train"]}
    validation_by_path = {entry["y_path"]: entry for entry in config["validation"]}

    train_row = train_by_path[train_entry.y_path]
    assert train_row["x_path"] == project.train_input
    assert train_row["delay"] == 15
    assert train_row["params"] == dict(train_entry.params)
    assert train_row["start_seconds"] == 1.0
    assert train_row["stop_seconds"] == 9.0
    assert train_row["ny"] == 4096

    validation_row = validation_by_path[validation_entry.y_path]
    assert validation_row["x_path"] == project.validation_input
    assert validation_row["delay"] == 3
    assert validation_row["start_seconds"] == 2.0
    assert validation_row["stop_seconds"] is None
    assert validation_row["ny"] == 8192

    no_delay_row = train_by_path[no_delay_entry.y_path]
    assert no_delay_row["delay"] == 0


def test_update_data_json_writes_file_matching_build_data_config(tmp_path: _Path):
    project = _project()
    entry = project.entries_for_split("train")[0]
    _mark_captured(entry, delay=4, qa=_QAModel())

    path = _update_data_json(project, tmp_path)

    assert path == tmp_path / _DATA_FILENAME
    assert path.is_file()
    with path.open() as fp:
        payload = _json.load(fp)
    assert payload == _build_data_config(project)


@_pytest.mark.parametrize(
    "builder,name",
    [
        (_build_concat_model_config, "ConcatWaveNet"),
        (_build_hyper_model_config, "HyperWaveNet"),
    ],
)
def test_build_model_config_shape_and_params_order(builder, name):
    project = _project()
    project.sample_rate = 48_000

    config = builder(project)

    assert config["net"]["name"] == name
    net_config = config["net"]["config"]

    params = net_config["params"]
    assert [p["name"] for p in params] == [knob.name for knob in project.knobs]
    for spec, knob in zip(params, project.knobs):
        assert spec["min"] == knob.min
        assert spec["max"] == knob.max
        assert spec["default"] == knob.default
        assert spec["step"] == knob.step
        assert spec["type"] == "continuous"
        assert "enum_names" not in spec
        assert "avoid_zero" not in spec

    assert net_config["sample_rate"] == 48_000.0
    assert isinstance(net_config["sample_rate"], float)

    assert net_config["head_scale"] == 0.01
    assert config["loss"] == {"val_loss": "esr", "mrstft_weight": 0.0005}
    assert config["optimizer"]["weight_decay"] == 3.17e-07
    assert config["lr_scheduler"] == {
        "class": "ExponentialLR",
        "kwargs": {"gamma": 0.994},
    }


@_pytest.mark.parametrize(
    "builder", [_build_concat_model_config, _build_hyper_model_config]
)
def test_build_model_config_omits_sample_rate_when_unset(builder):
    project = _project()
    assert project.sample_rate is None

    config = builder(project)

    assert "sample_rate" not in config["net"]["config"]


def test_concat_model_config_derives_sizes_and_channels():
    project = _project()

    config = _build_concat_model_config(project)
    net_config = config["net"]["config"]

    # 8 + 2 per knob past the first; the two-knob project is the narrowest useful case.
    assert _concat_wavenet_channels(len(project.knobs)) == 10
    assert config["optimizer"]["lr"] == 0.003
    for layer in net_config["layers"]:
        assert layer["channels"] == 10
        assert "input_size" not in layer
        assert "condition_size" not in layer


def test_concat_wavenet_channels_matches_reference_five_knob_run():
    assert _concat_wavenet_channels(5) == 16


def test_hyper_model_config_template_and_hypernet():
    project = _project()

    config = _build_hyper_model_config(project)
    net_config = config["net"]["config"]

    assert config["optimizer"]["lr"] == 0.002
    assert net_config["hypernet"] == {
        "hidden_sizes": [16],
        "activation": "LeakyReLU",
        "selector": {"exclude_suffixes": ["_conv.weight"]},
    }
    # The knobs act through the hypernetwork, so the template stays a stock audio-in
    # channels_8 WaveNet regardless of knob count.
    for layer in net_config["layers"]:
        assert layer["channels"] == 8
        assert layer["input_size"] == 1
        assert layer["condition_size"] == 1


def test_build_model_config_is_loadable_by_its_net():
    from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
    from nam.models.parametric import HyperWaveNet as _HyperWaveNet

    project = _project()

    assert (
        _ConcatWaveNet.init_from_config(
            _build_concat_model_config(project)["net"]["config"]
        )
        is not None
    )
    assert (
        _HyperWaveNet.init_from_config(
            _build_hyper_model_config(project)["net"]["config"]
        )
        is not None
    )


@_pytest.mark.parametrize(
    "builder,gradient_clip_val",
    [
        (_build_concat_learning_config, 0.5),
        (_build_hyper_learning_config, 1.0),
    ],
)
def test_build_learning_config_matches_its_architecture(builder, gradient_clip_val):
    project = _project()

    config = builder(project)

    assert config["trainer"]["accelerator"] == "auto"
    assert config["trainer"]["max_epochs"] == 200
    assert config["trainer"]["gradient_clip_val"] == gradient_clip_val
    assert config["train_dataloader"]["batch_size"] == 32
    assert config["train_dataloader"]["num_workers"] == 0
    assert config["val_dataloader"]["batch_size"] == 32
    assert config["torch_compile"] == {"enabled": True, "mode": "reduce-overhead"}


@_pytest.mark.parametrize(
    "writer,model_filename,learning_filename,model_builder,learning_builder",
    [
        (
            _write_concat_training_configs,
            _CONCAT_MODEL_CONFIG_FILENAME,
            _CONCAT_LEARNING_CONFIG_FILENAME,
            _build_concat_model_config,
            _build_concat_learning_config,
        ),
        (
            _write_hyper_training_configs,
            _HYPER_MODEL_CONFIG_FILENAME,
            _HYPER_LEARNING_CONFIG_FILENAME,
            _build_hyper_model_config,
            _build_hyper_learning_config,
        ),
    ],
)
def test_write_training_configs_writes_both_files(
    tmp_path: _Path,
    writer,
    model_filename,
    learning_filename,
    model_builder,
    learning_builder,
):
    project = _project()

    paths = writer(project, tmp_path)

    assert paths == [tmp_path / model_filename, tmp_path / learning_filename]
    for path in paths:
        assert path.is_file()

    with (tmp_path / model_filename).open() as fp:
        assert _json.load(fp) == model_builder(project)
    with (tmp_path / learning_filename).open() as fp:
        assert _json.load(fp) == learning_builder(project)


def test_both_architectures_export_side_by_side(tmp_path: _Path):
    project = _project()

    _write_concat_training_configs(project, tmp_path)
    _write_hyper_training_configs(project, tmp_path)

    written = {path.name for path in tmp_path.iterdir()}
    assert written == {
        _CONCAT_MODEL_CONFIG_FILENAME,
        _CONCAT_LEARNING_CONFIG_FILENAME,
        _HYPER_MODEL_CONFIG_FILENAME,
        _HYPER_LEARNING_CONFIG_FILENAME,
    }
