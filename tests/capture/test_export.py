import json as _json
from pathlib import Path as _Path

from nam.capture.export import build_data_config as _build_data_config
from nam.capture.export import build_learning_config as _build_learning_config
from nam.capture.export import build_model_config as _build_model_config
from nam.capture.export import update_data_json as _update_data_json
from nam.capture.export import write_training_configs as _write_training_configs
from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.project import DATA_FILENAME as _DATA_FILENAME
from nam.capture.project import LEARNING_CONFIG_FILENAME as _LEARNING_CONFIG_FILENAME
from nam.capture.project import MODEL_CONFIG_FILENAME as _MODEL_CONFIG_FILENAME
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


def test_build_model_config_shape_and_params_order():
    project = _project()
    project.sample_rate = 48_000

    config = _build_model_config(project)

    assert config["net"]["name"] == "ConcatWaveNet"
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

    assert net_config["sample_rate"] == 48_000.0
    assert isinstance(net_config["sample_rate"], float)

    for layer in net_config["layers"]:
        assert "input_size" not in layer
        assert "condition_size" not in layer


def test_build_model_config_omits_sample_rate_when_unset():
    project = _project()
    assert project.sample_rate is None

    config = _build_model_config(project)

    assert "sample_rate" not in config["net"]["config"]


def test_build_model_config_is_loadable_by_concat_wavenet():
    from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet

    project = _project()
    config = _build_model_config(project)

    net = _ConcatWaveNet.init_from_config(config["net"]["config"])
    assert net is not None


def test_build_learning_config_accelerator_and_batch_sizes():
    project = _project()

    config = _build_learning_config(project)

    assert config["trainer"]["accelerator"] == "auto"
    assert config["train_dataloader"]["batch_size"] == 16
    assert config["val_dataloader"]["batch_size"] == 16


def test_write_training_configs_writes_both_files(tmp_path: _Path):
    project = _project()

    paths = _write_training_configs(project, tmp_path)

    assert paths == [tmp_path / _MODEL_CONFIG_FILENAME, tmp_path / _LEARNING_CONFIG_FILENAME]
    for path in paths:
        assert path.is_file()
        with path.open() as fp:
            _json.load(fp)  # valid JSON

    with (tmp_path / _MODEL_CONFIG_FILENAME).open() as fp:
        assert _json.load(fp) == _build_model_config(project)
    with (tmp_path / _LEARNING_CONFIG_FILENAME).open() as fp:
        assert _json.load(fp) == _build_learning_config(project)
