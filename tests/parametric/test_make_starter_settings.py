import importlib.util as _importlib_util
from pathlib import Path as _Path

import pytest as _pytest

from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric import decode_named_params as _decode_named_params
from nam.models.parametric import format_param_value as _format_param_value


def _load_script_module():
    repo_root = _Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "make_starter_settings.py"
    spec = _importlib_util.spec_from_file_location("make_starter_settings", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load starter-settings script from {script_path}")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mixed_specs() -> tuple[_ParamSpec, ...]:
    return (
        _ParamSpec(name="Gain", min=0.0, max=10.0, default=5.0),
        _ParamSpec(name="Tone", min=0.0, max=10.0, default=5.0),
        _ParamSpec(
            name="Boost",
            min=0,
            max=1,
            default=0,
            type="switch",
            enum_names=("Off", "On"),
        ),
    )


def _two_switch_specs() -> tuple[_ParamSpec, ...]:
    return (
        _ParamSpec(
            name="A",
            min=0,
            max=1,
            default=0,
            type="switch",
            enum_names=("Off", "On"),
        ),
        _ParamSpec(
            name="B",
            min=0,
            max=1,
            default=0,
            type="switch",
            enum_names=("Low", "High"),
        ),
    )


def test_sample_raw_settings_balances_switches_and_stays_in_range():
    module = _load_script_module()
    specs = _mixed_specs()

    raw_settings = module.sample_raw_settings(specs, 10, seed=0)
    decoded = [_decode_named_params(raw, specs) for raw in raw_settings]

    assert len(decoded) == 10
    assert sum(entry["Boost"] == "Off" for entry in decoded) == 5
    assert sum(entry["Boost"] == "On" for entry in decoded) == 5
    for entry in decoded:
        gain = entry["Gain"]
        tone = entry["Tone"]
        assert isinstance(gain, float)
        assert isinstance(tone, float)
        assert 0.0 <= gain <= 10.0
        assert 0.0 <= tone <= 10.0


def test_sample_raw_settings_stratifies_across_joint_switch_combinations():
    module = _load_script_module()
    specs = _two_switch_specs()

    raw_settings = module.sample_raw_settings(specs, 4, seed=12)
    decoded = [_decode_named_params(raw, specs) for raw in raw_settings]

    assert len(decoded) == 4
    assert len({tuple(entry.items()) for entry in decoded}) == 4
    assert {tuple(entry.items()) for entry in decoded} == {
        (("A", "Off"), ("B", "Low")),
        (("A", "Off"), ("B", "High")),
        (("A", "On"), ("B", "Low")),
        (("A", "On"), ("B", "High")),
    }


def test_sample_raw_settings_repeats_joint_switch_cycles_only_after_exhausting_unique_combos():
    module = _load_script_module()
    specs = _two_switch_specs()

    raw_settings = module.sample_raw_settings(specs, 6, seed=12)
    decoded = [_decode_named_params(raw, specs) for raw in raw_settings]

    first_cycle = {tuple(entry.items()) for entry in decoded[:4]}
    assert len(first_cycle) == 4
    assert all(tuple(entry.items()) in first_cycle for entry in decoded[4:])


def test_build_starter_data_full_grid_covers_every_switch_state():
    module = _load_script_module()
    specs = _mixed_specs()

    data_config = module.build_starter_data(
        specs,
        n=3,
        seed=7,
        full_grid=True,
        y_path_prefix="starter_",
        n_validation=0,
    )

    assert data_config["type"] == "parametric"
    assert data_config["common"] == {"x_path": "input.wav", "delay": 0}
    assert len(data_config["train"]) == 6
    assert data_config["validation"] == []
    # y_path now encodes the decoded params (unique-prefix abbreviation + value), so
    # paired Off/On rows share the same Gain/Tone stem and differ only in the Boost token.
    for entry in data_config["train"]:
        gain = _format_param_value(entry["params"]["Gain"])
        tone = _format_param_value(entry["params"]["Tone"])
        boost = entry["params"]["Boost"]
        assert entry["y_path"] == f"starter_G{gain}_T{tone}_B{boost}.wav"

    boost_values = [entry["params"]["Boost"] for entry in data_config["train"]]
    assert boost_values.count("Off") == 3
    assert boost_values.count("On") == 3

    paired_rows = list(zip(data_config["train"][::2], data_config["train"][1::2]))
    assert len(paired_rows) == 3
    for off_entry, on_entry in paired_rows:
        assert off_entry["params"]["Boost"] == "Off"
        assert on_entry["params"]["Boost"] == "On"
        assert off_entry["params"]["Gain"] == _pytest.approx(on_entry["params"]["Gain"])
        assert off_entry["params"]["Tone"] == _pytest.approx(on_entry["params"]["Tone"])


def test_build_starter_data_seeds_held_out_validation_by_default():
    # Regression: an empty validation list is routed through ConcatDataset by
    # nam.data.init_dataset and blows up _make_lookup, so the generated config must seed a
    # non-empty, held-out validation split by default.
    module = _load_script_module()
    specs = _mixed_specs()

    data_config = module.build_starter_data(specs, n=8, seed=3)

    validation = data_config["validation"]
    assert isinstance(validation, list)
    assert len(validation) == module._DEFAULT_N_VALIDATION
    assert len(validation) > 0

    # Held-out: validation settings are a different stream than the train settings.
    train_settings = {
        tuple(entry["params"].items()) for entry in data_config["train"]
    }
    validation_settings = {
        tuple(entry["params"].items()) for entry in validation
    }
    assert validation_settings.isdisjoint(train_settings)

    # Validation captures use the tail-of-audio windowing, distinct from train.
    for entry in validation:
        assert entry["y_path"].startswith(module._DEFAULT_VALIDATION_Y_PATH_PREFIX)
        assert entry["start_seconds"] == module._DEFAULT_VALIDATION_START_SECONDS
        assert entry["stop_seconds"] == module._DEFAULT_VALIDATION_STOP_SECONDS
        assert entry["ny"] == module._DEFAULT_VALIDATION_NY
        gain = entry["params"]["Gain"]
        assert isinstance(gain, float)
        assert 0.0 <= gain <= 10.0


def test_default_window_lengths_stay_within_single_lstm_block():
    # Regression: validation entries used to default to a full EOF-length window (ny None),
    # which spans several ConcatLSTM blocks and crashes the LSTM on Apple MPS. Both ny
    # defaults must be finite, exceed the loss mask_first (8192), and stay below one block.
    module = _load_script_module()
    _CONCAT_LSTM_BLOCK = 65_535
    _MASK_FIRST = 8192
    for default_ny in (module._DEFAULT_NY, module._DEFAULT_VALIDATION_NY):
        assert default_ny is not None
        assert _MASK_FIRST < default_ny < _CONCAT_LSTM_BLOCK


def test_build_starter_data_validation_can_be_disabled():
    module = _load_script_module()
    specs = _mixed_specs()

    data_config = module.build_starter_data(specs, n=5, seed=3, n_validation=0)

    assert data_config["validation"] == []


def test_build_starter_data_validation_is_reproducible_from_seed():
    module = _load_script_module()
    specs = _mixed_specs()

    first = module.build_starter_data(specs, n=5, seed=11)
    second = module.build_starter_data(specs, n=5, seed=11)

    assert first["validation"] == second["validation"]


def test_build_starter_data_rejects_negative_validation_count():
    module = _load_script_module()
    specs = _mixed_specs()

    with _pytest.raises(ValueError):
        module.build_starter_data(specs, n=5, seed=0, n_validation=-1)


def test_build_starter_data_rounds_continuous_params_to_nearest_half_by_default():
    module = _load_script_module()
    specs = _mixed_specs()

    data_config = module.build_starter_data(specs, n=10, seed=0)

    for entry in data_config["train"]:
        gain = entry["params"]["Gain"]
        tone = entry["params"]["Tone"]
        assert isinstance(gain, float)
        assert isinstance(tone, float)
        assert gain * 2 == _pytest.approx(round(gain * 2))
        assert tone * 2 == _pytest.approx(round(tone * 2))


def test_build_starter_data_can_disable_rounding():
    module = _load_script_module()
    specs = _mixed_specs()

    rounded = module.build_starter_data(specs, n=10, seed=0)
    unrounded = module.build_starter_data(specs, n=10, seed=0, round_to_nearest=None)

    rounded_gain_values = [entry["params"]["Gain"] for entry in rounded["train"]]
    unrounded_gain_values = [entry["params"]["Gain"] for entry in unrounded["train"]]
    assert rounded_gain_values != unrounded_gain_values
    assert any(
        value * 2 != _pytest.approx(round(value * 2)) for value in unrounded_gain_values
    )
