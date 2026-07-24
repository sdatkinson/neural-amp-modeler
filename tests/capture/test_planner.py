import pytest as _pytest

from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.planner import corner_settings as _corner_settings
from nam.capture.planner import plan_captures as _plan_captures
from nam.capture.planner import plan_corner_captures as _plan_corner_captures


def _knobs() -> list[_KnobSpec]:
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]


def _specs(knobs):
    return tuple(knob.to_param_spec() for knob in knobs)


def _keys(param_dicts, specs):
    return {tuple(params[spec.name] for spec in specs) for params in param_dicts}


def test_plan_captures_counts_and_grid():
    train, validation = _plan_captures(_knobs(), n_train=8, n_validation=3, seed=0)

    assert len(train) == 8
    assert len(validation) == 3
    for planned in train + validation:
        gain = planned.params["Gain"]
        tone = planned.params["Tone"]
        assert 0.0 <= gain <= 10.0
        assert 0.0 <= tone <= 10.0
        assert gain * 2 == _pytest.approx(round(gain * 2))
        assert tone == _pytest.approx(round(tone))


def test_plan_captures_avoids_zero_on_flagged_knob():
    knobs = [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5, avoid_zero=True),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]
    train, validation = _plan_captures(knobs, n_train=40, n_validation=8, seed=3)

    for planned in train + validation:
        assert planned.params["Gain"] != 0.0
    # Tone, without the flag, is still allowed to reach zero.
    assert any(planned.params["Tone"] == 0.0 for planned in train + validation)


def test_plan_captures_is_reproducible_and_held_out():
    first_train, first_val = _plan_captures(_knobs(), n_train=6, n_validation=2, seed=7)
    second_train, second_val = _plan_captures(_knobs(), n_train=6, n_validation=2, seed=7)

    assert first_train == second_train
    assert first_val == second_val

    train_settings = {tuple(planned.params.items()) for planned in first_train}
    val_settings = {tuple(planned.params.items()) for planned in first_val}
    assert val_settings.isdisjoint(train_settings)


def test_plan_captures_filenames_encode_split_index_and_params():
    train, validation = _plan_captures(_knobs(), n_train=2, n_validation=1, seed=0)

    for index, planned in enumerate(train):
        assert planned.split == "train"
        assert planned.index == index
        assert planned.y_path.startswith(f"captures/lhs_{index:03d}_G")
        assert planned.y_path.endswith(".wav")
    assert validation[0].y_path.startswith("captures/val_lhs_000_G")

    all_paths = [planned.y_path for planned in train + validation]
    assert len(set(all_paths)) == len(all_paths)


def test_plan_captures_supports_zero_validation():
    train, validation = _plan_captures(_knobs(), n_train=3, n_validation=0, seed=0)
    assert len(train) == 3
    assert validation == []


def test_plan_captures_rejects_bad_counts():
    with _pytest.raises(ValueError):
        _plan_captures(_knobs(), n_train=0, n_validation=2, seed=0)
    with _pytest.raises(ValueError):
        _plan_captures(_knobs(), n_train=3, n_validation=-1, seed=0)


def test_plan_captures_matches_starter_script_stream():
    # The starter script and the app planner must draw from the same LHS streams so a
    # plan generated either way selects the same settings for the same seed.
    import importlib.util as _importlib_util
    from pathlib import Path as _Path

    repo_root = _Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "make_starter_settings.py"
    spec = _importlib_util.spec_from_file_location("make_starter_settings", script_path)
    assert spec is not None and spec.loader is not None
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)

    knobs = _knobs()
    param_specs = tuple(knob.to_param_spec() for knob in knobs)
    starter = module.build_starter_data(
        param_specs, n=4, seed=5, n_validation=2, round_to_nearest=None
    )
    train, validation = _plan_captures(knobs, n_train=4, n_validation=2, seed=5)

    for script_entry, planned in zip(
        starter["train"] + starter["validation"], train + validation
    ):
        for name, value in planned.params.items():
            # The planner snaps to each knob's grid; the raw draws must match to
            # within half a step for these to be the same underlying settings.
            assert abs(script_entry["params"][name] - value) <= 0.5 * 1.0 + 1e-9


def _one(name):
    return [_KnobSpec(name=name, min=0.0, max=10.0, step=1.0)]


def _three():
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=1.0),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
        _KnobSpec(name="Bass", min=0.0, max=10.0, step=1.0),
    ]


def test_corner_settings_single_knob_is_just_min_and_max():
    settings = _corner_settings(_specs(_one("Gain")))
    assert settings == [{"Gain": 0.0}, {"Gain": 10.0}]
    # A gain marking does not change a single knob: still only its two extremes.
    assert _corner_settings(_specs(_one("Gain")), gain_index=0) == settings


def test_corner_settings_two_knobs_is_four_regardless_of_gain():
    knobs = _knobs()
    assert len(_corner_settings(_specs(knobs))) == 4
    assert len(_corner_settings(_specs(knobs), gain_index=0)) == 4


def test_corner_settings_three_knobs_without_gain_is_four():
    settings = _corner_settings(_specs(_three()))
    assert len(settings) == 4
    assert {"Gain": 0.0, "Tone": 0.0, "Bass": 0.0} in settings
    assert {"Gain": 10.0, "Tone": 10.0, "Bass": 10.0} in settings


def test_corner_settings_three_knobs_with_gain_is_eight_distinct():
    settings = _corner_settings(_specs(_three()), gain_index=0)
    assert len(settings) == 8
    assert len({tuple(sorted(s.items())) for s in settings}) == 8
    # E: gain min / others max, and F: gain max / others min.
    assert {"Gain": 0.0, "Tone": 10.0, "Bass": 10.0} in settings
    assert {"Gain": 10.0, "Tone": 0.0, "Bass": 0.0} in settings


def test_corner_settings_respects_avoid_zero_on_gain_min():
    knobs = [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=1.0, avoid_zero=True),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]
    settings = _corner_settings(_specs(knobs), gain_index=0)
    # No corner may set the avoid-zero gain knob to exactly zero.
    assert all(s["Gain"] != 0.0 for s in settings)
    assert {"Gain": 1.0, "Tone": 0.0} in settings


def test_plan_corner_captures_names_and_indexes():
    knobs = _three()
    planned, skipped = _plan_corner_captures(knobs)
    assert skipped == 0
    assert len(planned) == 4
    for offset, capture in enumerate(planned):
        assert capture.split == "train"
        assert capture.index == offset
        assert capture.y_path.startswith(f"captures/corner_{offset:03d}_")


def test_plan_corner_captures_offsets_index_and_filename():
    knobs = _three()
    planned, _ = _plan_corner_captures(knobs, index_offset=10, filename_start=2)
    assert planned[0].index == 10
    assert planned[0].y_path.startswith("captures/corner_002_")


def test_plan_corner_captures_skips_excluded_settings():
    knobs = _three()
    all_corners = _corner_settings(_specs(knobs))
    exclude = _keys(all_corners[:1], _specs(knobs))
    planned, skipped = _plan_corner_captures(knobs, exclude=exclude)
    assert skipped == 1
    assert len(planned) == len(all_corners) - 1
