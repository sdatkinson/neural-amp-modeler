import pytest as _pytest

from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.planner import plan_captures as _plan_captures


def _knobs() -> list[_KnobSpec]:
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]


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
        assert planned.y_path.startswith(f"captures/train_{index:03d}_G")
        assert planned.y_path.endswith(".wav")
    assert validation[0].y_path.startswith("captures/validation_000_G")

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
