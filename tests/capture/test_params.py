import pytest as _pytest

from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.params import validate_knobs as _validate_knobs


def test_knob_defaults_to_snapped_midpoint():
    knob = _KnobSpec(name="Gain", min=0.0, max=11.0, step=0.5)
    assert knob.default == 5.5

    coarse = _KnobSpec(name="Tone", min=0.0, max=10.0, step=3.0)
    assert coarse.default == 6.0
    assert coarse.min <= coarse.default <= coarse.max


def test_knob_round_trips_through_dict():
    knob = _KnobSpec(name="Gain", min=1.0, max=9.0, step=0.25, default=3.0)
    assert _KnobSpec.from_dict(knob.to_dict()) == knob


def test_knob_converts_to_param_spec_with_step():
    knob = _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5)
    spec = knob.to_param_spec()
    assert spec.name == "Gain"
    assert spec.type == "continuous"
    assert spec.min == 0.0
    assert spec.max == 10.0
    assert spec.step == 0.5
    assert spec.default == 5.0


@_pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "min": 0.0, "max": 10.0},
        {"name": "  ", "min": 0.0, "max": 10.0},
        {"name": "Gain", "min": 10.0, "max": 0.0},
        {"name": "Gain", "min": 0.0, "max": 10.0, "step": 0.0},
        {"name": "Gain", "min": 0.0, "max": 10.0, "step": -1.0},
        {"name": "Gain", "min": 0.0, "max": 10.0, "step": 11.0},
        {"name": "Gain", "min": 0.0, "max": 10.0, "default": 12.0},
        {"name": "Gain", "min": 0.0, "max": float("inf")},
        {"name": "Gain", "min": 0.0, "max": "loud"},
    ],
)
def test_knob_rejects_invalid_specs(kwargs):
    with _pytest.raises(ValueError):
        _KnobSpec(**kwargs)


def test_validate_knobs_rejects_case_insensitive_duplicates():
    knobs = [
        _KnobSpec(name="Gain", min=0.0, max=10.0),
        _KnobSpec(name="gain", min=0.0, max=5.0),
    ]
    with _pytest.raises(ValueError):
        _validate_knobs(knobs)


def test_validate_knobs_rejects_empty():
    with _pytest.raises(ValueError):
        _validate_knobs([])


def test_from_dict_rejects_missing_fields():
    with _pytest.raises(ValueError):
        _KnobSpec.from_dict({"name": "Gain", "min": 0.0})
