import pytest as _pytest
from typing import Any as _Any
from typing import cast as _cast

from nam.models.parametric import ParamSpec as _ParamSpec


def test_continuous_spec_round_trip():
    spec = _ParamSpec(name="gain", min=0.0, max=10.0, default=5.0)

    exported = spec.to_dict()
    restored = _ParamSpec.from_dict(exported)

    assert restored == spec
    assert restored.num_inputs == 1
    assert restored.normalized_min == -1.0
    assert restored.normalized_max == 1.0
    assert exported["type"] == "continuous"
    assert exported["enum_names"] is None
    assert "normalization" not in exported
    assert "normalized_min" not in exported
    assert "normalized_max" not in exported


def test_switch_spec_round_trip():
    spec = _ParamSpec(
        name="mode",
        min=0,
        max=2,
        default=1,
        type="switch",
        enum_names=("clean", "crunch", "lead"),
    )

    exported = spec.to_dict()
    restored = _ParamSpec.from_dict(exported)

    assert restored == spec
    assert restored.num_inputs == 3
    assert restored.normalized_min == 0.0
    assert restored.normalized_max == 1.0
    assert exported["enum_names"] == ["clean", "crunch", "lead"]


def test_from_dict_defaults_to_continuous_signed_normalization():
    spec = _ParamSpec.from_dict({"name": "depth", "min": 1.0, "max": 9.0, "default": 4.0})

    assert spec.type == "continuous"
    assert spec.normalization == "min_max_signed"


def test_switch_specs_default_to_unit_normalization():
    spec = _ParamSpec(
        name="mode",
        min=0,
        max=1,
        default=0,
        type="switch",
        enum_names=("clean", "lead"),
    )

    assert spec.normalization == "min_max_unit"
    assert spec.normalized_min == 0.0
    assert spec.normalized_max == 1.0


@_pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "", "min": 0.0, "max": 1.0, "default": 0.5}, "name must be non-empty"),
        (
            {"name": "gain", "min": 1.0, "max": 1.0, "default": 1.0},
            "must satisfy min < max",
        ),
        (
            {"name": "gain", "min": 0.0, "max": 1.0, "default": 2.0},
            "default must satisfy min <= default <= max",
        ),
        (
            {"name": "mode", "min": 0, "max": 1, "default": 0, "type": "switch"},
            "requires enum_names",
        ),
        (
            {
                "name": "mode",
                "min": 1,
                "max": 2,
                "default": 1,
                "type": "switch",
                "enum_names": ("clean", "lead"),
            },
            "must use min/max index bounds",
        ),
        (
            {
                "name": "mode",
                "min": 0,
                "max": 1,
                "default": 0.5,
                "type": "switch",
                "enum_names": ("clean", "lead"),
            },
            "must be integer indices",
        ),
        (
            {
                "name": "mode",
                "min": 0,
                "max": 1,
                "default": 0,
                "type": "switch",
                "enum_names": "clean",
            },
            "must be a sequence of names, not a string",
        ),
        (
            {
                "name": "mode",
                "min": 0,
                "max": 1,
                "default": 0,
                "type": "switch",
                "enum_names": ("clean", "clean"),
            },
            "must be unique",
        ),
        (
            {
                "name": "gain",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "enum_names": ("bad", "idea"),
            },
            "cannot define enum_names",
        ),
    ],
)
def test_validation_errors(kwargs, match):
    with _pytest.raises(ValueError, match=match):
        _ParamSpec(**kwargs)


def test_constructor_rejects_internal_normalization_argument():
    with _pytest.raises(TypeError, match="unexpected keyword argument 'normalization'"):
        _cast(_Any, _ParamSpec)(
            name="gain", min=0.0, max=1.0, default=0.5, normalization="bad"
        )


def test_from_dict_rejects_missing_required_fields():
    with _pytest.raises(ValueError, match="missing required field\\(s\\): name"):
        _ParamSpec.from_dict({"min": 0.0, "max": 1.0, "default": 0.5})


def test_from_dict_rejects_string_enum_names():
    with _pytest.raises(ValueError, match="must be a sequence of names, not a string"):
        _ParamSpec.from_dict(
            {
                "name": "mode",
                "min": 0,
                "max": 1,
                "default": 0,
                "type": "switch",
                "enum_names": "clean",
            }
        )
