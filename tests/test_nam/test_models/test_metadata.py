"""
Tests for model metadata
"""

import pytest

from nam.models import metadata

# The TONE3000 gear categories, with "_" in place of "-" per the NAM convention.
_TONE3000_GEAR = {"amp", "amp_cab", "pedal", "outboard", "cab", "space", "experimental"}


def test_gear_type_matches_tone3000_categories():
    assert {g.value for g in metadata.GearType} == _TONE3000_GEAR


@pytest.mark.parametrize(
    "legacy,expected",
    (
        ("pedal_amp", metadata.GearType.AMP),
        ("amp_pedal_cab", metadata.GearType.AMP_CAB),
        ("preamp", metadata.GearType.AMP),
        ("studio", metadata.GearType.OUTBOARD),
    ),
)
def test_gear_type_accepts_legacy_values(legacy: str, expected: metadata.GearType):
    assert metadata.GearType(legacy) is expected
    # ...including through pydantic validation
    assert metadata.UserMetadata(gear_type=legacy).gear_type is expected


def test_gear_type_rejects_unknown_values():
    with pytest.raises(ValueError):
        metadata.GearType("bogus")
