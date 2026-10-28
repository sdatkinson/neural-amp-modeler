# File: metadata.py
# Created Date: Wednesday April 12th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Metadata about models
"""

from enum import Enum as _Enum
from typing import Optional as _Optional

from pydantic import BaseModel as _BaseModel


# Note: if you change this enum, you need to update the options in easy_colab.ipynb!
class GearType(_Enum):
    """
    What kind of gear a model captures.

    The values follow the TONE3000 gear categories so that metadata is consistent
    between local and cloud trainings.
    """

    AMP = "amp"  # Amp head / combo / preamp without a speaker cabinet (any pedals in front)
    AMP_CAB = "amp_cab"  # Amp through a speaker cabinet (any pedals in front)
    PEDAL = "pedal"
    OUTBOARD = "outboard"  # EQs, compressors, and other studio gear
    CAB = "cab"  # Speaker cabinet only
    SPACE = "space"  # A room, hall, or other physical space (e.g. a room IR)
    EXPERIMENTAL = "experimental"

    @classmethod
    def _missing_(cls, value):
        # Accept the gear types used before the TONE3000 categories were adopted.
        legacy = {
            "pedal_amp": cls.AMP,
            "amp_pedal_cab": cls.AMP_CAB,
            "preamp": cls.AMP,
            "studio": cls.OUTBOARD,
        }
        return legacy.get(value)


# Note: if you change this enum, you need to update the options in easy_colab.ipynb!
class ToneType(_Enum):
    CLEAN = "clean"
    OVERDRIVE = "overdrive"
    CRUNCH = "crunch"
    HI_GAIN = "hi_gain"
    FUZZ = "fuzz"


class Date(_BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int


class UserMetadata(_BaseModel):
    """
    Metadata that users provide for a NAM model

    :param name: A "human-readable" name for the model.
    :param modeled_by: Who made the model
    :param gear_type: Type of gear.
    :param gear_make: Make of the gear.
    :param gear_model: Model of the gear.
    :param tone_type: What sort of tone the gear has.
    :input_level_dbu: What analog loudness, in dBu, corresponds to 0 dbFS input to the
        model.
    :output_level_dbu: What analog loudness, in dBu, corresponds to 0 dbFS outputted by
        the model.
    """

    name: _Optional[str] = None
    modeled_by: _Optional[str] = None
    gear_type: _Optional[GearType] = None
    gear_make: _Optional[str] = None
    gear_model: _Optional[str] = None
    tone_type: _Optional[ToneType] = None
    input_level_dbu: _Optional[float] = None
    output_level_dbu: _Optional[float] = None
