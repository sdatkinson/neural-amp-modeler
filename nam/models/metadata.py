# File: metadata.py
# Created Date: Wednesday April 12th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Metadata about models
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


# Note: if you change this enum, you need to update the options in easy_colab.ipynb!
class GearType(Enum):
    AMP = "amp"
    PEDAL = "pedal"
    PEDAL_AMP = "pedal_amp"
    AMP_CAB = "amp_cab"
    AMP_PEDAL_CAB = "amp_pedal_cab"
    PREAMP = "preamp"
    STUDIO = "studio"


# Note: if you change this enum, you need to update the options in easy_colab.ipynb!
class ToneType(Enum):
    CLEAN = "clean"
    OVERDRIVE = "overdrive"
    CRUNCH = "crunch"
    HI_GAIN = "hi_gain"
    FUZZ = "fuzz"


class Date(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int


class UserMetadata(BaseModel):
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

    name: Optional[str] = None
    modeled_by: Optional[str] = None
    gear_type: Optional[GearType] = None
    gear_make: Optional[str] = None
    gear_model: Optional[str] = None
    tone_type: Optional[ToneType] = None
    input_level_dbu: Optional[float] = None
    output_level_dbu: Optional[float] = None
