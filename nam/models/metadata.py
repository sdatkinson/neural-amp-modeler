# File: metadata.py
# Created Date: Wednesday April 12th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Metadata about models
"""

from enum import Enum

# from pydantic import BaseModel


class GearType(Enum):
    AMP = "amp"
    PEDAL = "pedal"
    AMP_CAB = "amp_cab"
    AMP_PEDAL_CAB = "amp_pedal_cab"
    PREAMP = "preamp"
    STUDIO = "studio"


class ToneType(Enum):
    CLEAN = "clean"
    OVERDRIVE = "overdrive"
    CRUNCH = "crunch"
    HI_GAIN = "hi_gain"
    FUZZ = "fuzz"


class Date(object):  # BaesModel):
    year: int
    month: int
    day: int


class Metadata(object):  # BaseModel):
    """
    Metadata that NAM models can export
    """
    name: str=""
    modeled_by: str=""
    model_date: Date
    gear_type: GearType
    gear_make: str
    gear_model: str
    tone_type: ToneType
    gain: float