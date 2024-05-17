# File: metadata.py
# Created Date: Wednesday April 12th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Metadata about models
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel

__all__ = ["GearType", "ToneType", "Date", "TrainingMetadata", "UserMetadata"]


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
    """

    name: Optional[str] = None
    modeled_by: Optional[str] = None
    gear_type: Optional[GearType] = None
    gear_make: Optional[str] = None
    gear_model: Optional[str] = None
    tone_type: Optional[ToneType] = None


class LatencyMetadata(BaseModel):
    """
    Metadata about the data latency when using a standardized trainer (GUI, Colab)

    :param user_samples: What the audio latency was provided as. If None, it was
        not provided and we attempted to figure it out automatically.
    :param estimation_algorithm_version: What algorithm was used to estimate latency.
        1: Introduced in version TK
    :param estimated_samples: What the algoirthm estimated the latency was. One
        estimation for each blip.
    :param safety_factor_samples: Latency safety factor
    """

    user_samples: Optional[int]
    estimation_algorithm_version: int
    estimated_samples: List[int]
    safety_factor_samples: int


class TrainingMetadata(BaseModel):
    """
    Metadata from training when using a standardized trainer.

    :param validation_esr: ESR on the standardized validation signal
    :fit_cab: Whether cab fitting was enabled
    :param ignored_checks: Whether the checks were ignored
    :param latency: Information about the input-output audio latency in the training
        data.
    """

    validation_esr: float
    fit_cab: bool
    ignored_checks: bool
    latency: LatencyMetadata
