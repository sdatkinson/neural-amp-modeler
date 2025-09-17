# File: metadata.py
# Created Date: Sunday May 19th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Information from the simplified trainers that is good to know about.
"""

# This isn't part of ../metadata because it's not necessarily worth knowing about--only
# if you're using the simplified trainers!

from typing import List as _List, Optional as _Optional

from pydantic import BaseModel as _BaseModel

# The key under which the metadata are saved in the .nam:
TRAINING_KEY = "training"


class Settings(_BaseModel):
    """
    User-provided settings
    """

    ignore_checks: bool


class LatencyCalibrationWarnings(_BaseModel):
    """
    Things that aren't necessarily wrong with the latency calibration but are
    worth looking into.

    :param matches_lookahead: The calibrated latency is as far forward as
        possible, i.e. the very first sample we looked at tripped the trigger.
        That's probably not a coincidence but the trigger is too sensitive.
    :param disagreement_too_high: The range of the latency estimates is greater
        than the max_disagreement_threshold. Indication that something may have
        gone wrong.\
    :param not_detected: The impulse responses used for latency calibration were
        not detected by the algorithm, so latency calibration could not be
        determined automatically.
    """

    matches_lookahead: bool
    disagreement_too_high: bool
    not_detected: bool


class LatencyCalibration(_BaseModel):
    """
    :param recommended: In samples. Positive values mean that the output lags
        behind the input and should be puleld forward in time. Negative values
        indicate that the latency is over-compensated (possibly by teh DAW or
        human error) and the output should be pushed back in time. None if
        calibration could not be determined automatically due to some error.
    """

    algorithm_version: int
    delays: _List[int]
    safety_factor: int
    recommended: _Optional[int]
    warnings: LatencyCalibrationWarnings


class Latency(_BaseModel):
    """
    Information about the latency
    """

    manual: _Optional[int]
    calibration: LatencyCalibration


class DataChecks(_BaseModel):
    version: int
    passed: bool


class Data(_BaseModel):
    latency: Latency
    checks: DataChecks


class TrainingMetadata(_BaseModel):
    settings: Settings
    data: Data
    validation_esr: _Optional[float]
