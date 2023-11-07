# File: _names.py
# Created Date: Monday November 6th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

from typing import NamedTuple

from ._version import Version

__all__ = ["INPUT_BASENAMES", "LATEST_VERSION", "VersionAndName"]


class VersionAndName(NamedTuple):
    version: Version
    name: str


# From most the least recently-released
INPUT_BASENAMES = (
    VersionAndName(Version(3, 0, 0), "v3_0_0.wav"),
    VersionAndName(Version(2, 0, 0), "v2_0_0.wav"),
    VersionAndName(Version(1, 1, 1), "v1_1_1.wav"),
    VersionAndName(Version(1, 0, 0), "v1.wav"),
)

LATEST_VERSION = INPUT_BASENAMES[0]
