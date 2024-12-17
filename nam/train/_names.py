# File: _names.py
# Created Date: Monday November 6th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

from typing import NamedTuple as _NamedTuple, Optional as _Optional, Set as _Set

from ._version import PROTEUS_VERSION as _PROTEUS_VERSION, Version


class VersionAndName(_NamedTuple):
    version: Version
    name: str
    other_names: _Optional[_Set[str]]


# From most- to the least-recently-released:
INPUT_BASENAMES = (
    VersionAndName(Version(3, 0, 0), "input.wav", {"v3_0_0.wav"}),
    # ==================================================================================
    # These are deprecated and will be removed in v0.11. If you still want them, you'll
    # need to write an extension.
    VersionAndName(Version(2, 0, 0), "v2_0_0.wav", None),
    VersionAndName(Version(1, 1, 1), "v1_1_1.wav", None),
    VersionAndName(Version(1, 0, 0), "v1.wav", None),
    VersionAndName(_PROTEUS_VERSION, "Proteus_Capture.wav", None),
    # ==================================================================================
)

LATEST_VERSION = INPUT_BASENAMES[0]
