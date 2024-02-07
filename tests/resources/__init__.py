# File: __init__.py
# Created Date: Thursday May 18th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Download the standardized reamping files to the directory containing this file.
See:
https://github.com/sdatkinson/neural-amp-modeler/tree/main#standardized-reamping-files
"""

from pathlib import Path

import pytest

__all__ = [
    "requires_proteus",
    "requires_v1_0_0",
    "requires_v1_1_1",
    "requires_v2_0_0",
    "requires_v3_0_0",
    "resource_path",
]


def _requires_v(name: str):
    path = Path(__file__).parent / Path(name)
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"Requires {name}, which hasn't been downloaded to {path}.",
    )


requires_v1_0_0 = _requires_v("v1.wav")
requires_v1_1_1 = _requires_v("v1_1_1.wav")
requires_v2_0_0 = _requires_v("v2_0_0.wav")
requires_v3_0_0 = _requires_v("v3_0_0.wav")
requires_proteus = _requires_v("Proteus_Capture.wav")


def resource_path(name: str) -> Path:
    return Path(__file__).absolute().parent / Path(name)
