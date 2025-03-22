# File: test_version.py
# Created Date: Saturday April 29th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Tests for version class
"""

import pytest as _pytest

from nam.train import _version


def test_eq():
    assert _version.Version(0, 0, 0) == _version.Version(0, 0, 0)
    assert _version.Version(0, 0, 0) != _version.Version(0, 0, 1)
    assert _version.Version(0, 0, 0) != _version.Version(0, 1, 0)
    assert _version.Version(0, 0, 0) != _version.Version(1, 0, 0)


def test_lt():
    assert _version.Version(0, 0, 0) < _version.Version(0, 0, 1)
    assert _version.Version(0, 0, 0) < _version.Version(0, 1, 0)
    assert _version.Version(0, 0, 0) < _version.Version(1, 0, 0)

    assert _version.Version(1, 2, 3) < _version.Version(2, 0, 0)

    assert not _version.Version(1, 2, 3) < _version.Version(0, 4, 5)


def test_current_version():
    """
    Test that the current version is valid
    """
    from nam import __version__

    # First off, assert that the current version can be understood by _version.Version.
    # Broken by PR 516 (pyproject.toml)--watch out!
    v = _version.Version.from_string(__version__)

    # Check comparisons like used in GUI:
    assert _version.Version(0, 0, 0) != v
    assert _version.Version(0, 0, 0) < v
    # Just checking both orders. If we're actually at this version, then fine, move it up!
    high_major_version_that_we_will_probably_never_get_to = 1000
    assert v < _version.Version(
        high_major_version_that_we_will_probably_never_get_to, 0, 0
    )
