# File: test_version.py
# Created Date: Saturday April 29th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

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
