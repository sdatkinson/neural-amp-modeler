# File: test_version.py
# Created Date: Monday December 16th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest as _pytest

import nam as _nam


def test_has_version():
    assert hasattr(_nam, "__version__")


if __name__ == "__main__":
    _pytest.main()
