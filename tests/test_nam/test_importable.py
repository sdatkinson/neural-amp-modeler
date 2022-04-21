# File: test_importable.py
# Created Date: Sunday December 12th 2021
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest


def test_importable():
    import nam  # noqa F401


if __name__ == "__main__":
    pytest.main()
