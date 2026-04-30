# File: test_install.py
# File Created: Tuesday, 2nd February 2021 9:46:01 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest


def test_torch():
    import torch  # noqa F401


if __name__ == "__main__":
    pytest.main()
