# File: test_install.py
# File Created: Tuesday, 2nd February 2021 9:46:01 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest


def test_torch():
    import torch  # noqa F401


def test_lightning_security():
    """
    See https://github.com/sdatkinson/neural-amp-modeler/issues/656
    """

    import pytorch_lightning as pl
    assert pl.__version__ not in {"2.6.2", "2.6.3"}


if __name__ == "__main__":
    pytest.main()
