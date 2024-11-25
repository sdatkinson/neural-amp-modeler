# File: _conv_mixin.py
# Created Date: Saturday November 23rd 2024
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Mix-in tests for models with a convolution layer
"""

import pytest as _pytest
import torch as _torch

from .base import Base as _Base


class Convolutional(_Base):
    @_pytest.mark.skipif(
        not _torch.backends.mps.is_available(), reason="MPS-specific test"
    )
    def test_process_input_longer_than_65536(self):
        """
        Processing inputs longer than 65,536 samples using the MPS backend can
        cause problems.

        See: https://github.com/sdatkinson/neural-amp-modeler/issues/505

        Assert that precautions are taken.
        """

        x = _torch.zeros((65_536 + 1,)).to("mps")

        model = self._construct().to("mps")
        model(x)
