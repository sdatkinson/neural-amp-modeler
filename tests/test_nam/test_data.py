# File: test_data.py
# Created Date: Friday May 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import torch

from nam import data


class TestDataset(object):
    def test_init(self):
        x, y = self._create_xy()
        data.Dataset(x, y, 3, None)

    def test_init_zero_delay(self):
        """
        Assert https://github.com/sdatkinson/neural-amp-modeler/issues/15 fixed
        """
        x, y = self._create_xy()
        data.Dataset(x, y, 3, None, delay=0)

    def _create_xy(self):
        return 0.99 * (2.0 * torch.rand((2, 7)) - 1.0)  # Don't clip
