# File: test_data.py
# Created Date: Friday May 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import math

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


    def test_input_gain(self):
        """
        Checks correctness of input gain parameter
        """
        x_scale = 2.0
        input_gain = 20.0 * math.log10(x_scale)
        x, y = self._create_xy()
        nx = 3
        ny = None
        args = (x, y, nx, ny)
        d1 = data.Dataset(*args)
        d2 = data.Dataset(*args, input_gain=input_gain)

        sample_x1 = d1[0][0]
        sample_x2 = d2[0][0]
        assert torch.allclose(sample_x1 * x_scale, sample_x2)

    def _create_xy(self):
        return 0.99 * (2.0 * torch.rand((2, 7)) - 1.0)  # Don't clip


if __name__ == "__main__":
    pytest.main()