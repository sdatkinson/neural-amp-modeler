# File: test_data.py
# Created Date: Friday May 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import math
from enum import Enum
from typing import Tuple

import numpy as np
import pytest
import torch

from nam import data


class _XYMethod(Enum):
    ARANGE = "arange"
    RAND = "rand"
    STEP = "step"


class TestDataset(object):
    """
    Assertions about nam.data.Dataset
    """

    def test_apply_delay_zero(self):
        """
        Assert proper function of Dataset._apply_delay() when zero delay is given, i.e.
        no change.
        """
        x, y = self._create_xy()
        x_out, y_out = data.Dataset._apply_delay(
            x, y, 0, data._DelayInterpolationMethod.CUBIC
        )
        assert torch.all(x == x_out)
        assert torch.all(y == y_out)

    @pytest.mark.parametrize("method", (data._DelayInterpolationMethod))
    def test_apply_delay_float_negative(self, method):
        n = 7
        delay = -2.5
        x_out, y_out = self._t_apply_delay_float(n, delay, method)

        assert torch.all(x_out == torch.Tensor([3, 4, 5, 6]))
        assert torch.all(y_out == torch.Tensor([0.5, 1.5, 2.5, 3.5]))

    @pytest.mark.parametrize("method", (data._DelayInterpolationMethod))
    def test_apply_delay_float_positive(self, method):
        n = 7
        delay = 2.5
        x_out, y_out = self._t_apply_delay_float(n, delay, method)

        assert torch.all(x_out == torch.Tensor([0, 1, 2, 3]))
        assert torch.all(y_out == torch.Tensor([2.5, 3.5, 4.5, 5.5]))

    def test_apply_delay_int_negative(self):
        """
        Assert proper function of Dataset._apply_delay() when a positive integer delay
        is given.
        """
        n = 7
        delay = -3
        x_out, y_out = self._t_apply_delay_int(n, delay)

        assert torch.all(x_out == torch.Tensor([3, 4, 5, 6]))
        assert torch.all(y_out == torch.Tensor([0, 1, 2, 3]))

    def test_apply_delay_int_positive(self):
        """
        Assert proper function of Dataset._apply_delay() when a positive integer delay
        is given.
        """
        n = 7
        delay = 3
        x_out, y_out = self._t_apply_delay_int(n, delay)

        assert torch.all(x_out == torch.Tensor([0, 1, 2, 3]))
        assert torch.all(y_out == torch.Tensor([3, 4, 5, 6]))

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

    @pytest.mark.parametrize(
        "n,start,valid",
        (
            (13, None, True),  # No start restrictions; nothing wrong
            (13, 2, True),  # Starts before the end; fine.
            (13, 12, True),  # Starts w/ one to go--ok
            (13, 13, False),  # Starts after the end
            (13, -5, True),  # Starts counting back from the end, fine
            (13, -13, True),  # Starts at the beginning of the array--ok
            (13, -14, False),  # Starts before the beginning of the array--invalid
        ),
    )
    def test_validate_start(self, n: int, start: int, valid: bool):
        def init():
            data.Dataset(x, y, nx, ny, start=start)

        nx = 1
        ny = None
        x, y = self._create_xy(n=n)
        if valid:
            init()
            assert True  # No problem!
        else:
            with pytest.raises(data.StartError):
                init()

    @pytest.mark.parametrize(
        "n,stop,valid",
        (
            (13, None, True),  # No stop restrictions; nothing wrong
            (13, 2, True),  # Stops before the end; fine.
            (13, 13, True),  # Stops at the end--ok
            (13, 14, False),  # Stops after the end--not ok
            (13, -5, True),  # Stops counting back from the end, fine
            (13, -12, True),  # Stops w/ one sample--ok
            (13, -13, False),  # Stops w/ no samples--not ok
        ),
    )
    def test_validate_stop(self, n: int, stop: int, valid: bool):
        def init():
            data.Dataset(x, y, nx, ny, stop=stop)

        nx = 1
        ny = None
        x, y = self._create_xy(n=n)
        if valid:
            init()
            assert True  # No problem!
        else:
            with pytest.raises(data.StopError):
                init()

    def _create_xy(
        self,
        n: int = 7,
        method: _XYMethod = _XYMethod.RAND,
        must_be_in_valid_range: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (n,), (n,)
        """
        if method == _XYMethod.ARANGE:
            # note: this isn't "valid" data in the sense that it's beyond (-1, 1).
            # But it is useful for the delay code.
            assert not must_be_in_valid_range
            return tuple(
                torch.tile(torch.arange(n, dtype=torch.float)[None, :], (2, 1))
            )
        elif method == _XYMethod.RAND:
            return tuple(0.99 * (2.0 * torch.rand((2, n)) - 1.0))  # Don't clip
        elif method == _XYMethod.STEP:
            return tuple(
                torch.tile((torch.linspace(0.0, 1.0, n) > 0.5)[None, :], (2, 1))
            )

    def _t_apply_delay_float(
        self, n: int, delay: int, method: data._DelayInterpolationMethod
    ):
        x, y = self._create_xy(
            n=n, method=_XYMethod.ARANGE, must_be_in_valid_range=False
        )

        x_out, y_out = data.Dataset._apply_delay(x, y, delay, method)
        # 7, +/-2.5 -> 4
        n_out = n - int(np.ceil(np.abs(delay)))
        assert len(x_out) == n_out
        assert len(y_out) == n_out

        return x_out, y_out

    def _t_apply_delay_int(self, n: int, delay: int):
        x, y = self._create_xy(
            n=n, method=_XYMethod.ARANGE, must_be_in_valid_range=False
        )

        x_out, y_out = data.Dataset._apply_delay(
            x, y, delay, data._DelayInterpolationMethod.CUBIC
        )
        n_out = n - np.abs(delay)
        assert len(x_out) == n_out
        assert len(y_out) == n_out

        return x_out, y_out


if __name__ == "__main__":
    pytest.main()
