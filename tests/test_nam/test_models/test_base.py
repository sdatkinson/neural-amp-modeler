# File: test_base.py
# Created Date: Thursday March 16th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Tests for the base network and Lightning module
"""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import pytest
import torch

from nam.models import base


class MockBaseNet(base.BaseNet):
    def __init__(self, gain: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = gain

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return 1

    def _export_config(self):
        pass

    def _export_weights(self) -> np.ndarray:
        pass

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * x


def test_metadata_gain():
    obj = MockBaseNet(1.0)
    g = obj._metadata_gain()
    # It's linear, so gain is zero.
    assert g == 0.0


def test_metadata_loudness():
    obj = MockBaseNet(1.0)
    y = obj._metadata_loudness()
    obj.gain = 2.0
    y2 = obj._metadata_loudness()
    assert isinstance(y, float)
    # 2x louder = +6dB
    assert y2 == pytest.approx(y + 20.0 * math.log10(2.0))


class TestSampleRate(object):
    """
    Tests for sample_rate interface
    """

    @pytest.mark.parametrize("expected_sample_rate", (None, 44_100.0, 48_000.0))
    def test_on_init(self, expected_sample_rate: Optional[float]):
        model = MockBaseNet(gain=1.0, sample_rate=expected_sample_rate)
        self._wrap_assert(model, expected_sample_rate)

    @pytest.mark.parametrize("expected_sample_rate", (None, 44_100.0, 48_000.0))
    def test_setter(self, expected_sample_rate: Optional[float]):
        model = MockBaseNet(gain=1.0)
        model.sample_rate = expected_sample_rate
        self._wrap_assert(model, expected_sample_rate)

    @pytest.mark.parametrize("expected_sample_rate", (None, 44_100.0, 48_000.0))
    def test_state_dict(self, expected_sample_rate: Optional[float]):
        """
        Assert that it makes it into the state dict

        https://github.com/sdatkinson/neural-amp-modeler/issues/351
        """
        model = MockBaseNet(gain=1.0, sample_rate=expected_sample_rate)
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)
            model2 = MockBaseNet(gain=1.0)
            model2.load_state_dict(torch.load(model_path))
        self._wrap_assert(model2, expected_sample_rate)

    @classmethod
    def _wrap_assert(cls, model: MockBaseNet, expected: Optional[float]):
        actual = model.sample_rate
        if expected is None:
            assert actual is None
        else:
            assert isinstance(actual, float)
            assert actual == expected


if __name__ == "__main__":
    pytest.main()
