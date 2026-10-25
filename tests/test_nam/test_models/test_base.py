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


class MockCompressorNet(MockBaseNet):
    """Memoryless soft clipper: compressive, so metadata gain should be > 0"""

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.gain * x)


class MockPerBurstGainNet(MockBaseNet):
    """
    Linear, but applies a different gain to each of the stimulus' 7 bursts.
    """

    def __init__(self, burst_gains_db, *args, **kwargs):
        super().__init__(1.0, *args, **kwargs)
        self.burst_gains_db = list(burst_gains_db)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        burst_len = x.shape[-1] // len(self.burst_gains_db)
        y = x.clone()
        for b, g_db in enumerate(self.burst_gains_db):
            start, stop = b * burst_len, (b + 1) * burst_len
            y[..., start:stop] = x[..., start:stop] * 10.0 ** (g_db / 20.0)
        return y


def test_metadata_gain():
    obj = MockBaseNet(1.0)
    g = obj._metadata_gain()
    # It's linear, so gain is zero.
    assert g == pytest.approx(0.0, abs=1e-6)


def test_metadata_gain_compressor():
    g = MockCompressorNet(30.0)._metadata_gain()
    assert 0.0 < g <= 1.0


def test_metadata_loudness():
    obj = MockBaseNet(1.0)
    y = obj._metadata_loudness()
    obj.gain = 2.0
    y2 = obj._metadata_loudness()
    assert isinstance(y, float)
    # 2x louder = +6dB
    assert y2 == pytest.approx(y + 20.0 * math.log10(2.0))


def test_metadata_loudness_stimulus():
    """
    The stimulus is a 7-burst level sweep: 1.75 s at 48 kHz, bursts at
    -39, -35, ..., -15 dBFS RMS (nominal; measured over the burst interior).
    """
    x = base.BaseNet._metadata_loudness_x()
    assert x.ndim == 1
    assert x.numel() == int(1.75 * 48_000)
    burst_len = x.numel() // 7
    trim = int(0.03 * 48_000)
    for b, nominal_db in enumerate(range(-39, -14, 4)):
        segment = x[b * burst_len + trim : (b + 1) * burst_len - trim]
        level_db = 20.0 * math.log10(torch.sqrt(torch.mean(segment**2)).item())
        assert level_db == pytest.approx(nominal_db, abs=0.5)
    # No meaningful DC offset
    assert abs(x.mean().item()) < 1e-3


def test_metadata_loudness_is_mean_of_burst_levels_in_db():
    """
    The number averages the per-burst levels in dB (not in energy), so shifting
    the bursts by different gains shifts the result by the mean of those gains.
    """
    reference = MockBaseNet(1.0)._metadata_loudness()
    burst_gains_db = [-12.0, 6.0, 0.0, 3.0, -6.0, 12.0, -3.0]
    shifted = MockPerBurstGainNet(burst_gains_db)._metadata_loudness()
    expected_shift = sum(burst_gains_db) / len(burst_gains_db)
    assert shifted - reference == pytest.approx(expected_shift, abs=0.05)


def test_metadata_loudness_is_insensitive_to_input_level():
    """
    For a compressive model, measuring over a range of input levels makes the
    result far less dependent on where the player sits than a single-level
    measurement would be: shifting the whole stimulus by +-6 dB should move the
    number by much less than 6 dB.
    """
    obj = MockCompressorNet(30.0)
    center = obj._metadata_loudness()
    quiet = obj._metadata_loudness(gain=10.0 ** (-6.0 / 20.0))
    loud = obj._metadata_loudness(gain=10.0 ** (6.0 / 20.0))
    assert abs(quiet - center) < 6.0
    assert abs(loud - center) < 6.0
    assert quiet < center < loud


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
