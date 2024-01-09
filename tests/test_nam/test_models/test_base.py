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
from auraloss.freq import MultiResolutionSTFTLoss

from nam.models import _base, base


class _MockBaseNet(_base.BaseNet):
    def __init__(self, gain: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = gain

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return 1

    def export_cpp_header(self, filename: Path):
        pass

    def _export_config(self):
        pass

    def _export_weights(self) -> np.ndarray:
        pass

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * x


def test_metadata_gain():
    obj = _MockBaseNet(1.0)
    g = obj._metadata_gain()
    # It's linear, so gain is zero.
    assert g == 0.0


def test_metadata_loudness():
    obj = _MockBaseNet(1.0)
    y = obj._metadata_loudness()
    obj.gain = 2.0
    y2 = obj._metadata_loudness()
    assert isinstance(y, float)
    # 2x louder = +6dB
    assert y2 == pytest.approx(y + 20.0 * math.log10(2.0))


@pytest.mark.parametrize(
    "batch_size,sequence_length", ((16, 8192), (3, 2048), (1, 4000))
)
def test_mrstft_loss(batch_size: int, sequence_length: int):
    obj = base.Model(
        _MockBaseNet(1.0), loss_config=base.LossConfig(mrstft_weight=0.0002)
    )
    preds = torch.randn((batch_size, sequence_length))
    targets = torch.randn(preds.shape)
    loss = obj._mrstft_loss(preds, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_mrstft_loss_cpu_fallback(mocker):
    """
    Assert that fallback to CPU happens on failure

    :param mocker: Provided by pytest-mock
    """

    def mocked_loss(
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss_func: Optional[MultiResolutionSTFTLoss] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        As if the device doesn't support it
        """
        if device != "cpu":
            raise RuntimeError("Trigger fallback")
        return torch.tensor(1.0)

    mocker.patch("nam.models.base.multi_resolution_stft_loss", mocked_loss)

    batch_size = 3
    sequence_length = 4096
    obj = base.Model(
        _MockBaseNet(1.0), loss_config=base.LossConfig(mrstft_weight=0.0002)
    )
    preds = torch.randn((batch_size, sequence_length))
    targets = torch.randn(preds.shape)

    assert obj._mrstft_device is None
    obj._mrstft_loss(preds, targets)  # Should trigger fallback
    assert obj._mrstft_device == "cpu"


class TestSampleRate(object):
    """
    Tests for sample_rate interface
    """

    @pytest.mark.parametrize("expected_sample_rate", (None, 44_100.0, 48_000.0))
    def test_on_init(self, expected_sample_rate: Optional[float]):
        model = _MockBaseNet(gain=1.0, sample_rate=expected_sample_rate)
        self._wrap_assert(model, expected_sample_rate)

    @pytest.mark.parametrize("expected_sample_rate", (None, 44_100.0, 48_000.0))
    def test_setter(self, expected_sample_rate: Optional[float]):
        model = _MockBaseNet(gain=1.0)
        model.sample_rate = expected_sample_rate
        self._wrap_assert(model, expected_sample_rate)

    @pytest.mark.parametrize("expected_sample_rate", (None, 44_100.0, 48_000.0))
    def test_state_dict(self, expected_sample_rate: Optional[float]):
        """
        Assert that it makes it into the state dict

        https://github.com/sdatkinson/neural-amp-modeler/issues/351
        """
        model = _MockBaseNet(gain=1.0, sample_rate=expected_sample_rate)
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)
            model2 = _MockBaseNet(gain=1.0)
            model2.load_state_dict(torch.load(model_path))
        self._wrap_assert(model2, expected_sample_rate)

    @classmethod
    def _wrap_assert(cls, model: _MockBaseNet, expected: Optional[float]):
        actual = model.sample_rate
        if expected is None:
            assert actual is None
        else:
            assert isinstance(actual, float)
            assert actual == expected


if __name__ == "__main__":
    pytest.main()
