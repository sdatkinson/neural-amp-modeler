# File: test_base.py
# Created Date: Thursday March 16th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import math
from pathlib import Path

import numpy as np
import pytest
import torch

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


def test_loudness():
    obj = _MockBaseNet(1.0)
    y = obj._loudness()
    obj.gain = 2.0
    y2 = obj._loudness()
    assert isinstance(y, float)
    # 2x louder = +6dB
    assert y2 == pytest.approx(y + 20.0 * math.log10(2.0))


@pytest.mark.parametrize(
    "batch_size,sequence_length", ((16, 8192), (3, 2048), (1, 4000))
)
def test_mrstft_loss(batch_size: int, sequence_length: int):
    obj = base.Model(
        _MockBaseNet(1.0), loss_config=base.LossConfig(mstft_weight=0.0002)
    )
    preds = torch.randn((batch_size, sequence_length))
    targets = torch.randn(preds.shape)
    loss = obj._mrstft_loss(preds, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


if __name__ == "__main__":
    pytest.main()
