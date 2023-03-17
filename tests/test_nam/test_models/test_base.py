# File: test_base.py
# Created Date: Thursday March 16th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nam.models._base import BaseNet

def test_loudness():
    class C(BaseNet):
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
        
    obj = C(1.0)
    y = obj._loudness()
    obj.gain = 2.0
    y2 = obj._loudness()
    assert isinstance(y, float)
    # 2x louder = +6dB
    assert y2 == pytest.approx(y + 20.0 * math.log10(2.0))




if __name__ == "__main__":
    pytest.main()