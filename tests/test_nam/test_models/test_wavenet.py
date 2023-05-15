# File: test_wavenet.py
# Created Date: Friday May 5th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import torch

from nam.models.wavenet import WaveNet
from nam.train.core import Architecture, _get_wavenet_config


# from .base import Base


class TestWaveNet(object):
    def test_import_weights(self):
        config = _get_wavenet_config(Architecture.FEATHER)
        model_1 = WaveNet.init_from_config(config)
        model_2 = WaveNet.init_from_config(config)

        batch_size = 2
        x = torch.randn(batch_size, model_1.receptive_field + 23)

        y1 = model_1(x)
        y2_before = model_2(x)

        model_2.import_weights(model_1._export_weights())
        y2_after = model_2(x)

        assert not torch.allclose(y2_before, y1)
        assert torch.allclose(y2_after, y1)
