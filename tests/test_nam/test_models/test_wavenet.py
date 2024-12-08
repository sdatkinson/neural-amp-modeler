# File: test_wavenet.py
# Created Date: Friday May 5th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest as _pytest
import torch as _torch

from nam.models.wavenet import WaveNet as _WaveNet
from nam.train.core import (
    Architecture as _Architecture,
    get_wavenet_config as _get_wavenet_config,
)

from .base import Base as _Base


class TestWaveNet(_Base):
    @classmethod
    def setup_class(cls):
        C = _WaveNet
        args = ()
        kwargs = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 1,
                    "kernel_size": 1,
                    "dilations": [1],
                }
            ]
        }
        super().setup_class(C, args, kwargs)

    def test_import_weights(self):
        config = _get_wavenet_config(_Architecture.FEATHER)
        model_1 = _WaveNet.init_from_config(config)
        model_2 = _WaveNet.init_from_config(config)

        batch_size = 2
        x = _torch.randn(batch_size, model_1.receptive_field + 23)

        y1 = model_1(x)
        y2_before = model_2(x)

        model_2.import_weights(model_1._export_weights())
        y2_after = model_2(x)

        assert not _torch.allclose(y2_before, y1)
        assert _torch.allclose(y2_after, y1)


if __name__ == "__main__":
    _pytest.main()
