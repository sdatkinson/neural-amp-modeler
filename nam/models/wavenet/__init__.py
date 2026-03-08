# File: wavenet.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
WaveNet implementation
https://arxiv.org/abs/1609.03499
"""

import logging as _logging
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Sequence as _Sequence

import numpy as _np
import torch as _torch

from .._abc import ImportsWeights as _ImportsWeights
from ..base import BaseNet as _BaseNet
from ._wavenet import WaveNet as _WaveNet

_logger = _logging.getLogger(__name__)


class WaveNet(_BaseNet, _ImportsWeights):
    def __init__(
        self, wavenet: _WaveNet, sample_rate: _Optional[float] = None, **kwargs
    ):
        super().__init__(sample_rate=sample_rate)
        self._net = wavenet

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        wavenet = _WaveNet.init_from_config(config)

        return {"sample_rate": sample_rate, "wavenet": wavenet}

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    def import_weights(self, weights: _Sequence[float], i: int = 0) -> int:
        weights_tensor = (
            weights if isinstance(weights, _torch.Tensor) else _torch.Tensor(weights)
        )
        return self._net.import_weights(weights_tensor, i)

    def _export_config(self):
        return self._net.export_config(sample_rate=self.sample_rate)

    def _export_weights(self) -> _np.ndarray:
        return self._net.export_weights()

    def _forward(self, x, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("WaveNet does not support kwargs")
        if x.ndim == 2:
            x = x[:, None, :]
        if self.training and self._net.is_slimmable():
            with self._net.context_adjust_to_random():
                y = self._net(x)
        else:
            y = self._net(x)
        assert y.shape[1] == 1
        return y[:, 0, :]
