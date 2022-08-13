# File: rcnn.py
# Created Date: Friday August 12th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Recurrent-convolutional neural network.

LSTM outputs a series of hidden states that are inputted to a WaveNet
"""

from pathlib import Path

import numpy as np
import torch

from ._base import BaseNet
from .recurrent import LSTMCore
from .wavenet import WaveNetCore


class RCNN(BaseNet):
    """
    LSTM into a WaveNet, basically
    """
    def __init__(self, recurrent_config, conv_config):
        super().__init__()
        self._recurrent = LSTMCore(**recurrent_config)
        self._conv = WaveNetCore(**conv_config)

    @property
    def receptive_field(self) -> int:
        return self._conv.receptive_field

    @property
    def pad_start_default(self) -> bool:
        return True

    def export_cpp_header(self, filename: Path):
        raise NotImplementedError()

    def _export_config(self):
        raise NotImplementedError()

    def _export_weights(self) -> np.ndarray:
        raise NotImplementedError()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B,L) or (B,L,D)
        :return: (B,L)
        """
        if x.ndim == 2:
            x = x[:, :, None]
        h = self._recurrent(x)  # (B,L,DH)
        # h_conv = torch.cat([x, h], dim=2).transpose(1, 2)  # (B,DH+DX,L)
        h_conv = h.transpose(1, 2)
        y = self._conv(h_conv)
        if y.shape[1] != 1:
            raise ValueError(
                f"WaveNet must output a 1-channel output; got {y.shape[1]} instead."
            )
        return y[:, 0, :]
