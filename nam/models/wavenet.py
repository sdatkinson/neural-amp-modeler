# File: wavenet.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
WaveNet implementation
https://arxiv.org/abs/1609.03499
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ._activations import get_activation
from ._base import BaseNet
from ._names import ACTIVATION_NAME, CONV_NAME

class _Layer(nn.Module):
    def __init__(
        self, input_size, channels: int, kernel_size: int, 
        dilation: int, first: bool, final: bool):
        super().__init__()
        # Input mixer takes care of the bias
        self._conv = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation, bias=not first)
        self._input_mixer = None if first else nn.Conv1d(input_size, 2 * channels, 1)
        self._1x1 = nn.Conv1d(channels, channels, 1)
        self._first = first
        self._final = final

    @property
    def conv(self) -> nn.Conv1d:
        return self._conv

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor], out_length: int
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        :param x: (B,C,L1) From last layer
        :param h: (B,DX,L2) Conditioning. If first, ignored.
        
        :return: 
            If not final:
                (B,C,L1-d) to next layer
                (B,C,L1-d) to mixer
            If final, next layer is None
        """
        zconv = self.conv(x)
        if self._first:
            assert h is None
            z1 = zconv
        else:
            z1 = zconv + self._input_mixer(h)[:, :, -zconv.shape[2]:]
        z2 = self._1x1(
            torch.tanh(z1[:, :self._channels]) * torch.sigmoid(z1[:, self._channels:])
        )
        return (None if self._final else x[:, :, -z2.shape[2]:] + z2), z2[:, :, -out_length:]

    @property
    def _channels(self) -> int:
        return self._1x1.in_channels



class _WaveNet(nn.Module):
    def __init__(self, input_size: int, output_size, channels: int, kernel_size: int, 
        dilations: Sequence[int], head_layers: int=0, head_channels: int=8, head_activation: str="ReLU"
    ):
        super().__init__()
        self._rechannel = nn.Conv1d(input_size, channels, 1)
        self._layers = self._make_layers(input_size, channels, kernel_size, dilations)
        self._head = self._make_head(channels, output_size, head_channels, head_layers, 
            head_activation)

    @property
    def receptive_field(self) -> int:
        return 1 + sum([layer.conv.dilation[0] for layer in self._layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B,Cx,L)
        :return: (B,Cy,L-R)
        """
        z = self._rechannel(x)
        out_length = x.shape[2] - self.receptive_field + 1
        head_input = None
        for i, layer in enumerate(self._layers):
            z, head_term = layer(z, None if i == 0 else x, out_length)
            head_input = head_term if head_input is None else head_input + head_term
        return self._head(head_input)

    def _make_head(self, in_channels: int, out_channels: int, channels: int, 
        num_layers: int, activation: str
    ) -> nn.Sequential:
        def block(cx, cy):
            net = nn.Sequential()
            net.add_module(ACTIVATION_NAME, get_activation(activation))
            net.add_module(CONV_NAME, nn.Conv1d(cx, cy, 1))
            return net

        head = nn.Sequential()
        cin = in_channels
        for i in range(num_layers):
            head.add_module(f"layer_{i}", block(cin, channels))
            cin = channels
        head.add_module("head", nn.Conv1d(cin, out_channels, 1))
        return head

    def _make_layers(self, input_size: int, channels: int, kernel_size: int, dilations: Sequence[int]) -> nn.ModuleList:
        return nn.ModuleList([_Layer(input_size, channels, kernel_size, d, i == 0, 
            i == len(dilations)-1) for i, d in enumerate(dilations)])


class WaveNet(BaseNet):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._net = _WaveNet(*args, **kwargs)

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    def export_cpp_header(self, filename: Path):
        raise NotImplementedError()

    def _export_config(self):
        raise NotImplementedError()

    def _export_weights(self) -> np.ndarray:
        raise NotImplementedError()

    def _forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._net(x)
        assert y.shape[1] == 1
        return y[:, 0, :]
