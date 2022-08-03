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


class Conv1d(nn.Conv1d):
    def export_weights(self) -> torch.Tensor:
        tensors = []
        if self.weight is not None:
            tensors.append(self.weight.data.flatten())
        if self.bias is not None:
            tensors.append(self.bias.data.flatten())
        if len(tensors) == 0:
            return torch.zeros((0,))
        else:
            return torch.cat(tensors)


class _Layer(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        channels: int, 
        kernel_size: int, 
        dilation: int, 
        activation: str, 
        gated: bool, 
        first: bool, 
        final: bool
    ):
        super().__init__()
        # Input mixer takes care of the bias
        mid_channels = 2 * channels if gated else channels
        self._conv = Conv1d(channels, mid_channels, kernel_size, dilation=dilation)
        self._input_mixer = None if first else Conv1d(
            input_size, mid_channels, 1, bias=False
        )
        self._activation = get_activation(activation)
        self._activation_name = activation
        self._1x1 = Conv1d(channels, channels, 1)
        self._first = first
        self._final = final
        self._gated = gated

    @property
    def activation_name(self) -> str:
        return self._activation_name

    @property
    def conv(self) -> Conv1d:
        return self._conv

    @property
    def gated(self) -> bool:
        return self._gated

    def export_weights(self) -> torch.Tensor:
        tensors = [self.conv.export_weights()]
        if self._input_mixer is not None:
            tensors.append(self._input_mixer.export_weights())
        # No params in activation
        tensors.append(self._1x1.export_weights())
        return torch.cat(tensors)   

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
        post_activation = self._activation(z1) if not self._gated else (
            self._activation(z1[:, :self._channels]) 
            * torch.sigmoid(z1[:, self._channels:])
        )
        z2 = self._1x1(post_activation)
        return (
            (None if self._final else x[:, :, -z2.shape[2]:] + z2), 
            z2[:, :, -out_length:]
        )

    @property
    def _channels(self) -> int:
        return self._1x1.in_channels


class _WaveNet(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        output_size, 
        channels: int, 
        kernel_size: int, 
        dilations: Sequence[int], 
        activation: str="Tanh", 
        gated: bool=True, 
        head_layers: int=0, 
        head_channels: int=8, 
        head_activation: str="ReLU",
    ):
        super().__init__()
        self._rechannel = Conv1d(input_size, channels, 1, bias=False)
        self._layers = self._make_layers(input_size, channels, kernel_size, dilations,
            activation, gated
        )
        self._head = self._make_head(channels, output_size, head_channels, head_layers, 
            head_activation
        )
        self._head_channels = head_channels
        self._head_activation = head_activation

    @property
    def activation(self) -> str:
        """
        Activation name
        """
        return self._layers[0].activation_name

    @property
    def channels(self) -> int:
        return self._rechannel.out_channels

    @property
    def dilations(self) -> Tuple[int]:
        return tuple(layer.conv.dilation[0] for layer in self._layers)

    @property
    def gated(self) -> bool:
        return self._layers[0].gated

    @property
    def head_activation(self) -> str:
        return self._head_activation

    @property
    def head_channels(self) -> int:
        return self._head_channels

    @property
    def head_layers(self) -> int:
        return len(self._head)  # They're (Act->Conv) blocks
        
    @property
    def input_size(self) -> int:
        return self._rechannel.in_channels

    @property
    def kernel_size(self) -> int:
        return self._layers[0].conv.kernel_size[0]

    @property
    def receptive_field(self) -> int:
        return 1 + sum(self.dilations)

    def export_weights(self) -> np.ndarray:
        """
        :return: 1D array
        """
        return torch.cat(
            [
                self._rechannel.export_weights(),
                self._export_layer_weights(),
                self._export_head_weights()
            ]
        ).detach().cpu().numpy()

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

    def _export_head_weights(self) -> torch.Tensor:
        """
        return: 1D array
        """
        return torch.cat([layer[1].export_weights() for layer in self._head])

    def _export_layer_weights(self) -> torch.Tensor:
        # Reminder: First layer doesn't have a mixin module!
        return torch.cat([layer.export_weights() for layer in self._layers])

    def _make_head(
        self, 
        in_channels: int, 
        out_channels: int, 
        channels: int, 
        num_layers: int, 
        activation: str
    ) -> nn.Sequential:
        def block(cx, cy):
            net = nn.Sequential()
            net.add_module(ACTIVATION_NAME, get_activation(activation))
            net.add_module(CONV_NAME, Conv1d(cx, cy, 1))
            return net

        assert num_layers > 0

        head = nn.Sequential()
        cin = in_channels
        for i in range(num_layers):
            head.add_module(f"layer_{i}", block(cin, channels if i != num_layers-1 else out_channels))
            cin = channels
        return head

    def _make_layers(self, input_size: int, channels: int, kernel_size: int, 
        dilations: Sequence[int], activation: str, gated: bool) -> nn.ModuleList:
        return nn.ModuleList(
            [
                _Layer(
                    input_size, 
                    channels, 
                    kernel_size, 
                    d, 
                    activation, 
                    gated,  
                    i == 0,
                    i == len(dilations)-1
                ) 
                for i, d in enumerate(dilations)
            ]
        )


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
        raise NotImplementedError("C++ header")

    def _export_config(self):
        return {
            "input_size": self._net.input_size,
            # "output_size": 1,
            "channels": self._net.channels,
            "kernel_size": self._net.kernel_size,
            "dilations": self._net.dilations,
            "activation": self._net.activation,
            "gated": self._net.gated,
            "head_layers": self._net.head_layers,
            "head_channels": self._net.head_channels,
            "head_activation": self._net.head_activation
        }

    def _export_weights(self) -> np.ndarray:
        return self._net.export_weights()

    def _forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._net(x)
        assert y.shape[1] == 1
        return y[:, 0, :]
