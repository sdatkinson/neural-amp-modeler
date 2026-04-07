# File: _head.py
# WaveNet output head: repeated (activation → Conv1d) blocks (no dilation / groups options).

from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Sequence as _Sequence
from typing import Union as _Union

import torch as _torch
import torch.nn as _nn

from .._activations import get_activation as _get_activation
from .._names import ACTIVATION_NAME as _ACTIVATION_NAME
from .._names import CONV_NAME as _CONV_NAME
from ._conv import Conv1d as _Conv1d


class Head(_nn.Module):
    """
    ``len(kernel_sizes)`` blocks, each ``activation`` then ``Conv1d`` with the
    corresponding kernel size (stride 1, padding 0, dilation 1, groups 1, bias True).

    The first block applies an activation before its convolution, matching the
    historical WaveNet head layout.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: _Union[str, _Dict[str, _Any]],
        out_channels: int,
        kernel_sizes: _Sequence[int],
    ) -> None:
        """
        :param in_channels: Must match the last layer array's ``head_channels``
            (``head_size``); supplied by ``WaveNet`` construction, not the JSON head dict.
        """
        super().__init__()
        ks = list(kernel_sizes)
        if len(ks) < 1:
            raise ValueError("kernel_sizes must be non-empty")
        self._kernel_sizes = ks

        def block(cx: int, cy: int, kernel_size: int) -> _nn.Sequential:
            net = _nn.Sequential()
            net.add_module(_ACTIVATION_NAME, _get_activation(activation))
            net.add_module(
                _CONV_NAME,
                _Conv1d(
                    cx,
                    cy,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=True,
                ),
            )
            return net

        num_layers = len(ks)
        layers = _nn.Sequential()
        cin = in_channels
        for i in range(num_layers):
            cout = channels if i != num_layers - 1 else out_channels
            layers.add_module(f"layer_{i}", block(cin, cout, ks[i]))
            cin = channels
        self._layers = layers

        # in_channels is implied by the last layer array's head_channels / head_size;
        # omit from export so configs stay single-source.
        self._config: _Dict[str, _Any] = {
            "channels": channels,
            "activation": activation,
            "out_channels": out_channels,
            "kernel_sizes": _deepcopy(ks),
        }

    @property
    def receptive_field(self) -> int:
        return 1 + sum(k - 1 for k in self._kernel_sizes)

    def export_config(self) -> _Dict[str, _Any]:
        return _deepcopy(self._config)

    def export_weights(self) -> _torch.Tensor:
        return _torch.cat([layer[1].export_weights() for layer in self._layers])

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        for layer in self._layers:
            i = layer[1].import_weights(weights, i)
        return i

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self._layers(x)
