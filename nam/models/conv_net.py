# File: conv_net.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import json as _json
import math as _math
from enum import Enum as _Enum
from functools import partial as _partial
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory
from typing import (
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import numpy as _np
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F


from .. import __version__
from ..data import wav_to_tensor as _wav_to_tensor
from ._activations import get_activation as _get_activation
from .base import BaseNet as _BaseNet
from ._names import (
    ACTIVATION_NAME as _ACTIVATION_NAME,
    BATCHNORM_NAME as _BATCHNORM_NAME,
    CONV_NAME as _CONV_NAME,
)


class TrainStrategy(_Enum):
    STRIDE = "stride"
    DILATE = "dilate"


default_train_strategy = TrainStrategy.DILATE


class _Functional(_nn.Module):
    """
    Define a layer by a function w/ no params
    """

    def __init__(self, op):
        super().__init__()
        self._op = op

    def forward(self, *args, **kwargs):
        return self._op(*args, **kwargs)


class _IR(_nn.Module):
    def __init__(self, filename: _Union[str, _Path]):
        super().__init__()
        self.register_buffer("_weight", reversed(_wav_to_tensor(filename))[None, None])

    @property
    def length(self) -> int:
        return self._weight.shape[-1]

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        :param x: (N,D)
        :return: (N,D-length+1)
        """
        return _F.conv1d(x[:, None], self._weight)[:, 0]


def _conv_net(
    channels: int = 32,
    dilations: _Sequence[int] = None,
    batchnorm: bool = False,
    activation: str = "Tanh",
) -> _nn.Sequential:
    def block(cin, cout, dilation):
        net = _nn.Sequential()
        net.add_module(
            _CONV_NAME, _nn.Conv1d(cin, cout, 2, dilation=dilation, bias=not batchnorm)
        )
        if batchnorm:
            net.add_module(_BATCHNORM_NAME, _nn.BatchNorm1d(cout))
        net.add_module(_ACTIVATION_NAME, _get_activation(activation))
        return net

    def check_and_expand(n, x):
        if x.shape[1] < n:
            raise ValueError(
                f"Input of length {x.shape[1]} is shorter than model receptive field ({n})"
            )
        return x[:, None, :]

    dilations = [1, 2, 4, 8] if dilations is None else dilations
    receptive_field = sum(dilations) + 1
    net = _nn.Sequential()
    net.add_module("expand", _Functional(_partial(check_and_expand, receptive_field)))
    cin = 1
    cout = channels
    for i, dilation in enumerate(dilations):
        net.add_module(f"block_{i}", block(cin, cout, dilation))
        cin = cout
    net.add_module("head", _nn.Conv1d(channels, 1, 1))
    net.add_module("flatten", _nn.Flatten())
    return net


class ConvNet(_BaseNet):
    """
    A straightforward convolutional neural network.

    Works surprisingly well!
    """

    def __init__(
        self,
        *args,
        train_strategy: TrainStrategy = default_train_strategy,
        ir: _Optional[_IR] = None,
        sample_rate: _Optional[float] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate)
        self._net = _conv_net(*args, **kwargs)
        assert train_strategy == TrainStrategy.DILATE, "Stride no longer supported"
        self._train_strategy = train_strategy
        self._num_blocks = self._get_num_blocks(self._net)
        self._pad_start_default = True
        self._ir = ir

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)
        config["train_strategy"] = TrainStrategy(
            config.get("train_strategy", default_train_strategy.value)
        )
        config["ir"] = (
            None if "ir_filename" not in config else _IR(config.pop("ir_filename"))
        )
        return config

    @property
    def pad_start_default(self) -> bool:
        return self._pad_start_default

    @property
    def receptive_field(self) -> int:
        net_rf = 1 + sum(
            self._net._modules[f"block_{i}"]._modules["conv"].dilation[0]
            for i in range(self._num_blocks)
        )
        # Minus 1 because it composes w/ the net
        ir_rf = 0 if self._ir is None else self._ir.length - 1
        return net_rf + ir_rf

    @property
    def _activation(self):
        return (
            self._net._modules["block_0"]._modules[_ACTIVATION_NAME].__class__.__name__
        )

    @property
    def _channels(self) -> int:
        return self._net._modules["block_0"]._modules[_CONV_NAME].weight.shape[0]

    @property
    def _num_layers(self) -> int:
        return self._num_blocks

    @property
    def _batchnorm(self) -> bool:
        return _BATCHNORM_NAME in self._net._modules["block_0"]._modules

    def _export_config(self):
        return {
            "channels": self._channels,
            "dilations": self._get_dilations(),
            "batchnorm": self._batchnorm,
            "activation": self._activation,
        }

    def _export_input_output(self, x=None) -> _Tuple[_np.ndarray, _np.ndarray]:
        """
        :return: (L,), (L,)
        """
        with _torch.no_grad():
            training = self.training
            self.eval()
            x = self._export_input_signal() if x is None else x
            y = self(x, pad_start=True)
            self.train(training)
            return tuple(z.detach().cpu().numpy() for z in (x, y))

    def _export_input_signal(self):
        """
        :return: (L,)
        """
        rate = self.sample_rate
        if rate is None:
            raise RuntimeError(
                "Cannot export model's input and output without a sample rate."
            )
        return _torch.cat(
            [
                _torch.zeros((rate,)),
                0.5
                * _torch.sin(
                    2.0 * _math.pi * 220.0 * _torch.linspace(0.0, 1.0, rate + 1)[:-1]
                ),
                _torch.zeros((rate,)),
            ]
        )

    def _export_weights(self) -> _np.ndarray:
        """
        weights are serialized to weights.npy in the following order:
        * (expand: no params)
        * loop blocks 0,...,L-1
            * conv:
                * weight (Cout, Cin, K)
                * bias (if no batchnorm) (Cout)
            * BN
                * running mean
                * running_var
                * weight (Cout)
                * bias (Cout)
                * eps ()
        * head
            * weight (C, 1, 1)
            * bias (1, 1)
        * (flatten: no params)
        """
        params = []
        for i in range(self._num_layers):
            block_name = f"block_{i}"
            block = self._net._modules[block_name]
            conv = block._modules[_CONV_NAME]
            params.append(conv.weight.flatten())
            if conv.bias is not None:
                params.append(conv.bias.flatten())
            if self._batchnorm:
                bn = block._modules[_BATCHNORM_NAME]
                params.append(bn.running_mean.flatten())
                params.append(bn.running_var.flatten())
                params.append(bn.weight.flatten())
                params.append(bn.bias.flatten())
                params.append(_torch.Tensor([bn.eps]).to(bn.weight.device))
        head = self._net._modules["head"]
        params.append(head.weight.flatten())
        params.append(head.bias.flatten())
        params = _torch.cat(params).detach().cpu().numpy()
        return params

    def _forward(self, x):
        y = self._net(x)
        if self._ir is not None:
            y = self._ir(y)
        return y

    def _get_dilations(self) -> _Tuple[int]:
        return tuple(
            self._net._modules[f"block_{i}"]._modules[_CONV_NAME].dilation[0]
            for i in range(self._num_blocks)
        )

    def _get_num_blocks(self, net: _nn.Sequential):
        i = 0
        while True:
            if f"block_{i}" not in net._modules:
                break
            else:
                i += 1
        return i
