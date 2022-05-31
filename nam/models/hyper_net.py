# File: hyper_net.py
# Created Date: Sunday May 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import (
    calculate_gain,
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
)

from ._base import ParametricBaseNet


class SpecialLayers(Enum):
    CONV = "conv"
    BATCHNORM = "batchnorm"


@dataclass
class LayerSpec:
    """
    Helps the hypernet
    """

    special_type: Optional[str]
    shapes: Tuple[Tuple[int]]
    norms: Tuple[float]
    biases: Tuple[float]


class _NetLayer(nn.Module, abc.ABC):
    @abc.abstractproperty
    def num_tensors(self) -> int:
        pass

    @abc.abstractmethod
    def get_spec(self) -> LayerSpec:
        pass


class _Conv(nn.Conv1d, _NetLayer):
    @property
    def num_tensors(self):
        return 2 if self.bias is not None else 1

    def forward(self, params, inputs):
        # Use depthwise convolution trick to process the convolutions together
        cout, cin, kernel_size = self.weight.shape
        n = len(params[0])
        weight = params[0].reshape((n * cout, cin, kernel_size))  # (N, CinCout)
        bias = params[1].flatten() if self.bias is not None else None
        groups = n
        return F.conv1d(
            inputs.reshape((1, n * cin, -1)),
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups,
        ).reshape((n, cout, -1))

    def get_spec(self):
        shapes = (
            (self.weight.shape,)
            if self.bias is None
            else (self.weight.shape, self.bias.shape)
        )
        norms = (
            (self._weight_norm(),)
            if self.bias is None
            else (self._weight_norm(), self._bias_norm())
        )
        biases = (0,) if self.bias is None else (0, 0)
        return LayerSpec(SpecialLayers("conv"), shapes, norms, biases)

    def _bias_norm(self):
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
        fan = _calculate_fan_in_and_fan_out(self.weight.data)[0]
        bound = 1.0 / math.sqrt(fan)
        std = math.sqrt(1.0 / 12.0) * (2 * bound)
        # LayerNorm handles division by number of dimensions...
        return std

    def _weight_norm(self):
        """
        Std of the unfiorm distribution used in initialization
        """
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d
        fan = _calculate_correct_fan(self.weight.data, "fan_in")
        # Kaiming uniform w/ a=sqrt(5)
        gain = calculate_gain("leaky_relu", 5.0)
        std = gain / math.sqrt(fan)
        # LayerNorm handles division by number of dimensions...
        return std


class _BatchNorm(nn.BatchNorm1d, _NetLayer):
    def __init__(self, num_features, *args, affine=True, **kwargs):
        # Handle affine params outside of parent class
        super().__init__(num_features, *args, affine=False, **kwargs)
        self._num_features = num_features
        assert affine
        self._affine = affine

    @property
    def num_tensors(self) -> int:
        return 2

    def get_spec(self) -> LayerSpec:
        return LayerSpec(
            SpecialLayers.BATCHNORM,
            ((self._num_features,), (self._num_features,)),
            (1.0e-5, 1.0e-5),
            (1.0, 0.0),
        )

    def forward(self, params, inputs):
        """
        Only change is we need to provide *params into F.batch_norm instead of 
        self.weight, self.bias
        """
        # Also use "inputs" instead of "input" to not collide w/ builtin (ew)
        weight, bias = [z[:, :, None] for z in params]
        pre_affine = super().forward(inputs)
        return weight * pre_affine + bias


class _Affine(nn.Module):
    def __init__(self, scale: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self._weight = nn.Parameter(scale)
        self._bias = nn.Parameter(bias)

    def forward(self, inputs):
        return self._weight * inputs + self._bias


class HyperNet(nn.Module):
    """
    MLP followed by layer norms on split-up dims
    """

    def __init__(self, d_in, net, numels, norms, biases):
        super().__init__()
        self._net = net
        # Figure out the scale factor empirically
        norm0 = net(torch.randn((10_000, d_in))).std(dim=0).mean().item()
        self._cum_numel = torch.cat(
            [torch.LongTensor([0]), torch.cumsum(torch.LongTensor(numels), dim=0)]
        )
        affine_scale = torch.cat(
            [torch.full((numel,), norm / norm0) for numel, norm in zip(numels, norms)]
        )
        affine_bias = torch.cat(
            [torch.full((numel,), bias) for numel, bias in zip(numels, biases)]
        )
        self._affine = _Affine(affine_scale, affine_bias)

    def forward(self, x) -> Tuple[torch.Tensor]:
        """
        Just return a flat array of param tensors for now
        """
        y = self._affine(self._net(x))
        return tuple(
            y[:, i:j] for i, j in zip(self._cum_numel[:-1], self._cum_numel[1:])
        )


def _extend_activation(C):
    class Activation(C, _NetLayer):
        @property
        def num_tensors(self) -> int:
            return 0

        def get_spec(self) -> LayerSpec:
            return LayerSpec(None, (), (), ())

        def forward(self, params, inputs):
            return super().forward(inputs)

    return Activation


_Tanh = _extend_activation(nn.Tanh)
_Flatten = _extend_activation(nn.Flatten)  # Hah, works


def _get_activation(name):
    return {"Tanh": _Tanh}[name]()


class HyperConvNet(ParametricBaseNet):
    """
    For parameteric data

    Conditioning is input to a hypernetwork that outputs the parameters of the conv net.
    """

    def __init__(
        self, hyper_net: nn.Module, net: Callable[[Any, torch.Tensor], torch.Tensor]
    ):
        super().__init__()
        self._hyper_net = hyper_net
        self._net = net

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)
        net, specs = cls._get_net(config["net"])
        hyper_net = cls._get_hyper_net(config["hyper_net"], specs)
        return {"hyper_net": hyper_net, "net": net}

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        # Last conv is the collapser--compensate w/ a minus 1
        return sum([m.dilation[0] for m in self._net if isinstance(m, _Conv)]) + 1 - 1

    def export(self, outdir: Path):
        raise NotImplementedError()

    def export_cpp_header(self, filename: Path):
        return NotImplementedError()

    @classmethod
    def _get_net(cls, config):
        channels = config["channels"]
        dilations = config["dilations"]
        batchnorm = config["batchnorm"]
        activation = config["activation"]

        layers = []
        layer_specs = []
        cin = 1
        for dilation in dilations:
            layer = _Conv(cin, channels, 2, dilation=dilation, bias=not batchnorm)
            layers.append(layer)
            layer_specs.append(layer.get_spec())
            if batchnorm:
                # Slow momentum on main net bc it's wild
                layer = _BatchNorm(channels, momentum=0.01)
                layers.append(layer)
                layer_specs.append(layer.get_spec())
            layer = _get_activation(activation)
            layers.append(layer)
            layer_specs.append(layer.get_spec())
            cin = channels
        layer = _Conv(cin, 1, 1)
        layers.append(layer)
        layer_specs.append(layer.get_spec())
        layer = _Flatten()
        layers.append(layer)
        layer_specs.append(layer.get_spec())

        return nn.ModuleList(layers), layer_specs

    @classmethod
    def _get_hyper_net(cls, config, specs):
        def block(dx, dy, batchnorm):
            layer_list = [nn.Linear(dx, dy)]
            if batchnorm:
                layer_list.append(nn.BatchNorm1d(dy))
            layer_list.append(nn.ReLU())
            return nn.Sequential(*layer_list)

        num_inputs = config["num_inputs"]
        num_layers = config["num_layers"]
        num_units = config["num_units"]
        batchnorm = config["batchnorm"]
        # Flatten specs
        numels = [np.prod(np.array(shape)) for spec in specs for shape in spec.shapes]
        norms = [norm for spec in specs for norm in spec.norms]
        biases = [bias for spec in specs for bias in spec.biases]
        num_outputs = sum(numels)

        din, layer_list = num_inputs, []
        for _ in range(num_layers):
            layer_list.append(block(din, num_units, batchnorm))
            din = num_units
        layer_list.append(nn.Linear(din, num_outputs))
        net = nn.Sequential(*layer_list)

        return HyperNet(num_inputs, net, numels, norms, biases)

    def _forward(self, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        net_params = self._hyper_net(params)
        i = 0
        for m in self._net:
            j = i + m.num_tensors
            x = m(net_params[i:j], x)
            i = j
        assert j == len(net_params)
        return x
