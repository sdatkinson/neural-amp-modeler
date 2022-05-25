# File: conv_net.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import json
from enum import Enum
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .. import __version__
from ..data import wav_to_tensor
from ._base import BaseNet

_CONV_NAME = "conv"
_BATCHNORM_NAME = "batchnorm"
_ACTIVATION_NAME = "activation"


class TrainStrategy(Enum):
    STRIDE = "stride"
    DILATE = "dilate"


default_train_strategy = TrainStrategy.DILATE


class _Functional(nn.Module):
    """
    Define a layer by a function w/ no params
    """

    def __init__(self, op):
        super().__init__()
        self._op = op

    def forward(self, *args, **kwargs):
        return self._op(*args, **kwargs)


class _IR(nn.Module):
    def __init__(self, filename: Union[str, Path]):
        super().__init__()
        self.register_buffer("_weight", reversed(wav_to_tensor(filename))[None, None])

    @property
    def length(self) -> int:
        return self._weight.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (N,D)
        :return: (N,D-length+1)
        """
        return F.conv1d(x[:, None], self._weight)[:, 0]


def _make_activation(name: str) -> nn.Module:
    return getattr(nn, name)()


def _conv_net(
    channels: int = 32,
    dilations: Sequence[int] = None,
    batchnorm: bool = False,
    activation: str = "Tanh",
) -> nn.Sequential:
    def block(cin, cout, dilation):
        net = nn.Sequential()
        net.add_module(
            _CONV_NAME, nn.Conv1d(cin, cout, 2, dilation=dilation, bias=not batchnorm)
        )
        if batchnorm:
            net.add_module(_BATCHNORM_NAME, nn.BatchNorm1d(cout))
        net.add_module(_ACTIVATION_NAME, _make_activation(activation))
        return net

    def check_and_expand(n, x):
        if x.shape[1] < n:
            raise ValueError(
                f"Input of length {x.shape[1]} is shorter than model receptive field ({n})"
            )
        return x[:, None, :]

    dilations = [1, 2, 4, 8] if dilations is None else dilations
    receptive_field = sum(dilations) + 1
    net = nn.Sequential()
    net.add_module("expand", _Functional(partial(check_and_expand, receptive_field)))
    cin = 1
    cout = channels
    for i, dilation in enumerate(dilations):
        net.add_module(f"block_{i}", block(cin, cout, dilation))
        cin = cout
    net.add_module("head", nn.Conv1d(channels, 1, 1))
    net.add_module("flatten", nn.Flatten())
    return net


class ConvNet(BaseNet):
    """
    A straightforward convolutional neural network.

    Works surprisingly well!
    """

    def __init__(
        self,
        *args,
        train_strategy: TrainStrategy = default_train_strategy,
        ir: Optional[_IR] = None,
        **kwargs,
    ):
        super().__init__()
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

    def export(self, outdir: Path):
        """
        Files created:
        * config.json
        * weights.npy
        * input.npy
        * output.npy

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

        A test input & output are also provided, input.npy and output.npy
        """
        training = self.training
        self.eval()
        with open(Path(outdir, "config.json"), "w") as fp:
            json.dump(
                {
                    "version": __version__,
                    "architecture": "ConvNet",
                    "config": {
                        "channels": self._channels,
                        "dilations": self._get_dilations(),
                        "batchnorm": self._batchnorm,
                        "activation": self._activation,
                    },
                },
                fp,
                indent=4,
            )

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
                params.append(torch.Tensor([bn.eps]).to(bn.weight.device))
        head = self._net._modules["head"]
        params.append(head.weight.flatten())
        params.append(head.bias.flatten())
        params = torch.cat(params).detach().cpu().numpy()
        # Hope I don't regret using np.save...
        np.save(Path(outdir, "weights.npy"), params)

        # And an input/output to verify correct computation:
        x, y = self._test_signal()
        np.save(Path(outdir, "input.npy"), x.detach().cpu().numpy())
        np.save(Path(outdir, "output.npy"), y.detach().cpu().numpy())

        # And resume training state
        self.train(training)

    def export_cpp_header(self, filename: Path):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self.export(Path(tmpdir))
            with open(Path(tmpdir, "config.json"), "r") as fp:
                _c = json.load(fp)
            version = _c["version"]
            config = _c["config"]
            with open(filename, "w") as f:
                f.writelines(
                    (
                        "#pragma once\n",
                        "// Automatically-generated model file\n",
                        "#include <vector>\n",
                        f'#define PYTHON_MODEL_VERSION "{version}"\n',
                        f"const int CHANNELS = {config['channels']};\n",
                        f"const bool BATCHNORM = {'true' if config['batchnorm'] else 'false'};\n",
                        "std::vector<int> DILATIONS{"
                        + ",".join([str(d) for d in config["dilations"]])
                        + "};\n",
                        f"const std::string ACTIVATION = \"{config['activation']}\";\n",
                        "std::vector<float> PARAMS{"
                        + ",".join(
                            [f"{w:.16f}" for w in np.load(Path(tmpdir, "weights.npy"))]
                        )
                        + "};\n",
                    )
                )

    def _forward(self, x):
        y = self._net(x)
        if self._ir is not None:
            y = self._ir(y)
        return y

    def _get_dilations(self) -> Tuple[int]:
        return tuple(
            self._net._modules[f"block_{i}"]._modules[_CONV_NAME].dilation[0]
            for i in range(self._num_blocks)
        )

    def _get_num_blocks(self, net: nn.Sequential):
        i = 0
        while True:
            if f"block_{i}" not in net._modules:
                break
            else:
                i += 1
        return i


class _SkipConnectConv1d(nn.Module):
    """
    Special skip-connect for 1D Convolutions that trims the input as it passes it over
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self._net = net

    def forward(self, x):
        """
        Require Lout<=Lin

        :param x: (B,C,Lin)
        :return: (B,C,Lout)
        """
        z = self._net(x)
        if z.ndim != 3:
            raise ValueError(f"Expected 3-dimensional tensor; found {z.ndim}")
        l_out = z.shape[-1]
        return z + x[..., -l_out:]


class _SkipIn(nn.Module):
    """
    Module to let the input skip in to a conv layer
    """

    def __init__(self, dim: int):
        super().__init__()
        # self._weight_inner = nn.Parameter(torch.zeros((dim,)))
        # self._weight_skip = nn.Parameter(torch.ones((dim,)))
        # self._raw_balance = nn.Parameter(torch.tensor(0.0))
        self._mix_in = nn.Conv1d(dim, dim, 1, groups=dim)

    # @property
    # def _balance(self) -> torch.Tensor:
    #     return torch.sigmoid(self._raw_balance)

    def forward(self, x_inner, x_skip):
        """
        Assumes 
        :param x_inner: (B,C,L1)
        :param x_skip: (B,C,L2)

        :return: (B,C,L1)
        """
        l_inner = x_inner.shape[-1]
        l_skip = x_skip.shape[-1]
        if l_skip < l_inner:
            raise ValueError(
                f"Skip input is of length {l_skip}, which is less than the inner input "
                f"({l_inner})"
            )
        return x_inner + self._mix_in(x_skip[:, :, -l_inner:])


class SkippyNet(BaseNet):
    """
    A convolutional architecture with skip-in, skip-around (residual), and skip-out 
    connections!
    """

    def __init__(
        self,
        channels: int = 8,
        dilations: Optional[Sequence[int]] = None,
        batchnorm: bool = False,
        activation: str = "Tanh",
        skip_ins: bool = False,
        skip_connections: bool = False,
    ):
        dilations = (
            [
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
            ]
            if dilations is None
            else dilations
        )
        super().__init__()
        self._receptive_field = 1 + sum(dilations)
        self._conv_in = nn.Conv1d(1, channels, 1)
        self._convs = nn.ModuleList(
            [
                self._make_conv(channels, d, activation, batchnorm, skip_connections)
                for d in dilations
            ]
        )
        self._skip_ins = (
            nn.ModuleList([_SkipIn(channels) for _ in dilations]) if skip_ins else None
        )
        self._head = nn.Conv1d(channels * (1 + len(self._convs)), 1, 1)

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def export(self, outdir: Path):
        """
        Files created:
        * config.json
        * weights.npy
        * input.npy
        * output.npy

        weights are serialized to weights.npy in the following order:
        * Conv-in (TODO remove)
            * weight (C,)
            * biases (C,)
        * loop blocks 0,...,L-1
            * skip-in
                * conv
                    * weight (C,)
                    * bias (C,)
            * conv:
                * weight (Cout, Cin, K)
                * bias (if no batchnorm) (Cout)
            * BN
                * running mean
                * running_var
                * weight (Cout)
                * bias (Cout)
                * eps ()
            * activation (nothing)
            * Skip connection
                * balance ()
        * head
            * weight (LC, 1, 1)
            * bias (1, 1)

        A test input & output are also provided, input.npy and output.npy
        """
        # FIXME
        training = self.training
        self.eval()
        with open(Path(outdir, "config.json"), "w") as fp:
            json.dump(
                {
                    "version": __version__,
                    "architecture": "ConvNet",
                    "config": {
                        "channels": self._channels,
                        "dilations": self._get_dilations(),
                        "batchnorm": self._batchnorm,
                        "activation": self._activation,
                        "skip_ins": self._skip_ins is not None,
                        "skip_connections": isinstance(
                            self._convs[0], _SkipConnectConv1d
                        ),
                    },
                },
                fp,
                indent=4,
            )

        params = [self._conv_in.weight.flatten(), self._conv_in.bias.flatten()]
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
                params.append(torch.Tensor([bn.eps]).to(bn.weight.device))
        head = self._net._modules["head"]
        params.append(head.weight.flatten())
        params.append(head.bias.flatten())
        params = torch.cat(params).detach().cpu().numpy()
        # Hope I don't regret using np.save...
        np.save(Path(outdir, "weights.npy"), params)

        # And an input/output to verify correct computation:
        x, y = self._test_signal()
        np.save(Path(outdir, "input.npy"), x.detach().cpu().numpy())
        np.save(Path(outdir, "output.npy"), y.detach().cpu().numpy())

        # And resume training state
        self.train(training)

    def export_cpp_header(self, filename: Path):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self.export(Path(tmpdir))
            with open(Path(tmpdir, "config.json"), "r") as fp:
                _c = json.load(fp)
            version = _c["version"]
            config = _c["config"]
            with open(filename, "w") as f:
                f.writelines(
                    (
                        "#pragma once\n",
                        "// Automatically-generated model file\n",
                        "#include <vector>\n",
                        f'#define PYTHON_MODEL_VERSION "{version}"\n',
                        f"const int CHANNELS = {config['channels']};\n",
                        f"const bool"
                        "std::vector<int> DILATIONS{"
                        + ",".join([str(d) for d in config["dilations"]])
                        + "};\n",
                        f"const bool BATCHNORM = {'true' if config['batchnorm'] else 'false'};\n",
                        f"const std::string ACTIVATION = \"{config['activation']}\";\n",
                        f"const bool SKIP_INS = {'true' if config['skip_ins'] else 'false'};\n",
                        f"const bool SKIP_CONNECTIONS = {'true' if config['skip_connections'] else 'false'};\n",
                        "std::vector<float> PARAMS{"
                        + ",".join(
                            [f"{w:.16f}" for w in np.load(Path(tmpdir, "weights.npy"))]
                        )
                        + "};\n",
                    )
                )

    @classmethod
    def _make_conv(
        cls,
        channels: int,
        dilation: int,
        activation: str,
        batchnorm: bool,
        skip_connection: bool,
    ):
        net = nn.Sequential()
        net.add_module(
            _CONV_NAME,
            nn.Conv1d(channels, channels, 2, dilation=dilation, bias=not batchnorm),
        )
        if batchnorm:
            net.add_module(_BATCHNORM_NAME, nn.BatchNorm1d(channels))
        net.add_module(_ACTIVATION_NAME, _make_activation(activation))
        if skip_connection:
            net = _SkipConnectConv1d(net)
        return net

    @property
    def _num_layers(self) -> int:
        return len(self._convs)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        B=Batch size
        L=Input length
        R=receptive field

        :param x: (B,L)
        :return: (B, L-R+1)
        """
        x = self._conv_in(x[:, None, :])
        if self._skip_ins is not None:
            x_skip_in = x
        layer_outputs = [x]
        for i in range(len(self._convs)):
            if self._skip_ins is not None:
                x = self._skip_ins[i](x, x_skip_in)
            c = self._convs[i]
            x = c(x)
            layer_outputs.append(x)
        n = layer_outputs[-1].shape[-1]
        head_in = torch.cat([lo[:, :, -n:] for lo in layer_outputs], dim=1)
        head_out = self._head(head_in)
        return head_out[:, 0]

    def _get_dilations(self) -> Tuple[int]:
        return tuple(
            self._convs[i]._modules[_CONV_NAME].dilation[0]
            for i in range(self._num_blocks)
        )
