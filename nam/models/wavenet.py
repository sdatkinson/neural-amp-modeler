# File: wavenet.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
WaveNet implementation
https://arxiv.org/abs/1609.03499
"""

import logging as _logging
from copy import deepcopy as _deepcopy
from typing import (
    Dict as _Dict,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import numpy as _np
import torch as _torch
import torch.nn as _nn
from pydantic import BaseModel as _BaseModel, model_validator as _model_validator

from ._abc import ImportsWeights as _ImportsWeights
from ._activations import (
    PairBlend as _PairBlend,
    PairMultiply as _PairMultiply,
    PairingActivation as _PairingActivation,
    export_activation_config as _export_activation_config,
    get_activation as _get_activation,
)
from .base import BaseNet as _BaseNet
from ._names import ACTIVATION_NAME as _ACTIVATION_NAME, CONV_NAME as _CONV_NAME

_logger = _logging.getLogger(__name__)


class _Head1x1Config(_BaseModel):
    active: bool = False
    # NOTE: NeuralAmpModelerCore requires non-null values for out_channels and groups,
    # even though they are not used if active is False.
    out_channels: int = 1
    groups: int = 1


class _Layer1x1Config(_BaseModel):
    active: bool = True
    groups: int = 1


class Conv1d(_nn.Conv1d):
    def export_weights(self) -> _torch.Tensor:
        tensors = []
        if self.weight is not None:
            tensors.append(self.weight.data.flatten())
        if self.bias is not None:
            tensors.append(self.bias.data.flatten())
        if len(tensors) == 0:
            return _torch.zeros((0,))
        else:
            return _torch.cat(tensors)

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        if self.weight is not None:
            n = self.weight.numel()
            self.weight.data = (
                weights[i : i + n].reshape(self.weight.shape).to(self.weight.device)
            )
            i += n
        if self.bias is not None:
            n = self.bias.numel()
            self.bias.data = (
                weights[i : i + n].reshape(self.bias.shape).to(self.bias.device)
            )
            i += n
        return i


class _Layer(_nn.Module):
    def __init__(
        self,
        condition_size: int,
        channels: int,
        kernel_size: int,
        dilation: int,
        activation: _nn.Module,
        bottleneck: int,
        head_1x1_config: _Head1x1Config,
        layer_1x1_config: _Layer1x1Config,
    ):
        super().__init__()
        # Input mixer takes care of the bias
        mid_channels = (
            2 * bottleneck if isinstance(activation, _PairingActivation) else bottleneck
        )
        self._conv = Conv1d(channels, mid_channels, kernel_size, dilation=dilation)
        # Custom init: favors direct input-output
        # self._conv.weight.data.zero_()
        self._input_mixer = Conv1d(condition_size, mid_channels, 1, bias=False)

        self._activation = activation
        self._bottleneck = bottleneck

        self._activation_name = activation
        if layer_1x1_config.active:
            self._layer1x1 = (
                Conv1d(bottleneck, channels, 1, groups=layer_1x1_config.groups)
                if layer_1x1_config.active
                else None
            )
        else:
            self._layer1x1 = None
            assert (
                bottleneck == channels
            ), "bottleneck must equal channels if layer1x1 is not active"
        self._head1x1 = (
            None
            if not head_1x1_config.active
            else Conv1d(
                bottleneck,
                head_1x1_config.out_channels,
                1,
                groups=head_1x1_config.groups,
            )
        )

    @property
    def activation_name(self) -> str:
        if isinstance(self._activation, _PairingActivation):
            return self._activation.name
        else:
            return self._activation.__class__.__name__

    @property
    def conv(self) -> Conv1d:
        return self._conv

    @property
    def gated(self) -> bool:
        if isinstance(self._activation, _PairMultiply):
            return True
        elif isinstance(self._activation, _PairBlend):
            raise ValueError("PairBlend is not a gating activation")
        else:
            return False

    @property
    def kernel_size(self) -> int:
        return self._conv.kernel_size[0]

    def export_weights(self) -> _torch.Tensor:
        tensors = [
            self.conv.export_weights(),
            self._input_mixer.export_weights(),
        ]
        if self._layer1x1 is not None:
            tensors.append(self._layer1x1.export_weights())
        if self._head1x1 is not None:
            tensors.append(self._head1x1.export_weights())
        return _torch.cat(tensors)

    def forward(
        self, x: _torch.Tensor, h: _Optional[_torch.Tensor], out_length: int
    ) -> _Tuple[_Optional[_torch.Tensor], _torch.Tensor]:
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
        z1 = zconv + self._input_mixer(h)[:, :, -zconv.shape[2] :]
        post_activation = self._activation(z1)
        head_output = (
            post_activation[:, :, -out_length:]
            if self._head1x1 is None
            else self._head1x1(post_activation)[:, :, -out_length:]
        )
        layer_output = (
            post_activation
            if self._layer1x1 is None
            else self._layer1x1(post_activation)
        )
        residual = x[:, :, -layer_output.shape[2] :] + layer_output
        return (residual, head_output)

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        i = self.conv.import_weights(weights, i)
        i = self._input_mixer.import_weights(weights, i)
        if self._layer1x1 is not None:
            i = self._layer1x1.import_weights(weights, i)
        if self._head1x1 is not None:
            i = self._head1x1.import_weights(weights, i)
        return i

    def export_activation_config(self):
        """
        Export activation in WaveNet Factory format: (primary, gating_mode, secondary).
        Parses the output of _activations.export_activation_config() into primary/
        secondary/gating_mode as expected by the C++ Factory.
        """
        out = _export_activation_config(self._activation)
        if isinstance(self._activation, _PairingActivation):
            if isinstance(self._activation, _PairMultiply):
                gating_mode = "gated"
            elif isinstance(self._activation, _PairBlend):
                gating_mode = "blended"
            else:
                raise ValueError(f"Unknown pairing activation: {self._activation}")
            return {
                "primary": out["primary"],
                "gating_mode": gating_mode,
                "secondary": out["secondary"],
            }
        else:
            return {
                "primary": out,
                "gating_mode": "none",
                "secondary": None,
            }


class _LayerArray(_nn.Module):
    """
    Takes in the input and condition (and maybe the head input so far); outputs the
    layer output and head input.

    The original WaveNet only uses one of these, but you can stack multiple of this
    module to vary the channels throughout with minimal extra channel-changing conv
    layers.
    """

    def __init__(
        self,
        input_size: int,
        condition_size: int,
        head_size,
        channels: int,
        kernel_size: int,
        dilations: _Sequence[int],
        activation: _Union[str, dict, _Sequence[_Union[str, dict]]] = "Tanh",
        head_bias: bool = True,
        bottleneck: _Optional[int] = None,
        head_1x1_config: _Optional[dict] = None,
        layer_1x1_config: _Optional[dict] = None,
    ):
        super().__init__()
        head_1x1_config = dict() if head_1x1_config is None else head_1x1_config
        head1x1_config_pydantic = _Head1x1Config(**head_1x1_config)
        layer_1x1_config = dict() if layer_1x1_config is None else layer_1x1_config
        layer1x1_config_pydantic = _Layer1x1Config(**layer_1x1_config)

        bottleneck = channels if bottleneck is None else bottleneck

        self._rechannel = Conv1d(input_size, channels, 1, bias=False)
        num_layers = len(dilations)

        # Broadcast configs to all layers:
        # Activation:
        if isinstance(activation, (str, dict)):
            activation = [activation] * num_layers
        else:
            assert isinstance(
                activation, _Sequence
            ), "activation must be a string, dict, or sequence"
        a_list = [_get_activation(a) for a in activation]

        self._layers = _nn.ModuleList(
            [
                _Layer(
                    condition_size,
                    channels,
                    kernel_size,
                    dilations[i],
                    activation=a_list[i],
                    bottleneck=bottleneck,
                    head_1x1_config=head1x1_config_pydantic,
                    layer_1x1_config=layer1x1_config_pydantic,
                )
                for i in range(num_layers)
            ]
        )
        # Convert the head input to head_size (from head1x1 out_channels or bottleneck)
        head_rechannel_in = (
            head1x1_config_pydantic.out_channels
            if head1x1_config_pydantic.active
            else bottleneck
        )
        self._head_rechannel = Conv1d(head_rechannel_in, head_size, 1, bias=head_bias)

        self._config = {
            "input_size": input_size,
            "condition_size": condition_size,
            "head_size": head_size,
            "channels": channels,
            "kernel_size": kernel_size,
            "dilations": dilations,
            "activation": activation,
            "head_bias": head_bias,
            "bottleneck": bottleneck,
            "head1x1": head1x1_config_pydantic.model_dump(),
            "layer1x1": layer1x1_config_pydantic.model_dump(),
        }

    @property
    def receptive_field(self) -> int:
        return 1 + (self._kernel_size - 1) * sum(self._dilations)

    def export_config(self):
        config = _deepcopy(self._config)
        # Override dilations and activations with programmatic values (C++ format)
        dilations = []
        activations = []
        gating_modes = []
        secondary_activations = []

        for layer in self._layers:
            d = layer.conv.dilation
            dilations.append(d if isinstance(d, int) else d[0])
            activation_config = layer.export_activation_config()
            activations.append(activation_config["primary"])
            gating_modes.append(activation_config["gating_mode"])
            secondary_activations.append(activation_config["secondary"])

        config["dilations"] = dilations
        config["activation"] = activations
        config["gating_mode"] = gating_modes
        config["secondary_activation"] = secondary_activations
        return config

    def export_weights(self) -> _torch.Tensor:
        return _torch.cat(
            [self._rechannel.export_weights()]
            + [layer.export_weights() for layer in self._layers]
            + [self._head_rechannel.export_weights()]
        )

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        i = self._rechannel.import_weights(weights, i)
        for layer in self._layers:
            i = layer.import_weights(weights, i)
        return self._head_rechannel.import_weights(weights, i)

    def forward(
        self,
        x: _torch.Tensor,
        c: _torch.Tensor,
        head_input: _Optional[_torch.Tensor] = None,
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """
        :param x: (B,Dx,L) layer input
        :param c: (B,Dc,L) condition

        :return:
            (B,Dc,L-R+1) head input
            (B,Dc,L-R+1) layer output
        """
        out_length = x.shape[2] - (self.receptive_field - 1)
        x = self._rechannel(x)
        for layer in self._layers:
            x, head_term = layer(x, c, out_length)  # Ensures head_term sample length
            head_input = (
                head_term
                if head_input is None
                else head_input[:, :, -out_length:] + head_term
            )
        return self._head_rechannel(head_input), x

    @property
    def _dilations(self) -> _Sequence[int]:
        return self._config["dilations"]

    @property
    def _kernel_size(self) -> int:
        return self._layers[0].kernel_size


class _Head(_nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: str,
        num_layers: int,
        out_channels: int,
    ):
        super().__init__()

        def block(cx, cy):
            net = _nn.Sequential()
            net.add_module(_ACTIVATION_NAME, _get_activation(activation))
            net.add_module(_CONV_NAME, Conv1d(cx, cy, 1))
            return net

        assert num_layers > 0

        layers = _nn.Sequential()
        cin = in_channels
        for i in range(num_layers):
            layers.add_module(
                f"layer_{i}",
                block(cin, channels if i != num_layers - 1 else out_channels),
            )
            cin = channels
        self._layers = layers

        self._config = {
            "channels": channels,
            "activation": activation,
            "num_layers": num_layers,
            "out_channels": out_channels,
        }

    def export_config(self):
        return _deepcopy(self._config)

    def export_weights(self) -> _torch.Tensor:
        return _torch.cat([layer[1].export_weights() for layer in self._layers])

    def forward(self, *args, **kwargs):
        return self._layers(*args, **kwargs)

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        for layer in self._layers:
            i = layer[1].import_weights(weights, i)
        return i


class _WaveNet(_nn.Module):
    def __init__(
        self,
        layers_configs: _Sequence[_Dict],
        head_config: _Optional[_Dict] = None,
        head_scale: float = 1.0,
    ):
        super().__init__()

        self._layer_arrays = _nn.ModuleList(
            [_LayerArray(**lc) for lc in layers_configs]
        )
        self._head = None if head_config is None else _Head(**head_config)
        self._head_scale = head_scale

    @property
    def receptive_field(self) -> int:
        return 1 + sum([(layer.receptive_field - 1) for layer in self._layer_arrays])

    def export_config(self):
        return {
            "layers": [
                layer_array.export_config() for layer_array in self._layer_arrays
            ],
            "head": None if self._head is None else self._head.export_config(),
            "head_scale": self._head_scale,
        }

    def export_weights(self) -> _np.ndarray:
        """
        :return: 1D array
        """
        weights = _torch.cat([layer.export_weights() for layer in self._layer_arrays])
        if self._head is not None:
            weights = _torch.cat([weights, self._head.export_weights()])
        weights = _torch.cat([weights.cpu(), _torch.Tensor([self._head_scale])])
        return weights.detach().cpu().numpy()

    def import_weights(self, weights: _torch.Tensor):
        if self._head is not None:
            raise NotImplementedError("Head importing isn't implemented yet.")
        i = 0
        for layer in self._layer_arrays:
            i = layer.import_weights(weights, i)

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        :param x: (B,Cx,L)
        :return: (B,Cy,L-R)
        """
        y, head_input = x, None
        for layer in self._layer_arrays:
            head_input, y = layer(y, x, head_input=head_input)
        head_input = self._head_scale * head_input
        return head_input if self._head is None else self._head(head_input)


class WaveNet(_BaseNet, _ImportsWeights):
    def __init__(self, *args, sample_rate: _Optional[float] = None, **kwargs):
        super().__init__(sample_rate=sample_rate)
        self._net = _WaveNet(*args, **kwargs)

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    def import_weights(self, weights: _Sequence[float]):
        if not isinstance(weights, _torch.Tensor):
            weights = _torch.Tensor(weights)
        self._net.import_weights(weights)

    def _export_config(self):
        return self._net.export_config()

    def _export_weights(self) -> _np.ndarray:
        return self._net.export_weights()

    def _forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._net(x)
        assert y.shape[1] == 1
        return y[:, 0, :]
