# File: wavenet.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
WaveNet implementation
https://arxiv.org/abs/1609.03499
"""

import abc as _abc
import logging as _logging
import random as _random
from contextlib import contextmanager as _contextmanager
from copy import deepcopy as _deepcopy
from datetime import datetime as _datetime
from typing import (
    Any as _Any,
    Dict as _Dict,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import numpy as _np
import torch as _torch
import torch.nn as _nn
from pydantic import BaseModel as _BaseModel

from .._abc import ImportsWeights as _ImportsWeights
from .._activations import (
    PairBlend as _PairBlend,
    PairMultiply as _PairMultiply,
    PairingActivation as _PairingActivation,
    export_activation_config as _export_activation_config,
    get_activation as _get_activation,
)
from ..base import BaseNet as _BaseNet
from .._constants import MODEL_VERSION as _EXPORT_VERSION
from ..metadata import Date as _Date
from .._names import ACTIVATION_NAME as _ACTIVATION_NAME, CONV_NAME as _CONV_NAME

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


class _FiLMParamsConfig(_BaseModel):
    """FiLM (Feature-wise Linear Modulation) params for one insertion point."""

    active: bool = False
    shift: bool = True
    groups: int = 1


def _film_params_from_dict(d: _Optional[_Dict]) -> _FiLMParamsConfig:
    """Parse FiLM params from config dict (e.g. from .nam or init)."""
    if d is None or d is False:
        return _FiLMParamsConfig(active=False)
    if isinstance(d, dict):
        return _FiLMParamsConfig(
            active=d.get("active", True),
            shift=d.get("shift", True),
            groups=d.get("groups", 1),
        )
    return _FiLMParamsConfig(active=False)


_SLIMMABLE_METHOD = "slice_channels_uniform"  # TODO other methods in the future


def _parse_slimmable_config(val: _Any) -> bool:
    """
    Parse slimmable config from layer config.
    Only {"method": "slice_channels_uniform", "kwargs": {...}} is supported.
    Raises NotImplementedError for any other value.
    """
    if val is None:
        return False
    if not isinstance(val, dict):
        raise NotImplementedError(
            f"Slimmable config must be a dict, got {type(val).__name__}"
        )
    if val.get("method") != _SLIMMABLE_METHOD:
        raise NotImplementedError(
            f"Slimmable config only supports method '{_SLIMMABLE_METHOD}', "
            f"got {val.get('method', 'missing')!r}"
        )
    if "kwargs" not in val:
        raise NotImplementedError("Slimmable config must include 'kwargs' key")
    if not isinstance(val["kwargs"], dict):
        raise NotImplementedError(
            f"Slimmable config 'kwargs' must be a dict, got {type(val['kwargs']).__name__}"
        )
    return True


def _validate_slimmable_config(
    layers_configs: _Sequence[_Dict],
    condition_dsp: _Optional[_Dict],
) -> None:
    """
    Validate that slimmable config does not use unsupported options.
    Raises NotImplementedError if any layer has slimmable and uses:
    - more than one layer array
    - condition_dsp
    - groups_input != 1 or groups_input_mixin != 1
    - head_1x1
    - FiLM
    - layer_1x1 with groups != 1
    """
    if any("slimmable" in lc for lc in layers_configs) and len(layers_configs) > 1:
        raise NotImplementedError(
            "Slimmable training with more than one layer array is not supported yet"
        )
    if condition_dsp is not None:
        raise NotImplementedError(
            "Slimmable training with condition_dsp is not supported"
        )
    for lc in layers_configs:
        if "slimmable" not in lc:
            continue
        if lc.get("groups_input", 1) != 1 or lc.get("groups_input_mixin", 1) != 1:
            raise NotImplementedError(
                "Slimmable training with groups > 1 is not supported"
            )
        head_1x1 = lc.get("head_1x1_config") or lc.get("head1x1") or {}
        if isinstance(head_1x1, dict) and head_1x1.get("active", False):
            raise NotImplementedError(
                "Slimmable training with head 1x1 is not supported"
            )
        layer_1x1 = lc.get("layer_1x1_config") or lc.get("layer1x1") or {}
        if isinstance(layer_1x1, dict) and layer_1x1.get("groups", 1) != 1:
            raise NotImplementedError(
                "Slimmable training with layer 1x1 groups != 1 is not supported"
            )
        film_params = lc.get("film_params") or {}
        film_keys = (
            "conv_pre_film",
            "conv_post_film",
            "input_mixin_pre_film",
            "input_mixin_post_film",
            "activation_pre_film",
            "activation_post_film",
            "layer1x1_post_film",
            "head1x1_post_film",
        )
        for key in film_keys:
            fp = film_params.get(key)
            if fp and _film_params_from_dict(fp).active:
                raise NotImplementedError(
                    "Slimmable training with FiLM is not supported"
                )


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


def _ratio_to_channels(ratio: float, max_size: int) -> int:
    """Convert ratio in [0, 1] to integer channel count, minimum 1."""
    return 1 + min(int(_np.floor((ratio * max_size))), max_size - 1)


class _Slimmable(_abc.ABC, _nn.Module):
    """
    Mixin for slimmable modules. Dictates at what channel size the layer operates.
    """

    def __init__(self):
        super().__init__()
        self._adjust_size: _Optional[float] = None

    def adjust_to(
        self,
        ratio: _Optional[float] = None,
        _recurse: bool = True,
    ):
        """
        Set the channel width. ratio in [0, 1] scales to channels; None means full width.
        """
        self._adjust_size = ratio
        if _recurse:
            for module in self.modules():
                if module is self:
                    continue
                if isinstance(module, _Slimmable):
                    module.adjust_to(ratio, _recurse=False)

    def adjust_to_random(self):
        ratio = _random.uniform(0.0, 1.0)
        self.adjust_to(ratio)

    @_contextmanager
    def context_adjust_to_random(self):
        self.adjust_to_random()
        try:
            yield
        finally:
            self.adjust_to(None)


class _SlimmableConv1dBase(_nn.Conv1d, _Slimmable):
    """Base for slimmable 1D conv layers. Subclasses implement _get_adjusted_weight_and_bias."""

    def export_weights(self) -> _torch.Tensor:
        tensors = []
        if self.weight is not None:
            tensors.append(self.weight.data.flatten())
        if self.bias is not None:
            tensors.append(self.bias.data.flatten())
        if len(tensors) == 0:
            return _torch.zeros((0,))
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

    def forward(self, input: _torch.Tensor) -> _torch.Tensor:
        w, b = (
            (self.weight, self.bias)
            if self._adjust_size is None
            else self._get_adjusted_weight_and_bias()
        )
        return _nn.functional.conv1d(
            input, w, b, self.stride, self.padding, self.dilation, self.groups
        )

    @_abc.abstractmethod
    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        """Get weight and bias tensors for the current adjust size."""
        pass


class _SlimmableRechannelIn(_SlimmableConv1dBase):
    """Rechannel into a layer array. First layer: slice out only. Later: slice in and out."""

    def __init__(self, *args, is_first: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_first = is_first

    def _max_adjust_size(self) -> int:
        return self.out_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        adj = _ratio_to_channels(self._adjust_size, self._max_adjust_size())
        if self._is_first:
            w = self.weight[:adj, :, :]
        else:
            w = self.weight[:adj, :adj, :]
        b = None if self.bias is None else self.bias[:adj]
        return w, b


class _SlimmableConvLayer(_SlimmableConv1dBase):
    """Layer conv: channels -> mid_channels. Gated: mid=2*ch, slice w[:2*adj,:adj,:]."""

    def __init__(self, *args, gated: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._gated = gated

    def _max_adjust_size(self) -> int:
        return self.in_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        adj = _ratio_to_channels(self._adjust_size, self._max_adjust_size())
        if self._gated:
            w = self.weight[: 2 * adj, :adj, :]
            b = None if self.bias is None else self.bias[: 2 * adj]
        else:
            w = self.weight[:adj, :adj, :]
            b = None if self.bias is None else self.bias[:adj]
        return w, b


class _SlimmableInputMixer(_SlimmableConv1dBase):
    """Input mixer: condition -> mid_channels. Slice output only."""

    def _max_adjust_size(self) -> int:
        return self.out_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        adj = _ratio_to_channels(self._adjust_size, self._max_adjust_size())
        w = self.weight[:adj, :, :]
        b = None if self.bias is None else self.bias[:adj]
        return w, b


class _Slimmable1x1(_SlimmableConv1dBase):
    """1x1 conv in residual path. Slice both in and out (must be equal for slimmable)."""

    def _max_adjust_size(self) -> int:
        if self.in_channels != self.out_channels:
            raise NotImplementedError(
                "Slimmable 1x1 conv with different input and output channels not implemented"
            )
        return self.in_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        adj = _ratio_to_channels(self._adjust_size, self._max_adjust_size())
        w = self.weight[:adj, :adj, :]
        b = None if self.bias is None else self.bias[:adj]
        return w, b


class _SlimmableHeadRechannel(_SlimmableConv1dBase):
    """Head rechannel: channels -> 1. Slice input channels only."""

    def _max_adjust_size(self) -> int:
        return self.in_channels

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        if self.out_channels != 1:
            raise NotImplementedError(
                "Slimmable head rechannel only implemented for 1 output channel"
            )
        adj = _ratio_to_channels(self._adjust_size, self._max_adjust_size())
        w = self.weight[:, :adj, :]
        b = self.bias
        return w, b


class _FiLM(_nn.Module, _ImportsWeights):
    """
    FiLM (Feature-wise Linear Modulation) module.

    Given input (B, input_dim, L) and condition (B, condition_dim, L), computes
    scale (and optionally shift) from condition via 1x1 conv, then
    output = input * scale + shift (or output = input * scale when shift=False).
    """

    def __init__(
        self,
        condition_size: int,
        input_dim: int,
        shift: bool = True,
        groups: int = 1,
    ):
        super().__init__()
        self._shift = shift
        out_channels = (2 if shift else 1) * input_dim
        self._film = Conv1d(condition_size, out_channels, 1, bias=True, groups=groups)

        # Initialize to identity:
        # self._film.weight.data.zero_()  # Independent of condition input
        # if self._film.bias is not None:
        #     self._film.bias.data.zero_()  # (Scales &) shifts to zero
        #     self._film.bias.data[:input_dim] = 1.0  # Scales back to 1

    def forward(self, x: _torch.Tensor, c: _torch.Tensor) -> _torch.Tensor:
        """
        :param x: (B, input_dim, L)
        :param c: (B, condition_size, L) condition
        :return: (B, input_dim, L)
        """
        film_out = self._film(c)
        if self._shift:
            scale, shift = film_out.chunk(2, dim=1)
            return scale * x + shift
        else:
            return film_out * x

    def export_weights(self) -> _torch.Tensor:
        return self._film.export_weights()

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        return self._film.import_weights(weights, i)


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
        film_params: _Optional[_Dict[str, _Any]] = None,
        groups_input: int = 1,
        groups_input_mixin: int = 1,
        slimmable: bool = False,
    ):
        super().__init__()
        film_params = dict() if film_params is None else film_params
        if slimmable:
            if not layer_1x1_config.active or bottleneck != channels:
                raise ValueError(
                    "Slimmable layers require layer1x1 active and bottleneck==channels"
                )
        # Input mixer takes care of the bias
        mid_channels = (
            2 * bottleneck if isinstance(activation, _PairingActivation) else bottleneck
        )
        gated = isinstance(activation, _PairingActivation)
        if slimmable:
            self._conv = _SlimmableConvLayer(
                channels,
                mid_channels,
                kernel_size,
                dilation=dilation,
                gated=gated,
            )
            self._input_mixer = _SlimmableInputMixer(
                condition_size, mid_channels, 1, bias=False
            )
        else:
            self._conv = Conv1d(
                channels,
                mid_channels,
                kernel_size,
                dilation=dilation,
                groups=groups_input,
            )
            self._input_mixer = Conv1d(
                condition_size, mid_channels, 1, bias=False, groups=groups_input_mixin
            )

        self._activation = activation
        self._bottleneck = bottleneck

        self._activation_name = activation
        if layer_1x1_config.active:
            if slimmable:
                self._layer1x1 = _Slimmable1x1(bottleneck, channels, 1)
            else:
                self._layer1x1 = Conv1d(
                    bottleneck, channels, 1, groups=layer_1x1_config.groups
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

        # FiLM modules (optional at each position)
        def _maybe_film(
            fp: _FiLMParamsConfig, cond_dim: int, in_dim: int
        ) -> _Optional[_FiLM]:
            if not fp.active:
                return None
            return _FiLM(
                condition_size=cond_dim,
                input_dim=in_dim,
                shift=fp.shift,
                groups=fp.groups,
            )

        fp_conv_pre = _film_params_from_dict(film_params.get("conv_pre_film"))
        fp_conv_post = _film_params_from_dict(film_params.get("conv_post_film"))
        fp_im_pre = _film_params_from_dict(film_params.get("input_mixin_pre_film"))
        fp_im_post = _film_params_from_dict(film_params.get("input_mixin_post_film"))
        fp_act_pre = _film_params_from_dict(film_params.get("activation_pre_film"))
        fp_act_post = _film_params_from_dict(film_params.get("activation_post_film"))
        fp_l1x1_post = _film_params_from_dict(film_params.get("layer1x1_post_film"))
        fp_h1x1_post = _film_params_from_dict(film_params.get("head1x1_post_film"))

        if fp_l1x1_post.active and not layer_1x1_config.active:
            raise ValueError(
                "layer1x1_post_film cannot be active when layer1x1 is not active"
            )
        if fp_h1x1_post.active and not head_1x1_config.active:
            raise ValueError(
                "head1x1_post_film cannot be active when head1x1 is not active"
            )

        self._conv_pre_film = _maybe_film(fp_conv_pre, condition_size, channels)
        self._conv_post_film = _maybe_film(fp_conv_post, condition_size, mid_channels)
        self._input_mixin_pre_film = _maybe_film(
            fp_im_pre, condition_size, condition_size
        )
        self._input_mixin_post_film = _maybe_film(
            fp_im_post, condition_size, mid_channels
        )
        self._activation_pre_film = _maybe_film(
            fp_act_pre, condition_size, mid_channels
        )
        self._activation_post_film = _maybe_film(
            fp_act_post, condition_size, bottleneck
        )
        self._layer1x1_post_film = (
            _maybe_film(fp_l1x1_post, condition_size, channels)
            if layer_1x1_config.active
            else None
        )
        self._head1x1_post_film = (
            _maybe_film(
                fp_h1x1_post,
                condition_size,
                head_1x1_config.out_channels if head_1x1_config.active else bottleneck,
            )
            if head_1x1_config.active
            else None
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
        for film in (
            self._conv_pre_film,
            self._conv_post_film,
            self._input_mixin_pre_film,
            self._input_mixin_post_film,
            self._activation_pre_film,
            self._activation_post_film,
            self._layer1x1_post_film,
            self._head1x1_post_film,
        ):
            if film is not None:
                tensors.append(film.export_weights())
        return _torch.cat(tensors)

    def forward(
        self, x: _torch.Tensor, h: _torch.Tensor, out_length: int
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

        # Helper: slice condition to match tensor time length (conv shortens sequence)
        def _c(t_len: int, tensor: _torch.Tensor = h) -> _torch.Tensor:
            return tensor[:, :, -t_len:]

        # Step 1: input convolution (with optional pre/post FiLM)
        conv_input = x
        if self._conv_pre_film is not None:
            pre_conv_length = min(conv_input.shape[2], h.shape[2])
            conv_input = self._conv_pre_film(
                _c(pre_conv_length, tensor=conv_input), _c(pre_conv_length)
            )
        zconv = self.conv(conv_input)
        if self._conv_post_film is not None:
            post_conv_length = min(zconv.shape[2], h.shape[2])
            zconv = self._conv_post_film(
                _c(post_conv_length, tensor=zconv), _c(post_conv_length)
            )

        # Input mixin (with optional pre/post FiLM)
        mixin_input = h
        if self._input_mixin_pre_film is not None:
            mixin_input = self._input_mixin_pre_film(mixin_input, h)
        mix_out = self._input_mixer(mixin_input)[:, :, -zconv.shape[2] :]
        if self._input_mixin_post_film is not None:
            mix_out = self._input_mixin_post_film(mix_out, _c(mix_out.shape[2]))

        z1len = min(zconv.shape[2], mix_out.shape[2])
        z1 = zconv[:, :, -z1len:] + mix_out[:, :, -z1len:]
        if self._activation_pre_film is not None:
            z1 = self._activation_pre_film(z1, _c(z1.shape[2]))

        post_activation = self._activation(z1)
        if self._activation_post_film is not None:
            post_activation = self._activation_post_film(
                post_activation, _c(post_activation.shape[2])
            )

        layer_output = post_activation
        if self._layer1x1 is not None:
            layer_output = self._layer1x1(layer_output)
            if self._layer1x1_post_film is not None:
                layer_output = self._layer1x1_post_film(
                    layer_output, _c(layer_output.shape[2])
                )

        head_output = post_activation
        if self._head1x1 is not None:
            head_output = self._head1x1(head_output)[:, :, -out_length:]
            if self._head1x1_post_film is not None:
                head_output = self._head1x1_post_film(
                    head_output, _c(head_output.shape[2])
                )
        else:
            head_output = head_output[:, :, -out_length:]

        residual = x[:, :, -layer_output.shape[2] :] + layer_output
        return (residual, head_output)

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        i = self.conv.import_weights(weights, i)
        i = self._input_mixer.import_weights(weights, i)
        if self._layer1x1 is not None:
            i = self._layer1x1.import_weights(weights, i)
        if self._head1x1 is not None:
            i = self._head1x1.import_weights(weights, i)
        for film in (
            self._conv_pre_film,
            self._conv_post_film,
            self._input_mixin_pre_film,
            self._input_mixin_post_film,
            self._activation_pre_film,
            self._activation_post_film,
            self._layer1x1_post_film,
            self._head1x1_post_film,
        ):
            if film is not None:
                i = film.import_weights(weights, i)
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
        film_params: _Optional[_Dict[str, _Any]] = None,
        groups_input: int = 1,
        groups_input_mixin: int = 1,
        slimmable: bool = False,
    ):
        super().__init__()
        head_1x1_config = dict() if head_1x1_config is None else head_1x1_config
        head1x1_config_pydantic = _Head1x1Config(**head_1x1_config)
        layer_1x1_config = dict() if layer_1x1_config is None else layer_1x1_config
        layer1x1_config_pydantic = _Layer1x1Config(**layer_1x1_config)
        film_params = dict() if film_params is None else film_params

        bottleneck = channels if bottleneck is None else bottleneck

        if slimmable:
            self._rechannel = _SlimmableRechannelIn(
                input_size, channels, 1, bias=False, is_first=(input_size == 1)
            )
        else:
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
                    film_params=film_params,
                    groups_input=groups_input,
                    groups_input_mixin=groups_input_mixin,
                    slimmable=slimmable,
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
        if slimmable:
            self._head_rechannel = _SlimmableHeadRechannel(
                head_rechannel_in, head_size, 1, bias=head_bias
            )
        else:
            self._head_rechannel = Conv1d(
                head_rechannel_in, head_size, 1, bias=head_bias
            )

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
            "film_params": film_params,
            "groups_input": groups_input,
            "groups_input_mixin": groups_input_mixin,
            "slimmable": slimmable,
        }

    @property
    def receptive_field(self) -> int:
        return 1 + (self._kernel_size - 1) * sum(self._dilations)

    def export_config(self):
        config = _deepcopy(self._config)
        # Drop internal keys; export flat FiLM keys for C++/loadmodel
        config.pop("film_params", None)
        slimmable = config.pop("slimmable", None)
        film_params = self._config.get("film_params", {}) or {}
        for key in film_params:
            fp = _film_params_from_dict(film_params[key])
            config[key] = (
                {"active": True, "shift": fp.shift, "groups": fp.groups}
                if fp.active
                else False
            )
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
        if slimmable:
            config["slimmable"] = {"method": _SLIMMABLE_METHOD, "kwargs": {}}
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
        out_length = min(x.shape[2], c.shape[2]) - (self.receptive_field - 1)
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


class _WaveNet(_Slimmable, _nn.Module):
    def __init__(
        self,
        layers_configs: _Sequence[_Dict],
        head_config: _Optional[_Dict] = None,
        head_scale: float = 1.0,
        condition_dsp: _Optional[_Dict] = None,
    ):
        super().__init__()

        if condition_dsp is not None:
            if condition_dsp.get("name") != "WaveNet":
                raise NotImplementedError("Only WaveNet condition DSP is supported")
            self._condition_dsp = _WaveNet(**condition_dsp.get("config", dict()))
        else:
            self._condition_dsp = None

        self._slimmable = any("slimmable" in lc for lc in layers_configs)
        if self._slimmable:
            _validate_slimmable_config(layers_configs, condition_dsp)

        processed_configs = []
        for lc in layers_configs:
            lc_copy = _deepcopy(lc)
            slimmable_val = lc_copy.pop("slimmable", None)
            slimmable = _parse_slimmable_config(slimmable_val)
            lc_copy["slimmable"] = slimmable
            processed_configs.append(lc_copy)

        self._layer_arrays = _nn.ModuleList(
            [_LayerArray(**lc) for lc in processed_configs]
        )
        self._head = None if head_config is None else _Head(**head_config)
        self._head_scale = head_scale

    @property
    def receptive_field(self) -> int:
        receptive_field = 1 + sum(
            [(layer.receptive_field - 1) for layer in self._layer_arrays]
        )
        if self._condition_dsp is not None:
            receptive_field += self._condition_dsp.receptive_field - 1
        return receptive_field

    def export_config(self, sample_rate: _Optional[float] = None):
        config = {
            "layers": [
                layer_array.export_config() for layer_array in self._layer_arrays
            ],
            "head": None if self._head is None else self._head.export_config(),
            "head_scale": self._head_scale,
        }
        if self._condition_dsp is not None:
            # Build condition_dsp export dict without running forward (condition_dsp
            # may have multiple output channels; WaveNet wrapper asserts 1 channel).
            assert isinstance(
                self._condition_dsp, _WaveNet
            ), "The following assumes that the condition DSP is a _WaveNet"
            t = _datetime.now()
            condition_dsp_dict = {
                "version": _EXPORT_VERSION,
                "metadata": {
                    "date": _Date(
                        year=t.year,
                        month=t.month,
                        day=t.day,
                        hour=t.hour,
                        minute=t.minute,
                        second=t.second,
                    ).model_dump()
                },
                "architecture": "WaveNet",
                "config": self._condition_dsp.export_config(),
                "weights": self._condition_dsp.export_weights().tolist(),
            }
            # C++ loadmodel requires condition_dsp sample_rate to match main model
            if sample_rate is not None:
                condition_dsp_dict["sample_rate"] = sample_rate
            config["condition_dsp"] = condition_dsp_dict
        return config

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
        c = x if self._condition_dsp is None else self._condition_dsp(x)
        y, head_input = x, None
        for layer_array in self._layer_arrays:
            head_input, y = layer_array(y, c, head_input=head_input)
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
        return self._net.export_config(sample_rate=self.sample_rate)

    def _export_weights(self) -> _np.ndarray:
        return self._net.export_weights()

    def _forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        if self.training and self._net._slimmable:
            with self._net.context_adjust_to_random():
                y = self._net(x)
        else:
            y = self._net(x)
        assert y.shape[1] == 1
        return y[:, 0, :]
