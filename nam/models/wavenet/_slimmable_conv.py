"""
A set of slimmable convolutional layers for use in WaveNet layers and layer arrays.

Implements the "channel slicing" method introduced in https://arxiv.org/abs/2511.07470
"""

import abc as _abc
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np
import torch as _torch
import torch.nn as _nn

from . import _conv
from ._slimmable import Slimmable as _Slimmable


def _ratio_to_channels(ratio: float, max_size: int) -> int:
    """Convert ratio in [0, 1] to integer channel count, minimum 1."""
    return 1 + min(int(_np.floor((ratio * max_size))), max_size - 1)


class _SlimmableConv1dBase(_conv.Conv1d, _Slimmable):
    """Base for slimmable 1D conv layers. Subclasses implement _get_adjusted_weight_and_bias."""

    def __init__(self, *args, groups: int = 1, **kwargs):
        if groups != 1:
            raise NotImplementedError(
                "Slimmable conv layers with groups != 1 are not implemented"
            )
        super().__init__(*args, **kwargs)

    def forward(self, input: _torch.Tensor) -> _torch.Tensor:
        w, b = (
            (self.weight, self.bias)
            if self._slimming_value is None
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


class _SlimmableRechannelIn(_SlimmableConv1dBase, _conv.RechannelIn):
    """Rechannel into a layer array. First layer: slice out only. Later: slice in and out."""

    def __init__(self, *args, is_first: bool = True, **kwargs):
        """
        :param is_first: Whether this is the first layer array in the WaveNet. If it is,
            then the input doesn't care about slimming, so you need to make sure that
            you always accept it at the same size.
        """
        super().__init__(*args, **kwargs)
        self._is_first = is_first

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        out_channels = _ratio_to_channels(self._slimming_value, self.out_channels)
        in_channels = (
            self.in_channels
            if self._is_first
            else _ratio_to_channels(self._slimming_value, self.in_channels)
        )
        w = self.weight[:out_channels, :in_channels, :]
        b = None if self.bias is None else self.bias[:out_channels]
        return w, b


class _SlimmableConvLayer(_SlimmableConv1dBase, _conv.LayerConv):
    """Layer conv: channels -> mid_channels. Gated: mid=2*ch, slice w[:2*adj,:adj,:]."""

    def __init__(self, *args, output_paired: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_paired = output_paired

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        max_channels = self.in_channels
        in_channels = _ratio_to_channels(self._slimming_value, max_channels)
        out_channels = 2 * in_channels if self._output_paired else in_channels
        w = self.weight[:out_channels, :in_channels, :]
        b = None if self.bias is None else self.bias[:out_channels]
        return w, b


class _SlimmableInputMixer(_SlimmableConv1dBase, _conv.InputMixer):
    """Input mixer: condition -> mid_channels. Slice output only."""

    def __init__(self, *args, output_paired: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_paired = output_paired

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        max_single_channels = (
            self.out_channels // 2 if self._output_paired else self.out_channels
        )
        out_single_channels = _ratio_to_channels(
            self._slimming_value, max_single_channels
        )
        out_channels = (
            2 * out_single_channels if self._output_paired else out_single_channels
        )
        w = self.weight[:out_channels, :, :]
        b = None if self.bias is None else self.bias[:out_channels]
        return w, b


class _SlimmableLayer1x1(_SlimmableConv1dBase):
    """1x1 conv in residual path. Slice both in and out (must be equal for slimmable)."""

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        def max_adjust_size() -> int:
            if self.in_channels != self.out_channels:
                raise NotImplementedError(
                    "Slimmable 1x1 conv with different input and output channels not implemented"
                )
            return self.in_channels

        adj = _ratio_to_channels(self._slimming_value, max_adjust_size())
        w = self.weight[:adj, :adj, :]
        b = None if self.bias is None else self.bias[:adj]
        return w, b


class _SlimmableHead1x1(_SlimmableConv1dBase):
    """
    1x1 conv to the head collector
    """

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        raise NotImplementedError("Slimmable head 1x1 not implemented")

        # Layer1x1 code...

        # def max_adjust_size() -> int:
        #     if self.in_channels != self.out_channels:
        #         raise NotImplementedError(
        #             "Slimmable 1x1 conv with different input and output channels not implemented"
        #         )
        #     return self.in_channels

        # adj = _ratio_to_channels(self._slimming_value, max_adjust_size())
        # w = self.weight[:adj, :adj, :]
        # b = None if self.bias is None else self.bias[:adj]
        # return w, b


class _SlimmableHeadRechannel(_SlimmableConv1dBase, _conv.HeadRechannel):
    """Head rechannel: channels -> 1. Slice input channels only."""

    def __init__(self, *args, is_last: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_last = is_last

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        in_channels = _ratio_to_channels(self._slimming_value, self.in_channels)
        out_channels = (
            self.out_channels
            if self._is_last
            else _ratio_to_channels(self._slimming_value, self.out_channels)
        )
        w = self.weight[:out_channels, :in_channels, :]
        b = None if self.bias is None else self.bias[:out_channels]
        return w, b


class_set = _conv.ClassSet(
    RechannelIn=_SlimmableRechannelIn,
    LayerConv=_SlimmableConvLayer,
    InputMixer=_SlimmableInputMixer,
    Layer1x1=_SlimmableLayer1x1,
    Head1x1=_SlimmableHead1x1,
    HeadRechannel=_SlimmableHeadRechannel,
)
