"""
A set of slimmable convolutional layers for use in WaveNet layers and layer arrays.

Implements the "channel slicing" method introduced in https://arxiv.org/abs/2511.07470
"""

import abc as _abc
from typing import Optional as _Optional
from typing import Sequence as _Sequence
from typing import Tuple as _Tuple

import numpy as _np
import torch as _torch
import torch.nn as _nn

from . import _conv
from ._slimmable import Slimmable as _Slimmable


def _ratio_to_channels(
    ratio: float, allowed_channels: _Sequence[int]
) -> _Tuple[int, int]:
    """
    Convert ratio in [0, 1] to integer channel count, minimum 1.
    Also return the index of the chosen channel entry.
    """
    i = min(int(_np.floor(ratio * len(allowed_channels))), len(allowed_channels) - 1)
    return allowed_channels[i], i


class _AllowedChannelsValueError(ValueError):
    """Error raised when allowed channels are invalid."""

    pass


def _init_smallest_and_zeros(
    module: _nn.Conv1d,
    allowed_in_channels: _Sequence[int],
    allowed_out_channels: _Sequence[int],
) -> None:
    """
    Initialize Conv1d so the slice seen at smallest size gets standard init,
    and the rest is zeroed. Uses allowed_channels[0] for in/out at smallest.
    """
    min_in = allowed_in_channels[0]
    min_out = allowed_out_channels[0]
    # Create a temp conv at smallest size; its reset_parameters gives standard init
    temp = _nn.Conv1d(
        min_in,
        min_out,
        module.kernel_size[0],
        stride=module.stride[0],
        padding=module.padding[0],
        dilation=module.dilation[0],
        groups=module.groups,
        bias=module.bias is not None,
    )
    temp.reset_parameters()
    # Copy initialized slice into full-weight tensor
    module.weight.data[:min_out, :min_in, :] = temp.weight.data
    # Zero the rest (extra output channels, extra input channels)
    if min_out < module.out_channels:
        module.weight.data[min_out:, :, :] = 0.0
    if min_in < module.in_channels:
        module.weight.data[:, min_in:, :] = 0.0
    if module.bias is not None:
        module.bias.data[:min_out] = temp.bias.data
        module.bias.data[min_out:] = 0.0


def _init_channel_causal(
    module: _nn.Conv1d,
    allowed_in_channels: _Sequence[int],
    allowed_out_channels: _Sequence[int],
) -> None:
    """
    Zero w[:c_out, c_in:, :] for each (c_in, c_out) so that smaller nets' outputs
    are not influenced by bigger nets' extra input channels. Supports differing
    allowed_in_channels and allowed_out_channels lengths by iterating through
    (c_in, c_out) pairs in order of the minimum ratio at which each would be
    used, alternating c_in and c_out increments according to which threshold
    is reached next.
    """
    n_in = len(allowed_in_channels)
    n_out = len(allowed_out_channels)
    # (ratio, 0=in, 1=out, index) so that 'in' is processed before 'out' on ties
    events: list[tuple[float, int, int]] = []
    for i in range(1, n_in):
        events.append((i / n_in, 0, i))
    for i in range(1, n_out):
        events.append((i / n_out, 1, i))
    events.sort(key=lambda e: (e[0], e[1]))
    i_in, i_out = 0, 0
    c_in = allowed_in_channels[i_in]
    c_out = allowed_out_channels[i_out]
    if c_out < module.out_channels and c_in < module.in_channels:
        module.weight.data[:c_out, c_in:, :] = 0.0
    for _ratio, _typ, idx in events:
        if _typ == 0:
            i_in = idx
        else:
            i_out = idx
        c_in = allowed_in_channels[i_in]
        c_out = allowed_out_channels[i_out]
        if c_out < module.out_channels and c_in < module.in_channels:
            module.weight.data[:c_out, c_in:, :] = 0.0


class SlimmableConv1dBase(_conv.Conv1d, _Slimmable):
    """Base for slimmable 1D conv layers. Subclasses implement _get_adjusted_weight_and_bias."""

    def __init__(
        self,
        *args,
        groups: int = 1,
        allowed_in_channels: _Optional[_Sequence[int]] = None,
        allowed_out_channels: _Optional[_Sequence[int]] = None,
        boosting: bool = False,
        init_strategy: _Optional[str] = None,
        **kwargs,
    ):
        if groups != 1:
            raise NotImplementedError(
                "Slimmable conv layers with groups != 1 are not implemented"
            )
        super().__init__(*args, groups=groups, **kwargs)
        self._allowed_in_channels = (
            allowed_in_channels
            if allowed_in_channels is not None
            else tuple(range(1, self.in_channels + 1))
        )
        self._allowed_out_channels = (
            allowed_out_channels
            if allowed_out_channels is not None
            else tuple(range(1, self.out_channels + 1))
        )
        self._boosting = boosting

        if init_strategy == "smallest_and_zeros":
            _init_smallest_and_zeros(
                self, self._allowed_in_channels, self._allowed_out_channels
            )
        elif init_strategy == "channel_causal":
            _init_channel_causal(
                self, self._allowed_in_channels, self._allowed_out_channels
            )

        def validate_allowed_channels(
            allowed_channels: _Sequence[int], max_channels: int
        ) -> None:
            if len(allowed_channels) == 0:
                raise _AllowedChannelsValueError("Allowed channels must be non-empty")
            if any(c > max_channels for c in allowed_channels):
                raise _AllowedChannelsValueError(
                    f"Allowed channels must be less than or equal to {max_channels}"
                )
            if any(c < 1 for c in allowed_channels):
                raise _AllowedChannelsValueError(
                    "Allowed channels must be greater than 0"
                )
            if len(allowed_channels) != len(set(allowed_channels)):
                raise _AllowedChannelsValueError("Allowed channels must be unique")
            if any(c % self.groups != 0 for c in allowed_channels):
                raise _AllowedChannelsValueError(
                    "Allowed channels must be divisible by groups"
                )

        validate_allowed_channels(self._allowed_in_channels, self.in_channels)
        validate_allowed_channels(self._allowed_out_channels, self.out_channels)

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


class _SlimmableRechannelIn(_conv.RechannelIn, SlimmableConv1dBase):
    """Rechannel into a layer array. First layer: slice out only. Later: slice in and out."""

    def __init__(
        self,
        in_channels: int,
        *args,
        allowed_in_channels: _Optional[_Sequence[int]] = None,
        is_first: bool = True,
        **kwargs,
    ):
        """
        :param is_first: Whether this is the first layer array in the WaveNet. If it is,
            then the input doesn't care about slimming, so you need to make sure that
            you always accept it at the same size.
        """
        if is_first:
            allowed_in_channels = (
                [in_channels] if allowed_in_channels is None else allowed_in_channels
            )
            if len(allowed_in_channels) != 1 or allowed_in_channels[0] != in_channels:
                raise ValueError(
                    "Input channels must be fixed for the first layer array's rechannel-in layer."
                )
        super().__init__(
            in_channels, *args, allowed_in_channels=allowed_in_channels, **kwargs
        )
        self._is_first = is_first

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        out_channels, i_out = _ratio_to_channels(
            self._slimming_value, self._allowed_out_channels
        )
        if self._is_first:
            in_channels = self.in_channels
            i_in = 0
        else:
            in_channels, i_in = _ratio_to_channels(
                self._slimming_value, self._allowed_in_channels
            )
        w_full = self.weight[:out_channels, :in_channels, :]
        b_full = None if self.bias is None else self.bias[:out_channels]
        if not self._boosting or (self._is_first or i_in == 0):
            return w_full, b_full
        in_channels_prev = self._allowed_in_channels[i_in - 1]
        out_channels_prev = self._allowed_out_channels[i_out - 1]
        w_prev = w_full[:out_channels_prev, :in_channels_prev, :]
        w_diff = _torch.zeros_like(w_full)
        w_diff[:out_channels_prev, :in_channels_prev, :] = (
            self.weight[:out_channels_prev, :in_channels_prev, :].detach() - w_prev
        )
        w = w_full + w_diff
        if self.bias is None:
            return w, None
        b_slice = self.bias[:out_channels]
        b_prev = b_slice[:out_channels_prev]
        b_diff = _torch.zeros_like(b_slice)
        b_diff[:out_channels_prev] = self.bias[:out_channels_prev].detach() - b_prev
        return w, b_slice + b_diff


class _SlimmableLayerConv(_conv.LayerConv, SlimmableConv1dBase):
    """Layer conv: channels -> mid_channels. Gated: mid=2*ch, slice w[:2*adj,:adj,:]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        output_paired: bool = False,
        allowed_in_channels: _Optional[_Sequence[int]] = None,
        allowed_out_channels: _Optional[_Sequence[int]] = None,
        **kwargs,
    ):
        if output_paired:
            allowed_in_channels = (
                tuple(range(1, in_channels + 1))
                if allowed_in_channels is None
                else allowed_in_channels
            )
            allowed_out_channels = (
                tuple(range(2, out_channels + 1, 2))
                if allowed_out_channels is None
                else allowed_out_channels
            )
            if any(c % 2 != 0 for c in allowed_out_channels):
                raise ValueError("Output channels must be even if output is paired")

        super().__init__(
            in_channels,
            out_channels,
            *args,
            allowed_in_channels=allowed_in_channels,
            allowed_out_channels=allowed_out_channels,
            **kwargs,
        )
        self._output_paired = output_paired

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        in_channels, i_in = _ratio_to_channels(
            self._slimming_value, self._allowed_in_channels
        )
        out_channels = 2 * in_channels if self._output_paired else in_channels
        w_full = self.weight[:out_channels, :in_channels, :]
        b_full = None if self.bias is None else self.bias[:out_channels]
        if not self._boosting or i_in == 0:
            return w_full, b_full
        in_channels_prev = self._allowed_in_channels[i_in - 1]
        out_channels_prev = (
            2 * in_channels_prev if self._output_paired else in_channels_prev
        )
        w_prev = w_full[:out_channels_prev, :in_channels_prev, :]
        w_diff = _torch.zeros_like(w_full)
        w_diff[:out_channels_prev, :in_channels_prev, :] = (
            self.weight[:out_channels_prev, :in_channels_prev, :].detach() - w_prev
        )
        w = w_full + w_diff
        if self.bias is None:
            return w, None
        b_slice = self.bias[:out_channels]
        b_prev = b_slice[:out_channels_prev]
        b_diff = _torch.zeros_like(b_slice)
        b_diff[:out_channels_prev] = self.bias[:out_channels_prev].detach() - b_prev
        return w, b_slice + b_diff


class _SlimmableInputMixer(_conv.InputMixer, SlimmableConv1dBase):
    """Input mixer: condition -> mid_channels. Slice output only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        allowed_in_channels: _Optional[_Sequence[int]] = None,
        allowed_out_channels: _Optional[_Sequence[int]] = None,
        output_paired: bool = False,
        **kwargs,
    ):
        allowed_in_channels = (
            [in_channels] if allowed_in_channels is None else allowed_in_channels
        )
        if len(allowed_in_channels) != 1 or allowed_in_channels[0] != in_channels:
            raise ValueError("Input channels must be 1 and equal to the input channels")
        if output_paired:
            allowed_out_channels = (
                tuple(range(2, out_channels + 1, 2))
                if allowed_out_channels is None
                else allowed_out_channels
            )
            if any(c % 2 != 0 for c in allowed_out_channels):
                raise ValueError("Output channels must be even if output is paired")

        super().__init__(
            in_channels,
            out_channels,
            *args,
            allowed_in_channels=allowed_in_channels,
            allowed_out_channels=allowed_out_channels,
            output_paired=output_paired,
            **kwargs,
        )
        self._output_paired = output_paired

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        out_channels, i_out = _ratio_to_channels(
            self._slimming_value, self._allowed_out_channels
        )
        w_full = self.weight[:out_channels, :, :]
        b_full = None if self.bias is None else self.bias[:out_channels]
        if not self._boosting or i_out == 0:
            return w_full, b_full
        out_channels_prev = self._allowed_out_channels[i_out - 1]
        w_prev = w_full[:out_channels_prev, :, :]
        w_diff = _torch.zeros_like(w_full)
        w_diff[:out_channels_prev, :, :] = (
            self.weight[:out_channels_prev, :, :].detach() - w_prev
        )
        w = w_full + w_diff
        if self.bias is None:
            return w, None
        b_slice = self.bias[:out_channels]
        b_prev = b_slice[:out_channels_prev]
        b_diff = _torch.zeros_like(b_slice)
        b_diff[:out_channels_prev] = self.bias[:out_channels_prev].detach() - b_prev
        return w, b_slice + b_diff


class _SlimmableLayer1x1(SlimmableConv1dBase):
    """1x1 conv in residual path. Slice both in and out (must be equal for slimmable)."""

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        in_channels, i_in = _ratio_to_channels(
            self._slimming_value, self._allowed_in_channels
        )
        out_channels, _ = _ratio_to_channels(
            self._slimming_value, self._allowed_out_channels
        )
        w_full = self.weight[:out_channels, :in_channels, :]
        b_full = None if self.bias is None else self.bias[:out_channels]
        if not self._boosting or i_in == 0:
            return w_full, b_full
        in_channels_prev = self._allowed_in_channels[i_in - 1]
        out_channels_prev = self._allowed_out_channels[i_in - 1]
        w_prev = w_full[:out_channels_prev, :in_channels_prev, :]
        w_diff = _torch.zeros_like(w_full)
        w_diff[:out_channels_prev, :in_channels_prev, :] = (
            self.weight[:out_channels_prev, :in_channels_prev, :].detach() - w_prev
        )
        w = w_full + w_diff
        if self.bias is None:
            return w, None
        b_slice = self.bias[:out_channels]
        b_prev = b_slice[:out_channels_prev]
        b_diff = _torch.zeros_like(b_slice)
        b_diff[:out_channels_prev] = self.bias[:out_channels_prev].detach() - b_prev
        return w, b_slice + b_diff


class _SlimmableHead1x1(SlimmableConv1dBase):
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


class _SlimmableHeadRechannel(_conv.HeadRechannel, SlimmableConv1dBase):
    """
    Head rechannel: output size si fixed on the last layer array."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        allowed_in_channels: _Optional[_Sequence[int]] = None,
        allowed_out_channels: _Optional[_Sequence[int]] = None,
        is_last: bool = False,
        **kwargs,
    ):
        if is_last:
            allowed_out_channels = (
                [out_channels] if allowed_out_channels is None else allowed_out_channels
            )
            if (
                len(allowed_out_channels) != 1
                or allowed_out_channels[0] != out_channels
            ):
                raise ValueError(
                    "Output channels must be fixed for the last layer array's head rechannel layer."
                )

        super().__init__(
            in_channels,
            out_channels,
            *args,
            allowed_in_channels=allowed_in_channels,
            allowed_out_channels=allowed_out_channels,
            **kwargs,
        )
        self._is_last = is_last

    def _get_adjusted_weight_and_bias(
        self,
    ) -> _Tuple[_torch.Tensor, _Optional[_torch.Tensor]]:
        in_channels, i_in = _ratio_to_channels(
            self._slimming_value, self._allowed_in_channels
        )
        if self._is_last:
            out_channels = self._allowed_out_channels[0]
            out_channels_prev = out_channels
            i_out = i_in
        else:
            out_channels, i_out = _ratio_to_channels(
                self._slimming_value, self._allowed_out_channels
            )
        w_full = self.weight[:out_channels, :in_channels, :]
        b_full = None if self.bias is None else self.bias[:out_channels]
        if not self._boosting or i_in == 0:
            return w_full, b_full
        in_channels_prev = self._allowed_in_channels[i_in - 1]
        out_channels_prev = (
            self._allowed_out_channels[0]
            if self._is_last
            else self._allowed_out_channels[i_out - 1]
        )
        w_prev = w_full[:out_channels_prev, :in_channels_prev, :]
        w_diff = _torch.zeros_like(w_full)
        w_diff[:out_channels_prev, :in_channels_prev, :] = (
            self.weight[:out_channels_prev, :in_channels_prev, :].detach() - w_prev
        )
        w = w_full + w_diff
        if self.bias is None:
            return w, None
        b_slice = self.bias[:out_channels]
        b_prev = b_slice[:out_channels_prev]
        b_diff = _torch.zeros_like(b_slice)
        b_diff[:out_channels_prev] = self.bias[:out_channels_prev].detach() - b_prev
        return w, b_slice + b_diff


class_set = _conv.ClassSet(
    RechannelIn=_SlimmableRechannelIn,
    LayerConv=_SlimmableLayerConv,
    InputMixer=_SlimmableInputMixer,
    Layer1x1=_SlimmableLayer1x1,
    Head1x1=_SlimmableHead1x1,
    HeadRechannel=_SlimmableHeadRechannel,
)
