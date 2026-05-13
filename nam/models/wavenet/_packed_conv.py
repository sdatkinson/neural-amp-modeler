"""
Masked convolution layers for packed WaveNet training.
"""

from typing import Sequence as _Sequence
from typing import Tuple as _Tuple

import torch as _torch
import torch.nn as _nn

from . import _conv

_Segment = _Tuple[int, int]


def segments_from_sizes(sizes: _Sequence[int]) -> _Tuple[_Segment, ...]:
    """Return ``(start, size)`` channel segments for contiguous channel blocks."""
    segments = []
    start = 0
    for size in sizes:
        if size < 1:
            raise ValueError("Packed channel sizes must be positive")
        segments.append((start, int(size)))
        start += int(size)
    return tuple(segments)


class PackedConv1dBase(_conv.Conv1d):
    """
    Conv1d with an explicit channel mask.

    ``shared_input=True`` means every packed output block can read the same input
    channels. Otherwise, output block ``i`` reads only input block ``i``.
    """

    def __init__(
        self,
        *args,
        input_segments: _Sequence[_Segment],
        output_segments: _Sequence[_Segment],
        shared_input: bool = False,
        groups: int = 1,
        **kwargs,
    ):
        if groups != 1:
            raise NotImplementedError("Packed convolutions require groups == 1")
        super().__init__(*args, groups=groups, **kwargs)
        self._input_segments = tuple((int(s), int(n)) for s, n in input_segments)
        self._output_segments = tuple((int(s), int(n)) for s, n in output_segments)
        self._shared_input = bool(shared_input)
        if not self._shared_input and len(self._input_segments) != len(
            self._output_segments
        ):
            raise ValueError(
                "Packed convolutions require matching input/output block counts"
            )
        if len(self._output_segments) == 0:
            raise ValueError("Packed convolutions require at least one output block")

        mask = _torch.zeros_like(self.weight)
        for i, (out_start, out_size) in enumerate(self._output_segments):
            in_start, in_size = self._input_segment_for(i)
            mask[out_start : out_start + out_size, in_start : in_start + in_size, :] = (
                1.0
            )
        self.register_buffer("_weight_mask", mask)
        self._reset_packed_parameters()
        self.apply_mask()

    @property
    def output_segments(self) -> _Tuple[_Segment, ...]:
        return self._output_segments

    @property
    def input_segments(self) -> _Tuple[_Segment, ...]:
        return self._input_segments

    @property
    def shared_input(self) -> bool:
        return self._shared_input

    def get_block_slices(self, submodel_index: int):
        out_start, out_size = self._output_segments[submodel_index]
        in_start, in_size = self._input_segment_for(submodel_index)
        return (
            slice(out_start, out_start + out_size),
            slice(in_start, in_start + in_size),
        )

    def forward(self, input: _torch.Tensor) -> _torch.Tensor:
        return _nn.functional.conv1d(
            input,
            self.weight * self._weight_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def import_weights(self, weights: _Sequence[float], i: int) -> int:
        i = super().import_weights(weights, i)
        self.apply_mask()
        return i

    def apply_mask(self) -> None:
        with _torch.no_grad():
            self.weight.mul_(self._weight_mask)

    def _assert_masked(self) -> None:
        invalid = self.weight * (1.0 - self._weight_mask)
        if not _torch.allclose(invalid, _torch.zeros_like(invalid)):
            raise AssertionError("Packed convolution has non-zero off-block weights")

    def _input_segment_for(self, submodel_index: int) -> _Segment:
        if self._shared_input:
            if len(self._input_segments) != 1:
                raise ValueError("Shared-input packed convs require one input segment")
            return self._input_segments[0]
        return self._input_segments[submodel_index]

    def _reset_packed_parameters(self) -> None:
        with _torch.no_grad():
            self.weight.zero_()
            if self.bias is not None:
                self.bias.zero_()
            for i, (out_start, out_size) in enumerate(self._output_segments):
                in_start, in_size = self._input_segment_for(i)
                temp = _nn.Conv1d(
                    in_size,
                    out_size,
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                    bias=self.bias is not None,
                    padding_mode=self.padding_mode,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                self.weight[
                    out_start : out_start + out_size,
                    in_start : in_start + in_size,
                    :,
                ].copy_(temp.weight)
                if self.bias is not None:
                    self.bias[out_start : out_start + out_size].copy_(temp.bias)


class PackedRechannelIn(_conv.RechannelIn, PackedConv1dBase):
    pass


class PackedLayerConv(_conv.LayerConv, PackedConv1dBase):
    pass


class PackedInputMixer(_conv.InputMixer, PackedConv1dBase):
    pass


class PackedLayer1x1(PackedConv1dBase):
    pass


class PackedHead1x1(PackedConv1dBase):
    pass


class PackedHeadRechannel(_conv.HeadRechannel, PackedConv1dBase):
    pass
