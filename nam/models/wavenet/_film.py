from typing import Sequence as _Sequence

import torch as _torch
import torch.nn as _nn

from .._abc import ImportsWeights as _ImportsWeights
from ._conv import Conv1d as _FiLMConv


class FiLM(_nn.Module, _ImportsWeights):
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
        self._film = _FiLMConv(
            condition_size, out_channels, 1, bias=True, groups=groups
        )

        # Initialize to identity:
        # self._film.weight.data.zero_()  # Independent of condition input
        # if self._film.bias is not None:
        #     self._film.bias.data.zero_()  # (Scales &) shifts to zero
        #     self._film.bias.data[:input_dim] = 1.0  # Scales back to 1

    @property
    def shift(self) -> bool:
        return self._shift

    @property
    def groups(self) -> int:
        return self._film.groups

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

    def import_weights(self, weights: _Sequence[float], i: int) -> int:
        return self._film.import_weights(weights, i)
