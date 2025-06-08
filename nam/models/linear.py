# File: linear.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Linear model
"""

import numpy as _np
import torch as _torch
import torch.nn as _nn

from .._version import __version__
from ._abc import ImportsWeights as _ImportsWeights
from .base import BaseNet as _BaseNet


class Linear(_BaseNet, _ImportsWeights):
    def __init__(self, receptive_field: int, *args, bias: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._net = _nn.Conv1d(1, 1, receptive_field, bias=bias)

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.weight.shape[2]

    def import_weights(self, weights):
        self._net.weight.data = (
            _torch.Tensor([w for w in weights[: self._net.weight.numel()]])
            .reshape(self._net.weight.shape)
            .to(self._net.weight.device)
        )
        if self._bias:
            self._net.bias.data = (
                _torch.Tensor([w for w in weights[self._net.weight.numel() :]])
                .reshape(self._net.bias.shape)
                .to(self._net.bias.device)
            )

    @property
    def _bias(self) -> bool:
        return self._net.bias is not None

    def _forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return self._net(x[:, None])[:, 0]

    def _export_config(self):
        return {
            "receptive_field": self.receptive_field,
            "bias": self._bias,
        }

    def _export_weights(self) -> _np.ndarray:
        params_list = [self._net.weight.flatten()]
        if self._bias:
            params_list.append(self._net.bias.flatten())
        params = _torch.cat(params_list).detach().cpu().numpy()
        return params
