# File: linear.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Linear model
"""

import numpy as np
import torch
import torch.nn as nn

from .._version import __version__
from .base import BaseNet


class Linear(BaseNet):
    def __init__(self, receptive_field: int, *args, bias: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._net = nn.Conv1d(1, 1, receptive_field, bias=bias)

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.weight.shape[2]

    def export_cpp_header(self):
        raise NotImplementedError()

    @property
    def _bias(self) -> bool:
        return self._net.bias is not None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x[:, None])[:, 0]

    def _export_config(self):
        return {
            "receptive_field": self.receptive_field,
            "bias": self._bias,
        }

    def _export_weights(self) -> np.ndarray:
        params_list = [self._net.weight.flatten()]
        if self._bias:
            params_list.append(self._net.bias.flatten())
        params = torch.cat(params_list).detach().cpu().numpy()
        return params
