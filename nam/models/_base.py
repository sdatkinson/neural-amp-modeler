# File: _base.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .._core import InitializableFromConfig
from ._exportable import Exportable


class BaseNet(nn.Module, InitializableFromConfig, Exportable):
    @abc.abstractproperty
    def pad_start_default(self) -> bool:
        pass

    @abc.abstractproperty
    def receptive_field(self) -> int:
        """
        Receptive field of the model
        """
        pass

    def forward(self, x: torch.Tensor, pad_start: bool = None):
        pad_start = self.pad_start_default if pad_start is None else pad_start
        scalar = x.ndim == 1
        if scalar:
            x = x[None]
        if pad_start:
            x = torch.cat((torch.zeros((len(x), self.receptive_field - 1)), x), dim=1)
        y = self._forward(x)
        if scalar:
            y = y[0]
        return y

    @abc.abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The true forward method.

        :param x: (N,L1)
        :return: (N,L1-RF+1) 
        """
        pass

    def _test_signal(
        self, seed=0, extra_length=13
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.Tensor(
            np.random.default_rng(seed).normal(
                size=(self.receptive_field + extra_length,)
            )
        )
        return x, self(x, pad_start=False)
