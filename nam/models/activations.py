# File: activations.py
# Created Date: Wednesday May 25th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import torch
import torch.nn as nn

from .core import Functional


def _softsign(x):
    return x / (1.0 + torch.abs(x))


class Product(nn.Module):
    """
    Product of two nonlienarities
    For conv inputs of shape (B,C,L)
    """

    def __init__(self, f1, f2):
        super().__init__()
        self._f1 = f1
        self._f2 = f2

    def forward(self, x):
        c = x.shape[1]
        x1, x2 = x[:, : c // 2], x[:, c // 2 :]
        return self._f1(x1) * self._f2(x2)


class GatedTanh(Product):
    def __init__(self):
        super().__init__(torch.sigmoid, torch.tanh)


class GatedSoftsign(Product):
    def __init__(self):
        super().__init__(torch.sigmoid, _softsign)


class Sine(Functional):
    def __init__(self):
        super().__init__(torch.sin)


class Swish(Functional):
    def __init__(self):
        super().__init__(lambda x: x * torch.sigmoid(x))
