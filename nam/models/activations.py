# File: activations.py
# Created Date: Wednesday May 25th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from turtle import forward
import torch
import torch.nn as nn


class Functional(nn.Module):
    def __init__(self, f):
        super().__init__()
        self._f = f

    def forward(self, *args, **kwargs):
        return self._f(*args, **kwargs)


class Sine(Functional):
    def __init__(self):
        super().__init__(torch.sin)


class Swish(Functional):
    def __init__(self):
        super().__init__(lambda x: x * torch.sigmoid(x))
