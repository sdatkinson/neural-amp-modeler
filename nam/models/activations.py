# File: activations.py
# Created Date: Wednesday May 25th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import torch

from .core import Functional


class Sine(Functional):
    def __init__(self):
        super().__init__(torch.sin)


class Softsign(Functional):
    def __init__(self):
        super().__init__(lambda x: x / (1.0 + torch.abs(x)))


class Swish(Functional):
    def __init__(self):
        super().__init__(lambda x: x * torch.sigmoid(x))
