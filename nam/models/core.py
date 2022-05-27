# File: core.py
# Created Date: Wednesday May 25th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Core stuff
"""

import torch
import torch.nn as nn


class Functional(nn.Module):
    def __init__(self, f):
        super().__init__()
        self._f = f

    def forward(self, *args, **kwargs):
        return self._f(*args, **kwargs)
