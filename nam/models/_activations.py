# File: _activations.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import torch.nn as _nn


def get_activation(name: str) -> _nn.Module:
    return getattr(_nn, name)()
