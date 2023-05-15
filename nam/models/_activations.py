# File: _activations.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    return getattr(nn, name)()
