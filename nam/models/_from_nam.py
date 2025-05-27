# File: _from_nam.py
# Created Date: Tuesday May 27th 2025
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Initialize models from .nam files
"""

import torch as _torch

from .base import BaseNet as _BaseNet
from .wavenet import WaveNet as _WaveNet

def _init_wavenet(config) -> _WaveNet:
    return _WaveNet(layers_configs=config["layers"], head_config=config["head"], head_scale=config["head_scale"])


def init_from_nam(config) -> _BaseNet:
    """
    Taking the contents of a .nam file, initialize a model

    E.g.
    >>> with open("model.nam", "r") as fp:
    ...     config = json.load(fp)
    ...     model = init_from_nam(config)
    """
    model = {
        "WaveNet": _init_wavenet
    }[config["architecture"]](config["config"])
    model.import_weights(_torch.Tensor(config["weights"]))
    return model
