# File: _from_nam.py
# Created Date: Tuesday May 27th 2025
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Initialize models from .nam files
"""

from typing import Optional as _Optional

import torch as _torch

from .base import BaseNet as _BaseNet
from .linear import Linear as _Linear
from .recurrent import LSTM as _LSTM
from .wavenet import WaveNet as _WaveNet


def _init_linear(config, sample_rate: _Optional[float]) -> _Linear:
    return _Linear(sample_rate=sample_rate, **config)


def _init_lstm(config, sample_rate: _Optional[float]) -> _LSTM:
    return _LSTM(sample_rate=sample_rate, **config)


def _init_wavenet(config, sample_rate: _Optional[float]) -> _WaveNet:
    return _WaveNet(
        layers_configs=config["layers"],
        head_config=config["head"],
        head_scale=config["head_scale"],
        sample_rate=sample_rate,
    )


def init_from_nam(config) -> _BaseNet:
    """
    Taking the contents of a .nam file, initialize a model

    E.g.
    >>> with open("model.nam", "r") as fp:
    ...     config = json.load(fp)
    ...     model = init_from_nam(config)
    """
    # NB: Some old .nam files don't have a sample_rate. Must .get()
    model = {"Linear": _init_linear, "WaveNet": _init_wavenet, "LSTM": _init_lstm}[
        config["architecture"]
    ](config=config["config"], sample_rate=config.get("sample_rate", None))
    model.import_weights(_torch.Tensor(config["weights"]))
    return model
