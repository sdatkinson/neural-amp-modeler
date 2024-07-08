# File: _base.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
The foundation of the model without the PyTorch Lightning attributes (losses, training
steps)
"""

import abc
import math
import pkg_resources
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .._core import InitializableFromConfig
from ..data import wav_to_tensor
from .exportable import Exportable


class _Base(nn.Module, InitializableFromConfig, Exportable):
    def __init__(self, sample_rate: Optional[float] = None):
        super().__init__()
        self.register_buffer(
            "_has_sample_rate", torch.tensor(sample_rate is not None, dtype=torch.bool)
        )
        self.register_buffer(
            "_sample_rate", torch.tensor(0.0 if sample_rate is None else sample_rate)
        )

    @property
    @abc.abstractmethod
    def pad_start_default(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def receptive_field(self) -> int:
        """
        Receptive field of the model
        """
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass

    @classmethod
    def _metadata_loudness_x(cls) -> torch.Tensor:
        return wav_to_tensor(
            pkg_resources.resource_filename(
                "nam", "models/_resources/loudness_input.wav"
            )
        )

    @property
    def device(self) -> Optional[torch.device]:
        """
        Helpful property, where the parameters of the model live.
        """
        # We can do this because the models are tiny and I don't expect a NAM to be on
        # multiple devices
        try:
            return next(self.parameters()).device
        except StopIteration:
            return None

    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate.item() if self._has_sample_rate else None

    @sample_rate.setter
    def sample_rate(self, val: Optional[float]):
        self._has_sample_rate = torch.tensor(val is not None, dtype=torch.bool)
        self._sample_rate = torch.tensor(0.0 if val is None else val)

    def _get_export_dict(self):
        d = super()._get_export_dict()
        sample_rate_key = "sample_rate"
        if sample_rate_key in d:
            raise RuntimeError(
                "Model wants to put 'sample_rate' into model export dict, but the key "
                "is already taken!"
            )
        d[sample_rate_key] = self.sample_rate
        return d

    def _metadata_loudness(self, gain: float = 1.0, db: bool = True) -> float:
        """
        How loud is this model when given a standardized input?
        In dB

        :param gain: Multiplies input signal
        """
        x = self._metadata_loudness_x().to(self.device)
        y = self._at_nominal_settings(gain * x)
        loudness = torch.sqrt(torch.mean(torch.square(y)))
        if db:
            loudness = 20.0 * torch.log10(loudness)
        return loudness.item()

    def _metadata_gain(self) -> float:
        """
        Between 0 and 1, how much gain / compression does the model seem to have?
        """
        x = np.linspace(0.0, 1.0, 11)
        y = np.array([self._metadata_loudness(gain=gain, db=False) for gain in x])
        #
        # O ^ o o o o o o
        # u | o       x   +-------------------------------------+
        # t | o     x     | x: Minimum gain (no compression)    |
        # p | o   x       | o: Max gain     (100% compression)  |
        # u | o x         +-------------------------------------+
        # t | o
        #   +------------->
        #       Input
        #
        max_gain = y[-1] * len(x)  # "Square"
        min_gain = 0.5 * max_gain  # "Triangle"
        gain_range = max_gain - min_gain
        this_gain = y.sum()
        normalized_gain = (this_gain - min_gain) / gain_range
        return np.clip(normalized_gain, 0.0, 1.0)

    def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
        # parametric?...
        raise NotImplementedError()

    @abc.abstractmethod
    def _forward(self, *args) -> torch.Tensor:
        """
        The true forward method.

        :param x: (N,L1)
        :return: (N,L1-RF+1)
        """
        pass

    def _export_input_output_args(self) -> Tuple[Any]:
        """
        Create any other args necessesary (e.g. params to eval at)
        """
        return ()

    def _export_input_output(self) -> Tuple[np.ndarray, np.ndarray]:
        args = self._export_input_output_args()
        rate = self.sample_rate
        if rate is None:
            raise RuntimeError(
                "Cannot export model's input and output without a sample rate."
            )
        x = torch.cat(
            [
                torch.zeros((rate,)),
                0.5
                * torch.sin(
                    2.0 * math.pi * 220.0 * torch.linspace(0.0, 1.0, rate + 1)[:-1]
                ),
                torch.zeros((rate,)),
            ]
        )
        # Use pad start to ensure same length as requested by ._export_input_output()
        return (
            x.detach().cpu().numpy(),
            self(*args, x, pad_start=True).detach().cpu().numpy(),
        )


class BaseNet(_Base):
    def forward(self, x: torch.Tensor, pad_start: Optional[bool] = None, **kwargs):
        pad_start = self.pad_start_default if pad_start is None else pad_start
        scalar = x.ndim == 1
        if scalar:
            x = x[None]
        if pad_start:
            x = torch.cat(
                (torch.zeros((len(x), self.receptive_field - 1)).to(x.device), x), dim=1
            )
        y = self._forward(x, **kwargs)
        if scalar:
            y = y[0]
        return y

    def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    @abc.abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The true forward method.

        :param x: (N,L1)
        :return: (N,L1-RF+1)
        """
        pass

    def _get_non_user_metadata(self) -> Dict[str, Union[str, int, float]]:
        d = super()._get_non_user_metadata()
        d["loudness"] = self._metadata_loudness()
        d["gain"] = self._metadata_gain()
        return d
