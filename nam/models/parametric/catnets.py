# File: catnets.py
# Created Date: Wednesday June 22nd 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
"Cat nets" -- parametric models where the parametric input is concatenated to the
input samples
"""

import abc
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .._base import ParametricBaseNet
from ..recurrent import LSTM
from ..wavenet import WaveNet
from .params import Param


class _ShapeType(Enum):
    CONV = "conv"  # (B,C,L)
    RNN = "rnn"  # (B,L,D)


class _CatMixin(ParametricBaseNet):
    """
    Parameteric nets that concatenate the params with the input at each time point
    Mix in with a non-parametric class like

    ```
    class CatLSTM(LSTM, _CatMixin):
        pass
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hacky, see .export()
        self._sidedoor_parametric_config = None

    @abc.abstractproperty
    def _shape_type(self) -> _ShapeType:
        pass

    @abc.abstractproperty
    def _single_class(self):
        """"
        The class for the non-parametric model that this is extending
        """
        # TODO verify that single class satisfies requirements
        # ._export_weights()
        # ._export_input_output()
        pass  # HACK

    def export(self, outdir: Path, parametric_config: Dict[str, Param]):
        """
        Interface for exporting.
        You should create at least a `config.json` containing the two fields:
        * "version" (str)
        * "architecture" (str)
        * "config": (dict w/ other necessary data like tensor shapes etc)

        :param outdir: Assumed to exist. Can be edited inside at will.
        """
        with self._use_parametric_config(parametric_config):
            return super().export(outdir)

    def export_cpp_header(self, filename: Path, parametric_config: Dict[str, Param]):
        with self._use_parametric_config(parametric_config):
            return super().export_cpp_header(filename)

    def _export_config(self):
        """
        Adds in the sidedoored parametric pieces

        :paramtric_config: the dict of parameter info (name, type, etc)
        """
        config = super()._export_config()
        if not isinstance(config, dict):
            raise TypeError(
                f"Parameteric models' base configs must be a dict; got {type(config)}"
            )
        parametric_key = "parametric"
        if parametric_key in config:
            raise ValueError(
                f'Already found parametric key "{parametric_key}" in base config dict.'
            )
        # Yucky sidedoor
        config[parametric_key] = {
            k: v.to_json() for k, v in self._sidedoor_parametric_config.items()
        }
        return config

    def _export_input_output_args(self) -> Tuple[torch.Tensor]:
        return (self._sidedoor_params_to_tensor(),)

    def _forward(self, params, x):
        """
        :param params: (N,D)
        :param x: (N,L1)

        :return: (N,L2)
        """
        sequence_length = x.shape[1]
        x_augmented = (
            torch.cat(
                [
                    x[..., None],
                    torch.tile(params[:, None, :], (1, sequence_length, 1)),
                ],
                dim=2,
            )
            if self._shape_type == _ShapeType.RNN
            else torch.cat(
                [x[:, None, :], torch.tile(params[..., None], (1, 1, sequence_length))],
                dim=1,
            )
        )
        return self._single_class._forward(self, x_augmented)

    def _sidedoor_params_to_tensor(self) -> torch.Tensor:
        param_names = sorted([k for k in self._sidedoor_parametric_config.keys()])
        params = torch.Tensor(
            [self._sidedoor_parametric_config[k].default_value for k in param_names]
        )
        return params

    @contextmanager
    def _use_parametric_config(self, c):
        """
        Sneaks in the parametric config while exporting
        """
        try:
            self._sidedoor_parametric_config = c
            yield None
        finally:
            self._sidedoor_parametric_config = None


class CatLSTM(_CatMixin, LSTM):
    @property
    def _shape_type(self) -> _ShapeType:
        return _ShapeType.RNN

    @property
    def _single_class(self):
        return LSTM

    def _append_default_params(self, x: torch.Tensor) -> torch.Tensor:
        """
        Requires sidedoor'd params

        :param x: (B,L)
        :return: (B,L,1+D)
        """
        assert x.ndim == 2
        params = self._sidedoor_params_to_tensor()
        sequence_length = x.shape[1]
        return torch.cat(
            [
                x[:, :, None],
                torch.tile(params[None, None, :], (1, sequence_length, 1)),
            ],
            dim=2,
        )

    def _get_initial_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self._append_default_params(torch.zeros((1, 48_000)))
        return super()._get_initial_state(inputs=inputs)


class CatWaveNet(_CatMixin, WaveNet):
    @property
    def _shape_type(self) -> _ShapeType:
        return _ShapeType.CONV

    @property
    def _single_class(self):
        return WaveNet
        