# File: recurrent.py
# Created Date: Saturday July 2nd 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Recurrent models (LSTM)

TODO batch_first=False (I get it...)
"""

import abc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BaseNet


class _L(nn.LSTM):
    """
    Tweaks to PyTorch LSTM module
    * Up the remembering
    """

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # https://danijar.com/tips-for-training-recurrent-neural-networks/
        # forget += 1
        # ifgo
        value = 2.0
        idx_input = slice(0, self.hidden_size)
        idx_forget = slice(self.hidden_size, 2 * self.hidden_size)
        for layer in range(self.num_layers):
            for input in ("i", "h"):
                # Balance out the scale of the cell w/ a -=1
                getattr(self, f"bias_{input}h_l{layer}").data[idx_input] -= value
                getattr(self, f"bias_{input}h_l{layer}").data[idx_forget] += value


# State:
# L: Number of LSTM layers
# DH: Hidden state dimension
# [0]: hidden (L,DH)
# [1]: cell (L,DH)
_LSTMHiddenType = torch.Tensor
_LSTMCellType = torch.Tensor
_LSTMHiddenCellType = Tuple[_LSTMHiddenType, _LSTMCellType]


# TODO get this somewhere more core-ish
class _ExportsWeights(abc.ABC):
    @abc.abstractmethod
    def export_weights(self) -> np.ndarray:
        """
        :return: a 1D array of weights
        """
        pass


class _Linear(nn.Linear, _ExportsWeights):
    def export_weights(self):
        return np.concatenate(
            [
                self.weight.data.detach().cpu().numpy().flatten(),
                self.bias.data.detach().cpu().numpy().flatten(),
            ]
        )


class LSTM(BaseNet):
    """
    ABC for recurrent architectures
    """

    def __init__(
        self,
        hidden_size,
        train_burn_in: Optional[int] = None,
        train_truncate: Optional[int] = None,
        input_size: int = 1,
        sample_rate: Optional[float] = None,
        **lstm_kwargs,
    ):
        """
        :param hidden_size: for LSTM
        :param train_burn_in: Detach calculations from first (this many) samples when
            training to burn in the hidden state.
        :param train_truncate: detach the hidden & cell states every this many steps
            during training so that backpropagation through time is faster + to simulate
            better starting states for h(t0)&c(t0) (instead of zeros)
            TODO recognition head to start the hidden state in a good place?
        :param input_size: Usually 1 (mono input). A catnet extending this might change
            it and provide the parametric inputs as additional input dimensions.
        """
        super().__init__(sample_rate=sample_rate)
        if "batch_first" in lstm_kwargs:
            raise ValueError("batch_first cannot be set.")
        self._input_size = input_size
        self._core = _L(self._input_size, hidden_size, batch_first=True, **lstm_kwargs)
        self._head = self._init_head(hidden_size)
        self._train_burn_in = train_burn_in
        self._train_truncate = train_truncate
        self._initial_cell = nn.Parameter(
            torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )
        self._initial_hidden = nn.Parameter(
            torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )
        self._get_initial_state_burn_in = 48_000

    @property
    def input_device(self) -> torch.device:
        """
        What device does the input need to be on?
        """
        return self._core.bias_ih_l0.device

    @property
    def receptive_field(self) -> int:
        return 1

    @property
    def pad_start_default(self) -> bool:
        # I should simplify this...
        return True

    def export_cpp_header(self, filename: Path):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            LSTM.export(self, Path(tmpdir))  # Hacky...need to work w/ CatLSTM
            with open(Path(tmpdir, "model.nam"), "r") as fp:
                _c = json.load(fp)
            version = _c["version"]
            config = _c["config"]
            s_parametric = self._export_cpp_header_parametric(config.get("parametric"))
            with open(filename, "w") as f:
                f.writelines(
                    (
                        "#pragma once\n",
                        "// Automatically-generated model file\n",
                        "#include <vector>\n",
                        '#include "json.hpp"\n',
                        '#include "lstm.h"\n',
                        f'#define PYTHON_MODEL_VERSION "{version}"\n',
                        f'const int NUM_LAYERS = {config["num_layers"]};\n',
                        f'const int INPUT_SIZE = {config["input_size"]};\n',
                        f'const int HIDDEN_SIZE = {config["hidden_size"]};\n',
                    )
                    + s_parametric
                    + (
                        "std::vector<float> PARAMS{"
                        + ", ".join([f"{w:.16f}f" for w in _c["weights"]])
                        + "};\n",
                    )
                )

    def _apply_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        :param features: (B,S,DH)
        :return: (B,S)
        """
        return self._head(features)[:, :, 0]

    def _forward(
        self, x: torch.Tensor, initial_state: Optional[_LSTMHiddenCellType] = None
    ) -> torch.Tensor:
        """
        :param x: (B,L) or (B,L,D)
        :return: (B,L)
        """

        def process_in_blocks(x, hidden_state=None):
            # See: https://github.com/sdatkinson/neural-amp-modeler/issues/450
            BLOCK_SIZE = 65_535
            outputs = []
            for i in range(0, x.shape[1], BLOCK_SIZE):
                out, hidden_state = self._core(
                    x[:, i : i + BLOCK_SIZE, :], hidden_state
                )
                outputs.append(out)
            return torch.cat(outputs, dim=1), hidden_state  # assert batch_first

        last_hidden_state = (
            self._initial_state(len(x)) if initial_state is None else initial_state
        )
        if x.ndim == 2:
            x = x[:, :, None]
        if not self.training or self._train_truncate is None:
            output_features = process_in_blocks(x, last_hidden_state)[0]
        else:
            output_features_list = []
            if self._train_burn_in is not None:
                last_output_features, last_hidden_state = process_in_blocks(
                    x[:, : self._train_burn_in, :], last_hidden_state
                )
                output_features_list.append(last_output_features.detach())
            burn_in_offset = 0 if self._train_burn_in is None else self._train_burn_in
            for i in range(burn_in_offset, x.shape[1], self._train_truncate):
                if i > burn_in_offset:
                    # Don't detach the burn-in state so that we can learn it.
                    last_hidden_state = tuple(z.detach() for z in last_hidden_state)
                last_output_features, last_hidden_state = process_in_blocks(
                    x[:, i : i + self._train_truncate, :], last_hidden_state
                )
                output_features_list.append(last_output_features)
            output_features = torch.cat(output_features_list, dim=1)
        return self._apply_head(output_features)

    def _export_cell_weights(
        self, i: int, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> np.ndarray:
        """
        * weight matrix (xh -> ifco)
        * bias vector
        * Initial hidden state
        * Initial cell state
        """

        tensors = [
            torch.cat(
                [
                    getattr(self._core, f"weight_ih_l{i}").data,
                    getattr(self._core, f"weight_hh_l{i}").data,
                ],
                dim=1,
            ),
            getattr(self._core, f"bias_ih_l{i}").data
            + getattr(self._core, f"bias_hh_l{i}").data,
            hidden_state,
            cell_state,
        ]
        return np.concatenate([z.detach().cpu().numpy().flatten() for z in tensors])

    def _export_config(self):
        return {
            "input_size": self._core.input_size,
            "hidden_size": self._core.hidden_size,
            "num_layers": self._core.num_layers,
        }

    def _export_cpp_header_parametric(self, config):
        # TODO refactor to merge w/ WaveNet implementation
        if config is not None:
            raise ValueError("Got non-None parametric config")
        return ("nlohmann::json PARAMETRIC {};\n",)

    def _export_weights(self):
        """
        * Loop over cells:
            * weight matrix (xh -> ifco)
            * bias vector
            * Initial hidden state
            * Initial cell state
        * Head weights
        * Head bias
        """
        return np.concatenate(
            [
                self._export_cell_weights(i, h, c)
                for i, (h, c) in enumerate(zip(*self._get_initial_state()))
            ]
            + [self._head.export_weights()]
        )

    def _get_initial_state(self, inputs=None) -> _LSTMHiddenCellType:
        """
        Convenience function to find a good hidden state to start the plugin at

        DX=input size
        L=num layers
        S=sequence length
        :param inputs: (1,S,DX)

        :return: (L,DH), (L,DH)
        """
        inputs = (
            torch.zeros((1, self._get_initial_state_burn_in, 1))
            if inputs is None
            else inputs
        ).to(self.input_device)
        _, (h, c) = self._core(inputs)
        return h, c

    def _init_head(self, hidden_size: int) -> _ExportsWeights:
        return _Linear(hidden_size, 1)

    def _initial_state(self, n: Optional[int]) -> _LSTMHiddenCellType:
        """
        Literally what the forward pass starts with.
        Default is zeroes; this should be better since it can be learned.
        """
        return (
            (self._initial_hidden, self._initial_cell)
            if n is None
            else (
                torch.tile(self._initial_hidden[:, None], (1, n, 1)),
                torch.tile(self._initial_cell[:, None], (1, n, 1)),
            )
        )
