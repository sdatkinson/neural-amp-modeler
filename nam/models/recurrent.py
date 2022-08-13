# File: recurrent.py
# Created Date: Saturday July 2nd 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Recurrent models (LSTM)

TODO batch_first=False (I get it...)
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ._base import BaseNet


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
        super().__init__()
        if "batch_first" in lstm_kwargs:
            raise ValueError("batch_first cannot be set.")
        self._input_size = input_size
        self._core = nn.LSTM(
            self._input_size, hidden_size, batch_first=True, **lstm_kwargs
        )
        self._head = nn.Linear(hidden_size, 1)
        self._train_burn_in = train_burn_in
        self._train_truncate = train_truncate
        self._initial_cell = nn.Parameter(
            torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )
        self._initial_hidden = nn.Parameter(
            torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )

    @property
    def receptive_field(self) -> int:
        return 1

    @property
    def pad_start_default(self) -> bool:
        # I should simplify this...
        return True

    def export_cpp_header(self, filename: Path):
        raise NotImplementedError()

    def _forward(self, x):
        """
        :param x: (B,L) or (B,L,D)
        :return: (B,L)
        """
        last_hidden_state = self._initial_state(len(x))
        if x.ndim==2:
            x = x[:, :, None]
        if not self.training or self._train_truncate is None:
            output_features = self._core(x, last_hidden_state)[0]
        else:
            output_features_list = []
            if self._train_burn_in is not None:
                last_output_features, last_hidden_state = self._core(
                    x[:, : self._train_burn_in, :], last_hidden_state
                )
                output_features_list.append(last_output_features.detach())
            burn_in_offset = 0 if self._train_burn_in is None else self._train_burn_in
            for i in range(burn_in_offset, x.shape[1], self._train_truncate):
                if i > burn_in_offset:
                    # Don't detach the burn-in state so that we can learn it.
                    last_hidden_state = tuple(z.detach() for z in last_hidden_state)
                last_output_features, last_hidden_state = self._core(
                    x[:, i : i + self._train_truncate, :,],
                    last_hidden_state,
                )
                output_features_list.append(last_output_features)
            output_features = torch.cat(output_features_list, dim=1)
        return self._head(output_features)[:, :, 0]

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
            + [
                self._head.weight.data.detach().cpu().numpy().flatten(),
                self._head.bias.data.detach().cpu().numpy().flatten(),
            ]
        )

    def _get_initial_state(self, inputs=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience function to find a good hidden state to start the plugin at

        DX=input size
        L=num layers
        S=sequence length
        :param inputs: (1,S,DX)

        :return: (L,DH), (L,DH)
        """
        inputs = torch.zeros((1, 48_000, 1)) if inputs is None else inputs
        _, (h, c) = self._core(inputs)
        return h, c

    def _initial_state(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tile(self._initial_hidden[:, None], (1, n, 1)),
            torch.tile(self._initial_cell[:, None], (1, n, 1))
        )