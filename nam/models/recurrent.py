# File: recurrent.py
# Created Date: Saturday July 2nd 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Recurrent models (LSTM)

TODO batch_first=False (I get it...)
"""

import abc as _abc
import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory
from typing import Optional as _Optional, Tuple as _Tuple

import numpy as _np
import torch as _torch
import torch.nn as _nn

from ._abc import ImportsWeights as _ImportsWeights
from .base import BaseNet as _BaseNet


class _L(_nn.LSTM):
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
_LSTMHiddenType = _torch.Tensor
_LSTMCellType = _torch.Tensor
_LSTMHiddenCellType = _Tuple[_LSTMHiddenType, _LSTMCellType]


# TODO get this somewhere more core-ish
class _ExportsWeights(_abc.ABC):
    @_abc.abstractmethod
    def export_weights(self) -> _np.ndarray:
        """
        :return: a 1D array of weights
        """
        pass


class _ImportsAndExportsWeights(_ImportsWeights, _ExportsWeights):
    pass


class _Linear(_nn.Linear, _ImportsAndExportsWeights):
    def export_weights(self):
        return _np.concatenate(
            [
                self.weight.data.detach().cpu().numpy().flatten(),
                self.bias.data.detach().cpu().numpy().flatten(),
            ]
        )

    def import_weights(self, weights: _np.ndarray):
        self.weight.data = (
            _torch.from_numpy(weights[: self.weight.numel()])
            .reshape(self.weight.shape)
            .to(self.weight.device)
        )
        # This will catch any mismatch in number of weights given vs expected:
        self.bias.data = (
            _torch.from_numpy(weights[self.weight.numel() :])
            .reshape(self.bias.shape)
            .to(self.bias.device)
        )


class LSTM(_BaseNet, _ImportsWeights):
    """
    ABC for recurrent architectures
    """

    def __init__(
        self,
        hidden_size,
        train_burn_in: _Optional[int] = None,
        train_truncate: _Optional[int] = None,
        input_size: int = 1,
        sample_rate: _Optional[float] = None,
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
        self._initial_cell = _nn.Parameter(
            _torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )
        self._initial_hidden = _nn.Parameter(
            _torch.zeros((lstm_kwargs.get("num_layers", 1), hidden_size))
        )
        self._get_initial_state_burn_in = 48_000

    @property
    def input_device(self) -> _torch.device:
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

    def import_weights(self, weights):
        def import_lstm_cell_weights(cell_index: int, i_weight: int) -> int:
            def assign(
                name: str, i_weight: int, given_weights: _Optional[_torch.Tensor] = None
            ) -> int:
                """
                Weights and biases of LSTM cell
                """
                x = getattr(self._core, name)
                assert isinstance(x, _torch.Tensor)

                weight_list = (
                    [w for w in weights[i_weight : i_weight + x.numel()]]
                    if given_weights is None
                    else given_weights.flatten().detach().cpu().numpy().tolist()
                )

                if len(weight_list) != x.numel():
                    raise ValueError(
                        f"Weight list containing {len(weight_list)} elements is the "
                        f"wrong numel for destination {name} with numel {x.numel()} "
                        f"and shape {x.shape}!"
                    )
                x.data = _torch.Tensor(weight_list).reshape(x.shape).to(x.device)
                return i_weight + x.numel()

            # Form the entire weight matrix then split assign to w_ih and w_hh
            # This is necessary because the concatenation is horizontal, but the arrays
            # are stored row-major.
            input_dim = (
                self._core.input_size if cell_index == 0 else self._core.hidden_size
            )
            hidden_dim = self._core.hidden_size
            w_shape = (4 * hidden_dim, input_dim + hidden_dim)
            nw = _np.prod(w_shape)
            w = _torch.Tensor([w for w in weights[i_weight : i_weight + nw]]).reshape(
                w_shape
            )
            w_ih = w[:, :input_dim]
            w_hh = w[:, input_dim:]
            i_weight = assign(f"weight_ih_l{cell_index}", i_weight, given_weights=w_ih)
            i_weight = assign(f"weight_hh_l{cell_index}", i_weight, given_weights=w_hh)

            # NOTE: "bias" vectors aren't associated with inputs.
            # Arbitrary choice: assign to ih and zero out the hh.
            i_weight = assign(f"bias_ih_l{cell_index}", i_weight)
            getattr(self._core, f"bias_hh_l{cell_index}").data.zero_()

            # hidden_state
            self._initial_hidden.data[cell_index] = _torch.Tensor(
                [w for w in weights[i_weight : i_weight + self._core.hidden_size]]
            ).to(self._initial_hidden.device)
            i_weight += self._core.hidden_size
            # cell_state
            self._initial_cell.data[cell_index] = _torch.Tensor(
                [w for w in weights[i_weight : i_weight + self._core.hidden_size]]
            ).to(self._initial_cell.device)
            i_weight += self._core.hidden_size
            return i_weight

        weight_index = 0
        for layer_index in range(self._core.num_layers):
            weight_index = import_lstm_cell_weights(layer_index, i_weight=weight_index)

        # This should handle any length mismatch:
        self._head.import_weights(_np.array([w for w in weights[weight_index:]]))

    def _apply_head(self, features: _torch.Tensor) -> _torch.Tensor:
        """
        :param features: (B,S,DH)
        :return: (B,S)
        """
        return self._head(features)[:, :, 0]

    def _forward(
        self, x: _torch.Tensor, initial_state: _Optional[_LSTMHiddenCellType] = None
    ) -> _torch.Tensor:
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
            return _torch.cat(outputs, dim=1), hidden_state  # assert batch_first

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
            output_features = _torch.cat(output_features_list, dim=1)
        return self._apply_head(output_features)

    def _export_cell_weights(
        self, i: int, hidden_state: _torch.Tensor, cell_state: _torch.Tensor
    ) -> _np.ndarray:
        """
        * weight matrix (xh -> ifco)
        * bias vector
        * Initial hidden state
        * Initial cell state
        """

        tensors = [
            # NOTE: storing the entire (input, hidden) -> ifgo matrix at once.
            _torch.cat(
                [
                    getattr(self._core, f"weight_ih_l{i}").data,
                    getattr(self._core, f"weight_hh_l{i}").data,
                ],
                dim=1,
            ),
            # NOTE: "bias" vectors aren't associated with inputs.
            # Sum together WLOG.
            getattr(self._core, f"bias_ih_l{i}").data
            + getattr(self._core, f"bias_hh_l{i}").data,
            hidden_state,
            cell_state,
        ]
        return _np.concatenate([z.detach().cpu().numpy().flatten() for z in tensors])

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
        return _np.concatenate(
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
            _torch.zeros((1, self._get_initial_state_burn_in, 1))
            if inputs is None
            else inputs
        ).to(self.input_device)
        _, (h, c) = self._core(inputs)
        return h, c

    def _init_head(self, hidden_size: int) -> _ImportsAndExportsWeights:
        return _Linear(hidden_size, 1)

    def _initial_state(self, n: _Optional[int]) -> _LSTMHiddenCellType:
        """
        Literally what the forward pass starts with.
        Default is zeroes; this should be better since it can be learned.
        """
        return (
            (self._initial_hidden, self._initial_cell)
            if n is None
            else (
                _torch.tile(self._initial_hidden[:, None], (1, n, 1)),
                _torch.tile(self._initial_cell[:, None], (1, n, 1)),
            )
        )
