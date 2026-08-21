"""
Parametric LSTM with PANAMA-style concatenative control conditioning.

The conditioning scheme follows PANAMA (arXiv 2509.26564v1): encode the control
vector, tile it across time, and concatenate it to the audio input at every
timestep before the recurrent core.
"""

from collections.abc import Sequence as _Sequence
from typing import Any as _Any
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import cast as _cast

import numpy as _np
import torch as _torch
import torch.nn as _nn

from ..recurrent import _L
from ..recurrent import _Linear
from ._base import ParametricNet as _ParametricNet
from ._spec import ParamSpec as _ParamSpec

_BLOCK_SIZE = 65_535
_LSTMHiddenType = _torch.Tensor
_LSTMCellType = _torch.Tensor
_LSTMHiddenCellType = _Tuple[_LSTMHiddenType, _LSTMCellType]
_WeightsLike = _Sequence[float] | _np.ndarray | _torch.Tensor


def _validate_positive_or_none(value: _Optional[int], name: str) -> _Optional[int]:
    """
    ``train_burn_in``/``train_truncate`` feed length-slicing and ``range(..., step)`` in
    the BPTT loop, where ``0`` produces an empty burn-in slice (``torch.cat`` of nothing)
    or a zero ``range`` step. Reject non-positive values up front so a bad hyperparameter
    fails at construction instead of on the first training step.
    """
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None; got {value!r}")
    return value


class ConcatLSTM(_ParametricNet):
    def __init__(
        self,
        *,
        param_specs: _Sequence[_ParamSpec],
        hidden_size: int,
        num_layers: int = 1,
        train_burn_in: _Optional[int] = None,
        train_truncate: _Optional[int] = None,
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(param_specs=param_specs, sample_rate=sample_rate)
        train_burn_in = _validate_positive_or_none(train_burn_in, "train_burn_in")
        train_truncate = _validate_positive_or_none(train_truncate, "train_truncate")
        input_size = 1 + self.encoded_param_dim
        self._core = _L(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self._head = _Linear(hidden_size, 1)
        self._train_burn_in = train_burn_in
        self._train_truncate = train_truncate
        self._initial_hidden = _nn.Parameter(
            _torch.zeros((num_layers, hidden_size))
        )
        self._initial_cell = _nn.Parameter(_torch.zeros((num_layers, hidden_size)))
        self._get_initial_state_burn_in = 48_000

    @classmethod
    def parse_config(cls, config: dict[str, _Any]) -> dict[str, _Any]:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        raw_params = config.pop("params", None)
        if raw_params is None:
            raise ValueError("ConcatLSTM config must define a params array")
        param_specs = tuple(_ParamSpec.from_dict(spec) for spec in raw_params)
        if len(param_specs) == 0:
            raise ValueError("ConcatLSTM config must define at least one ParamSpec")
        if "hidden_size" not in config:
            raise ValueError("ConcatLSTM config must define hidden_size")
        return {
            "param_specs": param_specs,
            "hidden_size": config.pop("hidden_size"),
            "num_layers": config.pop("num_layers", 1),
            "train_burn_in": config.pop("train_burn_in", None),
            "train_truncate": config.pop("train_truncate", None),
            "sample_rate": sample_rate,
        }

    @property
    def input_device(self) -> _torch.device:
        return _cast(_torch.Tensor, self._core.bias_ih_l0).device

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return 1

    def _run_conditioned(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        if p.ndim == 1:
            p = p[None].expand(x.shape[0], -1)
        elif p.shape[0] != x.shape[0]:
            raise ValueError(
                f"Input batch size {x.shape[0]} must match encoded params batch size {p.shape[0]}"
            )

        length = x.shape[1]
        p_t = p[:, None, :].expand(-1, length, -1)
        seq = _torch.cat([x[..., None], p_t], dim=-1)
        last_hidden_state = self._initial_state(len(x))
        if not self.training or self._train_truncate is None:
            output_features = self._process_in_blocks(seq, last_hidden_state)[0]
        else:
            output_features_list = []
            if self._train_burn_in is not None:
                last_output_features, last_hidden_state = self._process_in_blocks(
                    seq[:, : self._train_burn_in, :], last_hidden_state
                )
                output_features_list.append(last_output_features.detach())
            burn_in_offset = 0 if self._train_burn_in is None else self._train_burn_in
            for i in range(burn_in_offset, seq.shape[1], self._train_truncate):
                if i > burn_in_offset:
                    last_hidden_state = _cast(
                        _LSTMHiddenCellType,
                        tuple(z.detach() for z in last_hidden_state),
                    )
                last_output_features, last_hidden_state = self._process_in_blocks(
                    seq[:, i : i + self._train_truncate, :], last_hidden_state
                )
                output_features_list.append(last_output_features)
            output_features = _torch.cat(output_features_list, dim=1)
        return self._apply_head(output_features)

    def _process_in_blocks(
        self,
        x: _torch.Tensor,
        hidden_state: _Optional[_LSTMHiddenCellType] = None,
    ) -> _Tuple[_torch.Tensor, _LSTMHiddenCellType]:
        outputs = []
        for i in range(0, x.shape[1], _BLOCK_SIZE):
            out, hidden_state = self._core(x[:, i : i + _BLOCK_SIZE, :], hidden_state)
            outputs.append(out)
        if hidden_state is None:
            raise RuntimeError("LSTM hidden state was not initialized")
        return _torch.cat(outputs, dim=1), hidden_state

    def _apply_head(self, features: _torch.Tensor) -> _torch.Tensor:
        return self._head(features)[:, :, 0]

    def _export_inner_config(self) -> dict[str, _Any]:
        # input_size is intentionally omitted: it is derived from the param specs
        # (1 + encoded_param_dim) at construction, so it is not a free config field.
        return {
            "hidden_size": self._core.hidden_size,
            "num_layers": self._core.num_layers,
            "train_burn_in": self._train_burn_in,
            "train_truncate": self._train_truncate,
        }

    def _export_cell_weights(
        self, i: int, hidden_state: _torch.Tensor, cell_state: _torch.Tensor
    ) -> _np.ndarray:
        """
        * weight matrix (xh -> ifco)
        * bias vector
        * Initial hidden state
        * Initial cell state

        Mirrors ``nam.models.recurrent.LSTM._export_cell_weights``; the conditioning params
        are already folded into ``x`` (and so into ``weight_ih_l{i}``'s input dimension), so
        no extra handling is needed here.
        """
        tensors = [
            _torch.cat(
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
        return _np.concatenate([z.detach().cpu().numpy().flatten() for z in tensors])

    def _export_weights(self) -> _np.ndarray:
        """
        * Loop over cells: weight matrix (xh -> ifco), bias vector, initial hidden, initial cell
        * Head weights, head bias
        """
        return _np.concatenate(
            [
                self._export_cell_weights(i, h, c)
                for i, (h, c) in enumerate(zip(*self._get_export_initial_state()))
            ]
            + [self._head.export_weights()]
        )

    def _get_export_initial_state(self) -> _LSTMHiddenCellType:
        """
        Burn the core in on silence at the nominal control settings to find a good hidden
        state to start the exported model at. Mirrors
        ``nam.models.recurrent.LSTM._get_initial_state``, but the burn-in input must also
        carry the encoded nominal params tiled across time, since this core's input
        dimension includes the conditioning channels.
        """
        burn_in = self._get_initial_state_burn_in
        x = _torch.zeros((1, burn_in, 1), device=self.input_device)
        p = self._encode_params(self.nominal_params)
        p_t = p[None, None, :].expand(1, burn_in, -1)
        seq = _torch.cat([x, p_t], dim=-1)
        _, (h, c) = self._core(seq)
        return h, c

    def _initial_state(self, n: _Optional[int]) -> _LSTMHiddenCellType:
        return (
            (self._initial_hidden, self._initial_cell)
            if n is None
            else (
                _torch.tile(self._initial_hidden[:, None], (1, n, 1)),
                _torch.tile(self._initial_cell[:, None], (1, n, 1)),
            )
        )

    def import_weights(self, weights: _WeightsLike, i: int = 0) -> int:
        """
        Inverse of `_export_weights`. Offset-based (rather than consuming a whole flat
        array like `nam.models.recurrent.LSTM.import_weights`) to match the
        `ParametricNet` contract, which lets a caller pack multiple sub-models' weights
        into one blob.
        """
        weights_tensor = _cast(
            _torch.Tensor,
            weights if isinstance(weights, _torch.Tensor) else _torch.tensor(weights),
        ).to(dtype=_torch.float32)
        if weights_tensor.ndim != 1:
            raise ValueError(
                f"ConcatLSTM weights must be a flat 1-D sequence; got shape {tuple(weights_tensor.shape)}"
            )

        def assign(name: str, i: int) -> int:
            x = getattr(self._core, name)
            assert isinstance(x, _torch.Tensor)
            n = x.numel()
            chunk = weights_tensor[i : i + n]
            if chunk.numel() != n:
                raise ValueError(
                    f"Weight list containing {chunk.numel()} elements is the wrong "
                    f"numel for destination {name} with numel {n} and shape {tuple(x.shape)}!"
                )
            x.data = chunk.reshape(x.shape).to(dtype=x.dtype, device=x.device)
            return i + n

        hidden_size = self._core.hidden_size
        for layer in range(self._core.num_layers):
            input_dim = self._core.input_size if layer == 0 else hidden_size
            w_shape = (4 * hidden_size, input_dim + hidden_size)
            nw = int(_np.prod(w_shape))
            chunk = weights_tensor[i : i + nw]
            if chunk.numel() != nw:
                raise ValueError(
                    f"Weight list containing {chunk.numel()} elements is the wrong "
                    f"numel for the layer {layer} weight matrix with numel {nw}!"
                )
            w = chunk.reshape(w_shape)
            i += nw
            getattr(self._core, f"weight_ih_l{layer}").data = (
                w[:, :input_dim]
                .clone()
                .to(dtype=_torch.float32, device=self.input_device)
            )
            getattr(self._core, f"weight_hh_l{layer}").data = (
                w[:, input_dim:]
                .clone()
                .to(dtype=_torch.float32, device=self.input_device)
            )

            i = assign(f"bias_ih_l{layer}", i)
            getattr(self._core, f"bias_hh_l{layer}").data.zero_()

            self._initial_hidden.data[layer] = weights_tensor[i : i + hidden_size].to(
                device=self._initial_hidden.device
            )
            i += hidden_size
            self._initial_cell.data[layer] = weights_tensor[i : i + hidden_size].to(
                device=self._initial_cell.device
            )
            i += hidden_size

        head_size = self._head.weight.numel() + self._head.bias.numel()
        chunk = weights_tensor[i : i + head_size]
        if chunk.numel() != head_size:
            raise ValueError(
                f"Weight list containing {chunk.numel()} elements is the wrong numel "
                f"for the head with numel {head_size}!"
            )
        self._head.import_weights(chunk.detach().cpu().numpy())
        return i + head_size
