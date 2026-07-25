"""
Parametric WaveNet with PANAMA-style concatenative control conditioning.

The conditioning scheme matches :class:`ConcatLSTM`: encode the control vector, tile it
across time, and concatenate it to the audio input channels before an otherwise stock
inner WaveNet. Because the stock WaveNet feeds its raw input to every layer array as the
condition, the encoded params also reach each layer's input mixin, not just the first
rechannel -- the same conditioning the old upstream ``CatWaveNet`` had.

The inner WaveNet is completely ordinary; the only structural difference from a stock
model is that the first layer array's ``input_size`` (and every layer array's
``condition_size``) is ``1 + encoded_param_dim`` instead of 1. Those two fields are
derived from the param specs at config-parse time, so configs may omit them.
"""

from collections.abc import Sequence as _Sequence
from typing import Any as _Any
from typing import Optional as _Optional
from typing import cast as _cast

import numpy as _np
import torch as _torch

from .._from_nam import convert_nam_wavenet_config as _convert_nam_wavenet_config
from ..wavenet._wavenet import WaveNet as _WaveNet
from ._base import ParametricNet as _ParametricNet
from ._spec import ParamSpec as _ParamSpec

_WeightsLike = _Sequence[float] | _np.ndarray | _torch.Tensor


class ConcatWaveNet(_ParametricNet):
    def __init__(
        self,
        *,
        wavenet: _WaveNet,
        param_specs: _Sequence[_ParamSpec],
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(param_specs=param_specs, sample_rate=sample_rate)
        self._validate_supported_wavenet(wavenet)
        self._wavenet = wavenet

    @classmethod
    def parse_config(cls, config: dict[str, _Any]) -> dict[str, _Any]:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        raw_params = config.pop("params", None)
        if raw_params is None:
            raise ValueError("ConcatWaveNet config must define a params array")
        param_specs = tuple(_ParamSpec.from_dict(spec) for spec in raw_params)
        if len(param_specs) == 0:
            raise ValueError("ConcatWaveNet config must define at least one ParamSpec")
        if "condition_dsp" in config:
            # Same rationale as HyperWaveNet: rebuilding from config via
            # convert_nam_wavenet_config()->WaveNet.init_from_config() would silently drop
            # the nested condition_dsp weights. Beyond that, a condition DSP would also
            # have to ingest the concatenated param channels, which has no story yet.
            raise NotImplementedError(
                "ConcatWaveNet does not support condition_dsp"
            )

        layers_configs = config.get("layers")
        if not layers_configs:
            raise ValueError("ConcatWaveNet config must define at least one layer array")

        in_channels = 1 + sum(spec.num_inputs for spec in param_specs)
        derivation = (
            f"{in_channels} (1 audio channel + {in_channels - 1} encoded param channels)"
        )
        for i, layer_config in enumerate(layers_configs):
            if layer_config.get("packing"):
                raise NotImplementedError(
                    "ConcatWaveNet does not support packed layer arrays"
                )
            if i == 0:
                declared = layer_config.get("input_size")
                if declared is None:
                    layer_config["input_size"] = in_channels
                elif declared != in_channels:
                    raise ValueError(
                        "ConcatWaveNet derives the first layer array's input_size from "
                        f"the param specs: expected {derivation}; got {declared}. Omit "
                        "the field or set it to the derived value."
                    )
            declared = layer_config.get("condition_size")
            if declared is None:
                layer_config["condition_size"] = in_channels
            elif declared != in_channels:
                raise ValueError(
                    f"ConcatWaveNet derives layer array {i}'s condition_size from the "
                    f"param specs: expected {derivation}; got {declared}. Omit the "
                    "field or set it to the derived value."
                )

        wavenet_config = _convert_nam_wavenet_config(config, sample_rate=sample_rate)
        wavenet = _WaveNet.init_from_config(wavenet_config)
        return {
            "wavenet": wavenet,
            "param_specs": param_specs,
            "sample_rate": sample_rate,
        }

    @staticmethod
    def _validate_supported_wavenet(wavenet: _WaveNet) -> None:
        if wavenet._condition_dsp is not None:
            raise NotImplementedError(
                "ConcatWaveNet does not support inner WaveNets with condition_dsp"
            )
        if wavenet.is_slimmable():
            raise NotImplementedError(
                "ConcatWaveNet does not support slimmable inner WaveNets"
            )

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._wavenet.receptive_field

    @property
    def supports_compiled_step(self) -> bool:
        # The conditioning is a tile-and-concatenate, so the whole conditioned forward is
        # straight-line tensor work with nothing for dynamo to graph-break on.
        return True

    def _compilable_step(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        """Tile the encoded params across time, concatenate, run the inner WaveNet."""
        p_t = p[:, :, None].expand(-1, -1, x.shape[1])
        return self._wavenet(_torch.cat([x[:, None, :], p_t], dim=1))

    def _run_conditioned(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        if p.ndim == 1:
            p = p[None].expand(x.shape[0], -1)
        elif p.shape[0] != x.shape[0]:
            raise ValueError(
                f"Input batch size {x.shape[0]} must match encoded params batch size {p.shape[0]}"
            )
        y = self._run_step(x, p)
        if y.shape[1] != 1:
            raise RuntimeError(
                f"Expected inner WaveNet to return one channel; got shape {tuple(y.shape)}"
            )
        return y[:, 0, :]

    def _export_inner_config(self) -> dict[str, _Any]:
        return self._wavenet.export_config(sample_rate=self.sample_rate)

    def _export_weights(self) -> _np.ndarray:
        return self._wavenet.export_weights()

    def import_weights(self, weights: _WeightsLike, i: int = 0) -> int:
        weights_tensor = _cast(
            _torch.Tensor,
            weights if isinstance(weights, _torch.Tensor) else _torch.tensor(weights),
        )
        if weights_tensor.ndim != 1:
            raise ValueError(
                f"ConcatWaveNet weights must be a flat 1-D sequence; got shape {tuple(weights_tensor.shape)}"
            )
        return self._wavenet.import_weights(weights_tensor, i)
