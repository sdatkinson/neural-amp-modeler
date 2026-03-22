"""
The basic WaveNet neural network
"""

from copy import deepcopy as _deepcopy
from datetime import datetime as _datetime
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Sequence as _Sequence

import numpy as _np
import torch as _torch
import torch.nn as _nn

from ..._core import InitializableFromConfig as _InitializableFromConfig
from .._activations import get_activation as _get_activation
from .._constants import MODEL_VERSION as _EXPORT_VERSION
from .._names import ACTIVATION_NAME as _ACTIVATION_NAME
from .._names import CONV_NAME as _CONV_NAME
from ..metadata import Date as _Date
from ._conv import Conv1d as _Conv1d
from ._conv import apply_stable as _apply_stable
from ._layer_array import LayerArray as _LayerArray
from ._layer_array import film_params_from_dict as _film_params_from_dict
from ._slimmable import Slimmable as _Slimmable


class _Head(_nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        activation: str,
        num_layers: int,
        out_channels: int,
        stable: bool = False,
    ):
        super().__init__()

        def block(cx, cy):
            net = _nn.Sequential()
            net.add_module(_ACTIVATION_NAME, _get_activation(activation))
            conv = _Conv1d(cx, cy, 1)
            if stable:
                _apply_stable(conv)
            net.add_module(_CONV_NAME, conv)
            return net

        assert num_layers > 0

        layers = _nn.Sequential()
        cin = in_channels
        for i in range(num_layers):
            layers.add_module(
                f"layer_{i}",
                block(cin, channels if i != num_layers - 1 else out_channels),
            )
            cin = channels
        self._layers = layers

        self._config = {
            "channels": channels,
            "activation": activation,
            "num_layers": num_layers,
            "out_channels": out_channels,
        }

    def export_config(self):
        return _deepcopy(self._config)

    def export_weights(self) -> _torch.Tensor:
        return _torch.cat([layer[1].export_weights() for layer in self._layers])

    def forward(self, *args, **kwargs):
        return self._layers(*args, **kwargs)

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        for layer in self._layers:
            i = layer[1].import_weights(weights, i)
        return i


_SLIMMABLE_TOP_KEYS = frozenset({"method", "kwargs"})
_SLIMMABLE_KWARGS_KEYS = frozenset({"allowed_channels", "boosting", "init_strategy"})


def _validate_slimmable_dict(s: _Dict, layer_idx: int) -> None:
    """
    Validate slimmable config dict for unrecognized keys.
    Raises ValueError if any top-level or kwargs keys are unrecognized.
    """
    extra_top = set(s) - _SLIMMABLE_TOP_KEYS
    if extra_top:
        raise ValueError(
            f"Layer {layer_idx} slimmable config: unrecognized keys {sorted(extra_top)!r}. "
            f"Allowed: {sorted(_SLIMMABLE_TOP_KEYS)!r}"
        )
    kwargs = s.get("kwargs")
    if isinstance(kwargs, dict):
        extra_kwargs = set(kwargs) - _SLIMMABLE_KWARGS_KEYS
        if extra_kwargs:
            raise ValueError(
                f"Layer {layer_idx} slimmable kwargs: unrecognized keys "
                f"{sorted(extra_kwargs)!r}. Allowed: {sorted(_SLIMMABLE_KWARGS_KEYS)!r}"
            )


def _validate_slimmable_config(
    layers_configs: _Sequence[_Dict],
    condition_dsp: _Optional[_Dict],
) -> None:
    """
    Validate that slimmable config does not use unsupported options.
    Raises NotImplementedError if any layer has slimmable and uses:
    - condition_dsp
    - groups_input != 1 or groups_input_mixin != 1
    - head_1x1
    - FiLM
    - layer_1x1 with groups != 1
    Raises ValueError if any slimmable config or kwargs has unrecognized keys.
    """

    def _has_slimmable(lc):
        s = lc.get("slimmable")
        return s is not None and (not isinstance(s, dict) or s.get("method"))

    if not any(_has_slimmable(lc) for lc in layers_configs):
        return

    if len(layers_configs) > 1:
        raise NotImplementedError(
            "Slimmable training with more than one layer array may have bugs"
        )
    if condition_dsp is not None:
        raise NotImplementedError(
            "Slimmable training with condition_dsp is not supported"
        )
    for i, lc in enumerate(layers_configs):
        if not _has_slimmable(lc):
            continue
        s = lc.get("slimmable")
        if isinstance(s, dict):
            _validate_slimmable_dict(s, i)
        if lc.get("groups_input", 1) != 1 or lc.get("groups_input_mixin", 1) != 1:
            raise NotImplementedError(
                "Slimmable training with groups > 1 is not supported"
            )
        head_1x1 = lc.get("head_1x1_config") or {}
        if isinstance(head_1x1, dict) and head_1x1.get("active", False):
            raise NotImplementedError(
                "Slimmable training with head 1x1 is not supported"
            )
        layer_1x1 = lc.get("layer_1x1_config") or {}
        if isinstance(layer_1x1, dict) and layer_1x1.get("groups", 1) != 1:
            raise NotImplementedError(
                "Slimmable training with layer 1x1 groups != 1 is not supported"
            )
        film_params = lc.get("film_params") or {}
        film_keys = (
            "conv_pre_film",
            "conv_post_film",
            "input_mixin_pre_film",
            "input_mixin_post_film",
            "activation_pre_film",
            "activation_post_film",
            "layer1x1_post_film",
            "head1x1_post_film",
        )
        for key in film_keys:
            fp = film_params.get(key)
            if fp and _film_params_from_dict(fp).active:
                raise NotImplementedError(
                    "Slimmable training with FiLM is not supported"
                )


class WaveNet(_Slimmable, _nn.Module, _InitializableFromConfig):
    def __init__(
        self,
        layer_arrays: _nn.ModuleList,
        head: _Optional[_Head],
        head_scale: float,
        condition_dsp: _Optional["WaveNet"] = None,
    ):
        super().__init__()

        self._layer_arrays = layer_arrays
        self._head = head
        self._head_scale = head_scale
        self._condition_dsp = condition_dsp
        self._is_slimmable = any(
            isinstance(m, _Slimmable) for m in self.modules() if m is not self
        )

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = super().parse_config(config)

        stable = config.pop("stable", False)
        condition_dsp_config = config.pop("condition_dsp", None)
        layers_configs = config.pop("layers_configs", [])
        _validate_slimmable_config(layers_configs, condition_dsp_config)
        head_config = config.pop("head", None)
        head_scale = config.pop("head_scale", 1.0)

        if condition_dsp_config is not None:
            if condition_dsp_config.get("name") != "WaveNet":
                raise NotImplementedError("Only WaveNet condition DSP is supported")
            cond_config = dict(condition_dsp_config["config"])
            cond_config["stable"] = stable
            condition_dsp = WaveNet.init_from_config(cond_config)
        else:
            condition_dsp = None

        for i, lc in enumerate(layers_configs):
            lc["is_first"] = i == 0
            lc["is_last"] = i == len(layers_configs) - 1
            lc["stable"] = stable
        layer_arrays = _nn.ModuleList(
            [_LayerArray.init_from_config(lc) for lc in layers_configs]
        )
        if head_config is not None:
            head_config = dict(head_config)
            head_config["stable"] = stable
        head = None if head_config is None else _Head(**head_config)

        return dict(
            condition_dsp=condition_dsp,
            layer_arrays=layer_arrays,
            head=head,
            head_scale=head_scale,
        )

    @property
    def receptive_field(self) -> int:
        receptive_field = 1 + sum(
            [(layer_array.receptive_field - 1) for layer_array in self._layer_arrays]
        )
        if self._condition_dsp is not None:
            receptive_field += self._condition_dsp.receptive_field - 1
        return receptive_field

    def export_config(self, sample_rate: _Optional[float] = None):
        config = {
            "layers": [
                layer_array.export_config() for layer_array in self._layer_arrays
            ],
            "head": None if self._head is None else self._head.export_config(),
            "head_scale": self._head_scale,
        }
        if self._condition_dsp is not None:
            # Build condition_dsp export dict without running forward (condition_dsp
            # may have multiple output channels; WaveNet wrapper asserts 1 channel).
            assert isinstance(
                self._condition_dsp, WaveNet
            ), "The following assumes that the condition DSP is a WaveNet"
            t = _datetime.now()
            condition_dsp_dict = {
                "version": _EXPORT_VERSION,
                "metadata": {
                    "date": _Date(
                        year=t.year,
                        month=t.month,
                        day=t.day,
                        hour=t.hour,
                        minute=t.minute,
                        second=t.second,
                    ).model_dump()
                },
                "architecture": "WaveNet",
                "config": self._condition_dsp.export_config(),
                "weights": self._condition_dsp.export_weights().tolist(),
            }
            # C++ loadmodel requires condition_dsp sample_rate to match main model
            if sample_rate is not None:
                condition_dsp_dict["sample_rate"] = sample_rate
            config["condition_dsp"] = condition_dsp_dict
        return config

    def export_weights(self) -> _np.ndarray:
        """
        :return: 1D array
        """
        weights = _torch.cat([layer.export_weights() for layer in self._layer_arrays])
        if self._head is not None:
            weights = _torch.cat([weights, self._head.export_weights()])
        weights = _torch.cat([weights.cpu(), _torch.Tensor([self._head_scale])])
        return weights.detach().cpu().numpy()

    def import_weights(self, weights: _torch.Tensor, i: int = 0) -> int:
        if self._head is not None:
            raise NotImplementedError("Head importing isn't implemented yet.")
        for layer in self._layer_arrays:
            i = layer.import_weights(weights, i)
        return i

    def is_slimmable(self) -> bool:
        return self._is_slimmable

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        :param x: (B,Cx,L)
        :return: (B,Cy,L-R)
        """
        c = x if self._condition_dsp is None else self._condition_dsp(x)
        y, head_input = x, None
        for layer_array in self._layer_arrays:
            head_input, y = layer_array(y, c, head_input=head_input)
        head_input = self._head_scale * head_input
        return head_input if self._head is None else self._head(head_input)
