# File: _from_nam.py
# Created Date: Tuesday May 27th 2025
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Initialize models from .nam files
"""

from copy import deepcopy as _deepcopy
from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Union as _Union

import torch as _torch

from .base import BaseNet as _BaseNet
from .linear import Linear as _Linear
from .recurrent import LSTM as _LSTM
from .wavenet import WaveNet as _WaveNet


def _init_linear(config, sample_rate: _Optional[float]) -> _Linear:
    return _Linear(sample_rate=sample_rate, **config)


def _init_lstm(config, sample_rate: _Optional[float]) -> _LSTM:
    return _LSTM(sample_rate=sample_rate, **config)


def _export_activation_to_init_format(
    export_d: _Dict[str, _Any],
) -> _Union[str, _Dict[str, _Any]]:
    """
    Convert one activation from .nam export format (type + kwargs) to get_activation
    init format (name + kwargs). Preserves params like negative_slope, min_val, etc.
    """
    d = _deepcopy(export_d)
    if "type" not in d:
        return d
    d["name"] = d.pop("type")
    # PReLU: export may have negative_slopes (list); nn.PReLU needs num_parameters.
    if d.get("name") == "PReLU" and "negative_slopes" in d:
        d["num_parameters"] = len(d["negative_slopes"])
        del d["negative_slopes"]
    if len(d) == 1:
        return d["name"]
    return d


def _nam_layer_activation_to_init(
    primary: _Union[str, _Dict[str, _Any]],
    gating_mode: str,
    secondary: _Optional[_Union[str, _Dict[str, _Any]]],
) -> _Union[str, _Dict[str, _Any]]:
    """
    Convert one layer's activation from .nam export format (primary, gating_mode,
    secondary) to _LayerArray init format (single str or dict for get_activation).
    """
    prim_init = (
        _export_activation_to_init_format(primary)
        if isinstance(primary, dict)
        else primary
    )
    if gating_mode == "none":
        return prim_init
    sec_init = (
        _export_activation_to_init_format(secondary)
        if isinstance(secondary, dict)
        else secondary
    )
    return {
        "name": "PairMultiply" if gating_mode == "gated" else "PairBlend",
        "primary": prim_init,
        "secondary": sec_init,
    }


def _convert_nam_layer_array_config(layer_config: _Dict[str, _Any]) -> _Dict[str, _Any]:
    """
    Convert a layer array config from .nam export format to _LayerArray __init__
    kwargs. Strips gating_mode / secondary_activation and re-parses activation.
    Renames head1x1 (export key) to head_1x1_config, layer1x1 to layer_1x1_config.
    """
    lc = _deepcopy(layer_config)
    gating_modes = lc.pop("gating_mode", None)
    secondary_activations = lc.pop("secondary_activation", None)
    if "head1x1" in lc:
        lc["head_1x1_config"] = lc.pop("head1x1")
    if "layer1x1" in lc:
        lc["layer_1x1_config"] = lc.pop("layer1x1")
    activations = lc.get("activation", [])

    if gating_modes is not None and secondary_activations is not None:
        n = max(len(activations), len(gating_modes), len(secondary_activations))
        new_activation = []
        for i in range(n):
            prim = activations[i] if i < len(activations) else {"type": "Tanh"}
            gmode = gating_modes[i] if i < len(gating_modes) else "none"
            sec = secondary_activations[i] if i < len(secondary_activations) else None
            new_activation.append(_nam_layer_activation_to_init(prim, gmode, sec))
        lc["activation"] = new_activation

    # FiLM params: collect flat keys into film_params for _LayerArray
    _film_keys = (
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    )
    film_params = {}
    for key in _film_keys:
        if key in lc:
            film_params[key] = lc.pop(key)
    if film_params:
        lc["film_params"] = film_params

    return lc


def _init_wavenet(config, sample_rate: _Optional[float]) -> _WaveNet:
    # This might have some issues with activation parameters no setting appropriately.
    # Need to look closer.
    layers_configs = [_convert_nam_layer_array_config(lc) for lc in config["layers"]]
    return _WaveNet(
        layers_configs=layers_configs,
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
