"""
Packed WaveNet config validation and layout construction.
"""

import json as _json
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Sequence as _Sequence

from .._activations import PairingActivation as _PairingActivation
from .._activations import get_activation as _get_activation

_FILM_NAMES = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


@_dataclass(frozen=True)
class PackedSubmodelSpec:
    name: str
    config: _Dict[str, _Any]


@_dataclass(frozen=True)
class PackedLayerArrayLayout:
    index: int
    is_first: bool
    is_last: bool
    input_channels: tuple[int, ...]
    channels: tuple[int, ...]
    bottleneck: tuple[int, ...]
    head1x1_out_channels: tuple[int, ...]
    head_rechannel_in_channels: tuple[int, ...]
    head_out_channels: tuple[int, ...]

    def as_packing_config(self) -> _Dict[str, _Any]:
        return {
            "num_submodels": len(self.channels),
            "input_channels": list(self.input_channels),
            "channels": list(self.channels),
            "bottleneck": list(self.bottleneck),
            "head1x1_out_channels": list(self.head1x1_out_channels),
            "head_rechannel_in_channels": list(self.head_rechannel_in_channels),
            "head_out_channels": list(self.head_out_channels),
        }


@_dataclass(frozen=True)
class PackedWaveNetSpec:
    submodels: tuple[PackedSubmodelSpec, ...]
    layer_layouts: tuple[PackedLayerArrayLayout, ...]
    export_config: _Dict[str, _Any]

    @property
    def num_submodels(self) -> int:
        return len(self.submodels)

    @property
    def submodel_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.submodels)

    @property
    def submodel_configs(self) -> tuple[_Dict[str, _Any], ...]:
        return tuple(_deepcopy(s.config) for s in self.submodels)

    def to_init_config(self, sample_rate: _Optional[float] = None) -> _Dict[str, _Any]:
        config = {
            "submodels": [
                {"name": s.name, "config": _deepcopy(s.config)} for s in self.submodels
            ],
            "export": _deepcopy(self.export_config),
        }
        if sample_rate is not None:
            config["sample_rate"] = sample_rate
        return config


def build_packed_wavenet_config(spec: PackedWaveNetSpec) -> _Dict[str, _Any]:
    reference_config = spec.submodels[0].config
    packed_layers = []
    for layout in spec.layer_layouts:
        ref = _deepcopy(reference_config["layers_configs"][layout.index])
        ref["input_size"] = (
            reference_config["layers_configs"][layout.index]["input_size"]
            if layout.is_first
            else sum(layout.input_channels)
        )
        ref["condition_size"] = reference_config["layers_configs"][layout.index][
            "condition_size"
        ]
        ref["channels"] = sum(layout.channels)
        ref["bottleneck"] = sum(layout.bottleneck)
        ref["head"]["out_channels"] = sum(layout.head_out_channels)
        head1x1 = ref.get("head_1x1_config", {"active": False})
        if head1x1.get("active", False):
            head1x1 = _deepcopy(head1x1)
            head1x1["out_channels"] = sum(layout.head1x1_out_channels)
            ref["head_1x1_config"] = head1x1
        ref["packing"] = layout.as_packing_config()
        packed_layers.append(ref)

    return {
        "layers_configs": packed_layers,
        "head": None,
        "head_scale": reference_config.get("head_scale", 1.0),
    }


def validate_and_build_packed_spec(
    submodels: _Sequence[_Dict[str, _Any]],
    export_config: _Optional[_Dict[str, _Any]] = None,
) -> PackedWaveNetSpec:
    if len(submodels) < 1:
        raise ValueError("PackedWaveNet requires at least one submodel")
    normalized = []
    for i, submodel in enumerate(submodels):
        name = str(submodel.get("name", f"submodel_{i}"))
        if "config" not in submodel:
            raise KeyError(f"Packed submodel {i} is missing 'config'")
        normalized.append(
            PackedSubmodelSpec(name, _normalize_model_config(submodel["config"]))
        )

    _validate_top_level(normalized)
    layer_layouts = _validate_layers(normalized)
    return PackedWaveNetSpec(
        submodels=tuple(normalized),
        layer_layouts=tuple(layer_layouts),
        export_config={} if export_config is None else _deepcopy(export_config),
    )


def _normalize_model_config(config: _Dict[str, _Any]) -> _Dict[str, _Any]:
    c = _deepcopy(config)
    if "layers_configs" not in c and "layers" in c:
        c["layers_configs"] = [
            _convert_export_layer_config(lc) for lc in c.pop("layers")
        ]
    if "layers_configs" not in c:
        raise KeyError("Packed submodel config must include 'layers_configs'")
    c["layers_configs"] = [_normalize_layer_config(lc) for lc in c["layers_configs"]]
    c.setdefault("head", None)
    c.setdefault("head_scale", 1.0)
    return c


def _normalize_layer_config(layer_config: _Dict[str, _Any]) -> _Dict[str, _Any]:
    lc = _convert_export_layer_config(layer_config)
    if "kernel_sizes" in lc and "kernel_size" not in lc:
        lc["kernel_size"] = lc.pop("kernel_sizes")
    if "head_1x1_config" not in lc:
        lc["head_1x1_config"] = {"active": False, "out_channels": 1, "groups": 1}
    if "layer_1x1_config" not in lc:
        lc["layer_1x1_config"] = {"active": True, "groups": 1}
    lc.setdefault("groups_input", 1)
    lc.setdefault("groups_input_mixin", 1)
    lc.setdefault("film_params", {})
    return lc


def _convert_export_layer_config(layer_config: _Dict[str, _Any]) -> _Dict[str, _Any]:
    lc = _deepcopy(layer_config)
    gating_modes = lc.pop("gating_mode", None)
    secondary_activations = lc.pop("secondary_activation", None)
    if "head1x1" in lc and "head_1x1_config" not in lc:
        lc["head_1x1_config"] = lc.pop("head1x1")
    if "layer1x1" in lc and "layer_1x1_config" not in lc:
        lc["layer_1x1_config"] = lc.pop("layer1x1")

    activations = lc.get("activation", [])
    if gating_modes is not None and secondary_activations is not None:
        if not isinstance(activations, list):
            activations = [activations] * len(gating_modes)
        n = max(len(activations), len(gating_modes), len(secondary_activations))
        converted = []
        for i in range(n):
            primary = activations[i] if i < len(activations) else {"type": "Tanh"}
            gating_mode = gating_modes[i] if i < len(gating_modes) else "none"
            secondary = (
                secondary_activations[i] if i < len(secondary_activations) else None
            )
            converted.append(
                _nam_layer_activation_to_init(primary, gating_mode, secondary)
            )
        lc["activation"] = converted

    film_params = lc.pop("film_params", {})
    for key in _FILM_NAMES:
        if key in lc:
            film_params[key] = lc.pop(key)
    if film_params:
        lc["film_params"] = film_params
    return lc


def _export_activation_to_init_format(value):
    if not isinstance(value, dict):
        return value
    d = _deepcopy(value)
    if "type" in d:
        d["name"] = d.pop("type")
    if d.get("name") == "PReLU" and "negative_slopes" in d:
        d["num_parameters"] = len(d["negative_slopes"])
        del d["negative_slopes"]
    if set(d) == {"name"}:
        return d["name"]
    return d


def _nam_layer_activation_to_init(primary, gating_mode: str, secondary):
    primary = _export_activation_to_init_format(primary)
    if gating_mode == "none":
        return primary
    secondary = _export_activation_to_init_format(secondary)
    return {
        "name": "PairMultiply" if gating_mode == "gated" else "PairBlend",
        "primary": primary,
        "secondary": secondary,
    }


def _validate_top_level(submodels: _Sequence[PackedSubmodelSpec]) -> None:
    ref = submodels[0].config
    if ref.get("condition_dsp") is not None:
        raise NotImplementedError("PackedWaveNet does not support condition_dsp")
    if ref.get("head") is not None:
        raise NotImplementedError("PackedWaveNet does not yet support top-level heads")
    for submodel in submodels:
        config = submodel.config
        if config.get("condition_dsp") is not None:
            raise NotImplementedError("PackedWaveNet does not support condition_dsp")
        if config.get("head") is not None:
            raise NotImplementedError(
                "PackedWaveNet does not yet support top-level heads"
            )
        if config.get("head_scale", 1.0) != ref.get("head_scale", 1.0):
            raise ValueError("PackedWaveNet submodels must use the same head_scale")
        if len(config["layers_configs"]) != len(ref["layers_configs"]):
            raise ValueError(
                "PackedWaveNet submodels must have the same number of layer arrays"
            )


def _validate_layers(
    submodels: _Sequence[PackedSubmodelSpec],
) -> list[PackedLayerArrayLayout]:
    ref_layers = submodels[0].config["layers_configs"]
    layouts = []
    previous_channels = None
    previous_head_channels = None
    for layer_array_index, ref in enumerate(ref_layers):
        is_first = layer_array_index == 0
        is_last = layer_array_index == len(ref_layers) - 1
        _validate_reference_layer(ref)
        ref_kernel_sizes = _kernel_sizes(ref)
        ref_dilations = tuple(ref["dilations"])
        ref_activation = _activation_signature(ref["activation"], len(ref_dilations))
        ref_head = ref["head"]
        ref_layer1x1 = ref["layer_1x1_config"]
        ref_head1x1 = ref["head_1x1_config"]

        input_channels = []
        channels = []
        bottlenecks = []
        head1x1_out_channels = []
        head_rechannel_in_channels = []
        head_out_channels = []

        for submodel in submodels:
            layer = submodel.config["layers_configs"][layer_array_index]
            _validate_reference_layer(layer)
            if len(layer["dilations"]) != len(ref_dilations):
                raise ValueError("PackedWaveNet layer arrays must have matching depth")
            if _kernel_sizes(layer) != ref_kernel_sizes:
                raise ValueError("PackedWaveNet layer arrays must match kernel sizes")
            if tuple(layer["dilations"]) != ref_dilations:
                raise ValueError("PackedWaveNet layer arrays must match dilations")
            if (
                _activation_signature(layer["activation"], len(ref_dilations))
                != ref_activation
            ):
                raise ValueError("PackedWaveNet layer activations must match")
            if layer["head"]["kernel_size"] != ref_head["kernel_size"]:
                raise ValueError("PackedWaveNet head rechannel kernels must match")
            if layer["head"].get("bias", True) != ref_head.get("bias", True):
                raise ValueError("PackedWaveNet head rechannel bias flags must match")
            if layer["layer_1x1_config"].get("active", True) != ref_layer1x1.get(
                "active", True
            ):
                raise ValueError("PackedWaveNet layer1x1 active flags must match")
            if layer["head_1x1_config"].get("active", False) != ref_head1x1.get(
                "active", False
            ):
                raise ValueError("PackedWaveNet head1x1 active flags must match")

            expected_input = 1 if is_first else previous_channels[len(input_channels)]
            if layer["input_size"] != expected_input:
                raise ValueError(
                    "PackedWaveNet layer input_size must match previous channels"
                )
            if layer["condition_size"] != ref["condition_size"]:
                raise ValueError("PackedWaveNet condition sizes must match")
            if is_first and layer["input_size"] != 1:
                raise NotImplementedError("PackedWaveNet currently supports mono input")
            if layer["condition_size"] != 1:
                raise NotImplementedError(
                    "PackedWaveNet currently supports mono conditioning audio"
                )
            if is_last and layer["head"]["out_channels"] != 1:
                raise NotImplementedError(
                    "PackedWaveNet currently supports one output channel per submodel"
                )

            c = int(layer["channels"])
            b = int(layer.get("bottleneck", c))
            h1 = int(layer["head_1x1_config"].get("out_channels", 1))
            h1_active = layer["head_1x1_config"].get("active", False)
            head_rechannel_in = h1 if h1_active else b
            if (
                not is_first
                and head_rechannel_in != previous_head_channels[len(input_channels)]
            ):
                raise ValueError(
                    "PackedWaveNet layer-array head channels must match the "
                    "previous layer-array head outputs"
                )
            input_channels.append(int(layer["input_size"]))
            channels.append(c)
            bottlenecks.append(b)
            head1x1_out_channels.append(h1 if h1_active else 0)
            head_rechannel_in_channels.append(head_rechannel_in)
            head_out_channels.append(int(layer["head"]["out_channels"]))

        layouts.append(
            PackedLayerArrayLayout(
                index=layer_array_index,
                is_first=is_first,
                is_last=is_last,
                input_channels=tuple(input_channels),
                channels=tuple(channels),
                bottleneck=tuple(bottlenecks),
                head1x1_out_channels=tuple(head1x1_out_channels),
                head_rechannel_in_channels=tuple(head_rechannel_in_channels),
                head_out_channels=tuple(head_out_channels),
            )
        )
        previous_channels = tuple(channels)
        previous_head_channels = tuple(head_out_channels)
    return layouts


def _validate_reference_layer(layer: _Dict[str, _Any]) -> None:
    if layer.get("slimmable") is not None:
        raise NotImplementedError("PackedWaveNet and slimmable WaveNet are exclusive")
    if layer.get("groups_input", 1) != 1 or layer.get("groups_input_mixin", 1) != 1:
        raise NotImplementedError("PackedWaveNet does not support grouped convolutions")
    if layer["layer_1x1_config"].get("groups", 1) != 1:
        raise NotImplementedError("PackedWaveNet does not support grouped layer1x1")
    if layer["head_1x1_config"].get("groups", 1) != 1:
        raise NotImplementedError("PackedWaveNet does not support grouped head1x1")
    for film_config in layer.get("film_params", {}).values():
        if isinstance(film_config, dict) and film_config.get("active", False):
            raise NotImplementedError("PackedWaveNet does not support FiLM")
    activations = layer["activation"]
    if not isinstance(activations, list):
        activations = [activations] * len(layer["dilations"])
    for activation in activations:
        if isinstance(_get_activation(activation), _PairingActivation):
            raise NotImplementedError(
                "PackedWaveNet does not yet support paired/gated activations"
            )


def _kernel_sizes(layer: _Dict[str, _Any]) -> tuple[int, ...]:
    kernel_sizes = layer["kernel_size"]
    if isinstance(kernel_sizes, int):
        return tuple([kernel_sizes] * len(layer["dilations"]))
    return tuple(int(k) for k in kernel_sizes)


def _activation_signature(activation, num_layers: int) -> tuple[str, ...]:
    if not isinstance(activation, list):
        activation = [activation] * num_layers
    return tuple(_json.dumps(a, sort_keys=True) for a in activation)
