"""
Packed WaveNet public wrapper implementation.
"""

import json as _json
from collections.abc import Mapping as _Mapping
from pathlib import Path as _Path
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Sequence as _Sequence

import numpy as _np
import torch as _torch

from .._abc import ImportsWeights as _ImportsWeights
from .._constants import MODEL_VERSION as _MODEL_VERSION
from ..base import BaseNet as _BaseNet
from ..exportable import _cast_enums
from ..metadata import UserMetadata as _UserMetadata
from ._packed import PackedWaveNetSpec as _PackedWaveNetSpec
from ._packed import build_packed_wavenet_config as _build_packed_wavenet_config
from ._packed import validate_and_build_packed_spec as _validate_and_build_packed_spec
from ._packed_conv import PackedConv1dBase as _PackedConv1dBase
from ._wavenet import WaveNet as _InternalWaveNet
from ._wavenet_wrapper import WaveNet as _PublicWaveNet


class PackedWaveNet(_BaseNet, _ImportsWeights):
    def __init__(
        self,
        wavenet: _InternalWaveNet,
        spec: _PackedWaveNetSpec,
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(sample_rate=sample_rate)
        self._net = wavenet
        self._spec = spec
        self.apply_mask()

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        submodels = config.pop("submodels")
        export_config = config.pop("export", {})
        if config:
            raise ValueError(f"Unexpected PackedWaveNet config keys: {sorted(config)}")
        spec = _validate_and_build_packed_spec(submodels, export_config=export_config)
        wavenet = _InternalWaveNet.init_from_config(_build_packed_wavenet_config(spec))
        return {"sample_rate": sample_rate, "wavenet": wavenet, "spec": spec}

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    @property
    def num_submodels(self) -> int:
        return self._spec.num_submodels

    @property
    def submodel_names(self) -> tuple[str, ...]:
        return self._spec.submodel_names

    @property
    def submodel_configs(self) -> tuple[_Dict, ...]:
        return self._spec.submodel_configs

    @property
    def _mps_fallback_cat_dim(self) -> int:
        return -1

    def import_weights(self, weights: _Sequence[float], i: int = 0) -> int:
        weights_tensor = (
            weights if isinstance(weights, _torch.Tensor) else _torch.Tensor(weights)
        )
        i = self._net.import_weights(weights_tensor, i)
        self.apply_mask()
        return i

    def import_submodel(self, submodel_index: int, submodel: _PublicWaveNet) -> None:
        packed_convs = list(_iter_wavenet_convs(self._net))
        ordinary_convs = list(_iter_wavenet_convs(submodel._net))
        if len(packed_convs) != len(ordinary_convs):
            raise ValueError("Packed and ordinary WaveNet structures do not match")
        for packed_conv, ordinary_conv in zip(packed_convs, ordinary_convs):
            _copy_ordinary_conv_to_packed(packed_conv, ordinary_conv, submodel_index)
        self.apply_mask()

    def extract_submodel(self, submodel_index: int) -> _PublicWaveNet:
        if submodel_index < 0 or submodel_index >= self.num_submodels:
            raise IndexError(submodel_index)
        self.apply_mask()

        submodel = _PublicWaveNet.init_from_config(
            self._spec.submodel_configs[submodel_index]
        )
        submodel.sample_rate = self.sample_rate
        if self.device is not None:
            submodel.to(self.device)
        packed_convs = list(_iter_wavenet_convs(self._net))
        ordinary_convs = list(_iter_wavenet_convs(submodel._net))
        if len(packed_convs) != len(ordinary_convs):
            raise ValueError("Packed and ordinary WaveNet structures do not match")
        for packed_conv, ordinary_conv in zip(packed_convs, ordinary_convs):
            _copy_packed_conv_to_ordinary(packed_conv, ordinary_conv, submodel_index)
        submodel.train(self.training)
        return submodel

    def export(
        self,
        outdir: _Path,
        include_snapshot: bool = False,
        basename: str = "model",
        user_metadata: _Optional[_UserMetadata] = None,
        other_metadata: _Optional[dict] = None,
    ):
        self.export_container(
            outdir,
            include_snapshot=include_snapshot,
            basename=basename,
            user_metadata=user_metadata,
            other_metadata=other_metadata,
        )

    def export_container(
        self,
        outdir: _Path,
        include_snapshot: bool = False,
        basename: str = "model",
        checkpoint_paths_by_submodel: _Optional[_Sequence[_Optional[str]]] = None,
        user_metadata: _Optional[_UserMetadata] = None,
        other_metadata: _Optional[dict] = None,
    ) -> _Dict:
        models = []
        checkpoint_paths = self._normalize_checkpoint_paths(
            checkpoint_paths_by_submodel
        )
        for i in range(self.num_submodels):
            source = self
            checkpoint_path = (
                checkpoint_paths[i] if checkpoint_paths is not None else None
            )
            if checkpoint_path is not None:
                source = self._load_packed_checkpoint(checkpoint_path)
            models.append(source.extract_submodel(i)._get_export_dict())

        container = {
            "version": _MODEL_VERSION,
            "metadata": self._get_non_user_metadata(),
            "architecture": "SlimmableContainer",
            "config": {
                "submodels": [
                    {"max_value": max_value, "model": model}
                    for max_value, model in zip(self._container_max_values(), models)
                ]
            },
            "weights": [],
        }
        self._sync_container_metadata_to_highest_quality_submodel(container)
        if self.sample_rate is not None:
            container["sample_rate"] = self.sample_rate
        if user_metadata is not None:
            container["metadata"].update(_cast_enums(user_metadata.model_dump()))
        if other_metadata is not None:
            container["metadata"].update(_cast_enums(other_metadata))
        container = self._apply_export_model_dict_post_hooks(model_dict=container)

        outdir = _Path(outdir)
        with open(_Path(outdir, f"{basename}{self.FILE_EXTENSION}"), "w") as fp:
            _json.dump(container, fp)
        if include_snapshot:
            x, y = self._export_input_output()
            _np.save(_Path(outdir, "test_inputs.npy"), x)
            _np.save(_Path(outdir, "test_outputs.npy"), y)
        return container

    def apply_mask(self) -> None:
        for module in self.modules():
            if isinstance(module, _PackedConv1dBase):
                module.apply_mask()

    def _assert_masked(self) -> None:
        for module in self.modules():
            if isinstance(module, _PackedConv1dBase):
                module._assert_masked()

    def _export_config(self):
        return self._spec.to_init_config(sample_rate=self.sample_rate)

    def _export_weights(self) -> _np.ndarray:
        self.apply_mask()
        return self._net.export_weights()

    def _forward(self, x, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("PackedWaveNet does not support kwargs")
        if x.ndim == 2:
            x = x[:, None, :]
        y = self._net(x)
        assert y.shape[1] == self.num_submodels
        return y

    def _container_max_values(self) -> list[float]:
        configured = self._spec.export_config.get("container_max_values", "uniform")
        if configured == "uniform":
            values = [(i + 1) / self.num_submodels for i in range(self.num_submodels)]
            values[-1] = 1.0
            return values
        if not isinstance(configured, list):
            raise ValueError("container_max_values must be 'uniform' or a list")
        if len(configured) != self.num_submodels:
            raise ValueError("container_max_values length must match submodels")
        values = [float(v) for v in configured]
        values[-1] = 1.0
        if values != sorted(values):
            raise ValueError("container_max_values must be sorted")
        return values

    @staticmethod
    def _sync_container_metadata_to_highest_quality_submodel(container: _Dict) -> None:
        submodels = container["config"]["submodels"]
        if len(submodels) == 0:
            return
        highest_quality = max(submodels, key=lambda submodel: submodel["max_value"])
        highest_quality_metadata = highest_quality["model"].get("metadata")
        if not isinstance(highest_quality_metadata, dict):
            return
        for key in ("loudness", "gain"):
            if key in highest_quality_metadata:
                container["metadata"][key] = highest_quality_metadata[key]

    def _normalize_checkpoint_paths(self, paths):
        if paths is None:
            return None
        if isinstance(paths, dict):
            return [paths.get(i) for i in range(self.num_submodels)]
        if len(paths) != self.num_submodels:
            raise ValueError("checkpoint_paths_by_submodel length must match submodels")
        return list(paths)

    def _load_packed_checkpoint(self, checkpoint_path: str) -> "PackedWaveNet":
        try:
            checkpoint = _torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = _torch.load(checkpoint_path, map_location="cpu")
        sample_rate = (
            checkpoint.get("sample_rate")
            if isinstance(checkpoint, _Mapping) and "sample_rate" in checkpoint
            else None
        )
        packed = self.__class__.init_from_config(
            self._spec.to_init_config(sample_rate=sample_rate)
        )
        state_dict = _extract_checkpoint_state_dict(checkpoint, checkpoint_path)
        state_dict = _normalize_packed_checkpoint_state_dict(
            state_dict, packed.state_dict().keys(), checkpoint_path
        )
        try:
            packed.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load packed checkpoint {checkpoint_path}: {e}"
            ) from e
        if sample_rate is not None:
            packed.sample_rate = sample_rate
        elif packed.sample_rate is None and self.sample_rate is not None:
            packed.sample_rate = self.sample_rate
        packed.apply_mask()
        return packed


def _extract_checkpoint_state_dict(checkpoint, checkpoint_path: str) -> _Mapping:
    if isinstance(checkpoint, _Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, _Mapping):
        raise RuntimeError(
            f"Packed checkpoint {checkpoint_path} does not contain a state_dict mapping"
        )
    non_string_keys = [k for k in state_dict if not isinstance(k, str)]
    if non_string_keys:
        raise RuntimeError(
            f"Packed checkpoint {checkpoint_path} has unexpected non-string "
            f"state_dict keys: {_summarize_keys(non_string_keys)}"
        )
    return state_dict


def _normalize_packed_checkpoint_state_dict(
    state_dict: _Mapping, expected_keys, checkpoint_path: str
) -> _Mapping:
    expected = set(expected_keys)
    source = set(state_dict.keys())
    if source == expected:
        return state_dict

    lightning_prefix = "_net."
    if source and all(k.startswith(lightning_prefix) for k in source):
        normalized = {k[len(lightning_prefix) :]: v for k, v in state_dict.items()}
        normalized_keys = set(normalized.keys())
        if normalized_keys == expected:
            return normalized
        raise RuntimeError(
            _format_checkpoint_key_mismatch(
                checkpoint_path,
                expected,
                normalized_keys,
                "after stripping one leading '_net.' prefix",
            )
        )

    raise RuntimeError(
        _format_checkpoint_key_mismatch(
            checkpoint_path,
            expected,
            source,
            "for raw PackedWaveNet format",
        )
    )


def _format_checkpoint_key_mismatch(
    checkpoint_path: str, expected: set[str], actual: set[str], context: str
) -> str:
    missing = expected - actual
    unexpected = actual - expected
    return (
        f"Packed checkpoint {checkpoint_path} does not match PackedWaveNet "
        f"state_dict keys {context}; missing keys: {_summarize_keys(missing)}; "
        f"unexpected keys: {_summarize_keys(unexpected)}"
    )


def _summarize_keys(keys, limit: int = 8) -> str:
    keys = sorted(str(k) for k in keys)
    if not keys:
        return "[]"
    shown = keys[:limit]
    suffix = "" if len(keys) <= limit else f", ... ({len(keys) - limit} more)"
    return "[" + ", ".join(shown) + suffix + "]"


def _iter_wavenet_convs(wavenet: _InternalWaveNet):
    if wavenet._condition_dsp is not None:
        raise NotImplementedError("PackedWaveNet does not support condition_dsp")
    if wavenet._head is not None:
        raise NotImplementedError("PackedWaveNet does not yet support top-level heads")
    for layer_array in wavenet._layer_arrays:
        yield layer_array._rechannel
        for layer in layer_array._layers:
            yield layer.conv
            yield layer.input_mixer
            if layer.layer1x1 is not None:
                yield layer.layer1x1
            if layer.head1x1 is not None:
                yield layer.head1x1
        yield layer_array._head_rechannel


def _copy_packed_conv_to_ordinary(
    packed_conv: _PackedConv1dBase, ordinary_conv, submodel_index: int
) -> None:
    out_slice, in_slice = packed_conv.get_block_slices(submodel_index)
    with _torch.no_grad():
        ordinary_conv.weight.copy_(packed_conv.weight[out_slice, in_slice, :])
        if ordinary_conv.bias is not None:
            ordinary_conv.bias.copy_(packed_conv.bias[out_slice])


def _copy_ordinary_conv_to_packed(
    packed_conv: _PackedConv1dBase, ordinary_conv, submodel_index: int
) -> None:
    out_slice, in_slice = packed_conv.get_block_slices(submodel_index)
    with _torch.no_grad():
        packed_conv.weight[out_slice, in_slice, :].copy_(ordinary_conv.weight)
        if packed_conv.bias is not None and ordinary_conv.bias is not None:
            packed_conv.bias[out_slice].copy_(ordinary_conv.bias)
