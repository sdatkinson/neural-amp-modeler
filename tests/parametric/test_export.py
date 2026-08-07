import json as _json
from collections.abc import Sequence as _Sequence
from typing import cast as _cast

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Dataset as _Dataset
from nam.models import factory as _factory
from nam.models._from_nam import init_from_nam as _init_from_nam
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric import bake as _bake
from nam.models.parametric import bake_to_files as _bake_to_files
from nam.models.parametric import export_parametric as _export_parametric
from nam.models.parametric import output_scale_from_datasets as _output_scale_from_datasets
from nam.models.parametric._dataset import ParametricDataset as _ParametricDataset
from nam.models.wavenet import WaveNet as _WaveNet

from .test_hyperwavenet import _hyperwavenet_config


def _load_nam(path) -> dict:
    with open(path) as fp:
        return _json.load(fp)


def _make_nonzero_model(*, slimmable: bool = False) -> _HyperWaveNet:
    model = _HyperWaveNet.init_from_config(
        _hyperwavenet_config(hypernet={"hidden_sizes": [5]}, slimmable=slimmable)
    )
    generator = _torch.Generator().manual_seed(0)
    with _torch.no_grad():
        for parameter in model._hypernet.parameters():
            parameter.add_(
                0.01
                * _torch.randn(
                    parameter.shape,
                    generator=generator,
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
    model.eval()
    return model


def _make_parametric_dataset(*, y_scale):
    dataset = _Dataset(
        x=_torch.zeros(16),
        y=_torch.zeros(16),
        nx=4,
        ny=4,
        sample_rate=48_000.0,
    )
    if y_scale is not None:
        dataset.scale_output(y_scale)
    return _ParametricDataset(
        dataset=dataset,
        params=_torch.tensor([5.0, 1.0], dtype=_torch.float32),
    )


def _template_export_offsets(model: _HyperWaveNet) -> dict[str, int]:
    offsets = {}
    offset = 0

    def add_conv(prefix: str, conv) -> None:
        nonlocal offset
        if conv.weight is not None:
            offsets[f"{prefix}.weight"] = offset
            offset += conv.weight.numel()
        if conv.bias is not None:
            offsets[f"{prefix}.bias"] = offset
            offset += conv.bias.numel()

    for layer_array_i, layer_array in enumerate(model._template._layer_arrays):
        prefix = f"_layer_arrays.{layer_array_i}"
        add_conv(f"{prefix}._rechannel", layer_array._rechannel)
        for layer_i, layer in enumerate(_cast(_Sequence, layer_array._layers)):
            layer_prefix = f"{prefix}._layers.{layer_i}"
            add_conv(f"{layer_prefix}._conv", layer._conv)
            add_conv(f"{layer_prefix}._input_mixer", layer._input_mixer)
            if layer._layer1x1 is not None:
                add_conv(f"{layer_prefix}._layer1x1", layer._layer1x1)
            if layer.head1x1 is not None:
                add_conv(f"{layer_prefix}._head1x1", layer.head1x1)
            for film_name in (
                "_conv_pre_film",
                "_conv_post_film",
                "_input_mixin_pre_film",
                "_input_mixin_post_film",
                "_activation_pre_film",
                "_activation_post_film",
                "_layer1x1_post_film",
                "_head1x1_post_film",
            ):
                film = getattr(layer, film_name)
                if film is not None:
                    add_conv(f"{layer_prefix}.{film_name}._film", film._film)
        add_conv(f"{prefix}._head_rechannel", layer_array._head_rechannel)

    if model._template._head is not None:
        for layer_i, layer in enumerate(_cast(_Sequence, model._template._head._layers)):
            add_conv(f"_head._layers.layer_{layer_i}.1", _cast(_Sequence, layer)[1])

    assert offset + 1 == len(model._template.export_weights())
    return offsets


def test_bake_matches_hyperwavenet_forward_for_named_settings():
    model = _make_nonzero_model()
    params = {"gain": 8.0, "mode": "crunch"}
    resolved = _ParametricDataset._resolve_params(params, model.param_specs)
    x = _torch.randn(model.receptive_field + 64)

    wrapper = _bake(model, params)

    assert isinstance(wrapper, _WaveNet)
    assert _torch.allclose(
        wrapper(x, pad_start=False),
        model(x, resolved, pad_start=False),
    )


def test_bake_matches_forward_for_slimmable_template():
    # Eval pins the full channel width, so the baked stock net must reproduce the
    # HyperWaveNet forward even when the template is slimmable.
    model = _make_nonzero_model(slimmable=True)
    model.eval()
    params = _torch.tensor([7.0, 2.0], dtype=_torch.float32)
    x = _torch.randn(model.receptive_field + 32)

    wrapper = _bake(model, params)

    assert _torch.allclose(
        wrapper(x, pad_start=False),
        model(x, params, pad_start=False),
    )


def test_bake_output_scale_leaves_in_memory_forward_unscaled():
    # The scale hook only rewrites the exported head_scale; the live wrapper is unchanged.
    model = _make_nonzero_model()
    params = [8.0, 1.0]
    x = _torch.randn(model.receptive_field + 32)

    plain = _bake(model, params)
    scaled = _bake(model, params, output_scale=0.25)

    assert _torch.allclose(
        plain(x, pad_start=False),
        scaled(x, pad_start=False),
    )


def test_bake_distinct_settings_produce_distinct_weights(tmp_path):
    model = _make_nonzero_model()
    _bake(model, [1.0, 0.0]).export(tmp_path, basename="low")
    _bake(model, [9.0, 2.0]).export(tmp_path, basename="high")

    low = _load_nam(tmp_path / "low.nam")["weights"]
    high = _load_nam(tmp_path / "high.nam")["weights"]

    assert len(low) == len(high) == 12146
    assert not _np.allclose(low, high)


def test_baked_export_is_stock_wavenet_compatible(tmp_path):
    model = _make_nonzero_model()
    params = [8.0, 1.0]
    wrapper = _bake(model, params)
    wrapper.export(tmp_path)

    model_dict = _load_nam(tmp_path / "model.nam")
    round_tripped = _init_from_nam(model_dict)
    x = _torch.randn(model.receptive_field + 48)

    assert model_dict["architecture"] == "WaveNet"
    assert len(model_dict["weights"]) == 12146
    assert set(model_dict["config"]) == {"head", "head_scale", "layers"}
    assert isinstance(round_tripped, _WaveNet)
    assert _torch.allclose(
        round_tripped(x, pad_start=False),
        wrapper(x, pad_start=False),
    )


def test_bake_scale_compensation_scales_stock_head_export(tmp_path):
    model = _make_nonzero_model()
    params = [8.0, 1.0]
    output_scale = 0.25

    _bake(model, params).export(tmp_path, basename="unscaled")
    _bake(model, params, output_scale=output_scale).export(
        tmp_path, basename="scaled"
    )

    unscaled = _load_nam(tmp_path / "unscaled.nam")
    scaled = _load_nam(tmp_path / "scaled.nam")

    assert scaled["config"]["head_scale"] == _pytest.approx(
        unscaled["config"]["head_scale"] / output_scale
    )
    assert scaled["weights"][-1] == _pytest.approx(
        unscaled["weights"][-1] / output_scale
    )


def test_bake_recovers_output_scale_from_parametric_datasets(tmp_path):
    model = _make_nonzero_model()
    params = [8.0, 1.0]
    datasets = [
        _make_parametric_dataset(y_scale=0.25),
        _make_parametric_dataset(y_scale=0.25),
    ]

    recovered = _output_scale_from_datasets(datasets)
    assert recovered == _pytest.approx(0.25)

    _bake(model, params).export(tmp_path, basename="unscaled")
    _bake(model, params, output_scale=recovered).export(tmp_path, basename="scaled")

    unscaled = _load_nam(tmp_path / "unscaled.nam")
    scaled = _load_nam(tmp_path / "scaled.nam")

    assert scaled["config"]["head_scale"] == _pytest.approx(
        unscaled["config"]["head_scale"] / 0.25
    )
    assert scaled["weights"][-1] == _pytest.approx(unscaled["weights"][-1] / 0.25)


def test_bake_to_files_exports_each_requested_setting(tmp_path):
    model = _make_nonzero_model()

    paths = _bake_to_files(
        model,
        [
            ("nominal", {"gain": 5.0, "mode": "crunch"}),
            ("lead", [8.0, 2.0]),
        ],
        tmp_path,
    )

    assert [path.name for path in paths] == ["nominal.nam", "lead.nam"]
    assert all(path.exists() for path in paths)


def test_bake_to_files_rejects_duplicate_basenames(tmp_path):
    model = _make_nonzero_model()
    with _pytest.raises(ValueError):
        _bake_to_files(
            model,
            [("dup", {"gain": 1.0, "mode": "clean"}), ("dup", [2.0, 1.0])],
            tmp_path,
        )


def test_export_parametric_round_trips_through_factory(tmp_path):
    model = _make_nonzero_model()
    with _torch.no_grad():
        _cast(_torch.Tensor, model._hypernet._low_rank_anchor).add_(0.125)
    _export_parametric(model, tmp_path)
    model_dict = _load_nam(tmp_path / "model.nam")
    base_len = len(model._template.export_weights())

    round_tripped = _cast(
        _HyperWaveNet,
        _factory.init(
            "HyperWaveNet",
            args=(
                {**model_dict["config"], "sample_rate": model_dict.get("sample_rate")},
            ),
        ),
    )
    end = round_tripped.import_weights(model_dict["weights"])
    params = _torch.tensor([6.5, 2.0], dtype=_torch.float32)
    x = _torch.randn(model.receptive_field + 32)

    assert model_dict["architecture"] == "HyperWaveNet"
    assert model_dict["config"]["params"] == [
        spec.to_dict() for spec in model.param_specs
    ]
    assert set(model_dict["config"]) == {"head", "head_scale", "hypernet", "layers", "params"}
    assert "layers_configs" not in model_dict["config"]
    assert "hypernet" in model_dict["config"]
    delta_map = model_dict["config"]["hypernet"]["delta_map"]
    expected_offsets = _template_export_offsets(model)
    assert [entry["name"] for entry in delta_map] == list(model._hypernet.target_names)
    for entry in delta_map:
        name = entry["name"]
        assert entry["export_offset"] == expected_offsets[name]
        assert entry["numel"] == model._template.get_parameter(name).numel()
        if entry["mode"] == "low_rank":
            assert set(entry) == {
                "export_offset",
                "mode",
                "name",
                "numel",
                "out_features",
                "rank",
                "rest_features",
            }
        else:
            assert set(entry) == {"export_offset", "mode", "name", "numel"}
    assert len(model_dict["weights"]) == base_len + model._hypernet.state_count()
    assert end == len(model_dict["weights"])
    for (model_name, model_tensor), (round_trip_name, round_trip_tensor) in zip(
        model._hypernet.state_dict().items(),
        round_tripped._hypernet.state_dict().items(),
    ):
        assert model_name == round_trip_name
        assert _torch.equal(model_tensor, round_trip_tensor)
    assert _torch.allclose(
        round_tripped(x, params, pad_start=False),
        model(x, params, pad_start=False),
    )


def test_parametric_scale_compensation_targets_base_head_scale_slot(tmp_path):
    model = _make_nonzero_model()
    output_scale = 0.5
    base_len = len(model._template.export_weights())
    original_hooks = list(model.export_model_dict_post_hooks)

    _export_parametric(model, tmp_path, basename="unscaled")
    _export_parametric(model, tmp_path, basename="scaled", output_scale=output_scale)

    unscaled = _load_nam(tmp_path / "unscaled.nam")
    scaled = _load_nam(tmp_path / "scaled.nam")

    assert scaled["config"]["head_scale"] == _pytest.approx(
        unscaled["config"]["head_scale"] / output_scale
    )
    assert scaled["weights"][base_len - 1] == _pytest.approx(
        unscaled["weights"][base_len - 1] / output_scale
    )
    assert _np.allclose(scaled["weights"][base_len:], unscaled["weights"][base_len:])
    assert model.export_model_dict_post_hooks == original_hooks


def test_export_parametric_recovers_output_scale_from_concat_dataset(tmp_path):
    model = _make_nonzero_model()
    output_scale = 0.5
    base_len = len(model._template.export_weights())
    wrappers = [
        _make_parametric_dataset(y_scale=output_scale),
        _make_parametric_dataset(y_scale=output_scale),
    ]
    datasets = _ConcatDataset(
        _cast(_Sequence[_Dataset], wrappers)
    )
    recovered = _output_scale_from_datasets(datasets)

    _export_parametric(model, tmp_path, basename="unscaled")
    _export_parametric(model, tmp_path, basename="scaled", output_scale=recovered)

    unscaled = _load_nam(tmp_path / "unscaled.nam")
    scaled = _load_nam(tmp_path / "scaled.nam")

    assert scaled["config"]["head_scale"] == _pytest.approx(
        unscaled["config"]["head_scale"] / output_scale
    )
    assert scaled["weights"][base_len - 1] == _pytest.approx(
        unscaled["weights"][base_len - 1] / output_scale
    )


class _DummyDataset:
    def __init__(self, y_scale):
        self._y_scale = y_scale


def test_output_scale_from_datasets_validates_shared_scale():
    assert _output_scale_from_datasets(
        [_DummyDataset(0.25), _DummyDataset(0.25)]
    ) == _pytest.approx(0.25)
    assert _output_scale_from_datasets([_DummyDataset(None), _DummyDataset(None)]) is None
    with _pytest.raises(ValueError):
        _output_scale_from_datasets([_DummyDataset(0.25), _DummyDataset(0.5)])


def test_output_scale_from_datasets_unwraps_parametric_and_concat_datasets():
    wrappers = [
        _make_parametric_dataset(y_scale=0.25),
        _make_parametric_dataset(y_scale=0.25),
    ]
    datasets = _ConcatDataset(_cast(_Sequence[_Dataset], wrappers))

    assert _output_scale_from_datasets(datasets) == _pytest.approx(0.25)
