import json as _json
from typing import cast as _cast

import pytest as _pytest
import torch as _torch

from nam.models import factory as _factory
from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric import export_parametric as _export_parametric
from nam.models.wavenet._wavenet import WaveNet as _InnerWaveNet


def _concat_wavenet_config() -> dict:
    return {
        "sample_rate": 48_000.0,
        "layers": [
            {
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 4,
                "kernel_size": 2,
                "dilations": [1, 2],
                "activation": "Tanh",
            },
            {
                "input_size": 4,  # = previous layer array's channels
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 4,
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            },
        ],
        "head_scale": 0.5,
        "params": [
            {"name": "gain", "min": 0.0, "max": 10.0, "default": 5.0, "type": "continuous"},
            {"name": "mode", "min": 0, "max": 2, "default": 1, "type": "switch",
             "enum_names": ["clean", "crunch", "lead"]},
        ],
    }


def _param_specs() -> tuple[_ParamSpec, ...]:
    return tuple(_ParamSpec.from_dict(spec) for spec in _concat_wavenet_config()["params"])


def _tiny_wavenet_internal_config() -> dict:
    # input_size/condition_size == 1: only used to build a condition_dsp WaveNet, whose
    # forward is never run in the constructor-rejection test.
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 2,
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            }
        ],
        "head_scale": 1.0,
    }


def _inner_wavenet_internal_config(*, slimmable: bool = False) -> dict:
    # Internal init config: derived channels == 5, so input_size/condition_size are set
    # explicitly to what parse_config would derive.
    layer = {
        "input_size": 5,
        "condition_size": 5,
        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
        "channels": 4,
        "kernel_size": 2,
        "dilations": [1, 2],
        "activation": "Tanh",
    }
    if slimmable:
        layer["slimmable"] = {"method": "slice_channels_uniform", "kwargs": {}}
    return {"layers_configs": [layer], "head_scale": 1.0}


def _init(config: dict) -> _ConcatWaveNet:
    return _cast(_ConcatWaveNet, _factory.init("ConcatWaveNet", args=(config,)))


def test_factory_init_derives_channels_and_receptive_field():
    model = _init(_concat_wavenet_config())

    assert isinstance(model, _ConcatWaveNet)
    assert model.receptive_field == 5
    exported = model._export_config()
    assert exported["layers"][0]["input_size"] == 5
    assert exported["layers"][0]["condition_size"] == 5
    assert exported["layers"][1]["condition_size"] == 5


def test_explicit_derived_channels_are_accepted():
    config = _concat_wavenet_config()
    config["layers"][0]["input_size"] = 5
    config["layers"][0]["condition_size"] = 5
    config["layers"][1]["condition_size"] = 5

    model = _init(config)

    assert isinstance(model, _ConcatWaveNet)
    assert model.receptive_field == 5


def test_wrong_first_input_size_is_rejected():
    config = _concat_wavenet_config()
    config["layers"][0]["input_size"] = 1
    with _pytest.raises(ValueError, match="input_size"):
        _init(config)


def test_wrong_condition_size_is_rejected():
    config = _concat_wavenet_config()
    config["layers"][1]["condition_size"] = 1
    with _pytest.raises(ValueError, match="condition_size"):
        _init(config)


def test_condition_dsp_config_is_rejected():
    config = _concat_wavenet_config()
    config["condition_dsp"] = {"name": "WaveNet", "config": _tiny_wavenet_internal_config()}
    with _pytest.raises(NotImplementedError, match="condition_dsp"):
        _init(config)


def test_condition_dsp_inner_wavenet_is_rejected():
    tiny = _tiny_wavenet_internal_config()
    inner = _InnerWaveNet.init_from_config(
        {**tiny, "condition_dsp": {"name": "WaveNet", "config": tiny}}
    )
    with _pytest.raises(NotImplementedError, match="condition_dsp"):
        _ConcatWaveNet(wavenet=inner, param_specs=_param_specs(), sample_rate=48_000.0)


def test_slimmable_inner_wavenet_is_rejected():
    inner = _InnerWaveNet.init_from_config(_inner_wavenet_internal_config(slimmable=True))
    assert inner.is_slimmable()
    with _pytest.raises(NotImplementedError, match="slimmable"):
        _ConcatWaveNet(wavenet=inner, param_specs=_param_specs(), sample_rate=48_000.0)


def test_packed_layer_array_is_rejected():
    config = _concat_wavenet_config()
    config["layers"][0]["packing"] = {"num_models": 2}
    with _pytest.raises(NotImplementedError, match="pack"):
        _init(config)


def test_forward_shape_contract():
    model = _init(_concat_wavenet_config())
    rf = model.receptive_field
    length = rf + 10
    out_length = length - rf + 1
    batch_params = _torch.tensor([[5.0, 1.0], [2.0, 0.0]], dtype=_torch.float32)
    shared_params = _torch.tensor([5.0, 1.0], dtype=_torch.float32)

    x_batched = _torch.randn(2, length)
    x_flat = _torch.randn(length)

    assert model(x_batched, batch_params, pad_start=False).shape == (2, out_length)
    assert model(x_batched, shared_params, pad_start=False).shape == (2, out_length)
    assert model(x_flat, batch_params, pad_start=False).shape == (2, out_length)
    assert model(x_flat, shared_params, pad_start=False).shape == (out_length,)


def test_forward_rejects_batch_mismatch():
    model = _init(_concat_wavenet_config())
    length = model.receptive_field + 10
    x = _torch.randn(2, length)
    params = _torch.tensor([[5.0, 1.0], [2.0, 0.0], [8.0, 2.0]], dtype=_torch.float32)
    with _pytest.raises(ValueError, match="batch size"):
        model(x, params, pad_start=False)


def test_params_condition_the_output():
    model = _init(_concat_wavenet_config())
    model.eval()
    x = _torch.randn(model.receptive_field + 32)

    low = model(x, _torch.tensor([0.0, 0.0], dtype=_torch.float32), pad_start=False)
    high = model(x, _torch.tensor([10.0, 2.0], dtype=_torch.float32), pad_start=False)

    assert not _torch.allclose(low, high)


def test_forward_matches_manual_concat_wiring():
    model = _init(_concat_wavenet_config())
    model.eval()
    length = model.receptive_field + 16
    x = _torch.randn(2, length)
    params = _torch.tensor([[3.0, 0.0], [9.0, 2.0]], dtype=_torch.float32)

    encoded = model._encode_params(params)
    p_t = encoded[:, :, None].expand(-1, -1, length)
    manual = model._wavenet(_torch.cat([x[:, None, :], p_t], dim=1))[:, 0, :]

    assert _torch.allclose(model(x, params, pad_start=False), manual)


def test_backward_reaches_inner_wavenet():
    model = _init(_concat_wavenet_config())
    x = _torch.randn(2, model.receptive_field + 8)
    params = _torch.tensor([[4.0, 0.0], [7.0, 2.0]], dtype=_torch.float32)

    model(x, params, pad_start=False).square().mean().backward()

    assert any(
        parameter.grad is not None and _torch.count_nonzero(parameter.grad) > 0
        for parameter in model._wavenet.parameters()
    )


def test_export_weights_round_trip():
    config = _concat_wavenet_config()
    first = _init(config)
    second = _init(_concat_wavenet_config())
    first.eval()
    second.eval()
    x = _torch.randn(first.receptive_field + 24)
    params = _torch.tensor([6.0, 1.0], dtype=_torch.float32)
    assert not _torch.allclose(
        first(x, params, pad_start=False), second(x, params, pad_start=False)
    )

    weights = first._export_weights()
    end = second.import_weights(weights)

    assert end == len(weights)
    assert _torch.allclose(
        first(x, params, pad_start=False), second(x, params, pad_start=False)
    )


def test_full_export_round_trip_through_factory():
    first = _init(_concat_wavenet_config())
    first.eval()
    round_tripped = _init({**first._export_config(), "sample_rate": first.sample_rate})
    end = round_tripped.import_weights(first._export_weights())
    round_tripped.eval()

    x = _torch.randn(first.receptive_field + 32)
    params = _torch.tensor([8.0, 2.0], dtype=_torch.float32)

    assert isinstance(round_tripped, _ConcatWaveNet)
    assert end == len(first._export_weights())
    assert _torch.allclose(
        round_tripped(x, params, pad_start=False),
        first(x, params, pad_start=False),
    )


def test_export_parametric_scale_compensation(tmp_path):
    model = _init(_concat_wavenet_config())
    output_scale = 0.5

    _export_parametric(model, tmp_path, basename="unscaled")
    _export_parametric(model, tmp_path, basename="scaled", output_scale=output_scale)

    with open(tmp_path / "unscaled.nam") as fp:
        unscaled = _json.load(fp)
    with open(tmp_path / "scaled.nam") as fp:
        scaled = _json.load(fp)

    assert scaled["architecture"] == "ConcatWaveNet"
    assert unscaled["architecture"] == "ConcatWaveNet"
    assert scaled["config"]["head_scale"] == _pytest.approx(
        unscaled["config"]["head_scale"] / output_scale
    )
    assert scaled["weights"][-1] == _pytest.approx(
        unscaled["weights"][-1] / output_scale
    )
