import json as _json
import random as _random
from collections.abc import Mapping as _Mapping
from contextlib import contextmanager as _contextmanager
from copy import deepcopy as _deepcopy
from functools import lru_cache as _lru_cache
from typing import cast as _cast

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.models import factory as _factory
from nam.models.parametric import Hypernetwork as _Hypernetwork
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.wavenet._wavenet import WaveNet as _InnerWaveNet


@_lru_cache(maxsize=1)
def _channels_8_wavenet_config() -> dict:
    with open("nam/train/_resources/config_model_packed.json") as fp:
        packed_config = _json.load(fp)
    return _deepcopy(
        next(
            submodel
            for submodel in packed_config["net"]["config"]["submodels"]
            if submodel["name"] == "channels_8"
        )["config"]
    )


def _hyperwavenet_config(
    *,
    hypernet: _Mapping[str, object] | None = None,
    slimmable: bool = False,
) -> dict:
    config = _InnerWaveNet.init_from_config(_channels_8_wavenet_config()).export_config()
    if slimmable:
        config = _InnerWaveNet.init_from_config(
            {
                "layers_configs": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                        "channels": 4,
                        "kernel_size": 2,
                        "dilations": [1, 2],
                        "activation": "Tanh",
                        "slimmable": {
                            "method": "slice_channels_uniform",
                            "kwargs": {},
                        },
                    }
                ],
                "head_scale": 1.0,
            }
        ).export_config()
    config["sample_rate"] = 48_000.0
    config["params"] = [
        {
            "name": "gain",
            "min": 0.0,
            "max": 10.0,
            "default": 5.0,
            "type": "continuous",
        },
        {
            "name": "mode",
            "min": 0,
            "max": 2,
            "default": 1,
            "type": "switch",
            "enum_names": ["clean", "crunch", "lead"],
        },
    ]
    if hypernet is not None:
        config["hypernet"] = dict(hypernet)
    return config


def _tiny_wavenet_internal_config() -> dict:
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


def _hyperwavenet_config_with_condition_dsp() -> dict:
    condition_dsp = _InnerWaveNet.init_from_config(_tiny_wavenet_internal_config())
    config = _hyperwavenet_config()
    config["condition_dsp"] = {
        "architecture": "WaveNet",
        "config": condition_dsp.export_config(),
        "weights": condition_dsp.export_weights().tolist(),
    }
    return config


def test_factory_init_registers_hyperwavenet_and_injects_default_selector():
    model = _factory.init("HyperWaveNet", args=(_hyperwavenet_config(),))

    assert isinstance(model, _HyperWaveNet)
    assert model.receptive_field == 6347
    assert model._hypernet.config["selector"] == {"exclude_suffixes": ["_conv.weight"]}
    assert len(model._hypernet.target_names) == 95


def test_condition_dsp_config_is_rejected():
    with _pytest.raises(NotImplementedError, match="condition_dsp"):
        _HyperWaveNet.init_from_config(_hyperwavenet_config_with_condition_dsp())


def test_condition_dsp_template_is_rejected():
    template = _InnerWaveNet.init_from_config(
        {
            **_tiny_wavenet_internal_config(),
            "condition_dsp": {
                "name": "WaveNet",
                "config": _tiny_wavenet_internal_config(),
            },
        }
    )
    param_specs = tuple(
        _ParamSpec.from_dict(spec) for spec in _hyperwavenet_config()["params"]
    )
    hypernet = _Hypernetwork.from_config(
        input_dim=sum(spec.num_inputs for spec in param_specs),
        named_shapes={
            name: parameter.shape for name, parameter in template.named_parameters()
        },
        config={"selector": {"exclude_suffixes": ["_conv.weight"]}},
    )

    with _pytest.raises(NotImplementedError, match="condition_dsp"):
        _HyperWaveNet(
            template=template,
            hypernet=hypernet,
            param_specs=param_specs,
            sample_rate=48_000.0,
        )


def test_zero_init_matches_bare_template_for_shared_and_batched_params():
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    x = _torch.randn(2, model.receptive_field + 32)
    params = _torch.tensor([[2.5, 0.0], [9.0, 2.0]], dtype=_torch.float32)

    actual = model(x, params, pad_start=False)
    expected = model._template(x[:, None, :])[:, 0, :]

    assert _torch.allclose(actual, expected)


def test_zero_init_matches_bare_template_for_broadcast_input():
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    x = _torch.randn(model.receptive_field + 16)
    params = _torch.tensor([[1.0, 0.0], [8.0, 1.0], [11.0, 2.0]], dtype=_torch.float32)

    actual = model(x, params, pad_start=False)
    expected = model._template(x.repeat(3, 1)[:, None, :])[:, 0, :]

    assert actual.shape == expected.shape == (3, 17)
    assert _torch.allclose(actual, expected)


def test_backward_reaches_base_template_and_hypernetwork():
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    x = _torch.randn(2, model.receptive_field + 8)
    params = _torch.tensor([[4.0, 0.0], [7.0, 2.0]], dtype=_torch.float32)

    loss = model(x, params, pad_start=False).square().mean()
    loss.backward()

    assert any(
        parameter.grad is not None and _torch.count_nonzero(parameter.grad) > 0
        for parameter in model._template.parameters()
    )
    assert any(
        parameter.grad is not None and _torch.count_nonzero(parameter.grad) > 0
        for parameter in model._hypernet.parameters()
    )


def test_batched_duplicate_params_share_conditioned_forward_calls():
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    x = _torch.randn(5, model.receptive_field + 8)
    params = _torch.tensor(
        [
            [4.0, 0.0],
            [4.0, 0.0],
            [7.0, 2.0],
            [1.0, 1.0],
            [7.0, 2.0],
        ],
        dtype=_torch.float32,
    )
    batch_sizes = []
    real_apply = model._apply_conditioned_weights

    def _counting_apply(
        x: _torch.Tensor, weight_dict: dict[str, _torch.Tensor]
    ) -> _torch.Tensor:
        batch_sizes.append(x.shape[0])
        return real_apply(x, weight_dict)

    setattr(model, "_apply_conditioned_weights", _counting_apply)

    y = model(x, params, pad_start=False)

    assert y.shape == (5, 9)
    assert sorted(batch_sizes) == [1, 2, 2]


def test_slimmable_batch_draws_one_random_width_per_forward():
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config(slimmable=True))
    x = _torch.randn(3, model.receptive_field + 8)
    params = _torch.tensor([[4.0, 0.0], [7.0, 2.0], [1.0, 1.0]], dtype=_torch.float32)
    entered = 0

    @_contextmanager
    def _counting_context():
        nonlocal entered
        entered += 1
        yield

    model.train()
    model._template.context_adjust_to_random = _counting_context

    y = model(x, params, pad_start=False)

    assert y.shape == (3, 9)
    assert entered == 1


def test_slimmable_real_forward_slices_channels_and_stays_trainable():
    # The plumbing test above mocks out `context_adjust_to_random`; this one runs the REAL
    # slimmable path so channel slicing actually flows through functional_call.
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config(slimmable=True))
    assert model._template.is_slimmable()
    x = _torch.randn(3, model.receptive_field + 16)
    params = _torch.tensor([[4.0, 0.0], [7.0, 2.0], [1.0, 1.0]], dtype=_torch.float32)

    # Training mode: real channel slicing, finite output, and width varies per forward.
    model.train()
    _random.seed(0)
    y0 = model(x, params, pad_start=False)
    assert y0.shape == (3, 17)
    assert _torch.isfinite(y0).all()
    _random.seed(1)
    assert not _torch.allclose(y0, model(x, params, pad_start=False))

    # Eval pins the full width, so it is deterministic.
    model.eval()
    assert _torch.allclose(
        model(x, params, pad_start=False), model(x, params, pad_start=False)
    )

    # A nonzero hypernetwork still receives gradients under slimming, and its generated
    # delta survives the channel slice (the slimmed output depends on the params).
    model.train()
    with _torch.no_grad():
        for parameter in model._hypernet.parameters():
            parameter.add_(0.01 * _torch.randn_like(parameter))
    _random.seed(2)
    model(x, params, pad_start=False).square().mean().backward()
    assert any(
        parameter.grad is not None and _torch.count_nonzero(parameter.grad) > 0
        for parameter in model._hypernet.parameters()
    )
    # Same seed => same random width, so params are the only difference between the two runs.
    _random.seed(3)
    low = model(x, _torch.zeros_like(params), pad_start=False)
    _random.seed(3)
    high = model(x, _torch.full_like(params, 2.0), pad_start=False)
    assert not _torch.allclose(low, high)


def test_slimmable_draws_one_width_across_mps_fallback_segments():
    # With BaseNet's MPS >65,536 stitching fallback forced on, a long input is processed in
    # several temporal segments. The slimming width must be drawn ONCE for the whole logical
    # forward (in `_forward_mps_safe`), not once per stitched segment.
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config(slimmable=True))
    model.train()
    model._mps_65536_fallback = True  # force the CPU stitching path
    x = _torch.randn(2, 65_600)  # > 65,536 => stitched into multiple segments
    params = _torch.tensor([[4.0, 0.0], [7.0, 2.0]], dtype=_torch.float32)

    entered = 0
    forward_calls = 0
    real_forward = model._forward

    @_contextmanager
    def _counting_context():
        nonlocal entered
        entered += 1
        yield

    def _counting_forward(x: _torch.Tensor, *, params: _torch.Tensor) -> _torch.Tensor:
        nonlocal forward_calls
        forward_calls += 1
        return real_forward(x, params=params)

    model._template.context_adjust_to_random = _counting_context
    setattr(model, "_forward", _counting_forward)

    y = model(x, params, pad_start=False)

    assert _torch.isfinite(y).all()
    assert forward_calls > 1  # the input really was stitched into multiple segments
    assert entered == 1  # but only one slimming width spanned the whole forward


def test_import_weights_accepts_stock_wavenet_blob_for_base_seed_only():
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    stock = _InnerWaveNet.init_from_config(_channels_8_wavenet_config())
    stock_weights = stock.export_weights()
    hypernet_before = {
        name: tensor.detach().clone()
        for name, tensor in model._hypernet.state_dict().items()
    }

    end = model.import_weights(stock_weights)

    assert end == len(stock_weights)
    assert _np.allclose(model._template.export_weights(), stock_weights)
    for name, tensor in model._hypernet.state_dict().items():
        assert _torch.equal(tensor, hypernet_before[name])


def test_imported_stock_head_scale_stays_in_sync_for_export():
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    stock_config = _channels_8_wavenet_config()
    stock_config["head_scale"] = 0.5
    stock = _InnerWaveNet.init_from_config(stock_config)
    stock_weights = stock.export_weights()

    model.import_weights(stock_weights)
    exported = model._export_inner_config()

    assert _np.isclose(exported["head_scale"], 0.5)
    assert _np.isclose(exported["head_scale"], model._export_weights()[len(stock_weights) - 1])


def test_export_weights_round_trip_restores_base_and_hypernetwork():
    config = _hyperwavenet_config(hypernet={"hidden_sizes": [5]})
    first = _HyperWaveNet.init_from_config(config)
    second = _HyperWaveNet.init_from_config(config)
    with _torch.no_grad():
        _cast(_torch.Tensor, first._hypernet._low_rank_anchor).add_(0.125)
    first_weights = first._export_weights()

    end = second.import_weights(first_weights)

    assert end == len(first_weights)
    assert _np.allclose(
        second._template.export_weights(), first._template.export_weights()
    )
    for (first_name, first_tensor), (second_name, second_tensor) in zip(
        first._hypernet.state_dict().items(),
        second._hypernet.state_dict().items(),
    ):
        assert first_name == second_name
        assert _torch.equal(first_tensor, second_tensor)


def _hyperwavenet_with_nonzero_deltas() -> _HyperWaveNet:
    # The readout is zero-initialized, so every delta starts at exactly zero and any two code
    # paths would agree trivially. Push it off zero first.
    model = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    with _torch.no_grad():
        model._hypernet._final.weight.normal_(0.0, 0.05)
        model._hypernet._final.bias.normal_(0.0, 0.05)
    return model


def _capture_pure_batch(model: _HyperWaveNet, batch_size: int):
    x = _torch.randn(batch_size, model.receptive_field - 1 + 64)
    params = _torch.tensor([[7.5, 2.0]]).expand(batch_size, -1).contiguous()
    return x, params


def test_uniform_batch_params_matches_the_general_grouping_path():
    _torch.manual_seed(0)
    model = _hyperwavenet_with_nonzero_deltas()
    x, params = _capture_pure_batch(model, 4)

    with _torch.no_grad():
        model.set_uniform_batch_params(False)
        general = model(x, params, pad_start=False)
        model.set_uniform_batch_params(True)
        uniform = model(x, params, pad_start=False)
        single = model(x, params[0], pad_start=False)

    assert _torch.equal(uniform, general)
    assert _torch.equal(uniform, single)


def test_uniform_batch_params_rejects_a_mixed_batch():
    _torch.manual_seed(0)
    model = _hyperwavenet_with_nonzero_deltas()
    x, _ = _capture_pure_batch(model, 4)
    mixed = _torch.tensor([[1.0, 0.0], [9.0, 2.0], [4.0, 1.0], [2.0, 0.0]])

    model.set_uniform_batch_params(True)
    with _pytest.raises(RuntimeError, match="mixed settings"):
        with _torch.no_grad():
            model(x, mixed, pad_start=False)


def test_set_compiled_round_trips_to_the_eager_path():
    _torch.manual_seed(0)
    model = _hyperwavenet_with_nonzero_deltas()
    x, params = _capture_pure_batch(model, 2)
    model.set_uniform_batch_params(True)

    eager = model(x, params, pad_start=False)
    # backend="eager" exercises the routing without paying for an inductor compile.
    model.set_compiled(True, backend="eager")
    compiled = model(x, params, pad_start=False)
    model.set_compiled(False)
    restored = model(x, params, pad_start=False)

    assert model._compiled_conditioned_forward is None
    assert _torch.allclose(compiled, eager, atol=1e-6)
    assert _torch.equal(restored, eager)


def test_compiled_path_is_skipped_when_grad_is_disabled():
    # Inductor cannot lower the template's convs with grad off, so validation must stay eager.
    _torch.manual_seed(0)
    model = _hyperwavenet_with_nonzero_deltas()
    x, params = _capture_pure_batch(model, 2)
    model.set_uniform_batch_params(True)
    model.set_compiled(True, backend="eager")

    calls = []
    compiled = model._compiled_conditioned_forward

    def _counting(*args, **kwargs):
        calls.append(1)
        return _cast(object, compiled)(*args, **kwargs)

    model._compiled_conditioned_forward = _counting
    with _torch.no_grad():
        model(x, params, pad_start=False)
    assert calls == []

    model(x, params, pad_start=False)
    assert calls == [1]
