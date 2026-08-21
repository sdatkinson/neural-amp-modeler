import json as _json
from typing import cast as _cast

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.models import factory as _factory
from nam.models.parametric import ConcatLSTM as _ConcatLSTM
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric import export_parametric as _export_parametric


def _concat_lstm_config() -> dict:
    return {
        "hidden_size": 4,
        "num_layers": 2,
        "train_burn_in": 8,
        "train_truncate": 8,
        "sample_rate": 48_000.0,
        "params": [
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
        ],
    }


def _param_specs() -> tuple[_ParamSpec, ...]:
    return tuple(_ParamSpec.from_dict(spec) for spec in _concat_lstm_config()["params"])


def _init(config: dict) -> _ConcatLSTM:
    return _cast(_ConcatLSTM, _factory.init("ConcatLSTM", args=(config,)))


def test_export_weights_round_trip():
    # Unlike ConcatWaveNet, ConcatLSTM is stateful: forward output depends on each model's
    # own `_initial_hidden`/`_initial_cell`, and import intentionally overwrites those with
    # the exported burnt-in-on-silence state rather than the donor's learned parameters (see
    # `_get_export_initial_state`). So a plain "run both models on the same input" comparison
    # isn't the right invariant here; instead verify the core/head weights transferred
    # correctly (forward equivalence from an explicit shared state) and that the initial
    # state was set to the expected burnt-in value.
    _torch.manual_seed(0)
    first = _init(_concat_lstm_config())
    second = _init(_concat_lstm_config())
    first.eval()
    second.eval()

    weights = first._export_weights()
    end = second.import_weights(weights)
    assert end == len(weights)

    x = _torch.randn(1, 40, first._core.input_size)
    h0 = _torch.randn(first._core.num_layers, 1, first._core.hidden_size)
    c0 = _torch.randn(first._core.num_layers, 1, first._core.hidden_size)
    y1, _ = first._process_in_blocks(x, (h0, c0))
    y2, _ = second._process_in_blocks(x, (h0.clone(), c0.clone()))
    assert _torch.allclose(first._apply_head(y1), second._apply_head(y2), atol=1e-6)

    expected_hidden, expected_cell = first._get_export_initial_state()
    assert _torch.allclose(second._initial_hidden, expected_hidden[:, 0, :])
    assert _torch.allclose(second._initial_cell, expected_cell[:, 0, :])


def test_import_weights_nonzero_offset():
    # ParametricNet.import_weights(weights, i) must be able to read a weight blob starting
    # at an arbitrary offset (the contract that lets a caller pack several submodels' weights
    # into one buffer) and return the offset just past what it consumed. Prepend a stand-in
    # prefix for "another submodel's weights" and import starting after it.
    _torch.manual_seed(0)
    first = _init(_concat_lstm_config())
    second = _init(_concat_lstm_config())
    first.eval()
    second.eval()

    prefix = _np.array([123.456, -7.0, 42.0])
    weights = first._export_weights()
    packed = _np.concatenate([prefix, weights])

    end = second.import_weights(packed, i=len(prefix))
    assert end == len(packed)

    x = _torch.randn(1, 40, first._core.input_size)
    h0 = _torch.randn(first._core.num_layers, 1, first._core.hidden_size)
    c0 = _torch.randn(first._core.num_layers, 1, first._core.hidden_size)
    y1, _ = first._process_in_blocks(x, (h0, c0))
    y2, _ = second._process_in_blocks(x, (h0.clone(), c0.clone()))
    assert _torch.allclose(first._apply_head(y1), second._apply_head(y2), atol=1e-6)


def test_full_export_round_trip_through_factory():
    _torch.manual_seed(0)
    first = _init(_concat_lstm_config())
    first.eval()
    round_tripped = _init({**first._export_config(), "sample_rate": first.sample_rate})
    end = round_tripped.import_weights(first._export_weights())
    round_tripped.eval()

    assert isinstance(round_tripped, _ConcatLSTM)
    assert end == len(first._export_weights())
    # A fixed point: re-exporting the round-tripped model (whose core weights and burnt-in
    # initial state both derive from `first`) reproduces the exact same weights blob.
    assert _np.allclose(round_tripped._export_weights(), first._export_weights())


def test_export_parametric_scale_compensation(tmp_path):
    _torch.manual_seed(0)
    model = _init(_concat_lstm_config())
    output_scale = 0.5

    _export_parametric(model, tmp_path, basename="unscaled")
    _export_parametric(model, tmp_path, basename="scaled", output_scale=output_scale)

    with open(tmp_path / "unscaled.nam") as fp:
        unscaled = _json.load(fp)
    with open(tmp_path / "scaled.nam") as fp:
        scaled = _json.load(fp)

    assert scaled["architecture"] == "ConcatLSTM"
    assert unscaled["architecture"] == "ConcatLSTM"

    head_size = model._head.weight.numel() + model._head.bias.numel()
    scaled_head = scaled["weights"][-head_size:]
    unscaled_head = unscaled["weights"][-head_size:]
    for scaled_value, unscaled_value in zip(scaled_head, unscaled_head):
        assert scaled_value == _pytest.approx(unscaled_value / output_scale)
    # Everything before the head (cell weights/biases/initial state) is untouched.
    assert scaled["weights"][:-head_size] == unscaled["weights"][:-head_size]
