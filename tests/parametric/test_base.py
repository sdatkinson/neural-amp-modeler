from typing import Optional as _Optional

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric._base import ParametricNet as _ParametricNet


class _DummyParametricNet(_ParametricNet):
    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return 1

    def _run_conditioned(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        if p.ndim == 1:
            return x + p.sum()
        return x + p.sum(dim=-1, keepdim=True)

    def _export_inner_config(self) -> dict[str, bool]:
        return {"dummy": True}

    def _export_weights(self) -> _np.ndarray:
        return _np.zeros((0,), dtype=_np.float32)

    def import_weights(self, weights, i: int = 0) -> int:
        return i


def _make_model(sample_rate: _Optional[float] = None) -> _DummyParametricNet:
    return _DummyParametricNet(
        param_specs=(
            _ParamSpec(name="gain", min=0.0, max=10.0, default=5.0),
            _ParamSpec(
                name="mode",
                min=0,
                max=2,
                default=1,
                type="switch",
                enum_names=("clean", "crunch", "lead"),
            ),
        ),
        sample_rate=sample_rate,
    )


def test_forward_accepts_scalar_x_and_vector_params():
    model = _make_model()
    x = _torch.tensor([0.0, 1.0, 2.0], dtype=_torch.float32)
    params = _torch.tensor([5.0, 2.0], dtype=_torch.float32)

    y = model(x, params)

    encoded = model._encode_params(params)
    assert y.shape == x.shape
    assert _torch.equal(y, x + encoded.sum())


def test_forward_accepts_batched_params():
    model = _make_model()
    x = _torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=_torch.float32)
    params = _torch.tensor([[5.0, 0.0], [10.0, 2.0]], dtype=_torch.float32)

    y = model(x, params)

    encoded = model._encode_params(params)
    expected = x + encoded.sum(dim=-1, keepdim=True)
    assert y.shape == x.shape
    assert _torch.equal(y, expected)


def test_forward_broadcasts_scalar_input_across_batched_params():
    model = _make_model()
    x = _torch.tensor([0.0, 1.0, 2.0], dtype=_torch.float32)
    params = _torch.tensor([[5.0, 0.0], [10.0, 2.0]], dtype=_torch.float32)

    y = model(x, params)

    encoded = model._encode_params(params)
    expected = x[None, :] + encoded.sum(dim=-1, keepdim=True)
    assert y.shape == (2, 3)
    assert _torch.equal(y, expected)


def test_forward_rejects_mismatched_batched_params():
    model = _make_model()
    x = _torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=_torch.float32)
    params = _torch.tensor([[5.0, 0.0]], dtype=_torch.float32)

    with _pytest.raises(ValueError, match="Input batch size 2 must match params batch size 1"):
        model(x, params)


def test_encode_params_mixed_continuous_and_switch():
    model = _make_model()
    params = _torch.tensor(
        [
            [0.0, 0.0],
            [5.0, 1.0],
            [15.0, 2.0],
        ],
        dtype=_torch.float32,
    )

    encoded = model._encode_params(params)

    assert model.param_specs == _make_model().param_specs
    assert model.param_names == ("gain", "mode")
    assert model.param_dim == 2
    assert model.encoded_param_dim == 4
    assert _torch.allclose(
        encoded,
        _torch.tensor(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 1.0],
            ],
            dtype=_torch.float32,
        ),
    )


def test_export_config_appends_params():
    model = _make_model()

    config = model._export_config()

    assert config == {
        "dummy": True,
        "params": [spec.to_dict() for spec in model.param_specs],
    }


def test_at_nominal_settings_uses_default_params():
    model = _make_model()
    x = _torch.tensor([0.25, -0.5, 0.75], dtype=_torch.float32)

    # nominal_params are the RAW defaults (gain=5, mode index=1), not the encoded vector.
    assert _torch.equal(model.nominal_params, _torch.tensor([5.0, 1.0]))

    # Those defaults must flow THROUGH encoding: gain=5 -> 0.0 (midpoint), mode=1 ->
    # one-hot [0, 1, 0]; the dummy adds the encoded sum (= 1.0) to x. Asserting the
    # concrete value proves the defaults are encoded, not just that the method calls itself.
    actual = model._at_nominal_settings(x)
    assert _torch.allclose(actual, x + 1.0)


def test_init_rejects_empty_and_duplicate_specs():
    with _pytest.raises(ValueError):
        _DummyParametricNet(param_specs=())

    with _pytest.raises(ValueError):
        _DummyParametricNet(
            param_specs=(
                _ParamSpec(name="gain", min=0.0, max=10.0, default=5.0),
                _ParamSpec(name="gain", min=0.0, max=1.0, default=0.5),
            )
        )


def test_encode_params_rejects_bad_shapes():
    model = _make_model()

    with _pytest.raises(ValueError):
        model._encode_params(_torch.zeros(3))  # trailing dim 3 != param_dim 2

    with _pytest.raises(ValueError):
        model._encode_params(_torch.zeros(1, 1, 2))  # 3-D not allowed


def test_encode_params_rejects_out_of_range_switch():
    model = _make_model()

    # mode has 3 states (indices 0..2); 9 is out of range.
    with _pytest.raises(ValueError):
        model._encode_params(_torch.tensor([5.0, 9.0]))


def test_encode_params_rejects_non_integer_switch():
    model = _make_model()

    with _pytest.raises(ValueError, match="must be an integer"):
        model._encode_params(_torch.tensor([5.0, 1.5]))


def test_export_input_output_runs_at_nominal_and_restores_mode():
    model = _make_model(sample_rate=8.0)
    model.train()

    x, y = model._export_input_output()

    assert x.shape == y.shape
    assert x.ndim == 1 and x.shape[0] == 3 * 8  # silence + 220Hz tone + silence
    assert _np.isfinite(y).all()
    # nominal encoding sum is 1.0 (gain midpoint -> 0.0, switch one-hot -> 1.0)
    assert _np.allclose(y, x + 1.0)
    # train/eval mode must be restored after export
    assert model.training is True


def test_export_input_output_requires_sample_rate():
    model = _make_model()  # no sample rate
    with _pytest.raises(RuntimeError):
        model._export_input_output()


def test_normalization_buffers_are_private_and_exposed_via_property():
    model = _make_model()

    keys = set(model.state_dict().keys())
    assert {"_param_mins", "_param_maxs", "_nominal_params"} <= keys
    # The un-prefixed names must NOT leak into the state_dict (checkpoint compat surface).
    assert not {"param_mins", "param_maxs", "nominal_params"} & keys

    # Public read access is via the property, returning the RAW defaults.
    assert _torch.equal(model.nominal_params, _torch.tensor([5.0, 1.0]))
