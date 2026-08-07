import math as _math

import pytest as _pytest
import torch as _torch

from nam.models.parametric import assemble_raw_params as _assemble_raw_params
from nam.models.parametric import decode_named_params as _decode_named_params
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric import quantize_to_capture_grid as _quantize_to_capture_grid
from nam.models.parametric import split_param_indices as _split_param_indices
from nam.models.parametric import switch_combinations as _switch_combinations
from nam.models.parametric._dataset import resolve_named_params as _resolve_named_params


def _mixed_specs() -> tuple[_ParamSpec, ...]:
    return (
        _ParamSpec(name="Gain", min=0.0, max=10.0, default=5.0),
        _ParamSpec(name="Tone", min=0.0, max=10.0, default=5.0),
        _ParamSpec(
            name="Boost",
            min=0,
            max=1,
            default=0,
            type="switch",
            enum_names=("Off", "On"),
        ),
    )


def test_decode_named_params_round_trips_resolved_named_params():
    specs = _mixed_specs()
    cases = (
        {"Gain": 0.0, "Tone": 10.0, "Boost": "Off"},
        {"Gain": 4.125, "Tone": 6.5, "Boost": "On"},
        {"Gain": 9.999999, "Tone": 2.333333, "Boost": "Off"},
    )

    for named in cases:
        raw = _resolve_named_params(named, specs)
        decoded = _decode_named_params(raw, specs)
        assert tuple(decoded.keys()) == tuple(spec.name for spec in specs)
        assert decoded["Boost"] == named["Boost"]
        assert decoded["Gain"] == _pytest.approx(named["Gain"], abs=1.0e-6)
        assert decoded["Tone"] == _pytest.approx(named["Tone"], abs=1.0e-6)


def test_decode_named_params_preserves_out_of_range_continuous_values():
    specs = _mixed_specs()

    decoded = _decode_named_params(_torch.tensor([-2.0, 13.0, 0.0]), specs)

    assert decoded == {"Gain": -2.0, "Tone": 13.0, "Boost": "Off"}


def test_decode_named_params_rounds_continuous_to_display_precision():
    specs = _mixed_specs()

    # float64 input so the 7th decimal survives to exercise the round(value, 6) step.
    raw = _torch.tensor([4.1234567, 6.7654329, 1.0], dtype=_torch.float64)
    decoded = _decode_named_params(raw, specs)

    assert decoded["Gain"] == 4.123457
    assert decoded["Tone"] == 6.765433
    assert decoded["Boost"] == "On"


@_pytest.mark.parametrize("bad", (_math.inf, -_math.inf, _math.nan))
def test_decode_named_params_rejects_non_finite_continuous(bad):
    specs = _mixed_specs()

    with _pytest.raises(ValueError, match="Continuous parameter 'Gain' must be finite"):
        _decode_named_params(_torch.tensor([bad, 5.0, 0.0]), specs)


@_pytest.mark.parametrize("bad", (_math.inf, -_math.inf, _math.nan))
def test_decode_named_params_rejects_non_finite_switch(bad):
    specs = _mixed_specs()

    with _pytest.raises(ValueError, match="Switch parameter 'Boost' index must be finite"):
        _decode_named_params(_torch.tensor([5.0, 5.0, bad]), specs)


def test_decode_named_params_all_switch_specs():
    specs = (
        _ParamSpec(name="Boost", min=0, max=1, default=0, type="switch", enum_names=("Off", "On")),
        _ParamSpec(name="Voice", min=0, max=2, default=1, type="switch", enum_names=("A", "B", "C")),
    )

    assert _decode_named_params(_torch.tensor([1.0, 2.0]), specs) == {"Boost": "On", "Voice": "C"}


def test_split_param_indices_partitions_mixed_specs():
    continuous_idx, switch_idx, switch_cardinalities = _split_param_indices(_mixed_specs())

    assert continuous_idx == (0, 1)
    assert switch_idx == (2,)
    assert switch_cardinalities == (2,)


def test_switch_combinations_returns_cartesian_product():
    specs = (
        _ParamSpec(name="Gain", min=0.0, max=10.0, default=5.0),
        _ParamSpec(
            name="Boost",
            min=0,
            max=1,
            default=0,
            type="switch",
            enum_names=("Off", "On"),
        ),
        _ParamSpec(
            name="Voice",
            min=0,
            max=2,
            default=1,
            type="switch",
            enum_names=("A", "B", "C"),
        ),
    )

    combos = _switch_combinations(specs)

    assert combos == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert len(combos) == _math.prod((2, 3))


def test_switch_combinations_without_switches_returns_single_empty_combo():
    specs = (
        _ParamSpec(name="Gain", min=0.0, max=10.0, default=5.0),
        _ParamSpec(name="Tone", min=0.0, max=10.0, default=5.0),
    )

    assert _switch_combinations(specs) == [()]


def test_assemble_raw_params_supports_vector_and_batch_shapes():
    specs = _mixed_specs()

    vector = _assemble_raw_params(_torch.tensor([0.0, 1.0]), (1,), specs)
    batch = _assemble_raw_params(_torch.tensor([[0.0, 1.0], [-1.0, 2.0]]), (0,), specs)

    assert vector.shape == (3,)
    assert batch.shape == (2, 3)
    assert vector[2].item() == 1.0
    assert _torch.equal(batch[:, 2], _torch.tensor([0.0, 0.0]))
    assert _torch.all((vector[:2] >= 0.0) & (vector[:2] <= 10.0))
    assert _torch.all((batch[:, :2] >= 0.0) & (batch[:, :2] <= 10.0))


def test_assemble_raw_params_all_switch_specs_uses_empty_latents():
    specs = (
        _ParamSpec(name="Boost", min=0, max=1, default=0, type="switch", enum_names=("Off", "On")),
        _ParamSpec(name="Voice", min=0, max=2, default=1, type="switch", enum_names=("A", "B", "C")),
    )

    raw = _assemble_raw_params(_torch.zeros(0), (1, 2), specs)

    assert _torch.equal(raw, _torch.tensor([1.0, 2.0]))


def test_assemble_raw_params_preserves_autograd_for_continuous_dims_only():
    specs = _mixed_specs()
    z = _torch.tensor([[0.0, 1.0], [2.0, -1.0]], requires_grad=True)

    raw = _assemble_raw_params(z, (1,), specs)
    loss = raw.sum()
    loss.backward()

    assert z.grad is not None
    assert _torch.isfinite(z.grad).all()
    assert _torch.all(raw[:, :2] >= 0.0)
    assert _torch.all(raw[:, :2] <= 10.0)
    assert _torch.equal(raw[:, 2].detach(), _torch.tensor([1.0, 1.0]))

    z_for_switch = _torch.tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)
    switch_grad = _torch.autograd.grad(
        _assemble_raw_params(z_for_switch, (0,), specs)[:, 2].sum(), z_for_switch
    )[0]
    assert _torch.equal(switch_grad, _torch.zeros_like(z_for_switch))


@_pytest.mark.parametrize(
    "raw, match",
    [
        (_torch.tensor([[1.0, 2.0, 0.0]]), "shape \\(P,\\)"),
        (_torch.tensor([1.0, 2.0]), "length 3"),
        (_torch.tensor([1.0, 2.0, 0.5]), "must be an integer"),
        (_torch.tensor([1.0, 2.0, 3.0]), "within \\[0, 1\\]"),
    ],
)
def test_decode_named_params_validation_errors(raw, match):
    with _pytest.raises(ValueError, match=match):
        _decode_named_params(raw, _mixed_specs())


@_pytest.mark.parametrize(
    "z, combo, match",
    [
        (_torch.tensor(0.0), (0,), "shape \\(C,\\) or \\(..., C\\)"),
        (_torch.tensor([0.0]), (0,), "trailing dimension 2"),
        (_torch.tensor([0.0, 1.0]), (), "combination length 1"),
        (_torch.tensor([0.0, 1.0]), (2,), "within \\[0, 1\\]"),
    ],
)
def test_assemble_raw_params_validation_errors(z, combo, match):
    with _pytest.raises(ValueError, match=match):
        _assemble_raw_params(z, combo, _mixed_specs())


def test_quantize_to_capture_grid_snaps_continuous_and_leaves_switches():
    specs = _mixed_specs()
    raw = _torch.tensor([3.3, 6.74, 1.0])

    quantized = _quantize_to_capture_grid(raw, specs, default_step=0.5)

    assert quantized[0].item() == _pytest.approx(3.5)
    assert quantized[1].item() == _pytest.approx(6.5)
    # Switch index is untouched.
    assert quantized[2].item() == _pytest.approx(1.0)


def test_quantize_to_capture_grid_is_idempotent():
    specs = _mixed_specs()
    raw = _torch.tensor([3.3, 6.74, 0.0])

    once = _quantize_to_capture_grid(raw, specs)
    twice = _quantize_to_capture_grid(once, specs)

    assert _torch.allclose(once, twice)


def test_quantize_to_capture_grid_clamps_into_spec_range():
    specs = _mixed_specs()
    raw = _torch.tensor([-2.0, 12.4, 1.0])

    quantized = _quantize_to_capture_grid(raw, specs, default_step=0.5)

    assert quantized[0].item() == _pytest.approx(0.0)
    assert quantized[1].item() == _pytest.approx(10.0)


def test_quantize_to_capture_grid_decodes_to_grid_aligned_user_values():
    specs = _mixed_specs()
    raw = _torch.tensor([3.3, 6.74, 1.0])

    decoded = _decode_named_params(_quantize_to_capture_grid(raw, specs), specs)

    assert decoded["Gain"] == _pytest.approx(3.5)
    assert decoded["Tone"] == _pytest.approx(6.5)
    assert decoded["Boost"] == "On"


def test_quantize_to_capture_grid_accepts_python_sequences():
    specs = _mixed_specs()

    quantized = _quantize_to_capture_grid([3.3, 6.74, 0.0], specs)

    assert quantized[0].item() == _pytest.approx(3.5)


@_pytest.mark.parametrize("bad_step", [0.0, -0.5, _math.inf, _math.nan])
def test_quantize_to_capture_grid_rejects_bad_default_step(bad_step):
    specs = _mixed_specs()
    with _pytest.raises(ValueError):
        _quantize_to_capture_grid(_torch.tensor([1.0, 2.0, 0.0]), specs, default_step=bad_step)


def test_quantize_to_capture_grid_rejects_wrong_length():
    specs = _mixed_specs()
    with _pytest.raises(ValueError):
        _quantize_to_capture_grid(_torch.tensor([1.0, 2.0]), specs)
