# File: test_activations.py
# Tests for get_activation() and activation modules in nam.models._activations

import pytest as _pytest
import torch as _torch

from nam.models._activations import (
    PairingActivation,
    get_activation,
)

# -----------------------------------------------------------------------------
# PyTorch core library activations
# -----------------------------------------------------------------------------


@_pytest.mark.parametrize(
    "name",
    [
        "ReLU",
        "Sigmoid",
        "Tanh",
        "LeakyReLU",
        "GELU",
        "ELU",
        "SiLU",
    ],
)
def test_get_activation_pytorch_core(name: str) -> None:
    """get_activation(name) returns the corresponding _torch.nn module."""
    act = get_activation(name)
    assert isinstance(act, _torch.nn.Module)
    x = _torch.randn(2, 4)
    y = act(x)
    # Basic activations preserve shape and dtype
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_get_activation_with_kwargs() -> None:
    """PyTorch activations that accept kwargs are passed through."""
    act = get_activation("LeakyReLU", negative_slope=0.2)
    assert isinstance(act, _torch.nn.LeakyReLU)
    assert act.negative_slope == _pytest.approx(0.2)


# -----------------------------------------------------------------------------
# Special NAM activations (Softsigmoid, Softsign)
# -----------------------------------------------------------------------------


@_pytest.mark.parametrize("name", ["LeakyHardtanh", "Softsigmoid", "Softsign"])
def test_get_activation_nam_special(name: str) -> None:
    """get_activation returns NAM-specific activation modules."""
    act = get_activation(name)
    assert isinstance(act, _torch.nn.Module)
    x = _torch.randn(2, 4)
    y = act(x)
    # These happen to preserve shape and dtype
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_softsign_bounds() -> None:
    """Softsign maps R to (-1, 1)."""
    act = get_activation("Softsign")
    for v in [-100.0, 0.0, 100.0]:
        x = _torch.tensor([[v]])
        y = act(x)
        assert -1.0 < y.item() < 1.0
    x = _torch.tensor([[0.0]])
    assert act(x).item() == _pytest.approx(0.0)


def test_softsigmoid_bounds() -> None:
    """Softsigmoid maps R to (0, 1)."""
    act = get_activation("Softsigmoid")
    for v in [-100.0, 0.0, 100.0]:
        x = _torch.tensor([[v]])
        y = act(x)
        assert 0.0 < y.item() < 1.0
    x = _torch.tensor([[0.0]])
    assert act(x).item() == _pytest.approx(0.5)


# -----------------------------------------------------------------------------
# LeakyHardtanh (matches NeuralAmpModelerCore leaky_hardtanh)
# -----------------------------------------------------------------------------


def test_leaky_hardtanh_forward() -> None:
    """LeakyHardtanh: identity in [min_val, max_val], linear slopes outside."""
    act = get_activation("LeakyHardtanh")  # defaults: min=-1, max=1, slopes=0.01
    # In range: pass through
    x_mid = _torch.tensor([[0.0, -0.5, 1.0]])
    assert (act(x_mid) == x_mid).all().item()
    # Below min_val: (x - min_val) * min_slope + min_val. For x=-2: (-2 - (-1))*0.01 + (-1) = -1.01
    x_low = _torch.tensor([[-2.0]])
    assert act(x_low).item() == _pytest.approx(-1.01)
    # Above max_val: (x - max_val) * max_slope + max_val. For x=2: (2 - 1)*0.01 + 1 = 1.01
    x_high = _torch.tensor([[2.0]])
    assert act(x_high).item() == _pytest.approx(1.01)


def test_leaky_hardtanh_with_kwargs() -> None:
    """LeakyHardtanh accepts min_val, max_val, min_slope, max_slope."""
    act = get_activation(
        "LeakyHardtanh",
        min_val=-2.0,
        max_val=2.0,
        min_slope=0.1,
        max_slope=0.1,
    )
    # Below -2: (x - min_val)*min_slope + min_val = (-3 - (-2))*0.1 + (-2) = -2.1
    assert act(_torch.tensor([[-3.0]])).item() == _pytest.approx(-2.1)
    # Above 2: (x - max_val)*max_slope + max_val = (3 - 2)*0.1 + 2 = 2.1
    assert act(_torch.tensor([[3.0]])).item() == _pytest.approx(2.1)
    # In range [-2, 2]: pass through
    y = act(_torch.tensor([[-2.0, 0.0, 2.0]]))
    assert y[0, 0].item() == _pytest.approx(-2.0)
    assert y[0, 1].item() == _pytest.approx(0.0)
    assert y[0, 2].item() == _pytest.approx(2.0)


# -----------------------------------------------------------------------------
# Pairing activations (PairMultiply, PairBlend)
# -----------------------------------------------------------------------------


@_pytest.mark.parametrize("pairing_name", ["PairMultiply", "PairBlend"])
def test_get_activation_pairing(pairing_name: str) -> None:
    """get_activation(name, primary=..., secondary=...) returns a PairingActivation."""
    act = get_activation(
        pairing_name,
        primary="ReLU",
        secondary="Sigmoid",
    )
    assert isinstance(act, PairingActivation)
    # Input has 2D channels; output has D channels.
    x = _torch.randn(2, 8)
    y = act(x)
    assert y.shape == (2, 4)


def test_pair_multiply_forward() -> None:
    """PairMultiply: out = primary(x1) * secondary(x2)."""
    act = get_activation(
        "PairMultiply",
        primary="ReLU",
        secondary="Sigmoid",
    )
    # x1 = [2, -1], x2 = [0, 0] -> sigmoid(0)=0.5 each
    # relu(2)=2, relu(-1)=0 -> 2*0.5=1, 0*0.5=0
    x = _torch.tensor([[2.0, -1.0, 0.0, 0.0]])
    y = act(x)
    assert y.shape == (1, 2)
    assert y[0, 0].item() == _pytest.approx(1.0)
    assert y[0, 1].item() == _pytest.approx(0.0)


def test_pair_blend_forward() -> None:
    """PairBlend: out = blend * primary(x1) + (1 - blend) * x1."""
    act = get_activation(
        "PairBlend",
        primary="ReLU",
        secondary="Sigmoid",
    )
    # x1 = [1, 1], x2 = [0, 0] -> blend = sigmoid(0) = 0.5
    # out = 0.5 * relu(1) + 0.5 * 1 = 0.5 + 0.5 = 1
    x = _torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    y = act(x)
    assert y.shape == (1, 2)
    assert y[0, 0].item() == _pytest.approx(1.0)
    assert y[0, 1].item() == _pytest.approx(1.0)


def test_pairing_with_nam_activations() -> None:
    """Pairing activations can use NAM special activations as primary/secondary."""
    act = get_activation(
        "PairMultiply",
        primary="Softsign",
        secondary="Softsigmoid",
    )
    assert isinstance(act, PairingActivation)
    x = _torch.randn(3, 6)
    y = act(x)
    assert y.shape == (3, 3)


def test_pairing_odd_channels_raises() -> None:
    """Pairing activations expect even number of channels; odd raises on split."""
    act = get_activation("PairMultiply", primary="ReLU", secondary="Sigmoid")
    x = _torch.randn(2, 7)  # 7 is odd -> split gives 3 tensors, unpack fails
    with _pytest.raises(ValueError):
        act(x)
