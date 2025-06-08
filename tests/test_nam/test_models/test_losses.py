# File: test_losses.py
# Created Date: Saturday January 28th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest as _pytest
import torch as _torch
import torch.nn as _nn

from nam.models import losses as _losses

from ..._skips import requires_mps as _requires_mps


@_pytest.mark.parametrize(
    "x,coef,y_expected",
    (
        (_torch.Tensor([0.0, 1.0, 2.0]), 1.0, _torch.Tensor([1.0, 1.0])),
        (_torch.Tensor([0.0, 1.0, 2.0]), 0.5, _torch.Tensor([1.0, 1.5])),
        (
            _torch.Tensor([[0.0, 1.0, 0.0], [1.0, 1.5, 2.0]]),
            0.5,
            _torch.Tensor([[1.0, -0.5], [1.0, 1.25]]),
        ),
    ),
)
def test_apply_pre_emphasis_filter_1d(
    x: _torch.Tensor, coef: float, y_expected: _torch.Tensor
):
    y_actual = _losses.apply_pre_emphasis_filter(x, coef)
    assert isinstance(y_actual, _torch.Tensor)
    assert y_actual.ndim == y_expected.ndim
    assert y_actual.shape == y_expected.shape
    assert _torch.allclose(y_actual, y_expected)


def test_esr():
    """
    Is the ESR calculation correct?
    """

    class Model(_nn.Module):
        def forward(self, x):
            return x

    batch_size, input_length = 3, 5
    inputs = (
        _torch.linspace(0.1, 1.0, batch_size)[:, None]
        * _torch.full((input_length,), 1.0)[None, :]
    )  # (batch_size, input_length)
    target_factor = _torch.linspace(0.37, 1.22, batch_size)
    targets = target_factor[:, None] * inputs  # (batch_size, input_length)
    # Do the algebra:
    # y=a*yhat
    # ESR=(y-yhat)^2 / y^2
    # ...
    # =(1/a-1)^2
    expected_esr = _torch.square(1.0 / target_factor - 1.0).mean()
    model = Model()
    preds = model(inputs)
    actual_esr = _losses.esr(preds, targets)
    assert _torch.allclose(actual_esr, expected_esr)


@_requires_mps
def test_mrstft_loss_doesnt_fall_back_to_cpu():
    preds, targets = _torch.randn((2, 2048))
    # Implicitly assert that this doesn't raise an exception that's usually caught and
    # fallback
    # If this raises a NotImplementedError, then that's what we don't want!
    _losses.multi_resolution_stft_loss(preds, targets, device="mps")


if __name__ == "__main__":
    _pytest.main()
