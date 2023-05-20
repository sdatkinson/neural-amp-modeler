# File: test_losses.py
# Created Date: Saturday January 28th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest
import torch
import torch.nn as nn

from nam.models import losses


@pytest.mark.parametrize(
    "x,coef,y_expected",
    (
        (torch.Tensor([0.0, 1.0, 2.0]), 1.0, torch.Tensor([1.0, 1.0])),
        (torch.Tensor([0.0, 1.0, 2.0]), 0.5, torch.Tensor([1.0, 1.5])),
        (
            torch.Tensor([[0.0, 1.0, 0.0], [1.0, 1.5, 2.0]]),
            0.5,
            torch.Tensor([[1.0, -0.5], [1.0, 1.25]]),
        ),
    ),
)
def test_apply_pre_emphasis_filter_1d(
    x: torch.Tensor, coef: float, y_expected: torch.Tensor
):
    y_actual = losses.apply_pre_emphasis_filter(x, coef)
    assert isinstance(y_actual, torch.Tensor)
    assert y_actual.ndim == y_expected.ndim
    assert y_actual.shape == y_expected.shape
    assert torch.allclose(y_actual, y_expected)


def test_esr():
    """
    Is the ESR calculation correct?
    """

    class Model(nn.Module):
        def forward(self, x):
            return x

    batch_size, input_length = 3, 5
    inputs = (
        torch.linspace(0.1, 1.0, batch_size)[:, None]
        * torch.full((input_length,), 1.0)[None, :]
    )  # (batch_size, input_length)
    target_factor = torch.linspace(0.37, 1.22, batch_size)
    targets = target_factor[:, None] * inputs  # (batch_size, input_length)
    # Do the algebra:
    # y=a*yhat
    # ESR=(y-yhat)^2 / y^2
    # ...
    # =(1/a-1)^2
    expected_esr = torch.square(1.0 / target_factor - 1.0).mean()
    model = Model()
    preds = model(inputs)
    actual_esr = losses.esr(preds, targets)
    assert torch.allclose(actual_esr, expected_esr)


if __name__ == "__main__":
    pytest.main()
