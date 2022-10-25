# File: test_base.py
# Created Date: Monday October 24th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import torch
import torch.nn as nn

from nam.models import base


def test_esr_batch():
    """
    Is the ESR calculation correct?
    """
    class Model(nn.Module):
        def forward(self, x):
            return x

    batch_size, input_length = 3, 5
    inputs = torch.linspace(0.1, 1.0, batch_size)[:, None] * torch.full((input_length,), 1.0)[None, :]
    target_factor = torch.linspace(0.37, 1.22, batch_size)
    targets = target_factor[:, None] * inputs
    # Do the algebra: 
    # y=a*yhat
    # ESR=(y-yhat)^2 / y^2
    # ...
    # =(1/a-1)^2
    expected_esr = torch.square(1.0 / target_factor - 1.0).mean()
    base_model = base.Model(Model())
    preds = base_model(inputs)
    actual_esr = base_model._esr_loss(preds, targets)
    assert torch.allclose(actual_esr, expected_esr)
