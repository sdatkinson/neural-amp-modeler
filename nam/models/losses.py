# File: losses.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Loss functions
"""

import torch


def esr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    ESR of (a batch of) predictions & targets

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    if preds.ndim == 1 and targets.ndim == 1:
        preds, targets = preds[None], targets[None]
    if preds.ndim != 2:
        raise ValueError(
            f"Expect 2D predictions (batch_size, num_samples). Got {preds.shape}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expect 2D targets (batch_size, num_samples). Got {targets.shape}"
        )
    return torch.mean(
        torch.mean(torch.square(preds - targets), dim=1)
        / torch.mean(torch.square(targets), dim=1)
    )
