# File: losses.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Loss functions
"""

from typing import Optional

import torch
from auraloss.freq import MultiResolutionSTFTLoss

___all__ = ["apply_pre_emphasis_filter", "esr", "multi_resolution_stft_loss"]


def apply_pre_emphasis_filter(x: torch.Tensor, coef: float) -> torch.Tensor:
    """
    Apply first-order pre-emphsis filter

    :param x: (*, L)
    :param coef: The coefficient

    :return: (*, L-1)
    """
    return x[..., 1:] - coef * x[..., :-1]


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


def multi_resolution_stft_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_func: Optional[MultiResolutionSTFTLoss] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
    B: Batch size
    L: Sequence length

    :param preds: (B,L)
    :param targets: (B,L)
    :param loss_func: A pre-initialized instance of the loss function module. Providing
        this saves time.
    :param device: If provided, send the preds and targets to the provided device.
    :return: ()
    """
    loss_func = MultiResolutionSTFTLoss() if loss_func is None else loss_func
    if device is not None:
        preds, targets = [z.to(device) for z in (preds, targets)]
    return loss_func(preds, targets)


def mse_fft(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Fourier loss

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    fp = torch.fft.fft(preds)
    ft = torch.fft.fft(targets)
    e = fp - ft
    return torch.mean(torch.square(e.abs()))
