# File: losses.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Loss functions
"""

from typing import Optional as _Optional

import torch as _torch
import torch.nn as _nn

from .._dependencies.auraloss.freq import (
    MultiResolutionSTFTLoss as _MultiResolutionSTFTLoss,
)


def apply_pre_emphasis_filter(x: _torch.Tensor, coef: float) -> _torch.Tensor:
    """
    Apply first-order pre-emphsis filter

    :param x: (*, L)
    :param coef: The coefficient

    :return: (*, L-1)
    """
    return x[..., 1:] - coef * x[..., :-1]


def esr(
    preds: _torch.Tensor, targets: _torch.Tensor, eps: float = 0.0
) -> _torch.Tensor:
    """
    ESR of (a batch of) predictions & targets

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :param eps: Added to the denominator ``mean(y^2)`` for numerical stability (0 keeps
        the classic ratio used for validation metrics).
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
    denom = _torch.mean(_torch.square(targets), dim=1) + eps
    return _torch.mean(_torch.mean(_torch.square(preds - targets), dim=1) / denom)


class ESRLoss(_nn.Module):
    """
    ``nn.Module`` wrapper around :func:`esr` for use with ``custom_losses`` in the
    training config (the generic loader expects a constructible module). Typical use is
    a small ``weight`` (e.g. ``0.05``) as an auxiliary term alongside MSE and MRSTFT.
    Name the custom loss key ``"ESR"`` if ``val_loss`` is ``"esr"`` so validation
    resolves that metric from the loss dict.

    Uses ``mean((p-y)^2) / (mean(y^2) + eps)`` with a small default ``eps`` so quiet
    targets do not explode the ratio.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self._eps = eps

    def forward(self, preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
        return esr(preds, targets, eps=self._eps)


def multi_resolution_stft_loss(
    preds: _torch.Tensor,
    targets: _torch.Tensor,
    loss_func: _Optional[_MultiResolutionSTFTLoss] = None,
    device: _Optional[_torch.device] = None,
) -> _torch.Tensor:
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

    def ensure_shape(z: _torch.Tensor) -> _torch.Tensor:
        """
        Required for auraloss v0.4

        :param z: (L,) or (B,L)
        :return: (B,C,L)
        """
        if z.ndim == 1:
            return z[None, None, :]
        elif z.ndim == 2:
            return z[:, None, :]
        else:
            assert z.ndim == 3, f"Expected 1D or 2D tensor. Got {z.shape}"
            return z

    loss_func = _MultiResolutionSTFTLoss() if loss_func is None else loss_func
    if device is not None:
        preds, targets = [z.to(device) for z in (preds, targets)]
    preds, targets = [ensure_shape(z) for z in (preds, targets)]
    return loss_func(preds, targets)


def mse(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    MSE loss
    """
    return _torch.nn.functional.mse_loss(preds, targets)


def mse_fft(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    Fourier loss

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    fp = _torch.fft.fft(preds)
    ft = _torch.fft.fft(targets)
    e = fp - ft
    return _torch.mean(_torch.square(e.abs()))
