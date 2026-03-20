# File: losses.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Loss functions
"""

from typing import Literal as _Literal
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


def esr(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
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
    return _torch.mean(
        _torch.mean(_torch.square(preds - targets), dim=1)
        / _torch.mean(_torch.square(targets), dim=1)
    )


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


class SpectralBandLoss(_nn.Module):
    """
    Penalize spectral error in a specific frequency band (e.g. 7–12 kHz for ring).
    Uses time-averaged spectrum to avoid transient-driven gradients.

    :param sample_rate: Sample rate in Hz
    :param fft_size: FFT size (4096 gives ~12 Hz resolution at 48 kHz)
    :param hop_length: STFT hop length
    :param low_hz: Bottom of problem band (Hz)
    :param high_hz: Top of problem band (Hz)
    :param weight: Loss weight
    :param penalize: "excess" = only when pred > target (fixes ring), "missing" = only when pred < target
    :param log_scale: If True, use log magnitude (ear-like, reduces transient dominance)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        fft_size: int = 4096,
        hop_length: _Optional[int] = None,
        low_hz: float = 8000,
        high_hz: float = 12000,
        weight: float = 1.0,
        penalize: _Literal["excess", "missing"] = "excess",
        log_scale: bool = True,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length if hop_length is not None else fft_size // 4
        self.weight = weight
        self.penalize = penalize
        self.log_scale = log_scale
        self._low_hz = low_hz
        self._high_hz = high_hz
        self._sample_rate = sample_rate

        freqs = _torch.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
        self.register_buffer(
            "band_mask",
            ((freqs >= low_hz) & (freqs <= high_hz)).float(),
        )

    def update_sample_rate(self, sample_rate: int) -> None:
        """Recompute band_mask for a different sample rate (e.g. after dataset handshake)."""
        if sample_rate == self._sample_rate:
            return
        self._sample_rate = sample_rate
        freqs = _torch.fft.rfftfreq(self.fft_size, d=1.0 / sample_rate)
        new_mask = ((freqs >= self._low_hz) & (freqs <= self._high_hz)).float()
        self.band_mask.copy_(new_mask.to(device=self.band_mask.device))

    def forward(
        self,
        pred: _torch.Tensor,
        target: _torch.Tensor,
        window: _Optional[_torch.Tensor] = None,
    ) -> _torch.Tensor:
        """
        :param pred: (B, L) or (B, 1, L)
        :param target: Same shape as pred
        :param window: Optional Hann window; created if None
        :return: Scalar loss
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        if window is None:
            window = _torch.hann_window(self.fft_size, device=pred.device)

        def mean_spectrum(x: _torch.Tensor) -> _torch.Tensor:
            stft = _torch.stft(
                x,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            )
            mag = stft.abs()  # (B, F, T)
            mean_mag = mag.mean(dim=-1)  # (B, F)
            if self.log_scale:
                return _torch.log(mean_mag + 1e-8)
            return mean_mag

        pred_spec = mean_spectrum(pred)
        target_spec = mean_spectrum(target)

        if self.penalize == "excess":
            band_error = _torch.relu(pred_spec - target_spec) * self.band_mask
        else:
            band_error = _torch.relu(target_spec - pred_spec) * self.band_mask

        return self.weight * _torch.mean(band_error**2)
