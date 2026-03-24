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


def bandpass_filter(
    x: _torch.Tensor,
    low_hz: float,
    high_hz: float,
    sample_rate: int,
) -> _torch.Tensor:
    """
    Band-limited signal via FFT mask with cosine rolloff at band edges (differentiable).

    Transition width is **5% of the band width** ``(high_hz - low_hz)`` on each edge.

    :param x: (B, L) or (B, 1, L)
    :return: Same shape as ``x``
    """
    squeeze_back = False
    if x.dim() == 3:
        x = x.squeeze(1)
        squeeze_back = True
    nyq = 0.5 * float(sample_rate)
    low_hz = max(float(low_hz), 0.0)
    high_hz = min(float(high_hz), nyq - 1.0)
    if low_hz >= high_hz:
        out = _torch.zeros_like(x)
        return out.unsqueeze(1) if squeeze_back else out

    band_width = high_hz - low_hz
    tw = max(band_width * 0.05, 1e-6)

    def _soft_edge(f: _torch.Tensor, center: float, width: float) -> _torch.Tensor:
        t = ((f - center) / width).clamp(-1.0, 1.0)
        return 0.5 * (1.0 + _torch.cos(_torch.pi * t))

    X = _torch.fft.rfft(x, dim=-1)
    freqs = _torch.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate, device=x.device)
    low_ramp = 1.0 - _soft_edge(freqs, low_hz, tw)
    high_ramp = 1.0 - _soft_edge(freqs, high_hz, -tw)
    mask = (low_ramp * high_ramp).clamp(0.0, 1.0).to(dtype=x.dtype)
    y = _torch.fft.irfft(X * mask, n=x.shape[-1], dim=-1)
    return y.unsqueeze(1) if squeeze_back else y


def band_esr(
    pred: _torch.Tensor,
    target: _torch.Tensor,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> _torch.Tensor:
    """
    ESR on bandpass-filtered signals. Same structure as :func:`esr`:
    ``mean(error^2) / mean(target^2)`` along time, then mean over batch.

    :param pred: (B, L) or (B, 1, L)
    :param target: Same shape as pred
    """
    pred_b = bandpass_filter(pred, low_hz, high_hz, sample_rate)
    target_b = bandpass_filter(target, low_hz, high_hz, sample_rate)
    error = pred_b - target_b
    esr = (error**2).mean(dim=-1) / (target_b**2).mean(dim=-1)
    return esr.mean()


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
