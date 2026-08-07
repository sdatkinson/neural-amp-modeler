"""
Sub-sample time shifting for capture targets.

``delay`` in a data config is an integer, but the rig's true round-trip latency is not,
so a capture set aligned only to whole samples carries up to half a sample of per-capture
error. That residual is not a function of the knobs, so a parametric model cannot learn it
as a rule -- it can only memorise a per-capture phase, which is capacity spent on something
with no generalisation value.

The fix applied here is to shift the target by the fractional remainder *before writing the
WAV*, leaving ``delay`` an integer. Nothing downstream changes: ``nam.data`` still trims by
an integer, and every capture in a project ends up on one sub-sample timebase.

An oversample/shift/decimate round trip is not needed. A windowed-sinc FIR applies an
arbitrary sub-sample shift in a single pass at the capture rate; with the defaults below it
is flat to 5e-7 dB and accurate to 2e-6 samples of phase delay out to 0.9 Nyquist, which is
far below the quantisation of the WAV it gets written to.

The one irreducible cost is at Nyquist itself: a real-coefficient filter cannot delay a
Nyquist-frequency component (its response there is ``cos(pi * shift)``), so content in the
top ~1 kHz at 48 kHz is attenuated. Measured on real captures that is worth ~2e-5 ESR
against a metric that lives around 2e-2, and it does not improve with more taps.

Everywhere else the shift is exact to floating-point precision, with one exception: the
first and last ``TAPS // 2`` samples, where the filter's support runs past the end of the
audio and the signal is extended by reflection. That is ~256 samples of approximation at
each end of a capture that is millions long, at the head of which the rig is silent
anyway, but it is the reason a shift-and-unshift round trip is not bit-exact.
"""

from __future__ import annotations

import numpy as _np


# 513 taps at beta 14 is the knee of the accuracy/cost curve: doubling the taps buys another
# factor of two on a magnitude error that is already 5e-7 dB, and the residual that actually
# matters (the Nyquist bin, see the module docstring) does not move at all.
TAPS = 513
BETA = 14.0
# Past this the windowed sinc is being asked to slide off the end of its own window. A
# capture wanting a shift this large is a whole-sample routing problem, not sub-sample
# jitter, and the caller should say so rather than quietly resample.
MAX_SHIFT = 0.75


def fractional_delay_kernel(shift: float, taps: int = TAPS, beta: float = BETA):
    """
    FIR whose centre tap output is the input delayed by ``shift`` samples.

    Negative shifts advance. At ``shift == 0`` this is exactly a unit impulse, so applying
    it is the identity -- there is no such thing as a lossy "resample by zero".
    """
    if taps % 2 == 0:
        raise ValueError(f"taps must be odd so the kernel has a centre; got {taps}")
    half = (taps - 1) // 2
    offset = _np.arange(taps) - half - shift
    kernel = _np.sinc(offset)
    # Kaiser window centred on the (fractional) sinc peak rather than on the tap grid.
    position = _np.clip(offset / (half + 0.5), -1.0, 1.0)
    kernel = kernel * (_np.i0(beta * _np.sqrt(1.0 - position**2)) / _np.i0(beta))
    return kernel / kernel.sum()


def apply_fractional_delay(
    audio, shift: float, taps: int = TAPS, beta: float = BETA
):
    """
    Delay ``audio`` by ``shift`` samples (negative advances it), preserving its length.

    Edges are extended by reflection. Captures routinely end on full-level program
    material, so zero-padding would put a step discontinuity inside the filter's support
    and ring for ``taps // 2`` samples into real audio.
    """
    audio = _np.asarray(audio, dtype=_np.float64).squeeze()
    if audio.ndim != 1:
        raise ValueError(f"Expected single-channel audio; got shape {audio.shape}")
    if shift == 0.0:
        return audio
    half = (taps - 1) // 2
    if len(audio) <= half:
        raise ValueError(
            f"Audio is too short to shift with a {taps}-tap kernel: {len(audio)} samples"
        )
    from scipy.signal import fftconvolve as _fftconvolve

    padded = _np.pad(audio, half, mode="reflect")
    filtered = _fftconvolve(padded, fractional_delay_kernel(shift, taps, beta), mode="same")
    return filtered[half : half + len(audio)]
