"""
Per-capture latency measurement.

Arbitrary user-chosen input WAVs have no timing marks, so the app prepends a short
synthesized preamble to every playback: a stretch of silence (whose tail doubles as
the noise-floor measurement window), two single-sample impulse blips one second
apart, and a short gap before the input audio starts. Detection wraps the impulse
calibration in ``nam.train.core`` by describing the preamble with the same
``_DataInfo`` structure the standardized NAM input files use.

The measured delay is the round-trip latency of the whole chain (interface output,
device under test, interface input) in samples. Detection scans up to ~10k samples
past each blip, so chains with more than that much latency (~208 ms at 48 kHz) are
reported as not detected.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Optional as _Optional

import numpy as _np


# Preamble layout, in seconds. The blip section mirrors the v2 NAM input file's
# proportions (noise window before the first blip, blips at 0.5 s and 1.5 s of a 2 s
# section) so the wrapped calibration sees the geometry it was written for.
_BLIP_SECTION_SECONDS = 2.0
_NOISE_INTERVAL_SECONDS = (0.25, 0.375)
_BLIP_TIMES_SECONDS = (0.5, 1.5)
_GAP_SECONDS = 0.25
DEFAULT_BLIP_AMPLITUDE = 0.9

# The wrapped NAM calibrator (``nam.train.core``) only scans from ``_LOOKAHEAD`` samples
# before to ``_LOOKBACK`` samples after each expected blip. Virtual/buffered routes
# (e.g. a loopback through an audio-bridge device) can add a fixed round-trip latency
# larger than ``_LOOKBACK`` — well within a healthy capture, but past what the fine
# calibrator can see — so we coarse-align first. These must match the constants inside
# ``_calibrate_latency_v_all``.
_CALIBRATOR_LOOKAHEAD = 1_000
_CALIBRATOR_LOOKBACK = 10_000

# Sub-sample peak location (see ``_peak_delay``). 129 steps across one sample resolves the
# peak to under 0.008 samples, which is well inside the spread the correction is there to
# remove (~0.1-0.2 samples measured across real capture sessions).
_PEAK_INTERPOLATION_TAPS = 257
_PEAK_INTERPOLATION_STEPS = 129


@_dataclass(frozen=True)
class BlipPreamble:
    sample_rate: int
    amplitude: float = DEFAULT_BLIP_AMPLITUDE

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive; got {self.sample_rate}")
        if not (0.0 < self.amplitude <= 1.0):
            raise ValueError(
                f"amplitude must be within (0, 1]; got {self.amplitude}"
            )

    @property
    def blip_section_samples(self) -> int:
        return int(_BLIP_SECTION_SECONDS * self.sample_rate)

    @property
    def blip_locations(self) -> tuple[int, int]:
        first, second = _BLIP_TIMES_SECONDS
        return (int(first * self.sample_rate), int(second * self.sample_rate))

    @property
    def noise_interval(self) -> tuple[int, int]:
        start, stop = _NOISE_INTERVAL_SECONDS
        return (int(start * self.sample_rate), int(stop * self.sample_rate))

    @property
    def n_samples(self) -> int:
        """
        Total preamble length including the post-blip gap; the input audio starts at
        this offset in the playback.
        """
        return self.blip_section_samples + int(_GAP_SECONDS * self.sample_rate)

    def render(self) -> _np.ndarray:
        playback = _np.zeros(self.n_samples, dtype=_np.float32)
        for location in self.blip_locations:
            playback[location] = self.amplitude
        return playback

    def _data_info(self) -> _Any:
        from ..train.core import _DataInfo

        # Only the fields the latency calibration reads are meaningful here; the
        # train/validation split fields describe standardized training files, which
        # this preamble is not, so they get placeholder values.
        return _DataInfo(
            major_version=0,
            rate=float(self.sample_rate),
            t_blips=self.blip_section_samples,
            first_blips_start=0,
            t_validate=1,
            train_start=0,
            validation_start=-1,
            noise_interval=self.noise_interval,
            blip_locations=(self.blip_locations,),
        )


@_dataclass(frozen=True)
class LatencyResult:
    delay: _Optional[int]
    detected: bool
    disagreement_too_high: bool
    safety_factor: int
    # Sub-sample position of the blip response's energy peak, in the same frame as
    # ``delay``. See :func:`_peak_delay` for why this is not simply ``delay`` with a
    # fractional part: the two are different estimators of the same arrival, and only
    # the *difference between captures* of this one is meaningful.
    peak_delay: _Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.detected and not self.disagreement_too_high


def _coarse_bulk_delay(recording: _np.ndarray, preamble: BlipPreamble) -> int:
    """
    Estimate a large fixed round-trip latency so the fine calibrator can still find the
    blips. Returns ``0`` unless there is a confident bulk delay past the calibrator's
    reach — for normal low-latency routes the calibrator handles it and this stays out
    of the way.

    The search is restricted to the silent blip section (never the input-audio region,
    which is just as loud as the blips) and only as far as leaves blip 2 inside the
    calibrator's window, so it locks onto the blips rather than program material.
    """
    locs = preamble.blip_locations
    max_shift = min(
        preamble.blip_section_samples - locs[-1],
        len(recording) - locs[-1],
    )
    if max_shift <= _CALIBRATOR_LOOKBACK:
        return 0
    score = _np.zeros(max_shift, dtype=_np.float64)
    for loc in locs:
        score += _np.abs(recording[loc : loc + max_shift])
    d = int(_np.argmax(score))
    if d <= _CALIBRATOR_LOOKBACK - _CALIBRATOR_LOOKAHEAD:
        return 0  # already within the fine calibrator's window; don't interfere
    peak = float(score[d])
    baseline = float(_np.median(score))
    # Require the winning shift to clearly stand out; otherwise it's noise and we let
    # the calibrator report "not detected" as before.
    if peak < 0.02 * len(locs) or peak < 5.0 * (baseline + 1e-9):
        return 0
    return d


def _peak_delay(scan: _np.ndarray, preamble: BlipPreamble) -> _Optional[float]:
    """
    Sub-sample arrival time of the blip response, from the continuous position of its
    energy peak. ``scan`` must already be coarse-aligned, as it is for the calibrator.

    This is deliberately a *different* estimator from the one behind ``delay``, which is
    the first sample to cross an amplitude threshold, minus a safety factor. A threshold
    crossing has no meaningful fractional part -- it is amplitude-biased, and a quieter
    return crosses later -- so there is nothing there to refine. The peak of the response
    does have one.

    The two therefore disagree by some fixed offset. That offset does not matter: the
    round trip is LTI and, on a loopback, its impulse response is fixed hardware, so the
    offset is identical for every capture in a project and is absorbed as a constant. What
    varies between captures -- and what this exists to remove -- is the fractional part.
    See :meth:`nam.capture.session.CaptureSession._subsample_shift`.

    Both blips are averaged, as in the calibrator: they are exact-integer impulses through
    the same LTI path, so they share a sub-sample phase and averaging halves the noise.

    The peak is located by sinc interpolation on a fine grid, not by fitting a parabola to
    the three samples around it. A converter round trip's impulse response is bandlimited,
    so its peak is sinc-shaped rather than parabolic, and a three-point parabola
    underestimates the offset by more than an order of magnitude there -- 0.009 for a true
    0.2 in the test that caught it. Sinc interpolation is the exact reconstruction for a
    bandlimited signal, so it has no such bias.
    """
    from .resample import fractional_delay_kernel as _kernel

    section = scan[: preamble.blip_section_samples]
    windows = []
    for location in preamble.blip_locations:
        start = location - _CALIBRATOR_LOOKAHEAD
        stop = location + _CALIBRATOR_LOOKBACK
        if start < 0 or stop > len(section):
            return None
        windows.append(section[start:stop])
    response = _np.mean(_np.stack(windows), axis=0).astype(_np.float64)

    peak = int(_np.argmax(response**2))
    half = (_PEAK_INTERPOLATION_TAPS - 1) // 2
    if peak - half < 0 or peak + half + 1 > len(response) or response[peak] == 0.0:
        return None
    neighbourhood = response[peak - half : peak + half + 1]

    # Interpolated value at a continuous offset `t` from the peak sample. The response is
    # not a bare impulse -- it is the round trip's own shape -- so the offset this finds
    # carries that shape's own bias. It is the same shape on every capture, so the bias is
    # a constant and cancels; only the drift between captures is used.
    offsets = _np.linspace(-0.5, 0.5, _PEAK_INTERPOLATION_STEPS)
    interpolated = _np.array(
        [
            _kernel(offset, taps=_PEAK_INTERPOLATION_TAPS) @ neighbourhood
            for offset in offsets
        ]
    )
    fraction = float(offsets[int(_np.argmax(_np.abs(interpolated)))])
    return float(peak - _CALIBRATOR_LOOKAHEAD + fraction)


def measure_delay(recording: _np.ndarray, preamble: BlipPreamble) -> LatencyResult:
    """
    Measure round-trip delay from a recording that is time-aligned with the playback
    (sample 0 of the recording was captured as sample 0 of the playback started).
    """
    from ..train import core as _core

    recording = _np.asarray(recording).squeeze()
    if recording.ndim != 1:
        raise ValueError(
            f"Expected a single-channel recording; got shape {recording.shape}"
        )
    minimum = preamble.blip_locations[-1] + _CALIBRATOR_LOOKBACK
    if len(recording) < minimum:
        raise ValueError(
            f"Recording too short to scan for blips: {len(recording)} < {minimum} samples"
        )

    # Coarse-align away any bulk latency past the fine calibrator's window, then let it
    # measure the residual on the shifted copy and add the bulk offset back in.
    coarse = _coarse_bulk_delay(recording, preamble)
    if coarse > 0:
        scan = _np.zeros_like(recording)
        scan[: len(recording) - coarse] = recording[coarse:]
    else:
        scan = recording

    calibration = _core._calibrate_latency_v_all(
        preamble._data_info(),
        scan,
        manual_available=True,
        show_plots=False,
        _override_suppress_plots=True,
    )
    recommended = calibration.recommended
    detected = not calibration.warnings.not_detected
    peak = _peak_delay(scan, preamble) if detected else None
    return LatencyResult(
        delay=None if recommended is None else int(recommended) + coarse,
        detected=detected,
        disagreement_too_high=calibration.warnings.disagreement_too_high,
        safety_factor=int(calibration.safety_factor),
        peak_delay=None if peak is None else peak + coarse,
    )
