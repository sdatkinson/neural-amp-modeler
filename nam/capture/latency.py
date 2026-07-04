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

    @property
    def ok(self) -> bool:
        return self.detected and not self.disagreement_too_high


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
    minimum = preamble.blip_locations[-1] + 10_000
    if len(recording) < minimum:
        raise ValueError(
            f"Recording too short to scan for blips: {len(recording)} < {minimum} samples"
        )

    calibration = _core._calibrate_latency_v_all(
        preamble._data_info(),
        recording,
        manual_available=True,
        show_plots=False,
        _override_suppress_plots=True,
    )
    recommended = calibration.recommended
    return LatencyResult(
        delay=None if recommended is None else int(recommended),
        detected=not calibration.warnings.not_detected,
        disagreement_too_high=calibration.warnings.disagreement_too_high,
        safety_factor=int(calibration.safety_factor),
    )
