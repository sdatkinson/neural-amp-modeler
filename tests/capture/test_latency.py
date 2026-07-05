import numpy as _np
import pytest as _pytest

from nam.capture.latency import BlipPreamble as _BlipPreamble
from nam.capture.latency import measure_delay as _measure_delay


_RATE = 48_000


def _simulate_chain(
    playback: _np.ndarray,
    *,
    delay: int,
    gain: float = 0.5,
    noise: float = 1e-5,
    seed: int = 0,
) -> _np.ndarray:
    recording = _np.zeros_like(playback)
    recording[delay:] = playback[: len(playback) - delay] * gain
    rng = _np.random.default_rng(seed)
    return recording + noise * rng.standard_normal(len(playback)).astype(_np.float32)


def test_preamble_renders_blips_at_declared_locations():
    preamble = _BlipPreamble(_RATE)
    playback = preamble.render()
    assert playback.dtype == _np.float32
    assert len(playback) == preamble.n_samples
    nonzero = _np.nonzero(playback)[0]
    assert tuple(nonzero) == preamble.blip_locations
    assert _np.all(playback[nonzero] == preamble.amplitude)


def test_measure_delay_recovers_known_shift():
    preamble = _BlipPreamble(_RATE)
    tail = _np.zeros(_RATE // 2, dtype=_np.float32)
    playback = _np.concatenate([preamble.render(), tail])
    for true_delay in (37, 500, 4_321):
        recording = _simulate_chain(playback, delay=true_delay)
        result = _measure_delay(recording, preamble)
        assert result.detected
        assert not result.disagreement_too_high
        assert result.delay == true_delay - result.safety_factor


def test_measure_delay_recovers_large_fixed_latency():
    # A loopback through a buffered virtual device (e.g. an audio-bridge) adds a fixed
    # round-trip latency past NAM's 208 ms scan window; the coarse pre-alignment must
    # still recover it. 12_288 samples = 256 ms at 48 kHz, the value seen in the field.
    preamble = _BlipPreamble(_RATE)
    tail = _np.zeros(_RATE // 2, dtype=_np.float32)
    playback = _np.concatenate([preamble.render(), tail])
    true_delay = 12_288
    recording = _simulate_chain(playback, delay=true_delay)
    result = _measure_delay(recording, preamble)
    assert result.detected
    assert not result.disagreement_too_high
    assert result.delay == true_delay - result.safety_factor


def test_measure_delay_reports_not_detected_on_silence():
    preamble = _BlipPreamble(_RATE)
    recording = 1e-6 * _np.random.default_rng(0).standard_normal(
        preamble.n_samples + _RATE
    ).astype(_np.float32)
    result = _measure_delay(recording, preamble)
    assert not result.detected
    assert result.delay is None
    assert not result.ok


def test_measure_delay_rejects_short_recordings():
    preamble = _BlipPreamble(_RATE)
    with _pytest.raises(ValueError):
        _measure_delay(_np.zeros(1_000, dtype=_np.float32), preamble)


def test_measure_delay_rejects_multichannel_recordings():
    preamble = _BlipPreamble(_RATE)
    with _pytest.raises(ValueError):
        _measure_delay(
            _np.zeros((preamble.n_samples + _RATE, 2), dtype=_np.float32), preamble
        )


def test_preamble_rejects_bad_parameters():
    with _pytest.raises(ValueError):
        _BlipPreamble(0)
    with _pytest.raises(ValueError):
        _BlipPreamble(_RATE, amplitude=0.0)
    with _pytest.raises(ValueError):
        _BlipPreamble(_RATE, amplitude=1.5)
