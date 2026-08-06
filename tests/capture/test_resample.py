import numpy as _np
import pytest as _pytest

from nam.capture.resample import apply_fractional_delay as _apply_fractional_delay
from nam.capture.resample import fractional_delay_kernel as _kernel


def _tone(n, frequency, rate=48_000, phase=0.0):
    t = _np.arange(n) / rate
    return _np.sin(2 * _np.pi * frequency * t + phase)


def test_zero_shift_is_exactly_the_identity():
    # Any interpolator with the Nyquist property is a unit impulse at zero shift, which
    # is why there is no such thing as a lossy "resample by zero" control.
    kernel = _kernel(0.0)
    expected = _np.zeros_like(kernel)
    expected[(len(kernel) - 1) // 2] = 1.0
    assert _np.allclose(kernel, expected, atol=1e-12)

    audio = _tone(4_000, 1_000.0)
    assert _apply_fractional_delay(audio, 0.0) is not None
    assert _np.array_equal(_apply_fractional_delay(audio, 0.0), audio)


def test_kernel_has_unit_dc_gain():
    for shift in (-0.5, -0.13, 0.25, 0.5):
        assert _kernel(shift).sum() == _pytest.approx(1.0)


@_pytest.mark.parametrize("shift", [-0.5, -0.25, 0.1, 0.375, 0.5])
@_pytest.mark.parametrize("frequency", [100.0, 1_000.0, 10_000.0])
def test_shift_matches_an_analytically_delayed_tone(shift, frequency):
    rate = 48_000
    n = 8_192
    shifted = _apply_fractional_delay(_tone(n, frequency, rate), shift)
    # A delay of `shift` samples is a phase lag of 2*pi*f*shift/rate.
    expected = _tone(n, frequency, rate, phase=-2 * _np.pi * frequency * shift / rate)
    interior = slice(1_000, n - 1_000)
    assert _np.abs(shifted[interior] - expected[interior]).max() < 1e-6


def test_length_is_preserved_and_edges_stay_bounded():
    rng = _np.random.default_rng(0)
    audio = 0.4 * rng.standard_normal(20_000)
    shifted = _apply_fractional_delay(audio, 0.4)
    assert shifted.shape == audio.shape
    # Reflection padding must not manufacture a transient at either end.
    assert _np.abs(shifted).max() < 1.5 * _np.abs(audio).max()


def _bandlimited_noise(n, cutoff, rate=48_000, seed=1):
    """Noise with nothing above ``cutoff``, as a real converter's output already is."""
    spectrum = _np.fft.rfft(_np.random.default_rng(seed).standard_normal(n))
    spectrum[_np.fft.rfftfreq(n, 1 / rate) > cutoff] = 0.0
    audio = _np.fft.irfft(spectrum, n)
    return 0.4 * audio / _np.abs(audio).max()


_HALF_SUPPORT = 300  # a little past TAPS // 2


def test_round_trip_is_exact_away_from_the_edges():
    # +shift then -shift is the identity for an ideal filter, and within the band this
    # one is ideal to floating-point precision. Everything that is not exact comes from
    # the reflection padding at the two ends, which is why the check is on the interior.
    audio = _bandlimited_noise(48_000, 20_000.0)
    back = _apply_fractional_delay(_apply_fractional_delay(audio, 0.37), -0.37)
    interior = slice(_HALF_SUPPORT, -_HALF_SUPPORT)
    assert _np.sum((back - audio)[interior] ** 2) / _np.sum(audio[interior] ** 2) < 1e-12


def test_edge_error_stays_inside_the_filter_support():
    audio = _bandlimited_noise(48_000, 20_000.0)
    error = (
        _apply_fractional_delay(_apply_fractional_delay(audio, 0.37), -0.37) - audio
    )
    edges = _np.sum(error[:_HALF_SUPPORT] ** 2) + _np.sum(error[-_HALF_SUPPORT:] ** 2)
    assert edges / _np.sum(error**2) > 0.999


def test_only_the_top_of_the_band_is_lost():
    # With energy all the way to Nyquist the loss is real but stays at the very top: no
    # real-coefficient fractional delay can carry the Nyquist bin (its response there is
    # cos(pi * shift)).
    full_band = 0.4 * _np.random.default_rng(2).standard_normal(48_000)
    error = (
        _apply_fractional_delay(_apply_fractional_delay(full_band, 0.37), -0.37)
        - full_band
    )[_HALF_SUPPORT:-_HALF_SUPPORT]
    spectrum = _np.abs(_np.fft.rfft(error)) ** 2
    freqs = _np.fft.rfftfreq(len(error), 1 / 48_000)
    assert spectrum[freqs < 22_000].sum() / spectrum.sum() < 0.01


def test_rejects_multichannel_and_too_short_audio():
    with _pytest.raises(ValueError):
        _apply_fractional_delay(_np.zeros((100, 2)), 0.25)
    with _pytest.raises(ValueError):
        _apply_fractional_delay(_np.zeros(16), 0.25)


def test_rejects_even_tap_counts():
    with _pytest.raises(ValueError):
        _kernel(0.25, taps=512)
