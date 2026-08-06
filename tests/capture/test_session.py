import json as _json
from pathlib import Path as _Path

import numpy as _np
import pytest as _pytest

from nam.capture.audio import AudioDropoutError as _AudioDropoutError
from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.project import find_recoverable_entries as _find_recoverable_entries
from nam.capture.project import load_project as _load_project
from nam.capture.project import new_project as _new_project
from nam.capture.project import save_project as _save_project
from nam.capture.session import CaptureSession as _CaptureSession
from nam.capture.session import CaptureSessionError as _CaptureSessionError
from nam.data import np_to_wav as _np_to_wav
from nam.data import wav_to_np as _wav_to_np


_RATE = 48_000


class _FakeRecorder:
    """
    Pretends to be an amp on a loop with a fixed round-trip delay: attenuates the
    playback, shifts it, and adds a small noise floor.

    When loopback channels are requested it also returns a clean (undistorted) copy of
    the loopback playback shifted by ``loopback_delay`` (defaulting to the amp delay),
    modelling the direct patch through the same interface.
    """

    def __init__(
        self,
        delay: int,
        gain: float = 0.5,
        noise: float = 1e-5,
        loopback_delay: int = None,
        loopback_silent: bool = False,
    ):
        self.delay = delay
        self.gain = gain
        self.noise = noise
        self.loopback_delay = delay if loopback_delay is None else loopback_delay
        # Models an unplugged loopback cable: channels are requested but nothing
        # (beyond noise) actually arrives on the loopback input.
        self.loopback_silent = loopback_silent
        self.calls: list[dict] = []

    @staticmethod
    def _shift(signal, delay, gain):
        out = _np.zeros_like(signal)
        out[delay:] = signal[: len(signal) - delay] * gain
        return out

    def playrec(self, playback, sample_rate, **kwargs):
        self.calls.append(dict(kwargs, sample_rate=sample_rate))
        progress = kwargs.get("progress")
        if progress is not None:
            progress(1.0)
        rng = _np.random.default_rng(0)
        noise = self.noise * rng.standard_normal(len(playback)).astype(_np.float32)
        recording = self._shift(playback, self.delay, self.gain) + noise
        loopback = None
        loopback_playback = kwargs.get("loopback_playback")
        if (
            kwargs.get("loopback_output_channel") is not None
            and kwargs.get("loopback_input_channel") is not None
            and loopback_playback is not None
        ):
            loopback = _np.zeros_like(loopback_playback) + noise
            if not self.loopback_silent:
                loopback += self._shift(loopback_playback, self.loopback_delay, 1.0)
        return recording, loopback


def _make_project_dir(tmp_path: _Path, *, validation_rate: int = _RATE) -> _Path:
    rng = _np.random.default_rng(42)
    train_x = (0.1 * rng.standard_normal(_RATE)).astype(_np.float32)
    validation_x = (0.1 * rng.standard_normal(_RATE // 2)).astype(_np.float32)
    _np_to_wav(train_x, tmp_path / "input_train.wav", rate=_RATE)
    _np_to_wav(validation_x, tmp_path / "input_validation.wav", rate=validation_rate)

    knobs = [_KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5)]
    project = _new_project(knobs, n_train=2, n_validation=1, seed=0)
    _save_project(project, tmp_path)
    return tmp_path


def test_capture_entry_records_measures_and_persists(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    true_delay = 480
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(true_delay))

    entry = project.pending_entries()[0]
    qa = session.capture_entry(entry)

    assert entry.status == "captured"
    assert qa.impulse_detected
    assert not qa.clipping
    assert not qa.delay_disagreement
    assert entry.delay == true_delay - 1

    wav_path = project_dir / entry.y_path
    assert wav_path.is_file()
    y = _np.asarray(_wav_to_np(wav_path)).squeeze()
    x = _np.asarray(_wav_to_np(project_dir / "input_train.wav")).squeeze()
    assert len(y) == len(x)

    reloaded = _load_project(project_dir)
    assert reloaded.entries_for_split("train")[0].status == "captured"
    assert reloaded.sample_rate == _RATE

    data = _json.loads((project_dir / "data.json").read_text())
    assert len(data["train"]) == 1
    assert data["train"][0]["y_path"] == entry.y_path
    assert data["train"][0]["delay"] == true_delay - 1
    assert data["train"][0]["x_path"] == "input_train.wav"
    assert data["validation"] == []


def test_capture_saves_response_content_aligned_by_delay(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    true_delay = 480
    gain = 0.5
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(true_delay, gain=gain, noise=0.0)
    )
    entry = project.pending_entries()[0]
    session.capture_entry(entry)

    x = _np.asarray(_wav_to_np(project_dir / "input_train.wav")).squeeze()
    y = _np.asarray(_wav_to_np(project_dir / entry.y_path)).squeeze()
    # The fake chain is y[n] = gain * x[n - delay]; after the loader shifts by the
    # measured delay (true_delay - safety factor 1), y should track gain * x.
    delay = entry.delay
    assert delay is not None
    aligned_y = y[delay:]
    aligned_x = x[: len(aligned_y)]
    _np.testing.assert_allclose(aligned_y[1000:2000], gain * aligned_x[999:1999], atol=2e-4)


def test_validation_capture_uses_validation_input(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(300))

    entry = project.entries_for_split("validation")[0]
    session.capture_entry(entry)

    data = _json.loads((project_dir / "data.json").read_text())
    assert data["train"] == []
    assert len(data["validation"]) == 1
    assert data["validation"][0]["x_path"] == "input_validation.wav"

    y = _np.asarray(_wav_to_np(project_dir / entry.y_path)).squeeze()
    x = _np.asarray(_wav_to_np(project_dir / "input_validation.wav")).squeeze()
    assert len(y) == len(x)


def test_delay_disagreement_flagged_when_routing_changes(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)

    first = project.entries_for_split("train")[0]
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    qa_first = session.capture_entry(first)
    assert not qa_first.delay_disagreement

    second = project.entries_for_split("train")[1]
    drifted = _CaptureSession(project, project_dir, recorder=_FakeRecorder(900))
    qa_second = drifted.capture_entry(second)
    assert qa_second.delay_disagreement
    assert any("differs" in message for message in qa_second.messages)


def test_stream_latency_is_passed_to_the_recorder(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    project.audio.latency = 0.002
    project.audio.blocksize = 64
    recorder = _FakeRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)

    session.capture_entry(project.pending_entries()[0])

    assert recorder.calls[0]["latency"] == 0.002
    assert recorder.calls[0]["blocksize"] == 64


def test_changing_the_stream_settings_does_not_flag_a_delay_disagreement(tmp_path):
    """
    Buffer size and latency are the user's to change mid-project. Doing so moves the
    round trip by hundreds of samples, which must not be reported as a routing fault:
    each capture stores its own delay, so the change costs nothing.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)

    first = project.entries_for_split("train")[0]
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(900))
    session.capture_entry(first)
    assert first.stream_config == project.audio.stream_fingerprint()

    project.audio.latency = 0.002
    second = project.entries_for_split("train")[1]
    faster = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    qa_second = faster.capture_entry(second)

    assert not qa_second.delay_disagreement
    assert not any("differs" in message for message in qa_second.messages)
    assert second.delay != first.delay
    assert second.stream_config != first.stream_config


def test_delay_disagreement_still_fires_after_a_settings_change(tmp_path):
    """
    Scoping the check by stream configuration must not switch it off: once a capture
    exists at the new settings, the next one is compared against it.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    project.audio.latency = 0.002

    entries = project.entries_for_split("train")
    _CaptureSession(project, project_dir, recorder=_FakeRecorder(480)).capture_entry(
        entries[0]
    )
    qa = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(900)
    ).capture_entry(entries[1])

    assert qa.delay_disagreement


def test_delay_disagreement_uses_entries_predating_the_stream_config(tmp_path):
    """
    A project captured before the configuration was recorded keeps its delay check on
    the next capture rather than silently losing it.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)

    entries = project.entries_for_split("train")
    _CaptureSession(project, project_dir, recorder=_FakeRecorder(480)).capture_entry(
        entries[0]
    )
    entries[0].stream_config = None

    qa = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(900)
    ).capture_entry(entries[1])

    assert qa.delay_disagreement


def test_a_dropout_saves_nothing_and_leaves_the_entry_pending(tmp_path):
    """
    A dropped block leaves a hole in the middle of the capture that no delay
    measurement or level check would notice, so it must not reach disk.
    """

    class _DroppingRecorder:
        def playrec(self, playback, sample_rate, **kwargs):
            raise _AudioDropoutError("the audio stream could not keep up")

    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_DroppingRecorder())
    entry = project.pending_entries()[0]

    with _pytest.raises(_AudioDropoutError):
        session.capture_entry(entry)

    assert entry.status == "pending"
    assert entry.delay is None
    assert not (project_dir / entry.y_path).exists()


def test_clipping_is_flagged(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(480, gain=20.0)
    )
    entry = project.pending_entries()[0]
    qa = session.capture_entry(entry)
    assert qa.clipping
    assert any("Clipping" in message for message in qa.messages)


def test_recover_captured_from_disk_restores_regenerated_plan(tmp_path):
    project_dir = _make_project_dir(tmp_path)

    # Capture everything so WAVs + data.json (with per-entry delays) land on disk.
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    for entry in list(project.pending_entries()):
        session.capture_entry(entry)
    captured_delays = {e.y_path: e.delay for e in project.captured_entries()}
    assert len(captured_delays) == 3

    # Regenerate the plan with the same seed: fresh entries, all pending, identical
    # y_paths; the WAVs and data.json are left in place.
    regenerated = _new_project(
        [_KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5)],
        n_train=2,
        n_validation=1,
        seed=0,
    )
    _save_project(regenerated, project_dir)
    reloaded = _load_project(project_dir)
    assert all(e.status == "pending" for e in reloaded.entries)

    recoverable = _find_recoverable_entries(reloaded, project_dir)
    assert len(recoverable) == 3

    session = _CaptureSession(reloaded, project_dir, recorder=_FakeRecorder(0))
    notes = session.recover_captured_from_disk(recoverable)
    assert len(notes) == 3

    for entry in reloaded.entries:
        assert entry.status == "captured"
        assert entry.delay == captured_delays[entry.y_path]
        assert entry.qa is not None
        assert entry.captured_at is not None

    # data.json is rewritten from the restored entries.
    data = _json.loads((project_dir / "data.json").read_text())
    assert len(data["train"]) == 2
    assert len(data["validation"]) == 1

    # The restored statuses survive a reload.
    assert all(e.status == "captured" for e in _load_project(project_dir).entries)


def test_recover_captured_from_disk_skips_unreadable_wav(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    entry = project.pending_entries()[0]
    # A file that exists and is in data.json but is not a valid WAV.
    (project_dir / entry.y_path).parent.mkdir(parents=True, exist_ok=True)
    (project_dir / entry.y_path).write_bytes(b"not a wav")

    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(0))
    notes = session.recover_captured_from_disk([(entry, 42)])

    assert entry.status == "pending"
    assert any("could not read" in note for note in notes)


def test_route_test_measures_delay_without_saving_anything(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    true_delay = 480
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(true_delay))

    result = session.route_test()

    assert result.ok
    assert result.latency.delay == true_delay - 1
    assert not (project_dir / "captures").exists()
    assert not (project_dir / "data.json").exists()


def _enable_loopback(project, *, output_channel=2, input_channel=2):
    project.audio.output_channel = 1
    project.audio.input_channel = 1
    project.audio.loopback_output_channel = output_channel
    project.audio.loopback_input_channel = input_channel


def test_capture_uses_loopback_delay_and_crosschecks_amp(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    # The distorted amp return's threshold crossing lands a sample later than the clean
    # loopback's, which is within LOOPBACK_CROSSCHECK_SAMPLES; the loopback is the one
    # that should be trusted, and the amp agrees within tolerance.
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(481, loopback_delay=480)
    )
    entry = project.pending_entries()[0]
    qa = session.capture_entry(entry)

    assert entry.delay == 480 - 1
    assert qa.impulse_detected
    assert qa.loopback_disagreement is False
    # Both measurements are kept, not just the boolean above.
    assert qa.loopback_delay == 480 - 1
    assert qa.amp_return_delay == 481 - 1


def test_capture_flags_loopback_disagreement(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    # Amp return and loopback disagree by far more than the cross-check tolerance
    # (e.g. a mispatched loopback): the delay still comes from the loopback, but QA warns.
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(900, loopback_delay=480)
    )
    entry = project.pending_entries()[0]
    qa = session.capture_entry(entry)

    assert entry.delay == 480 - 1
    assert qa.loopback_disagreement is True
    assert any("loopback" in message for message in qa.messages)


def test_route_test_reports_loopback(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    # Amp return lands a sample later than the loopback, within
    # LOOPBACK_CROSSCHECK_SAMPLES, so the two should agree.
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(481, loopback_delay=480)
    )

    result = session.route_test()

    assert result.ok
    assert result.loopback_used
    assert not result.loopback_failed
    assert result.latency.delay == 480 - result.latency.safety_factor
    assert result.crosscheck is not None
    assert not result.loopback_disagreement


def test_capture_refuses_when_loopback_enabled_but_not_detected(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    # Channels are configured (as if the cable were plugged in and selected) but
    # nothing comes back on the loopback input -- an unplugged cable.
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(480, loopback_silent=True)
    )
    entry = project.pending_entries()[0]

    with _pytest.raises(_CaptureSessionError, match="[Ll]oopback"):
        session.capture_entry(entry)

    assert entry.status == "pending"
    assert entry.delay is None
    assert not (project_dir / entry.y_path).exists()
    assert not (project_dir / "data.json").exists()


def test_route_test_flags_loopback_not_detected(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(480, loopback_silent=True)
    )

    result = session.route_test()

    assert not result.ok
    assert result.loopback_failed
    # The amp return itself was fine; the failure must not be reported as if the
    # measured delay came from (or agreed with) the loopback.
    assert not result.latency.detected


def test_mismatched_input_rates_are_rejected(tmp_path):
    project_dir = _make_project_dir(tmp_path, validation_rate=44_100)
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    with _pytest.raises(_CaptureSessionError):
        session.load_inputs()


def test_missing_input_is_rejected(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    (project_dir / "input_validation.wav").unlink()
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    with _pytest.raises(_CaptureSessionError):
        session.load_inputs()


def _converter_response(taps: int = 31, cutoff: float = 0.8):
    """
    A compact bandlimited pulse standing in for a converter's round-trip impulse
    response. ``_FakeRecorder`` returns a bare unit impulse, which is not usable for
    timing tests: a fractionally delayed *ideal* impulse rings as a full sinc, decaying
    only as 1/n, so its leading lobes cross the calibrator's threshold a hundred samples
    early and the reported delay swings wildly with sub-sample phase. Real converters
    ring for a handful of samples, so their threshold crossing moves by at most one.
    """
    offset = _np.arange(taps) - (taps - 1) // 2
    return _np.sinc(cutoff * offset) * _np.hanning(taps) * cutoff


class _DriftingRecorder(_FakeRecorder):
    """
    A rig with a realistic loopback response whose round trip is ``delay + drift``
    samples, ``drift`` being a fraction the integer delay measurement cannot represent.
    It is read fresh on every call so a test can move it between captures.
    """

    def __init__(self, delay: int, drift: float = 0.0, **kwargs):
        super().__init__(delay, **kwargs)
        self.drift = drift

    def playrec(self, playback, sample_rate, **kwargs):
        from scipy.signal import fftconvolve as _fftconvolve

        from nam.capture.resample import apply_fractional_delay as _apply

        recording, loopback = super().playrec(playback, sample_rate, **kwargs)
        if loopback is not None:
            loopback = _fftconvolve(loopback, _converter_response(), mode="same")
            if self.drift:
                loopback = _apply(loopback, self.drift)
            loopback = loopback.astype(_np.float32)
        return recording, loopback


def test_first_capture_sets_the_timebase_and_is_not_shifted(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    assert project.alignment_reference is None
    qa = session.capture_entry(project.pending_entries()[0])

    assert project.alignment_reference is not None
    assert qa.subsample_shift is None
    # It is persisted, so reopening the project keeps the same timebase.
    assert _load_project(project_dir).alignment_reference == project.alignment_reference


def test_later_captures_are_shifted_back_onto_the_timebase(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)

    entries = project.pending_entries()
    session.capture_entry(entries[0])

    # The rig's sub-sample phase moves between captures; the correction should take it
    # straight back out rather than leaving it for the model to memorise.
    recorder.drift = 0.3
    qa = session.capture_entry(entries[1])

    assert qa.subsample_shift is not None
    assert qa.subsample_shift == _pytest.approx(-0.3, abs=0.05)


def test_no_correction_without_a_loopback(tmp_path):
    # The amp return carries the tone stack's own knob-dependent group delay, which is
    # real behaviour to be learned, not jitter to be removed.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    project.audio.loopback_output_channel = None
    project.audio.loopback_input_channel = None
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    qa = session.capture_entry(project.pending_entries()[0])

    assert qa.subsample_shift is None
    assert project.alignment_reference is None
    assert qa.loopback_delay is None
    assert qa.amp_return_delay == 480 - 1


def test_implausible_shift_is_refused_and_flagged(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    # A timebase from a rig that no longer exists: correcting onto it would slide the
    # capture far enough to do real damage.
    project.alignment_reference = 500.0

    qa = session.capture_entry(project.pending_entries()[0])

    assert qa.subsample_shift is None
    assert any("Alignment correction" in message for message in qa.messages)


def test_whole_sample_part_of_a_shift_is_taken_losslessly(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    entries = project.pending_entries()

    session.capture_entry(entries[0])
    baseline = _wav_to_np(project_dir / entries[0].y_path)

    # Same rig, but the timebase asks for exactly one whole sample of delay. That is a
    # slice offset, not a resample, so the samples themselves must be untouched.
    project.alignment_reference += 1.0
    qa = session.capture_entry(entries[1])
    shifted = _wav_to_np(project_dir / entries[1].y_path)

    assert qa.subsample_shift == _pytest.approx(1.0, abs=1e-6)
    assert _np.abs(shifted[1:] - baseline[:-1]).max() < 1e-6
