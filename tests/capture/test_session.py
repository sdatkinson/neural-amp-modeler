import json as _json
from pathlib import Path as _Path

import numpy as _np
import pytest as _pytest

from nam.capture import CAPTURE_APP_VERSION as _CAPTURE_APP_VERSION
from nam.capture import RAW_RECORDING_SINCE_VERSION as _RAW_RECORDING_SINCE_VERSION
from nam.capture.audio import AudioDropoutError as _AudioDropoutError
from nam.capture.latency import BlipPreamble as _BlipPreamble
from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.planner import CAPTURES_RAW_DIRNAME as _CAPTURES_RAW_DIRNAME
from nam.capture.planner import RAW_MANIFEST_FILENAME as _RAW_MANIFEST_FILENAME
from nam.capture.project import clear_captures as _clear_captures
from nam.capture.project import find_clearable_entries as _find_clearable_entries
from nam.capture.project import find_recoverable_entries as _find_recoverable_entries
from nam.capture.project import load_project as _load_project
from nam.capture.project import new_project as _new_project
from nam.capture.project import save_project as _save_project
from nam.capture.session import CaptureSession as _CaptureSession
from nam.capture.session import CaptureSessionError as _CaptureSessionError
from nam.capture.session import ALIGNMENT_LEAD_SAMPLES as _LEAD
from nam.capture.session import MAX_ALIGNMENT_SHIFT as _MAX_ALIGNMENT_SHIFT
from nam.capture.session import (
    MAX_LEGACY_ALIGNMENT_REFERENCE as _MAX_LEGACY_REFERENCE,
)
from nam.capture.session import playback_input_path as _playback_input_path
from nam.capture.session import audit_captures as _audit_captures
from nam.capture.session import audit_problems as _audit_problems
from nam.capture.session import has_captures as _has_captures
from nam.capture.session import raw_paths as _raw_paths
from nam.capture.session import shared_offset as _shared_offset
from nam.capture.session import timebase_problem as _timebase_problem
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
        # What was handed back, so a test can compare it against what reached disk.
        self.returns: list[tuple] = []

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
        self.returns.append((recording, loopback))
        return recording, loopback


def _make_project_dir(tmp_path: _Path, *, validation_rate: int = _RATE) -> _Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    assert entry.delay == true_delay - _LEAD

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
    assert data["train"][0]["delay"] == true_delay - _LEAD
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
    # The fake chain is y[n] = gain * x[n - delay]. The label sits ALIGNMENT_LEAD_SAMPLES
    # ahead of the true latency, so after the loader shifts by it the response lags the
    # input by exactly that lead -- the same amount on every capture, which is the point.
    delay = entry.delay
    assert delay is not None
    lead = int(_LEAD)
    aligned_y = y[delay:]
    aligned_x = x[: len(aligned_y)]
    _np.testing.assert_allclose(
        aligned_y[1000:2000], gain * aligned_x[1000 - lead : 2000 - lead], atol=2e-4
    )


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

    assert entry.delay == 480 - _LEAD
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

    assert entry.delay == 480 - _LEAD
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


# 24-bit WAVs are what the raw recordings are written as, so anything that survives the
# round trip untouched comes back within one quantization step.
_QUANTIZATION = 2.0**-23


def _read_manifest(project_dir):
    return _json.loads(
        (project_dir / _CAPTURES_RAW_DIRNAME / _RAW_MANIFEST_FILENAME).read_text()
    )


def _manifest_record(project_dir, entry):
    manifest = _read_manifest(project_dir)
    return next(
        record for record in manifest["captures"] if record["y_path"] == entry.y_path
    )


def test_raw_recordings_are_saved_exactly_as_recorded(tmp_path):
    """
    The point of captures_raw is that nothing has been done to it: no delay applied, no
    alignment shift, no resampling, nothing trimmed off either end.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)

    entries = project.pending_entries()
    session.capture_entry(entries[0])
    # The rig's sub-sample phase moves, so this capture's WAV really is shifted onto the
    # project's timebase and the raw copy has something to be different from.
    recorder.drift = 0.3
    entry = entries[1]
    qa = session.capture_entry(entry)
    assert qa.subsample_shift  # the capture was moved; the raw files must not have been

    recorded_main, recorded_loopback = recorder.returns[-1]
    amp_path, loopback_path = _raw_paths(project_dir, entry.y_path)
    for path, recorded in (
        (amp_path, recorded_main),
        (loopback_path, recorded_loopback),
    ):
        saved = _np.asarray(_wav_to_np(path)).squeeze()
        assert len(saved) == len(recorded)
        assert _np.abs(saved - recorded).max() < _QUANTIZATION

    # The whole stream, not just the part the capture WAV was cut from.
    x = _np.asarray(_wav_to_np(project_dir / "input_train.wav")).squeeze()
    assert len(recorded_main) > len(x)


def test_the_loopback_carries_the_capture_audio(tmp_path):
    """
    The loopback is played the same stream the amp is, so its recording is a clean,
    delay-bearing copy of the input as the rig actually played it.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _FakeRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)

    entry = project.pending_entries()[0]
    session.capture_entry(entry)

    x = _np.asarray(_wav_to_np(project_dir / "input_train.wav")).squeeze()
    _, loopback_path = _raw_paths(project_dir, entry.y_path)
    saved = _np.asarray(_wav_to_np(loopback_path)).squeeze()

    # Program material, not the silence a blips-only loopback would have recorded.
    assert float(_np.max(_np.abs(saved[-len(x) :]))) > 0.1 * float(_np.max(_np.abs(x)))
    # ...and the blips are still at the head of it, so the delay is still measurable.
    assert entry.delay is not None


def test_raw_amp_return_is_saved_without_a_loopback(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    project.audio.loopback_output_channel = None
    project.audio.loopback_input_channel = None
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    entry = project.pending_entries()[0]
    session.capture_entry(entry)

    amp_path, loopback_path = _raw_paths(project_dir, entry.y_path)
    assert amp_path.is_file()
    assert not loopback_path.exists()


def test_a_refused_capture_leaves_no_raw_recordings(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(
        project, project_dir, recorder=_FakeRecorder(480, loopback_silent=True)
    )
    entry = project.pending_entries()[0]

    with _pytest.raises(_CaptureSessionError):
        session.capture_entry(entry)

    # The played-stream copies belong to the inputs, not to this capture; what must not
    # exist is a recording, or a manifest record claiming one.
    assert not any(path.exists() for path in _raw_paths(project_dir, entry.y_path))
    assert not (project_dir / _CAPTURES_RAW_DIRNAME / _RAW_MANIFEST_FILENAME).exists()


def test_the_manifest_locates_the_reamp_inside_a_raw_recording(tmp_path):
    """
    What captures_raw has to support: take a raw file, trim off the preamble and tail the
    manifest describes, and be left with exactly the region the input WAV was played
    into. Everything else about the capture can be re-derived from there.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    entry = project.entries_for_split("validation")[0]
    session.capture_entry(entry)

    record = _manifest_record(project_dir, entry)
    assert record["x_path"] == "input_validation.wav"
    x = _np.asarray(_wav_to_np(project_dir / record["x_path"])).squeeze()

    for name in ("raw", "raw_loopback", "playback"):
        path = project_dir / _CAPTURES_RAW_DIRNAME / record[name]
        audio = _np.asarray(_wav_to_np(path)).squeeze()
        start = record["preamble_samples"]
        stop = len(audio) - record["tail_samples"]
        assert stop - start == len(x)


def test_the_manifest_only_describes_captures_that_happened(tmp_path):
    """
    Records are written with the audio, not with the plan, so the manifest never claims a
    file that is not there.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    manifest_path = project_dir / _CAPTURES_RAW_DIRNAME / _RAW_MANIFEST_FILENAME
    assert not manifest_path.exists()

    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)

    assert [record["y_path"] for record in _read_manifest(project_dir)["captures"]] == [
        entry.y_path
    ]


def test_recapturing_replaces_its_manifest_record(tmp_path):
    """
    A record describes the file currently on disk, so recapturing rewrites the two
    together rather than leaving a second record for the same capture.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    entry = project.pending_entries()[0]

    session.capture_entry(entry)
    session.capture_entry(entry)

    records = _read_manifest(project_dir)["captures"]
    assert [record["y_path"] for record in records] == [entry.y_path]


def test_manifest_records_name_the_files_the_capture_actually_writes(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    entry = project.pending_entries()[0]
    session.capture_entry(entry)

    record = _manifest_record(project_dir, entry)
    amp_path, loopback_path = _raw_paths(project_dir, entry.y_path)
    assert record["raw"] == amp_path.name
    assert record["raw_loopback"] == loopback_path.name
    assert amp_path.is_file() and loopback_path.is_file()
    # The pair sorts together, so a file listing can be read straight into DAW tracks.
    assert sorted((amp_path.name, loopback_path.name)) == [
        amp_path.name,
        loopback_path.name,
    ]


def test_other_manifest_records_survive_a_later_capture(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    entries = project.pending_entries()
    session.capture_entry(entries[0])
    session.capture_entry(entries[1])

    assert [record["y_path"] for record in _read_manifest(project_dir)["captures"]] == [
        entries[0].y_path,
        entries[1].y_path,
    ]


def test_a_project_from_before_raw_recordings_says_so_in_the_manifest(tmp_path):
    """
    A project carried over from a version that never wrote raw recordings will have
    captures with nothing behind them. The folder explains itself rather than looking
    like it lost files.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    # What an older project file looks like: no version was recorded when it was made.
    project.created_with_version = None
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    entries = project.pending_entries()
    session.capture_entry(entries[0])
    note = _read_manifest(project_dir)["note"]
    assert _RAW_RECORDING_SINCE_VERSION in note

    # It describes the folder's history, so it is written once and not repeated or
    # revised as more captures arrive.
    session.capture_entry(entries[1])
    assert _read_manifest(project_dir)["note"] == note


def test_a_project_created_here_has_nothing_to_explain(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    assert project.created_with_version == _CAPTURE_APP_VERSION

    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    session.capture_entry(project.pending_entries()[0])

    assert "note" not in _read_manifest(project_dir)


def test_a_project_file_without_a_version_still_opens(tmp_path):
    """
    Projects made before the version was recorded must open and carry on. The schema is
    unchanged, so this is a load, not a migration.
    """
    project_dir = _make_project_dir(tmp_path)
    path = project_dir / "capture_project.json"
    payload = _json.loads(path.read_text())
    del payload["created_with_version"]
    path.write_text(_json.dumps(payload))

    project = _load_project(project_dir)

    assert project.created_with_version is None
    assert len(project.entries) == 3
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))
    session.capture_entry(project.pending_entries()[0])
    assert project.captured_entries()


def test_the_played_stream_is_saved_for_each_input(tmp_path):
    """
    A copy of each input as it was actually played -- preamble, input, tail -- so a
    recording can be correlated against exactly what produced it.
    """
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    session = _CaptureSession(project, project_dir, recorder=_FakeRecorder(480))

    session.load_inputs()

    for name in ("input_train", "input_validation"):
        x = _np.asarray(_wav_to_np(project_dir / f"{name}.wav")).squeeze()
        path = _playback_input_path(project_dir, f"{name}.wav")
        played = _np.asarray(_wav_to_np(path)).squeeze()
        # The input sits inside it untouched, at the offset the capture uses.
        preamble = _BlipPreamble(_RATE).n_samples
        assert len(played) == preamble + len(x) + int(0.5 * _RATE)
        assert _np.abs(played[preamble : preamble + len(x)] - x).max() < _QUANTIZATION
        # ...behind the blips, so a chain that mangles the program still times.
        assert float(_np.max(_np.abs(played[:preamble]))) > 0.5


def test_a_replaced_input_regenerates_its_played_stream(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _CaptureSession(project, project_dir, recorder=_FakeRecorder(480)).load_inputs()

    path = _playback_input_path(project_dir, "input_train.wav")
    before = len(_np.asarray(_wav_to_np(path)).squeeze())

    rng = _np.random.default_rng(7)
    _np_to_wav(
        (0.1 * rng.standard_normal(2 * _RATE)).astype(_np.float32),
        project_dir / "input_train.wav",
        rate=_RATE,
    )
    _CaptureSession(project, project_dir, recorder=_FakeRecorder(480)).load_inputs()

    after = len(_np.asarray(_wav_to_np(path)).squeeze())
    assert after == before + _RATE


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
        self.returns[-1] = (recording, loopback)
        return recording, loopback


class _GlitchedBlipRecorder(_DriftingRecorder):
    """
    A rig that drops samples during the count-in, so the second blip comes back later and
    much quieter -- the real fault behind these tests: blips 129 samples apart gave a
    delay measured off one and a peak off the other, and nothing caught it.
    """

    def __init__(self, delay: int, jump: int = 130, survives: float = 0.02, **kwargs):
        super().__init__(delay, **kwargs)
        self.jump = jump
        self.survives = survives

    def playrec(self, playback, sample_rate, **kwargs):
        recording, loopback = super().playrec(playback, sample_rate, **kwargs)
        if loopback is not None:
            loopback = loopback.copy()
            second = _BlipPreamble(sample_rate=sample_rate).blip_locations[1]
            lo, hi = second, second + 20_000
            response = loopback[lo:hi].copy()
            loopback[lo:hi] = 0.0
            loopback[lo - self.jump : hi - self.jump] += (
                response * self.survives
            ).astype(_np.float32)
            loopback = loopback.astype(_np.float32)
        self.returns[-1] = (recording, loopback)
        return recording, loopback


def _timebase(entry) -> float:
    """
    Where a written capture sits: its label less the shift applied to it. Two captures of
    the same rig are aligned exactly when this tracks the rig's true latency one-for-one.
    """
    return entry.delay - (entry.qa.subsample_shift or 0.0)


def test_captures_share_a_timebase_without_sharing_any_state(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)

    entries = project.pending_entries()
    session.capture_entry(entries[0])
    # The rig's sub-sample phase moves between captures (the interface re-clocked). The
    # written timebase has to follow it exactly, so that what the model sees does not.
    recorder.drift = 0.3
    session.capture_entry(entries[1])

    assert _timebase(entries[1]) - _timebase(entries[0]) == _pytest.approx(
        0.3, abs=0.05
    )
    # Nothing measures a timebase into the project file any more.
    assert (
        _json.loads((project_dir / "capture_project.json").read_text())[
            "alignment_reference"
        ]
        is None
    )


def test_a_captures_alignment_does_not_depend_on_what_was_captured_before(tmp_path):
    # The regression this change is for: a capture whose blips disagree used to define the
    # project's timebase, so one bad measurement silently moved every good one after it.
    # Alignment must now depend on nothing but the capture's own recording.
    def timebase_of_second_entry(directory, first_recorder) -> float:
        project = _load_project(directory)
        _enable_loopback(project)
        recorder = first_recorder
        session = _CaptureSession(project, directory, recorder=recorder)
        entries = project.pending_entries()
        session.capture_entry(entries[0])
        session._recorder = _DriftingRecorder(480, drift=0.25)
        session.capture_entry(entries[1])
        return _timebase(entries[1])

    after_good = timebase_of_second_entry(
        _make_project_dir(tmp_path / "good"), _DriftingRecorder(480)
    )
    after_glitched = timebase_of_second_entry(
        _make_project_dir(tmp_path / "glitched"), _GlitchedBlipRecorder(480)
    )

    assert after_glitched == _pytest.approx(after_good, abs=1e-9)


def test_disagreeing_blips_are_flagged_with_both_readings(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(
        project, project_dir, recorder=_GlitchedBlipRecorder(480, jump=130)
    )

    qa = session.capture_entry(project.pending_entries()[0])

    assert len(qa.blip_delays) == 2
    assert abs(qa.blip_delays[0] - qa.blip_delays[1]) >= 20
    # The message has to name the two readings and say which capture is at fault, since
    # the failure it describes is local to this one.
    (message,) = [m for m in qa.messages if "timing blips came back" in m]
    assert all(str(d) in message for d in qa.blip_delays)
    assert "Record this one again" in message
    # The loopback cross-check is downstream of the same broken measurement, so it
    # must not also fire and send the user after a cable.
    assert not any("check the loopback patch" in m.lower() for m in qa.messages)


def test_shift_is_only_ever_the_rounding_residue_of_the_label(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)

    for entry, drift in zip(project.pending_entries(), (0.0, 0.4, -0.45)):
        recorder.drift = drift
        qa = session.capture_entry(entry)
        assert abs(qa.subsample_shift or 0.0) <= _MAX_ALIGNMENT_SHIFT + 1e-9


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
    assert qa.loopback_delay is None
    assert qa.amp_return_delay == 480 - 1


def test_a_legacy_project_keeps_the_timebase_its_captures_were_written_against(tmp_path):
    # A project part-captured before the timebase became stateless must reproduce its own
    # recorded offset, or captures from before and after the upgrade land (reference-lead)
    # samples apart -- the phase error alignment exists to remove.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480, drift=0.2)
    session = _CaptureSession(project, project_dir, recorder=recorder)
    # Stand in for what the old app left behind: a capture, and the offset it was
    # written against.
    session.capture_entry(project.pending_entries()[0])
    legacy = 11.0
    project.alignment_reference = legacy

    entry = project.pending_entries()[0]
    session.capture_entry(entry)

    # Labelled against the project's own offset rather than the constant, so it sits on
    # the same timebase as whatever was captured before the upgrade.
    assert _timebase(entry) == _pytest.approx(480.2 - legacy, abs=0.05)
    # And it is still only ever read: nothing measures a new one in.
    assert project.alignment_reference == legacy


def test_a_poisoned_legacy_timebase_is_refused_before_anything_is_recorded(tmp_path):
    # The original fault wrote 129 samples, from a first capture whose blips disagreed.
    # Reproducing it would spread that one bad measurement into every remaining capture.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)
    # A capture already in the project, then the poisoned reference it was written
    # against: that is the state a project part-captured under the old scheme is in.
    session.capture_entry(project.pending_entries()[0])
    project.alignment_reference = 129.0
    recorder.calls.clear()
    entry = project.pending_entries()[0]

    with _pytest.raises(_CaptureSessionError) as excinfo:
        session.capture_entry(entry)

    assert "129" in str(excinfo.value)
    assert "record them again" in str(excinfo.value)
    # Refused before the rig was driven at all, not after the user sat through a capture.
    assert recorder.calls == []
    assert entry.status == "pending"
    assert not (project_dir / entry.y_path).exists()


def test_the_legacy_bound_is_the_boundary(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    # The reference only applies while there are captures written against it.
    session.capture_entry(project.pending_entries()[0])

    project.alignment_reference = _MAX_LEGACY_REFERENCE
    assert session._alignment_lead() == _MAX_LEGACY_REFERENCE
    project.alignment_reference = _MAX_LEGACY_REFERENCE + 0.01
    with _pytest.raises(_CaptureSessionError):
        session._alignment_lead()
    # A project with no reference is not a legacy project; it uses the constant.
    project.alignment_reference = None
    assert session._alignment_lead() == _LEAD


def test_an_unusable_reference_describing_no_captures_is_dropped_not_refused(tmp_path):
    # A poisoned reference with nothing written against it cannot be acted on by the
    # advice the refusal gives ("clear the captures"), because there are none to clear.
    # It describes nothing, so it is dropped and the project starts fresh.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    project.alignment_reference = 129.0
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    assert not project.captured_entries()

    qa = session.capture_entry(project.pending_entries()[0])

    assert qa.peak_delay is not None
    # Dropped, not just skipped -- otherwise reopening the project would put a reference
    # back in force over captures that were never written against it.
    assert project.alignment_reference is None
    assert _load_project(project_dir).alignment_reference is None


def test_a_recorded_lead_survives_having_no_captures_right_now(tmp_path):
    # The hazard of releasing on empty: regenerating with a different seed renames every
    # entry, so the files stop matching any planned y_path and the project reads as empty.
    # Releasing there and taking it back when the old seed returns would strand whatever
    # was captured in between on a different timebase, silently.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    legacy = 4.43
    project.alignment_reference = legacy
    assert not _has_captures(project, project_dir)

    # Used even with nothing to match right now, and still there afterwards.
    assert session._alignment_lead() == _pytest.approx(legacy)
    entry = project.pending_entries()[0]
    session.capture_entry(entry)
    assert project.alignment_reference == _pytest.approx(legacy)
    # And the capture that was just made sits on it, not on the constant.
    assert _timebase(entry) == _pytest.approx(480.0 - legacy, abs=0.05)


def test_captures_made_while_entries_are_pending_join_the_recorded_timebase(tmp_path):
    # What plan regeneration produces: the lead carried forward, files on disk but their
    # entries pending, and a new capture made before the restore offer is accepted. It has
    # to land on the same timebase as the files waiting to be imported, or importing them
    # afterwards mixes two timebases in one project.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entries = project.pending_entries()
    # A project already on a recorded lead, with a capture written against it.
    legacy = 4.43
    project.alignment_reference = legacy
    session.capture_entry(entries[0])
    first_timebase = _timebase(entries[0])

    # Post-regeneration: the entry is pending again but its WAV is still on disk, and
    # the lead was carried forward with the rebuilt project file.
    entries[0].status = "pending"
    assert not project.captured_entries()

    session.capture_entry(entries[1])

    # Both on the recorded lead, so re-importing the first one later is consistent.
    assert _timebase(entries[1]) == _pytest.approx(first_timebase, abs=0.05)
    assert project.alignment_reference == _pytest.approx(legacy)


def test_nothing_a_capture_does_writes_a_new_reference(tmp_path):
    # No capture measures a timebase into the project any more, in any state. The only
    # things that set it are carrying it forward and the user adopting a measured one.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)

    for entry, drift in zip(project.pending_entries(), (0.0, 0.3, -0.2)):
        recorder.drift = drift
        session.capture_entry(entry)
        assert project.alignment_reference is None
    assert _load_project(project_dir).alignment_reference is None


def test_a_healthy_project_reports_no_timebase_problem(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))

    assert _timebase_problem(project, project_dir) is None
    session.capture_entry(project.pending_entries()[0])
    assert _timebase_problem(project, project_dir) is None

    # A plausible legacy offset is a timebase to honour, not a problem to report.
    project.alignment_reference = 11.0
    assert _timebase_problem(project, project_dir) is None


def test_timebase_problem_is_answerable_without_running_a_capture(tmp_path):
    # The GUI asks this before capturing, so it must not need anything a capture
    # produces -- and it must agree with the refusal the capture itself would raise.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)
    session.capture_entry(project.pending_entries()[0])
    project.alignment_reference = 129.0
    recorder.calls.clear()

    problem = _timebase_problem(project, project_dir)
    assert problem is not None
    assert recorder.calls == []
    with _pytest.raises(_CaptureSessionError) as excinfo:
        session.capture_entry(project.pending_entries()[0])
    assert str(excinfo.value) == problem


def test_clearing_captures_deletes_the_files_and_the_timebase(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)
    project.alignment_reference = 129.0
    raw, raw_loopback = _raw_paths(project_dir, entry.y_path)
    assert (project_dir / entry.y_path).is_file() and raw.is_file()

    notes = _clear_captures(project, project_dir)

    assert notes
    assert entry.status == "pending"
    assert entry.delay is None and entry.qa is None
    # The WAV must actually go: a pending entry whose file is still there gets offered
    # back as recoverable and would be marked captured again, undoing this.
    assert not (project_dir / entry.y_path).exists()
    assert not _find_recoverable_entries(project, project_dir)
    # The raw recordings stay -- they are the only record of what the rig did.
    assert raw.is_file() and raw_loopback.is_file()
    # And the timebase those captures were written against goes with them.
    assert project.alignment_reference is None
    assert _timebase_problem(project, project_dir) is None


def test_a_timebase_survives_its_captures_being_pending_pre_recovery(tmp_path):
    # Regenerating the plan leaves entries pending while their WAVs stay on disk. Reading
    # only the statuses in that window released the timebase and let the next capture be
    # made against a different one -- which is how a real project ended up with two
    # captures on one timebase and a third on another, undetectably.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)
    legacy = 11.0
    project.alignment_reference = legacy

    # The pre-recovery window: pending again, but the capture file is still there.
    entry.status = "pending"
    assert not project.captured_entries()
    assert (project_dir / entry.y_path).is_file()

    assert session._alignment_lead() == legacy
    assert project.alignment_reference == legacy
    # And a poisoned one is still refused rather than quietly released.
    project.alignment_reference = 129.0
    assert _timebase_problem(project, project_dir) is not None
    with _pytest.raises(_CaptureSessionError):
        session.capture_entry(project.pending_entries()[0])


def test_clear_captures_reaches_a_pending_entry_whose_file_is_still_on_disk(tmp_path):
    # The refusal above says to clear captures and retry, so that must work from the same
    # pending-but-file-present state: a surviving WAV is what has_captures() keys off, so
    # clear_captures() has to reach it too or the fix it points at is a no-op.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)
    project.alignment_reference = 129.0

    # The pre-recovery window: pending again, but the capture file is still there.
    entry.status = "pending"
    assert not project.captured_entries()
    assert (project_dir / entry.y_path).is_file()
    assert _find_clearable_entries(project, project_dir) == [entry]

    notes = _clear_captures(project, project_dir)

    assert notes
    assert not (project_dir / entry.y_path).exists()
    assert project.alignment_reference is None
    assert _timebase_problem(project, project_dir) is None
    assert _find_clearable_entries(project, project_dir) == []


def test_audit_finds_a_misaligned_capture_without_any_stored_timing(tmp_path):
    # The check that works when everything else is lost: a project whose plan was
    # regenerated has no measured QA left, so nothing in the project file can say whether
    # its captures agree -- but the raw recordings can, and they cannot go stale.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)
    entries = project.pending_entries()
    session.capture_entry(entries[0])
    session.capture_entry(entries[1])
    # Put the second one somewhere else on the timebase, as a capture written under a
    # different scheme would be, and strip the QA the way plan regeneration does.
    entries[1].delay += 3
    for entry in project.captured_entries():
        entry.qa.peak_delay = None
        entry.qa.subsample_shift = None
        entry.qa.blip_delays = []

    audits = _audit_captures(project, project_dir)
    problems = _audit_problems(audits)

    assert all(a.residual is not None for a in audits)
    # Only the capture that actually moved is named. The one that is where the project's
    # alignment puts it must not be reported alongside it.
    assert any(entries[1].y_path in p and "samples away" in p for p in problems)
    assert not any(entries[0].y_path in p for p in problems)


def test_audit_passes_a_healthy_set_and_flags_disagreeing_blips(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)
    entries = project.pending_entries()
    session.capture_entry(entries[0])
    # A different sub-sample phase is not a misalignment: the shift takes it out, which
    # is the whole point, so the audit must not report it.
    recorder.drift = 0.3
    session.capture_entry(entries[1])

    assert _audit_problems(_audit_captures(project, project_dir)) == []

    session._recorder = _GlitchedBlipRecorder(480)
    session.capture_entry(entries[2])
    problems = _audit_problems(_audit_captures(project, project_dir))
    assert any("timing blips came back" in p and entries[2].y_path in p
               for p in problems)


def test_audit_reports_captures_it_cannot_check_rather_than_passing_them(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)
    # As a project made before captures_raw/ existed would be.
    _, raw_loopback = _raw_paths(project_dir, entry.y_path)
    raw_loopback.unlink()

    (audit,) = _audit_captures(project, project_dir)

    assert audit.residual is None
    assert audit.unchecked is not None
    assert any("not checked" in p for p in _audit_problems([audit]))


def _put_set_on_a_larger_lead(project, by: float) -> None:
    """
    Put every capture on a lead ``by`` samples larger than the project's, the way a set
    written under an older scheme sits -- uniformly, agreeing with each other exactly.
    The audit then reads each one as ``-by`` from where the project would put a capture,
    which is the shape of the real failure (an iD44 set reading -0.43 against a lead of
    4.0, having been written against 4.43).
    """
    for entry in project.captured_entries():
        entry.qa.subsample_shift = (entry.qa.subsample_shift or 0.0) + by


def test_a_uniformly_offset_set_is_reported_as_one_fact_not_as_disagreement(tmp_path):
    # The real project this came from: three captures on an older lead, agreeing with each
    # other to 0.000000 samples, every one reported as "away from the rest" -- away from
    # captures it matched exactly. Such a set has one thing wrong with it, not three.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    recorder = _DriftingRecorder(480)
    session = _CaptureSession(project, project_dir, recorder=recorder)
    for entry, drift in zip(project.pending_entries(), (0.0, 0.3, -0.2)):
        recorder.drift = drift
        session.capture_entry(entry)
    _put_set_on_a_larger_lead(project, 0.43)

    audits = _audit_captures(project, project_dir)
    offsets = [a.offset for a in audits]
    assert max(offsets) - min(offsets) < 1e-6  # they agree with each other exactly
    assert _shared_offset(audits) == _pytest.approx(-0.43, abs=0.02)

    problems = _audit_problems(audits)

    (problem,) = problems
    assert "agree with each other" in problem
    # Not named individually, and not told to recapture what is not damaged.
    assert not any(a.y_path in problem for a in audits)
    assert "Record it again" not in problem


def test_captures_from_different_schemes_are_named_individually(tmp_path):
    # A genuine mixture -- some captures on one timebase, some on another -- is not a
    # uniform offset, so it must fall back to naming the captures that disagree rather
    # than reporting the set as coherent.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entries = project.pending_entries()
    for entry in entries[:3]:
        session.capture_entry(entry)
    # One capture left where it is; the other two moved onto an older lead.
    for entry in entries[1:3]:
        entry.qa.subsample_shift = (entry.qa.subsample_shift or 0.0) + 0.43

    audits = _audit_captures(project, project_dir)
    assert _shared_offset(audits) is None
    problems = _audit_problems(audits)

    assert all(
        any(entry.y_path in p for p in problems) for entry in entries[1:3]
    )
    assert not any(entries[0].y_path in p for p in problems)


def _played(rate=_RATE, blips=(0.5, 1.5), n_pre=None, amplitude=0.9):
    """A preamble laid out however we like, plus the length it was played at."""
    from nam.capture.latency import PlayedPreamble as _PlayedPreamble

    n_pre = n_pre or int(2.25 * rate)
    playback = _np.zeros(n_pre + rate, dtype=_np.float32)
    for t in blips:
        playback[int(t * rate)] = amplitude
    return playback, n_pre, _PlayedPreamble


def test_the_preamble_layout_is_read_from_the_signal_not_from_constants(tmp_path):
    from nam.capture.latency import BlipPreamble as _BP
    from nam.capture.latency import PlayedPreamble as _PP

    from nam.capture.latency import _NOISE_INTERVAL_SECONDS as _NOISE

    # The layout this version happens to ship reproduces exactly, so nothing about an
    # existing project's measurement moves. Checked against the constants themselves,
    # since BlipPreamble no longer carries a second description of them to compare with.
    for rate in (44_100, _RATE, 96_000):
        spec = _BP(sample_rate=rate)
        read = spec.as_played()
        assert read.blip_locations == spec.blip_locations
        assert read.noise_interval == (
            int(_NOISE[0] * rate), int(_NOISE[1] * rate)
        )


def test_a_changed_preamble_needs_no_version_gate(tmp_path):
    # The point of reading the layout back: blips elsewhere, more of them, at another
    # amplitude is not a migration, because the signal says where they are.
    playback, n_pre, _PP = _played(blips=(0.2, 0.9, 1.7), amplitude=0.4)
    read = _PP.from_playback(playback, n_pre, _RATE)

    assert read.blip_locations == (
        int(0.2 * _RATE), int(0.9 * _RATE), int(1.7 * _RATE)
    )
    # The noise window still lands in the silence before the first blip, wherever it is.
    assert read.noise_interval[1] <= read.blip_locations[0]


def test_a_silent_second_half_yields_one_reading_and_no_false_disagreement(tmp_path):
    # Why the preamble is not split blindly in two: a second half carrying nothing would
    # correlate noise against silence and report the capture's blips as disagreeing.
    # Reading the layout finds one impulse, so there is nothing to compare and none is.
    playback, n_pre, _PP = _played(blips=(0.5,))
    read = _PP.from_playback(playback, n_pre, _RATE)

    assert read.blip_locations == (int(0.5 * _RATE),)

    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    session.capture_entry(project.pending_entries()[0])
    (audit,) = _audit_captures(project, project_dir)
    single = [a for a in [audit] if len(a.blip_delays) < 2]
    assert not any(a.blips_disagree for a in single)


def test_a_preamble_with_no_impulses_is_refused_rather_than_guessed(tmp_path):
    from nam.capture.latency import PlayedPreamble as _PP

    with _pytest.raises(ValueError):
        _PP.from_playback(_np.zeros(1000, dtype=_np.float32), 1000, _RATE)


def test_audit_measures_against_the_preamble_the_capture_was_played(tmp_path):
    # It reads captures_raw/, so removing the played copy is what makes it unmeasurable
    # -- not anything about which version wrote the capture.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)
    assert _audit_problems(_audit_captures(project, project_dir)) == []

    # Drop the played copy the manifest points at.
    import json as _j
    man = _j.loads(
        (project_dir / _CAPTURES_RAW_DIRNAME / _RAW_MANIFEST_FILENAME).read_text()
    )
    record = next(r for r in man['captures'] if r['y_path'] == entry.y_path)
    (project_dir / _CAPTURES_RAW_DIRNAME / record['playback']).unlink()
    (audit,) = _audit_captures(project, project_dir)
    assert audit.unchecked is not None


def test_recovery_re_measures_instead_of_inventing_a_result(tmp_path):
    # Recovery used to fabricate a LatencyResult from data.json's delay, leaving the
    # restored QA with no peak_delay, blip_delays or subsample_shift -- so an aligned
    # capture came back looking like one that never was, and the audit called it
    # misaligned. Detection is one path now, so recovery matches what the capture wrote.
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480, drift=0.3))
    entries = project.pending_entries()
    for entry in entries[:2]:
        session.capture_entry(entry)
    original = {e.y_path: (e.delay, e.qa.subsample_shift, e.qa.peak_delay)
                for e in project.captured_entries()}
    assert _audit_problems(_audit_captures(project, project_dir)) == []

    # Plan regenerated: entries pending, WAVs and data.json still on disk.
    for entry in project.entries:
        entry.status, entry.qa = "pending", None
    session.recover_captured_from_disk(
        _find_recoverable_entries(project, project_dir)
    )

    for entry in project.captured_entries():
        delay, shift, peak = original[entry.y_path]
        assert entry.delay == delay
        assert entry.qa.peak_delay == _pytest.approx(peak, abs=1e-9)
        assert (entry.qa.subsample_shift or 0.0) == _pytest.approx(shift or 0.0, abs=1e-9)
        assert len(entry.qa.blip_delays) == 2
    # And the audit still passes, rather than reporting the whole set as offset because
    # the shift it was written with had been forgotten.
    assert _audit_problems(_audit_captures(project, project_dir)) == []


def test_recovery_still_works_when_nothing_was_kept_to_re_measure(tmp_path):
    project_dir = _make_project_dir(tmp_path)
    project = _load_project(project_dir)
    _enable_loopback(project)
    session = _CaptureSession(project, project_dir, recorder=_DriftingRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)
    delay = entry.delay
    # As a project made before captures_raw/ existed.
    _, raw_loopback = _raw_paths(project_dir, entry.y_path)
    raw_loopback.unlink()
    for e in project.entries:
        e.status, e.qa = "pending", None

    notes = session.recover_captured_from_disk(
        _find_recoverable_entries(project, project_dir)
    )

    restored = project.captured_entries()[0]
    assert restored.delay == delay
    assert any("not re-measured" in n for n in notes)
