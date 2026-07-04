import json as _json
from pathlib import Path as _Path

import numpy as _np
import pytest as _pytest

from nam.capture.params import KnobSpec as _KnobSpec
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
    """

    def __init__(self, delay: int, gain: float = 0.5, noise: float = 1e-5):
        self.delay = delay
        self.gain = gain
        self.noise = noise
        self.calls: list[dict] = []

    def playrec(self, playback, sample_rate, **kwargs):
        self.calls.append(dict(kwargs, sample_rate=sample_rate))
        progress = kwargs.get("progress")
        if progress is not None:
            progress(1.0)
        recording = _np.zeros_like(playback)
        recording[self.delay :] = playback[: len(playback) - self.delay] * self.gain
        rng = _np.random.default_rng(0)
        return recording + self.noise * rng.standard_normal(len(playback)).astype(
            _np.float32
        )


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
