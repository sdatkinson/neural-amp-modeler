"""
Capture-session orchestration: turn one planned entry at a time into a recorded,
delay-measured, QA-checked WAV, persisting after every step.

Per capture: build the playback (blip preamble + input audio + tail), play/record it,
measure the round-trip delay from the blips, run QA, save the capture WAV, then
rewrite both ``capture_project.json`` and ``data.json`` so a crash or app close
never loses a completed capture.
"""

from __future__ import annotations

import os as _os
import tempfile as _tempfile
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Sequence as _Sequence

import numpy as _np

from .audio import PlaybackRecorder as _PlaybackRecorder
from .export import update_data_json as _update_data_json
from .latency import BlipPreamble as _BlipPreamble
from .latency import LatencyResult as _LatencyResult
from .latency import measure_delay as _measure_delay
from .project import CaptureEntryModel as _CaptureEntryModel
from .project import CaptureProject as _CaptureProject
from .project import QAModel as _QAModel
from .project import mark_captured as _mark_captured
from .project import save_project as _save_project


# The parametric dataset refuses clipped output outright (nam.data
# _validate_inputs_after_processing raises at abs(y).max() >= 1.0), so a clipping QA
# flag means "re-record at a lower level", not just "sounds bad".
CLIPPING_THRESHOLD = 0.999
# Matches the ensemble-disagreement threshold inside the wrapped NAM calibration.
DELAY_DISAGREEMENT_SAMPLES = 20
# Extra playback beyond the input audio so the delayed response tail is still inside
# the stream when the recording stops.
TAIL_SECONDS = 0.5


class CaptureSessionError(RuntimeError):
    pass


@_dataclass(frozen=True)
class RouteTestResult:
    latency: _LatencyResult
    peak: float

    @property
    def ok(self) -> bool:
        return self.latency.detected


class CaptureSession:
    """
    Drives captures for one project. The recorder is injected so tests (and a future
    simulator mode) can run the full pipeline without audio hardware.
    """

    def __init__(
        self,
        project: _CaptureProject,
        project_dir: _Path,
        recorder: _Optional[_PlaybackRecorder] = None,
    ):
        self.project = project
        self.project_dir = _Path(project_dir)
        if recorder is None:
            from .audio import SounddeviceRecorder

            recorder = SounddeviceRecorder()
        self._recorder = recorder
        self._inputs: dict[str, _np.ndarray] = {}

    def load_inputs(self) -> int:
        """
        Load (and cache) both input WAVs, verify they are mono and share one sample
        rate, record that rate on the project, and return it.
        """
        from ..data import wav_to_np

        rates = {}
        for split in ("train", "validation"):
            relative = self.project.input_for_split(split)
            path = self.project_dir / relative
            if not path.is_file():
                raise CaptureSessionError(f"Missing {split} input WAV: {path}")
            x, info = wav_to_np(path, info=True)
            self._inputs[split] = _np.asarray(x, dtype=_np.float32).squeeze()
            rates[relative] = info.rate
        if len(set(rates.values())) != 1:
            summary = ", ".join(f"{name}: {rate} Hz" for name, rate in rates.items())
            raise CaptureSessionError(
                f"Train and validation inputs must share one sample rate; got {summary}"
            )
        rate = int(next(iter(rates.values())))
        self.project.sample_rate = rate
        return rate

    def _input_for_split(self, split: str) -> tuple[_np.ndarray, int]:
        if split not in self._inputs:
            self.load_inputs()
        assert self.project.sample_rate is not None
        return self._inputs[split], self.project.sample_rate

    def _resolve_devices(self) -> dict[str, _Optional[int]]:
        from .audio import find_device

        audio = self.project.audio
        resolved: dict[str, _Optional[int]] = {}
        for kind, name in (
            ("output", audio.output_device),
            ("input", audio.input_device),
        ):
            resolved[f"{kind}_device"] = (
                None
                if name is None
                else find_device(name, kind=kind, host_api=audio.host_api).index
            )
        return resolved

    def _playrec(
        self,
        playback: _np.ndarray,
        sample_rate: int,
        progress: _Optional[_Callable[[float], None]],
        cancel: _Optional[_Callable[[], bool]],
    ) -> _np.ndarray:
        recording = self._recorder.playrec(
            playback,
            sample_rate,
            output_channel=self.project.audio.output_channel,
            input_channel=self.project.audio.input_channel,
            blocksize=self.project.audio.blocksize,
            progress=progress,
            cancel=cancel,
            **self._resolve_devices(),
        )
        recording = _np.asarray(recording, dtype=_np.float32).squeeze()
        if recording.shape != playback.shape:
            raise CaptureSessionError(
                f"Recorder returned shape {recording.shape}; expected {playback.shape}"
            )
        return recording

    def route_test(
        self,
        sample_rate: _Optional[int] = None,
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> RouteTestResult:
        """
        Play just the blip preamble through the configured route and check that the
        impulses come back. Verifies routing and measures delay without a full pass.
        """
        if sample_rate is None:
            sample_rate = (
                self.project.sample_rate
                if self.project.sample_rate is not None
                else self.load_inputs()
            )
        preamble = _BlipPreamble(sample_rate)
        tail = _np.zeros(int(TAIL_SECONDS * sample_rate), dtype=_np.float32)
        playback = _np.concatenate([preamble.render(), tail])
        recording = self._playrec(playback, sample_rate, progress, cancel)
        return RouteTestResult(
            latency=_measure_delay(recording, preamble),
            peak=float(_np.max(_np.abs(recording))),
        )

    def capture_entry(
        self,
        entry: _CaptureEntryModel,
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> _QAModel:
        """
        Record one planned entry and persist everything. Returns the QA report; the
        entry is marked captured even when QA raises flags (the WAV exists and is
        saved), and the flags tell the user whether to recapture.
        """
        x, sample_rate = self._input_for_split(entry.split)
        preamble = _BlipPreamble(sample_rate)
        tail = _np.zeros(int(TAIL_SECONDS * sample_rate), dtype=_np.float32)
        playback = _np.concatenate([preamble.render(), x, tail])

        recording = self._playrec(playback, sample_rate, progress, cancel)

        latency = _measure_delay(recording, preamble)
        y = recording[preamble.n_samples : preamble.n_samples + len(x)]
        qa = self._qa(entry, y, latency)

        self._write_capture_wav(entry, y, sample_rate)
        _mark_captured(entry, delay=latency.delay, qa=qa)
        _save_project(self.project, self.project_dir)
        _update_data_json(self.project, self.project_dir)
        return qa

    def recover_captured_from_disk(
        self,
        recoverable: _Sequence[tuple[_CaptureEntryModel, _Optional[int]]],
    ) -> list[str]:
        """
        Restore entries whose capture WAV already exists on disk to "captured" without
        recapturing (see :func:`nam.capture.project.find_recoverable_entries`).

        The delay comes from data.json; QA is reconstructed from the WAV and
        ``captured_at`` is stamped now, since data.json records neither. Persists the
        project and data.json once at the end. Returns a note per entry.
        """
        from ..data import wav_to_np

        notes: list[str] = []
        for entry, delay in recoverable:
            path = self.project_dir / entry.y_path
            try:
                y = _np.asarray(wav_to_np(path), dtype=_np.float32).squeeze()
            except Exception as exc:
                notes.append(
                    f"{entry.y_path}: could not read WAV to restore ({exc})."
                )
                continue
            # data.json proves the delay was measured, so treat the impulse as detected;
            # only the disagreement-vs-history check inside _qa needs re-running.
            latency = _LatencyResult(
                delay=delay,
                detected=delay is not None,
                disagreement_too_high=False,
                safety_factor=0,
            )
            qa = self._qa(entry, y, latency)
            _mark_captured(entry, delay=delay, qa=qa)
            notes.append(f"{entry.y_path}: restored from disk (delay={delay}).")
        if notes:
            _save_project(self.project, self.project_dir)
            _update_data_json(self.project, self.project_dir)
        return notes

    def _qa(
        self,
        entry: _CaptureEntryModel,
        y: _np.ndarray,
        latency: _LatencyResult,
    ) -> _QAModel:
        messages: list[str] = []

        peak = float(_np.max(_np.abs(y))) if len(y) else 0.0
        clipping = peak >= CLIPPING_THRESHOLD
        if clipping:
            messages.append(
                f"Clipping: peak {peak:.3f}. Training rejects clipped captures — "
                "lower the level and recapture."
            )

        if not latency.detected:
            messages.append(
                "No impulse detected: the delay could not be measured. Check routing "
                "and levels, then recapture."
            )
        elif latency.disagreement_too_high:
            messages.append(
                "The two timing blips disagree about the delay; the measurement may "
                "be unreliable."
            )

        delay_disagreement = False
        if latency.delay is not None:
            other_delays = [
                other.delay
                for other in self.project.captured_entries()
                if other.delay is not None and other is not entry
            ]
            if other_delays and (
                abs(latency.delay - int(_np.median(other_delays)))
                >= DELAY_DISAGREEMENT_SAMPLES
            ):
                delay_disagreement = True
                messages.append(
                    f"Delay {latency.delay} differs from this project's typical "
                    f"{int(_np.median(other_delays))} by {DELAY_DISAGREEMENT_SAMPLES}+ "
                    "samples. Did the routing or device settings change?"
                )

        if peak < 1e-4:
            messages.append(
                f"Capture is near-silent (peak {peak:.6f}). Is the return input "
                "connected and the device under test on?"
            )

        return _QAModel(
            peak=peak,
            clipping=clipping,
            impulse_detected=latency.detected,
            delay_disagreement=delay_disagreement,
            messages=messages,
        )

    def _write_capture_wav(
        self, entry: _CaptureEntryModel, y: _np.ndarray, sample_rate: int
    ) -> None:
        from ..data import np_to_wav

        path = self.project_dir / entry.y_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = _tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp.wav", dir=path.parent
        )
        _os.close(fd)
        try:
            np_to_wav(y, tmp_path, rate=sample_rate)
            _os.replace(tmp_path, path)
        except BaseException:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            raise
