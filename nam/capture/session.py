"""
Capture-session orchestration: turn one planned entry at a time into a recorded,
delay-measured, QA-checked WAV, persisting after every step.

Per capture: build the playback (blip preamble + input audio + tail), play/record it,
measure the round-trip delay from the blips, shift the target onto the project's
timebase, run QA, save the capture WAV, then rewrite both ``capture_project.json`` and
``data.json`` so a crash or app close never loses a completed capture.

The alignment step exists because ``delay`` is an integer while the rig's latency is not.
A capture set aligned only to whole samples carries up to half a sample of per-capture
error, and that error is not a function of the knobs, so a parametric model cannot learn
it as a rule -- it can only memorise a per-capture phase. Correcting it at write time
keeps ``delay`` an integer and leaves everything downstream unchanged. See
:meth:`CaptureSession._alignment_shift` for what is measured and why, and
:mod:`nam.capture.resample` for the filter.
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
from .audio import peak_to_dbfs as _peak_to_dbfs
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
DELAY_DISAGREEMENT_SAMPLES = 10
# When a clean loopback is used to measure the delay, the amp-return blip is still
# detected as a cross-check. On a correctly patched rig the two travel the same route and
# agree to within a sample; a larger gap means a mispatched loopback or a genuine routing
# problem. This was 3, which is wider than the ~3-sample drift a mispatched loopback
# actually produced, so the check never fired across a whole 93-capture project. It has to
# sit below the drift it is meant to catch to be worth anything.
LOOPBACK_CROSSCHECK_SAMPLES = 1
# Shared between the route test and the actual capture so the two surfaces never
# describe this failure differently.
LOOPBACK_NOT_DETECTED_MESSAGE = (
    "Loopback is enabled but no blip was detected on the loopback input, so the "
    "measured delay cannot be trusted as coming from the loopback. Check the "
    "loopback cable and patch, or uncheck 'Use a second I/O pair as a clean "
    "delay-detection loopback' in Audio settings to capture from the amp return "
    "only."
)
# Extra playback beyond the input audio so the delayed response tail is still inside
# the stream when the recording stops.
TAIL_SECONDS = 0.5
# Alignment corrections beyond this are refused rather than applied: at that size the
# measurement itself is suspect, and silently sliding a capture several samples is worse
# than leaving it where it is and saying so.
MAX_ALIGNMENT_SHIFT = 4.0


class CaptureSessionError(RuntimeError):
    pass


@_dataclass(frozen=True)
class RouteTestResult:
    # ``latency`` is the authoritative measurement (the clean loopback when one is
    # configured, otherwise the amp return). ``crosscheck`` is the amp-return
    # measurement when a loopback was used, kept so the UI can show both.
    latency: _LatencyResult
    peak: float  # dBFS, not linear amplitude
    loopback_used: bool = False
    crosscheck: _Optional[_LatencyResult] = None
    loopback_disagreement: bool = False
    # A loopback was configured but its blips were not detected (e.g. an unplugged
    # cable). Distinct from ``not latency.detected``: that also covers "no loopback
    # configured and the amp return itself found nothing", which needs a different
    # message since there is no loopback to blame.
    loopback_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.latency.detected and not self.loopback_failed


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

    @staticmethod
    def _loopback_playback(
        playback: _np.ndarray, preamble: _BlipPreamble
    ) -> _np.ndarray:
        """
        The signal for the loopback output channel: only the timing blips, followed by
        silence for the rest of the playback. It carries no program material, so the
        loopback stays clean regardless of the input audio.
        """
        loopback = _np.zeros_like(playback)
        loopback[: preamble.n_samples] = preamble.render()
        return loopback

    def _playrec(
        self,
        playback: _np.ndarray,
        sample_rate: int,
        progress: _Optional[_Callable[[float], None]],
        cancel: _Optional[_Callable[[], bool]],
        loopback_playback: _Optional[_np.ndarray] = None,
    ) -> tuple[_np.ndarray, _Optional[_np.ndarray]]:
        audio = self.project.audio
        use_loopback = audio.loopback_enabled and loopback_playback is not None
        main, loopback = self._recorder.playrec(
            playback,
            sample_rate,
            output_channel=audio.output_channel,
            input_channel=audio.input_channel,
            loopback_output_channel=(
                audio.loopback_output_channel if use_loopback else None
            ),
            loopback_input_channel=(
                audio.loopback_input_channel if use_loopback else None
            ),
            loopback_playback=loopback_playback if use_loopback else None,
            blocksize=audio.blocksize,
            latency=audio.latency,
            progress=progress,
            cancel=cancel,
            **self._resolve_devices(),
        )
        main = _np.asarray(main, dtype=_np.float32).squeeze()
        if main.shape != playback.shape:
            raise CaptureSessionError(
                f"Recorder returned shape {main.shape}; expected {playback.shape}"
            )
        if loopback is not None:
            loopback = _np.asarray(loopback, dtype=_np.float32).squeeze()
            if loopback.shape != playback.shape:
                raise CaptureSessionError(
                    f"Recorder returned loopback shape {loopback.shape}; expected "
                    f"{playback.shape}"
                )
        return main, loopback

    def _resolve_latency(
        self,
        main_recording: _np.ndarray,
        loopback_recording: _Optional[_np.ndarray],
        preamble: _BlipPreamble,
    ) -> tuple[_LatencyResult, _Optional[_LatencyResult], bool, bool]:
        """
        Measure the delay. Returns ``(authoritative, crosscheck, disagreement,
        loopback_failed)``.

        With a loopback the clean loopback delay is authoritative (it is undistorted no
        matter how hard the amp is driven) and the amp return is measured only as a
        cross-check. Without one the amp return is authoritative and there is no
        cross-check. ``disagreement`` is set when both are detected but their delays
        differ by more than :data:`LOOPBACK_CROSSCHECK_SAMPLES`.

        A loopback that is configured but whose blips are not detected (e.g. an
        unplugged cable) is a hard failure, not a fallback: silently substituting the
        amp-return delay would report a "loopback" measurement that was never actually
        checked. ``loopback_failed`` signals this so the caller refuses the capture
        instead of saving one; ``authoritative`` is the (undetected) loopback result in
        that case, so ``.detected`` stays correctly ``False``.
        """
        amp_latency = _measure_delay(main_recording, preamble)
        if loopback_recording is None:
            return amp_latency, None, False, False
        loopback_latency = _measure_delay(loopback_recording, preamble)
        if not loopback_latency.detected:
            return loopback_latency, amp_latency, False, True
        disagreement = (
            amp_latency.delay is not None
            and abs(loopback_latency.delay - amp_latency.delay)
            > LOOPBACK_CROSSCHECK_SAMPLES
        )
        return loopback_latency, amp_latency, disagreement, False

    @staticmethod
    def _aligned_target(
        main: _np.ndarray,
        preamble: _BlipPreamble,
        length: int,
        shift: float,
    ) -> _np.ndarray:
        """
        Cut the target out of the recording, delayed by ``shift`` samples.

        The whole part is taken by starting the cut that many samples earlier, which is
        exact -- the recording extends well past the target on both sides (the preamble
        before it, ``TAIL_SECONDS`` after). Only the remaining fraction goes through the
        resampling filter, so a shift of exactly 1.0 costs nothing at all.
        """
        whole = int(round(shift))
        fraction = shift - whole
        start = preamble.n_samples - whole
        if start < 0 or start + length > len(main):
            # Cannot borrow the samples the shift needs; fall back to the plain cut
            # rather than truncating the target.
            start, fraction = preamble.n_samples, 0.0
        y = main[start : start + length]
        if fraction:
            from .resample import apply_fractional_delay as _apply_fractional_delay

            y = _apply_fractional_delay(y, fraction).astype(_np.float32)
        return y

    def _alignment_shift(
        self, latency: _LatencyResult, loopback_used: bool
    ) -> tuple[float, _Optional[str]]:
        """
        How far to shift this capture's target so it lands on the project's timebase.
        Returns ``(shift, note)``; ``note`` is a QA message worth surfacing.

        What has to hold for a capture set to be mutually aligned is that

            (true latency of the written target) - (delay it is labelled with)

        is the same for every capture. ``latency.peak_delay`` measures the first term to a
        fraction of a sample and ``latency.delay`` is the second, so the shift that
        restores the invariant is ``reference + delay - peak_delay``. The first capture to
        measure both defines ``reference``; its own shift is therefore zero.

        Both terms are needed, and this is why the shift is not clamped below a sample.
        ``delay`` comes from a threshold crossing on the blip's leading edge, so it moves
        with the response's pre-ringing and can step by a whole sample (or further) while
        the peak barely moves. When it does, that step is a genuine misalignment of the
        written data, and compensating for it is the correct response -- the whole part of
        the shift costs nothing, since it is taken by slicing the recording one sample
        over rather than by filtering.

        Only done when a loopback is in use. The amp return carries the tone stack's
        genuine knob-dependent group delay, which the model is supposed to learn; taking
        that out per capture would be correcting away real amp behaviour, which is worse
        than leaving the jitter in.

        What this is for, measured rather than assumed: while the interface's clock stays
        put there is no sub-sample drift at all (52 route tests on an iD44 read the same
        offset to sd 0.0000). Re-clocking it is what moves the phase -- taking the device
        to 44.1 kHz and back to 48 kHz shifted the round trip by 0.48 samples while the
        integer delay stayed at 8442, invisible to any whole-sample measurement. A project
        captured over several days crosses clock epochs whenever another application
        touches the interface, which is why ``alignment_reference`` lives in the project
        file rather than in a session.
        """
        if not loopback_used or latency.peak_delay is None or latency.delay is None:
            return 0.0, None

        offset = latency.peak_delay - latency.delay
        if self.project.alignment_reference is None:
            self.project.alignment_reference = offset
            return 0.0, None

        shift = self.project.alignment_reference - offset
        if abs(shift) > MAX_ALIGNMENT_SHIFT:
            return 0.0, (
                f"Alignment correction of {shift:+.2f} samples was needed but not "
                f"applied: past {MAX_ALIGNMENT_SHIFT:g} samples the timing measurement "
                "itself is in doubt. Re-run the route test and check the loopback patch "
                "before trusting this capture."
            )
        note = None
        if abs(shift) >= 0.5:
            note = (
                f"Timing moved {shift:+.2f} samples from this project's reference; the "
                "capture was realigned, but a sub-sample shift this large usually means "
                "the interface re-clocked or the routing changed mid-session."
            )
        return float(shift), note

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
        loopback_playback = (
            self._loopback_playback(playback, preamble)
            if self.project.audio.loopback_enabled
            else None
        )
        main, loopback = self._playrec(
            playback, sample_rate, progress, cancel, loopback_playback
        )
        latency, crosscheck, disagreement, loopback_failed = self._resolve_latency(
            main, loopback, preamble
        )
        return RouteTestResult(
            latency=latency,
            peak=_peak_to_dbfs(float(_np.max(_np.abs(main)))),
            loopback_used=loopback is not None,
            crosscheck=crosscheck,
            loopback_disagreement=disagreement,
            loopback_failed=loopback_failed,
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

        Raises :class:`CaptureSessionError` -- writing nothing and leaving the entry
        pending -- if a loopback is enabled but its blips were not detected (e.g. an
        unplugged cable). That case is not a QA flag on a saved capture: the delay
        cannot be trusted as coming from the loopback at all, so nothing is saved.
        """
        x, sample_rate = self._input_for_split(entry.split)
        preamble = _BlipPreamble(sample_rate)
        tail = _np.zeros(int(TAIL_SECONDS * sample_rate), dtype=_np.float32)
        playback = _np.concatenate([preamble.render(), x, tail])
        loopback_playback = (
            self._loopback_playback(playback, preamble)
            if self.project.audio.loopback_enabled
            else None
        )

        main, loopback = self._playrec(
            playback, sample_rate, progress, cancel, loopback_playback
        )

        latency, crosscheck, loopback_disagreement, loopback_failed = (
            self._resolve_latency(main, loopback, preamble)
        )
        if loopback_failed:
            # Refuse the capture outright rather than quietly falling back to the amp
            # return: nothing is written and the entry stays pending, so the user must
            # fix the loopback patch or uncheck it before this entry can be captured.
            raise CaptureSessionError(LOOPBACK_NOT_DETECTED_MESSAGE)
        loopback_used = loopback is not None
        shift, shift_note = self._alignment_shift(latency, loopback_used)
        y = self._aligned_target(main, preamble, len(x), shift)

        qa = self._qa(
            entry,
            y,
            latency,
            loopback_used=loopback_used,
            loopback_disagreement=loopback_disagreement,
            crosscheck=crosscheck,
            alignment_shift=shift,
            alignment_note=shift_note,
        )

        self._write_capture_wav(entry, y, sample_rate)
        _mark_captured(
            entry,
            delay=latency.delay,
            qa=qa,
            stream_config=self.project.audio.stream_fingerprint(),
        )
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

    def _comparable_delays(self, entry: _CaptureEntryModel) -> list[int]:
        """
        Delays of the already-captured entries this one's delay may be compared against:
        those recorded through the same stream configuration.

        Buffer size and stream latency are the user's to change mid-project -- a room or
        a machine that needs a bigger buffer today should not cost a project -- and doing
        so moves the round trip by hundreds or thousands of samples at once. That is not
        a fault: ``delay`` is measured and written per capture, and the sub-sample
        timebase is untouched by it (a buffer change is a whole number of samples, so the
        blip response and its energy peak move together and ``_alignment_shift`` sees
        nothing). Only the delay-consistency check would notice, and comparing across
        configurations would make it fire on every capture after the change while saying
        "did the routing change?" -- training the user to ignore the one message that
        catches a genuinely mispatched rig.

        Entries from before the configuration was recorded (``stream_config is None``)
        are used only when nothing matches the current one, so an existing project keeps
        its delay check on the next capture instead of silently losing it.
        """
        current = self.project.audio.stream_fingerprint()
        others = [
            other
            for other in self.project.captured_entries()
            if other.delay is not None and other is not entry
        ]
        matching = [other.delay for other in others if other.stream_config == current]
        if matching:
            return matching
        return [other.delay for other in others if other.stream_config is None]

    def _qa(
        self,
        entry: _CaptureEntryModel,
        y: _np.ndarray,
        latency: _LatencyResult,
        loopback_used: bool = False,
        loopback_disagreement: bool = False,
        crosscheck: _Optional[_LatencyResult] = None,
        alignment_shift: float = 0.0,
        alignment_note: _Optional[str] = None,
    ) -> _QAModel:
        messages: list[str] = []

        peak = float(_np.max(_np.abs(y))) if len(y) else 0.0
        peak_dbfs = _peak_to_dbfs(peak)
        clipping = peak >= CLIPPING_THRESHOLD
        if clipping:
            messages.append(
                f"Clipping: peak {peak_dbfs:.1f} dBFS. Training rejects clipped "
                "captures — lower the level and recapture."
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
            other_delays = self._comparable_delays(entry)
            if other_delays and (
                abs(latency.delay - int(_np.median(other_delays)))
                >= DELAY_DISAGREEMENT_SAMPLES
            ):
                delay_disagreement = True
                messages.append(
                    f"Delay {latency.delay} differs from this project's typical "
                    f"{int(_np.median(other_delays))} by {DELAY_DISAGREEMENT_SAMPLES}+ "
                    "samples at these audio settings. Did the routing change?"
                )

        if peak < 1e-4:
            messages.append(
                f"Capture is near-silent (peak {peak_dbfs:.1f} dBFS). Is the return "
                "input connected and the device under test on?"
            )

        if loopback_disagreement:
            messages.append(
                f"The loopback and amp-return timing blips disagree by more than "
                f"{LOOPBACK_CROSSCHECK_SAMPLES} samples. Check the loopback patch: the "
                "two routes are no longer travelling the same chain."
            )

        if alignment_note:
            messages.append(alignment_note)

        return _QAModel(
            peak=peak_dbfs,
            clipping=clipping,
            impulse_detected=latency.detected,
            delay_disagreement=delay_disagreement,
            loopback_disagreement=loopback_disagreement if loopback_used else None,
            loopback_delay=latency.delay if loopback_used else None,
            amp_return_delay=(
                crosscheck.delay if loopback_used and crosscheck else latency.delay
            ),
            subsample_shift=alignment_shift or None,
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
