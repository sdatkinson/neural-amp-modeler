"""
Capture-session orchestration: turn one planned entry at a time into a recorded,
delay-measured, QA-checked WAV, persisting after every step.

Per capture: build the playback (blip preamble + input audio + tail), play/record it,
measure the round-trip delay from the blips, shift the target onto the project's
timebase, run QA, save the capture WAV, then rewrite both ``capture_project.json`` and
``data.json`` so a crash or app close never loses a completed capture.

The recordings themselves are also kept, untouched, in ``captures_raw/`` -- the amp
return under the capture's own filename and the loopback under the same name with an
``_lb`` suffix. The same playback goes out both the amp and the loopback channels, so
the pair is a full record of the session: what was played, what came back, and the blips
that time them. Alongside them sit each input WAV as it was actually played (preamble,
input, tail) and ``manifest.json``, which says how many samples of preamble and tail
surround the input audio in each recording. Together that is enough to trim a raw file
back to the reamp and re-measure its delay without trusting anything measured live. See
:meth:`CaptureSession._write_raw_recordings`, :meth:`CaptureSession._write_playback_inputs`
and :func:`update_raw_manifest`.

The alignment step exists because ``delay`` is an integer while the rig's latency is not.
A capture set aligned only to whole samples carries up to half a sample of per-capture
error, and that error is not a function of the knobs, so a parametric model cannot learn
it as a rule -- it can only memorise a per-capture phase. Correcting it at write time
keeps ``delay`` an integer and leaves everything downstream unchanged. Every capture
derives its own timebase from its own loopback blip, holding nothing in common with the
others but a fixed lead, so no capture can put another one out. See
:meth:`CaptureSession._alignment` for what is measured and why,
:meth:`CaptureSession._alignment_lead` for the one case where that lead comes from the
project rather than a constant (a project part-captured before this, whose existing
captures are already written against its own offset), and :mod:`nam.capture.resample`
for the filter.
"""

from __future__ import annotations

import json as _json
import os as _os
import tempfile as _tempfile
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Sequence as _Sequence

import numpy as _np

from . import RAW_RECORDING_SINCE_VERSION as _RAW_RECORDING_SINCE_VERSION
from .audio import PlaybackRecorder as _PlaybackRecorder
from .audio import peak_to_dbfs as _peak_to_dbfs
from .export import update_data_json as _update_data_json
from .latency import BlipPreamble as _BlipPreamble
from .latency import PlayedPreamble as _PlayedPreamble
from .latency import LatencyResult as _LatencyResult
from .latency import measure_delay as _measure_delay
from .planner import CAPTURES_RAW_DIRNAME as _CAPTURES_RAW_DIRNAME
from .planner import PLAYBACK_INPUT_SUFFIX as _PLAYBACK_INPUT_SUFFIX
from .planner import RAW_LOOPBACK_SUFFIX as _RAW_LOOPBACK_SUFFIX
from .planner import RAW_MANIFEST_FILENAME as _RAW_MANIFEST_FILENAME
from .project import CaptureEntryModel as _CaptureEntryModel
from .project import CaptureProject as _CaptureProject
from .project import QAModel as _QAModel
from .project import atomic_write_json as _atomic_write_json
from .project import find_clearable_entries as _find_clearable_entries
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
# How far ahead of the loopback blip's energy peak a capture is labelled, in samples.
#
# The label has to sit before the response: a converter's anti-alias filter spreads energy
# ahead of the peak, and labelling the peak itself would ask the model to account for
# input it has not been shown. Any fixed value puts the project on one timebase (see
# :meth:`CaptureSession._alignment`); only never varying between captures matters.
#
# Nearly all of it is a change of reference point rather than caution. The peak is the
# maximum, not the onset, and on an iD44 sits 3.4 samples past the first sample to clear
# the noise floor. NAM labels from that onset less one sample of padding
# (``_DELAY_CALIBRATION_SAFETY_FACTOR``), so reaching the same instant from the peak means
# backing off 3.4 + 1 -- this constant. It lands where NAM does exactly: a blip peak of
# 1194.42 labels 1190, which is what NAM's threshold-minus-one gives, so the latency baked
# into everything exported is unchanged. What it buys is a peak that is stable and
# resolvable below a sample, where a threshold crossing moves with amplitude -- which is
# what let one bad capture take a project's timebase with it.
#
# The cost: 3.4 is a property of the converter, so one whose response spreads further
# would be labelled fractionally late. ``CaptureSession._qa`` checks each capture against
# its own onset and says so, rather than leaving it to be found in a soft-sounding model.
ALIGNMENT_LEAD_SAMPLES = 4.0
# Past this a legacy ``alignment_reference`` is not a converter's offset but a broken
# measurement -- a first capture whose blips disagreed writes exactly that -- and
# reproducing its timebase would inherit the fault. Above any plausible converter, well
# below the 129 samples the failure that prompted this wrote.
MAX_LEGACY_ALIGNMENT_REFERENCE = 32.0
# A shift is the rounding residue of the label, so it cannot leave this range. Past it the
# arithmetic is broken rather than the rig, so it is checked rather than trusted.
MAX_ALIGNMENT_SHIFT = 0.5
# How far a capture may sit from the rest before it is reported as misaligned. Half a
# sample is the error the alignment removes, so the check must sit below that to be worth
# running; a healthy set agrees to a few hundredths.
ALIGNMENT_MISMATCH_SAMPLES = 0.25


# Schema version of captures_raw/manifest.json, independent of the project file's.
RAW_MANIFEST_VERSION = 1


class CaptureSessionError(RuntimeError):
    pass


def recorded_lead(project: _CaptureProject) -> _Optional[float]:
    """
    This project's own recorded lead, if it has one that could be a converter's timing
    offset. ``None`` when there is none or it is out of range.

    A value outside the range is not a timebase: it is a broken measurement (a first
    capture whose two timing blips disagreed writes exactly that), and the callers treat
    it as a fault to report rather than a number to use.
    """
    reference = project.alignment_reference
    if reference is None:
        return None
    if 0.0 < reference <= MAX_LEGACY_ALIGNMENT_REFERENCE:
        return float(reference)
    return None


def alignment_lead(project: _CaptureProject) -> float:
    """
    The lead every capture in this project is labelled against, without refusing or
    changing anything -- :meth:`CaptureSession._alignment_lead` does both, and the audit
    needs to ask the same question of a project it is only inspecting.

    A usable recorded lead is the project's lead unconditionally, whether or not it has
    captures right now. Conditioning it on that made the answer depend on state that
    moves underneath it: regenerating the plan with a different seed renames every entry,
    so the files stop matching any planned ``y_path`` and the project reads as empty.
    Releasing the lead there and taking it back when the old seed returns would strand
    whatever was captured in between on a third timebase, silently. Holding it costs
    nothing -- the lead is arbitrary, only constancy matters. It is dropped only by
    :func:`nam.capture.project.clear_captures`, or by
    :meth:`CaptureSession._alignment_lead` when unusable *and* describing nothing.
    """
    recorded = recorded_lead(project)
    return ALIGNMENT_LEAD_SAMPLES if recorded is None else recorded


def has_captures(project: _CaptureProject, project_dir: _Path) -> bool:
    """
    Whether this project has captures the legacy timebase could still describe.

    Deliberately not just ``captured_entries()``: an entry goes pending while its WAV
    stays on disk, after the plan is regenerated and until the user accepts the offer to
    restore it (:func:`nam.capture.project.find_recoverable_entries`), and the project
    then looks empty while being one dialog from having every capture back.

    Delegates to :func:`nam.capture.project.find_clearable_entries` so that the refusal
    and the "Clear captures" it points at cannot disagree -- two answers to "does this
    project have captures" is how a user could be told to clear a project that Clear
    captures then called empty.
    """
    return bool(_find_clearable_entries(project, project_dir))


def timebase_problem(
    project: _CaptureProject, project_dir: _Path
) -> _Optional[str]:
    """
    Why this project cannot be captured into, or ``None`` if it can.

    Split out from :meth:`CaptureSession._alignment_lead`, which refuses the capture, so
    the GUI can ask the same question without starting one -- the answer does not depend
    on anything a capture produces, and arriving as a failure *after* a recording is a
    poor way to learn that the recording could never have been kept.

    An unusable reference is only a problem while captures are written against it. With
    none it describes nothing, so it is not reported and
    :meth:`CaptureSession._alignment_lead` drops it rather than refusing forever -- there
    is no way out otherwise, since a project with no captures cannot clear any.
    """
    reference = project.alignment_reference
    if reference is None or recorded_lead(project) is not None:
        return None
    if not has_captures(project, project_dir):
        return None
    return (
        "The captures already in this project have a bad timing measurement, so new "
        "captures can't be lined up with them. Clear the captures made so far and "
        f"record them again. (Timing offset {reference:.0f} samples, usually caused by "
        "an audio dropout during the first capture.)"
    )


@_dataclass(frozen=True)
class CaptureAudit:
    """
    One capture re-measured from the recording it was made from. See
    :func:`audit_captures`.
    """

    y_path: str
    # Where this capture sits: the delay it is labelled with, less the shift applied to
    # it, less the arrival its own loopback shows. ``None`` when it could not be
    # re-measured.
    residual: _Optional[float] = None
    # How far that is from where this project's alignment puts a capture, which is
    # ``-lead`` exactly for anything written under it. This is what is judged, rather
    # than each capture's distance from the middle of the set: the middle is not a fixed
    # point, so with two captures 0.42 apart it sits between them and reports both as
    # 0.21 out, naming the good one alongside the bad and identifying neither.
    offset: _Optional[float] = None
    blip_delays: tuple[int, ...] = ()
    blips_disagree: bool = False
    # Why this capture could not be checked, if it could not be.
    unchecked: _Optional[str] = None


def audit_captures(
    project: _CaptureProject,
    project_dir: _Path,
    progress: _Optional[_Callable[[float], None]] = None,
) -> list[CaptureAudit]:
    """
    Re-measure every captured entry from its own raw loopback recording and report where
    each one sits on the project's timebase.

    This asks the recordings rather than the project file, which is what makes it worth
    having: it needs no ``alignment_reference``, no ``peak_delay`` in the QA, and no
    knowledge of which scheme a capture was made under. A project whose timebase was lost
    -- regenerating the plan and restoring the files from disk drops the QA that was
    measured live -- can still be checked completely, because ``captures_raw/`` holds
    what the rig actually did and that cannot go stale.

    Captures from before ``captures_raw/`` existed have nothing to re-measure and are
    reported as unchecked rather than as passing.
    """
    from ..data import wav_to_np

    audits: list[CaptureAudit] = []
    expected = -alignment_lead(project)
    entries = project.captured_entries()
    for index, entry in enumerate(entries):
        if progress is not None:
            progress(index / max(len(entries), 1))
        try:
            latency = measure_from_raw(project, project_dir, entry.y_path)
        except Exception as exc:  # unreadable, too short, or nothing kept to measure
            audits.append(
                CaptureAudit(y_path=entry.y_path, unchecked=f"could not measure ({exc})")
            )
            continue
        if latency.peak_delay is None or entry.delay is None:
            audits.append(
                CaptureAudit(
                    y_path=entry.y_path,
                    blip_delays=latency.blip_delays,
                    blips_disagree=latency.disagreement_too_high,
                    unchecked="no timing blip was detected in it",
                )
            )
            continue
        shift = (entry.qa.subsample_shift if entry.qa else None) or 0.0
        residual = entry.delay - shift - latency.peak_delay
        audits.append(
            CaptureAudit(
                y_path=entry.y_path,
                residual=residual,
                offset=residual - expected,
                blip_delays=latency.blip_delays,
                blips_disagree=latency.disagreement_too_high,
            )
        )
    if progress is not None:
        progress(1.0)
    return audits


def shared_offset(audits: _Sequence[CaptureAudit]) -> _Optional[float]:
    """
    The one offset every measurable capture shares, or ``None`` if they share none.

    A whole set sitting the same distance from the project's lead agrees with itself and
    was written under another timebase -- harmless to train on, since the constant is the
    same on every capture. Captures at *different* distances genuinely disagree, which is
    the fault that hurts. The two need saying differently.
    """
    offsets = [
        audit.offset
        for audit in audits
        if audit.offset is not None and not audit.blips_disagree
    ]
    if not offsets:
        return None
    if max(offsets) - min(offsets) >= ALIGNMENT_MISMATCH_SAMPLES:
        return None
    return sum(offsets) / len(offsets)


def audit_problems(audits: _Sequence[CaptureAudit]) -> list[str]:
    """
    The captures in an :func:`audit_captures` result that need attention, in plain words.
    Empty when they all sit where this project's alignment puts them.

    Captures that disagree are judged against the project's alignment rather than against
    the set, so one that is where it should be is never named because something else is
    not. A set that agrees with itself but sits together away from that alignment is
    reported once instead: naming each capture as "away from the rest" would accuse every
    one of them of a discrepancy with captures it matches exactly.
    """
    problems: list[str] = []
    for audit in audits:
        if audit.unchecked is not None:
            problems.append(f"{audit.y_path}: not checked -- {audit.unchecked}.")
        elif audit.blips_disagree:
            gap = max(audit.blip_delays) - min(audit.blip_delays)
            problems.append(
                f"{audit.y_path}: its two timing blips came back {gap} samples apart, so "
                "its timing was never measured reliably. Record it again."
            )

    offset = shared_offset(audits)
    if offset is not None:
        if abs(offset) >= ALIGNMENT_MISMATCH_SAMPLES:
            measured = sum(1 for a in audits if a.offset is not None)
            problems.append(
                f"All {measured} measurable capture(s) agree with each other, but the "
                f"set sits {offset:+.2f} samples from where a new capture in this "
                "project would land. They were recorded on an older timebase that this "
                "project no longer records. Training on them as they are is fine -- the "
                "offset is the same on every one -- but a capture added now would be "
                f"{abs(offset):.2f} samples out from them."
            )
        return problems

    for audit in audits:
        if audit.unchecked is not None or audit.blips_disagree:
            continue
        if audit.offset is not None and abs(audit.offset) >= ALIGNMENT_MISMATCH_SAMPLES:
            problems.append(
                f"{audit.y_path}: sits {audit.offset:+.2f} samples away from where this "
                "project's alignment puts a capture, and the rest of the set does not "
                "agree with it. Record it again."
            )
    return problems


def measure_from_raw(
    project: _CaptureProject, project_dir: _Path, y_path: str
) -> _LatencyResult:
    """
    Measure a capture's delay from the loopback recording it kept.

    The one detection path, shared with a live capture: same function, same preamble
    description, same numbers. So the audit and recovery cannot disagree with what the
    capture itself recorded -- which they did when recovery invented a result from
    ``data.json`` rather than measuring one.

    Raises when nothing was kept to measure, as for a capture predating
    ``captures_raw/``.
    """
    from ..data import wav_to_np

    project_dir = _Path(project_dir)
    _, raw_loopback = raw_paths(project_dir, y_path)
    if not raw_loopback.is_file():
        raise CaptureSessionError(f"no raw loopback recording was kept for {y_path}")
    recording = _np.asarray(wav_to_np(raw_loopback), dtype=_np.float32).squeeze()
    rate = project.sample_rate or _read_rate(raw_loopback)
    return _measure_delay(recording, played_preamble(project_dir, y_path, rate))


def played_preamble(
    project_dir: _Path, y_path: str, sample_rate: int
) -> _PlayedPreamble:
    """
    The preamble a capture was played, rebuilt from ``captures_raw/``: ``manifest.json``
    names the playback file and its preamble length, and the file itself is kept. See
    :class:`nam.capture.latency.PlayedPreamble` for why the layout is read rather than
    assumed.

    Raises if either is missing, so such a capture is reported as unchecked rather than
    measured against a guess.
    """
    project_dir = _Path(project_dir)
    manifest_path = project_dir / _CAPTURES_RAW_DIRNAME / _RAW_MANIFEST_FILENAME
    manifest = _json.loads(manifest_path.read_text())
    record = next(
        (r for r in manifest.get("captures", []) if r.get("y_path") == y_path), None
    )
    if record is None:
        raise CaptureSessionError(f"{y_path} is not in {_RAW_MANIFEST_FILENAME}")
    playback_path = project_dir / _CAPTURES_RAW_DIRNAME / record["playback"]
    from ..data import wav_to_np

    playback = _np.asarray(wav_to_np(playback_path), dtype=_np.float32).squeeze()
    return _PlayedPreamble.from_playback(
        playback, int(record["preamble_samples"]), sample_rate
    )


def _read_rate(path: _Path) -> int:
    import wave as _wave

    with _wave.open(str(path), "rb") as handle:
        return handle.getframerate()


def raw_paths(project_dir: _Path, y_path: str) -> tuple[_Path, _Path]:
    """
    Where a capture's untouched recordings live: ``(amp return, loopback)``, both in
    ``captures_raw/`` and named after the capture itself so the correspondence is
    readable without consulting anything else.
    """
    name = _Path(y_path).name
    raw = _Path(project_dir) / _CAPTURES_RAW_DIRNAME / name
    return raw, raw.with_name(f"{raw.stem}{_RAW_LOOPBACK_SUFFIX}{raw.suffix}")


def build_playback(
    x: _np.ndarray, sample_rate: int
) -> tuple[_np.ndarray, _PlayedPreamble]:
    """
    The stream that goes out to the amp: blip preamble, then the input audio, then
    enough tail for the delayed response to land inside the recording.

    Single source of this layout. The copy saved in ``captures_raw/`` is what a raw
    recording is correlated against, so it has to be built the same way as the stream
    that was played, not merely the same way in two places.

    The preamble handed back describes the signal as rendered rather than as specified,
    so a live capture is measured by the same code that re-measures an old recording:
    the path that has to survive a preamble change is one every capture already runs.
    """
    rendered = _BlipPreamble(sample_rate)
    tail = _np.zeros(int(TAIL_SECONDS * sample_rate), dtype=_np.float32)
    playback = _np.concatenate([rendered.render(), x, tail])
    preamble = _PlayedPreamble.from_playback(playback, rendered.n_samples, sample_rate)
    return playback, preamble


def _wav_frame_count(path: _Path) -> _Optional[int]:
    """
    Frames in a WAV from its header, without decoding it, or ``None`` if it cannot be
    read at all.
    """
    import wave

    try:
        with wave.open(str(path)) as fp:
            return fp.getnframes()
    except (OSError, wave.Error):
        return None


def playback_input_path(project_dir: _Path, x_path: str) -> _Path:
    """
    Where the "as played" copy of an input WAV lives: the input with the blip preamble in
    front and the tail behind, in ``captures_raw/``. Named after the input it was built
    from, so a project with more than two inputs gets one of these per input.
    """
    stem = _Path(x_path).stem
    return (
        _Path(project_dir)
        / _CAPTURES_RAW_DIRNAME
        / f"{stem}{_PLAYBACK_INPUT_SUFFIX}.wav"
    )


def _predates_raw_recording_note(project: _CaptureProject) -> _Optional[str]:
    """
    The one-line explanation for a ``captures_raw/`` that cannot possibly be complete,
    or ``None`` when there is nothing to explain.

    A project with no recorded version was created before the capture app stamped one,
    which is also before raw recordings existed -- so anything captured back then has
    nothing here, and the folder will look like it lost files it never had. Saying so in
    the folder itself beats a dialog: it is read exactly when someone is looking at the
    gap, and it survives the folder being copied away from the project.
    """
    if project.created_with_version is not None:
        return None
    return (
        f"Raw recordings were first saved in capture app {_RAW_RECORDING_SINCE_VERSION}. "
        f"This project was created before that, so captures made earlier have no files "
        f"here; everything captured from {_RAW_RECORDING_SINCE_VERSION} on is listed "
        "below."
    )


def update_raw_manifest(
    project_dir: _Path, record: dict, note: _Optional[str] = None
) -> None:
    """
    Insert or replace one capture's record in ``captures_raw/manifest.json``, keyed by
    ``y_path``. Written as each capture is recorded, so every record describes a file
    that exists and the geometry that file was actually recorded with -- a recapture
    rewrites both together.

    What a record holds is where the input audio sits inside the raw recordings: how many
    samples of preamble precede it and how many of tail follow. The delay is deliberately
    not among them. A raw loopback is a delayed copy of a signal that is also on disk, so
    the delay can always be re-measured from it, by correlation against the played stream
    or from the blips when the preamble carries them. What cannot be recovered from the
    audio alone is which part of it is the reamp, and that is what a recoverer needs to
    trim a raw file back to something that lines up with the input WAV sample for sample.

    ``note`` is recorded once, when the manifest is created, and left alone afterwards:
    it explains something about the folder's history, not about any one capture.
    """
    path = _Path(project_dir) / _CAPTURES_RAW_DIRNAME / _RAW_MANIFEST_FILENAME
    manifest: dict = {"version": RAW_MANIFEST_VERSION, "captures": []}
    if note:
        manifest["note"] = note
    if path.is_file():
        try:
            with path.open() as fp:
                existing = _json.load(fp)
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict) and isinstance(existing.get("captures"), list):
            manifest = existing

    records = manifest["captures"]
    for index, existing_record in enumerate(records):
        if existing_record.get("y_path") == record["y_path"]:
            records[index] = record
            break
    else:
        records.append(record)
    _atomic_write_json(path, manifest)


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
        self._write_playback_inputs(rate)
        return rate

    def _write_playback_inputs(self, sample_rate: int) -> None:
        """
        Save each input WAV as it will actually be played -- preamble, input, tail -- into
        ``captures_raw/``.

        This is the reference a raw recording is recovered against: correlating a return
        against the exact stream that produced it gives the delay whether or not the
        blips survived the chain, which is what a capture reamped in a DAW (or an input
        file with no blips in it) will need. Written here rather than when the inputs are
        chosen because this is the one place that has both the audio and the validated
        sample rate, and it runs before any route test or capture; inputs dropped into
        the project folder by hand are covered for free.

        Rewritten only when what is on disk is not the right length, so replacing an
        input WAV regenerates it and an unchanged project does no work.
        """
        for split in ("train", "validation"):
            x_path = self.project.input_for_split(split)
            playback, _ = build_playback(self._inputs[split], sample_rate)
            path = playback_input_path(self.project_dir, x_path)
            if _wav_frame_count(path) == len(playback):
                continue
            self._write_wav(path, playback, sample_rate)

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
    def _loopback_playback(playback: _np.ndarray) -> _np.ndarray:
        """
        The signal for the loopback output channel: the same playback the amp gets,
        blips and all.

        It used to be the blips alone, on the grounds that the loopback only had to
        carry timing. Sending the program material too costs the measurement nothing --
        detection and the sub-sample peak both look only inside the blip section, which
        is silent either way (see :mod:`nam.capture.latency`) -- and it makes the
        recorded loopback a clean, delay-bearing copy of the input as the rig actually
        played it, which is what makes a session recoverable from ``captures_raw/``
        rather than only re-recordable.
        """
        return _np.asarray(playback, dtype=_np.float32)

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

    def _alignment_lead(self) -> float:
        """
        How far ahead of the blip peak this project's captures are labelled.

        :data:`ALIGNMENT_LEAD_SAMPLES` for any project made under the stateless timebase,
        which is all of them from here on. One part-captured before it has captures
        written against its own ``alignment_reference``, and finishing it with a different
        lead would leave the two groups a fraction of a sample -- or, on a rig whose
        response spreads further, many samples -- apart, which is the per-capture phase
        error the alignment exists to remove. So it keeps using its recorded offset,
        read from the file and never written back.

        Once recorded, a usable lead applies to every capture whether or not the project
        has any right now (see :func:`alignment_lead` for why that must not be
        conditional), and is released only by
        :func:`nam.capture.project.clear_captures`.

        A reference too large to be a converter's offset is refused rather than
        reproduced: that is what the original fault wrote, and honouring it would spread
        one bad measurement into every remaining capture.
        """
        if self.project.alignment_reference is None:
            return ALIGNMENT_LEAD_SAMPLES
        recorded = recorded_lead(self.project)
        if recorded is not None:
            return recorded
        problem = timebase_problem(self.project, self.project_dir)
        if problem is not None:
            raise CaptureSessionError(problem)
        # Unusable *and* describing no captures, so there is nothing it could keep this
        # capture aligned with and nothing for the user to clear. Dropped rather than
        # refused, or the project would be permanently uncapturable with no way out.
        self.project.alignment_reference = None
        return ALIGNMENT_LEAD_SAMPLES

    @staticmethod
    def _alignment(
        latency: _LatencyResult, loopback_used: bool, lead: float
    ) -> tuple[_Optional[int], float]:
        """
        The delay this capture is labelled with and the shift applied to its target,
        derived from this capture alone.

        Static, and deliberately so: given the project's fixed ``lead`` it reads nothing
        else, which is the property that makes one capture unable to disturb another. See
        :meth:`_alignment_lead` for where ``lead`` comes from.

        What has to hold for a capture set to be mutually aligned is that

            (true latency of the written target) - (delay it is labelled with)

        is the same for every capture. Writing the target as ``y[n] = a(n - shift - D)``
        for a true latency ``D``, and pairing it downstream as ``x[n]`` against
        ``y[n + delay]``, that quantity is ``r = delay - shift - D``.

        ``peak_delay`` locates the blip response's energy peak to a fraction of a sample,
        so taking the label from it directly::

            target = peak_delay - lead
            delay  = round(target)
            shift  = delay - target

        gives ``r = target - D = (peak_delay - D) - lead``. The first term is fixed
        hardware -- through a loopback the round trip is LTI, so its peak sits the same
        distance past the true arrival every time -- and the second is a constant, so
        ``r`` is the same for every capture with nothing from any other capture entering.
        The shift is only the label's rounding residue, always within half a sample.

        This used to take the label from the calibrator's threshold crossing, which is
        amplitude-biased: a quieter return crosses later, stepping by whole samples while
        the peak barely moves. Holding ``r`` constant then meant cancelling that step
        against a project-wide offset taken from whichever capture was recorded first,
        which silently made that one capture's measurement the timebase for every later
        one. A first capture whose blips disagreed (see ``blip_delays``) put it 129
        samples out, and every later capture was told *it* had drifted.

        Only done when a loopback is in use. The amp return carries the tone stack's
        knob-dependent group delay, which the model is supposed to learn; removing that
        per capture would correct away real amp behaviour.

        What the sub-sample part is for, measured rather than assumed: while the clock
        stays put there is no drift (52 route tests on an iD44, sd 0.0000). Re-clocking
        moves the phase -- 44.1 kHz and back to 48 shifted the round trip 0.48 samples
        while the integer delay stayed at 8442, invisible to any whole-sample measure. A
        project captured over days crosses clock epochs whenever another application
        touches the interface, and each capture now measures its own.
        """
        if not loopback_used or latency.peak_delay is None:
            return latency.delay, 0.0

        target = latency.peak_delay - lead
        delay = int(round(target))
        shift = float(delay - target)
        if abs(shift) > MAX_ALIGNMENT_SHIFT + 1e-9:  # pragma: no cover - invariant
            raise CaptureSessionError(
                f"Internal error: alignment shift {shift:+.4f} exceeds the half-sample "
                "rounding residue it is defined as."
            )
        return delay, shift

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
        playback, preamble = build_playback(
            _np.zeros(0, dtype=_np.float32), sample_rate
        )
        loopback_playback = (
            self._loopback_playback(playback)
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
        # Resolved before anything is played: a project whose recorded timebase cannot be
        # honoured is refused now rather than after the user has sat through a capture.
        lead = self._alignment_lead()

        x, sample_rate = self._input_for_split(entry.split)
        playback, preamble = build_playback(x, sample_rate)
        loopback_playback = (
            self._loopback_playback(playback)
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
        delay, shift = self._alignment(latency, loopback_used, lead)
        y = self._aligned_target(main, preamble, len(x), shift)

        qa = self._qa(
            entry,
            y,
            latency,
            loopback_used=loopback_used,
            loopback_disagreement=loopback_disagreement,
            crosscheck=crosscheck,
            alignment_shift=shift,
            delay=delay,
        )

        self._write_raw_recordings(
            entry,
            main,
            loopback,
            sample_rate,
            preamble_samples=preamble.n_samples,
            input_samples=len(x),
        )
        self._write_capture_wav(entry, y, sample_rate)
        _mark_captured(
            entry,
            delay=delay,
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

        The delay comes from data.json, which is what the WAV on disk is labelled with.
        Everything else is re-measured from the recording the capture kept, through
        :func:`measure_from_raw`.

        This used to fabricate a ``LatencyResult`` from the delay alone, leaving the
        restored QA with no ``peak_delay``, ``blip_delays`` or ``subsample_shift`` -- so a
        sub-sample aligned capture came back looking like one that never was, and the
        audit, reading a shift of zero where a real one had been applied, called a
        healthy capture misaligned.

        The shift is derived from that delay and the capture's own measured peak, which
        reproduces what was applied when it was written as long as the project's lead has
        not moved. One landing past the half-sample residue it is defined as means it has,
        so that is reported rather than recorded.

        ``captured_at`` is stamped now, since data.json records neither it nor QA.
        Persists the project and data.json once at the end. Returns a note per entry.
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
            shift = 0.0
            loopback_used = False
            try:
                latency = measure_from_raw(self.project, self.project_dir, entry.y_path)
            except Exception as exc:
                # Nothing kept to measure. data.json proves the delay was measured once,
                # so the entry is restored on that alone with less in its QA.
                latency = _LatencyResult(
                    delay=delay,
                    detected=delay is not None,
                    disagreement_too_high=False,
                    safety_factor=0,
                )
                notes.append(f"{entry.y_path}: timing not re-measured ({exc}).")
            else:
                loopback_used = True
                if delay is not None and latency.peak_delay is not None:
                    shift = float(delay - (latency.peak_delay - self._alignment_lead()))
                    if abs(shift) > MAX_ALIGNMENT_SHIFT + 1e-9:
                        notes.append(
                            f"{entry.y_path}: its recorded delay does not match its own "
                            f"recording (off by {shift:+.2f} samples); restored without "
                            "a sub-sample shift, and 'Check captures' will report it."
                        )
                        shift = 0.0
            qa = self._qa(
                entry,
                y,
                latency,
                loopback_used=loopback_used,
                alignment_shift=shift,
                delay=delay,
            )
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
        timebase is untouched by it (a buffer change is a whole number of samples, so it
        lands entirely in the rounded label and ``_alignment`` sees no change in the
        residue). Only the delay-consistency check would notice, and comparing across
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
        delay: _Optional[int] = None,
    ) -> _QAModel:
        messages: list[str] = []
        # The delay actually written for this capture, which is what later captures are
        # compared against. Falls back to the measurement for callers that do not derive
        # a label of their own (recovery from disk).
        if delay is None:
            delay = latency.delay

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
            # Name the two readings. This fires when one blip came back at a different
            # time from the other, which means a glitch during the preamble rather than
            # anything about the rig's steady-state latency -- and the pair of numbers is
            # what makes that recognisable instead of a bare "may be unreliable".
            if latency.blip_delays:
                gap = max(latency.blip_delays) - min(latency.blip_delays)
                detail = (
                    f"{gap} samples apart "
                    f"({' and '.join(str(d) for d in latency.blip_delays)})"
                )
            else:
                detail = "at different times"
            messages.append(
                "This capture's timing couldn't be measured reliably: the two timing "
                f"blips came back {detail}. That usually means an audio dropout at the "
                "start of the recording. Record this one again -- your other captures "
                "aren't affected."
            )

        # The label is placed ALIGNMENT_LEAD_SAMPLES ahead of the blip's energy peak,
        # which is only conservative if that clears the converter's pre-ringing. That
        # holds on the rigs this was measured on, but it is an assumption about someone
        # else's hardware, so it is checked against this capture's own independently
        # detected onset rather than trusted -- otherwise a converter that rings longer
        # would quietly produce targets that start before the input they are paired with.
        if loopback_used and delay is not None and latency.delay is not None:
            if delay > latency.delay:
                messages.append(
                    "This capture's response starts before the point it's timed from "
                    f"({latency.delay} against {delay} samples), which can soften the "
                    "model. Your audio interface needs a bigger timing margin than the "
                    "app allows for -- please report this."
                )

        delay_disagreement = False
        if delay is not None:
            other_delays = self._comparable_delays(entry)
            if other_delays and (
                abs(delay - int(_np.median(other_delays)))
                >= DELAY_DISAGREEMENT_SAMPLES
            ):
                delay_disagreement = True
                messages.append(
                    f"Delay {delay} differs from this project's typical "
                    f"{int(_np.median(other_delays))} by {DELAY_DISAGREEMENT_SAMPLES}+ "
                    "samples at these audio settings. Did the routing change?"
                )

        if peak < 1e-4:
            messages.append(
                f"Capture is near-silent (peak {peak_dbfs:.1f} dBFS). Is the return "
                "input connected and the device under test on?"
            )

        # Not reported when this capture's own blips disagreed: the loopback delay is
        # then a blend of two arrival times, so it will differ from the amp return no
        # matter how the rig is patched. Saying "check the loopback patch" there sends
        # the user after a cable when the fault is a dropout during the count-in.
        if loopback_disagreement and not latency.disagreement_too_high:
            messages.append(
                f"The loopback and amp-return timing blips disagree by more than "
                f"{LOOPBACK_CROSSCHECK_SAMPLES} samples. Check the loopback patch: the "
                "two routes are no longer travelling the same chain."
            )

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
            # The measurement the timebase is actually built from, and the per-blip
            # readings behind it. Recorded because the one failure this had to be
            # diagnosed from left neither behind: the project stored a shift and a
            # complaint, and the numbers that would have identified the real culprit
            # had to be recovered by re-measuring captures_raw/ by hand.
            peak_delay=latency.peak_delay,
            blip_delays=list(latency.blip_delays),
            messages=messages,
        )

    def _write_raw_recordings(
        self,
        entry: _CaptureEntryModel,
        main: _np.ndarray,
        loopback: _Optional[_np.ndarray],
        sample_rate: int,
        *,
        preamble_samples: int,
        input_samples: int,
    ) -> None:
        """
        Save the recordings as they came off the interface: the whole stream, blips and
        tail included, with no delay applied, no alignment shift and no resampling, and
        record where the input audio sits inside them.

        That is the point of them. The capture WAV is already interpreted -- cut to the
        input's length, slid onto the project's timebase and labelled with a measured
        delay -- so if any of that turns out to be wrong, the only way back is a copy no
        such decision has touched. Nothing in training reads these.

        The manifest record is written here, with the audio, so it can only ever describe
        files that exist; see :func:`update_raw_manifest`.
        """
        amp_path, loopback_path = raw_paths(self.project_dir, entry.y_path)
        self._write_wav(amp_path, main, sample_rate)
        if loopback is not None:
            self._write_wav(loopback_path, loopback, sample_rate)

        x_path = self.project.input_for_split(entry.split)
        update_raw_manifest(
            self.project_dir,
            {
                "y_path": entry.y_path,
                "x_path": x_path,
                "playback": playback_input_path(self.project_dir, x_path).name,
                "raw": amp_path.name,
                "raw_loopback": loopback_path.name if loopback is not None else None,
                "preamble_samples": preamble_samples,
                "tail_samples": len(main) - preamble_samples - input_samples,
            },
            note=_predates_raw_recording_note(self.project),
        )

    def _write_capture_wav(
        self, entry: _CaptureEntryModel, y: _np.ndarray, sample_rate: int
    ) -> None:
        self._write_wav(self.project_dir / entry.y_path, y, sample_rate)

    @staticmethod
    def _write_wav(path: _Path, y: _np.ndarray, sample_rate: int) -> None:
        from ..data import np_to_wav

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
