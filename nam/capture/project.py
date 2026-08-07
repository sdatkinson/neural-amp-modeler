"""
The capture project: the app's resumable source of truth.

``capture_project.json`` lives in the project folder next to the training-facing
``data.json`` and records the knob specs, the full capture plan (with per-entry
status and measured delay), and the audio-routing settings. It is saved atomically
after every state change so closing the app mid-session never loses work.
"""

from __future__ import annotations

import json as _json
import os as _os
import tempfile as _tempfile
from datetime import datetime as _datetime
from datetime import timezone as _timezone
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Literal as _Literal
from typing import Optional as _Optional
from typing import Sequence as _Sequence
from typing import Union as _Union

import numpy as _np
from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field

from .params import DEFAULT_KNOB_STEP
from .params import KnobSpec
from .planner import CAPTURES_DIRNAME
from .planner import plan_captures as _plan_captures
from .planner import plan_corner_captures as _plan_corner_captures


PROJECT_FILENAME = "capture_project.json"
DATA_FILENAME = "data.json"
MODEL_CONFIG_FILENAME = "model.json"
LEARNING_CONFIG_FILENAME = "learning.json"
# The two parametric architectures are exported side by side, so each gets its own pair.
CONCAT_MODEL_CONFIG_FILENAME = "model_concat.json"
CONCAT_LEARNING_CONFIG_FILENAME = "learning_concat.json"
HYPER_MODEL_CONFIG_FILENAME = "model_hyper.json"
HYPER_LEARNING_CONFIG_FILENAME = "learning_hyper.json"

_PROJECT_VERSION = 1
# Keep every window below one ConcatLSTM processing block (65535 samples) and above the
# active-learning loss mask_first (8192) if using for active learning; else, use
# 8192 for ConcatWavenet or HyperWavenet trainng. see scripts/make_starter_settings.py.
_DEFAULT_NY = 8192


def atomic_write_json(path: _Path, payload: _Any) -> None:
    """
    Write JSON so a crash mid-write never leaves a truncated file: write to a
    temporary file in the destination directory, then atomically replace.
    """
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = _tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with _os.fdopen(fd, "w") as fp:
            _json.dump(payload, fp, indent=4)
            fp.write("\n")
        _os.replace(tmp_path, path)
    except BaseException:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise


class KnobModel(_BaseModel):
    name: str
    min: float
    max: float
    step: float = DEFAULT_KNOB_STEP
    default: _Optional[float] = None
    avoid_zero: bool = False
    is_gain: bool = False

    def to_knob_spec(self) -> KnobSpec:
        return KnobSpec(
            name=self.name,
            min=self.min,
            max=self.max,
            step=self.step,
            default=self.default,
            avoid_zero=self.avoid_zero,
            is_gain=self.is_gain,
        )

    @classmethod
    def from_knob_spec(cls, knob: KnobSpec) -> "KnobModel":
        return cls(**knob.to_dict())


class WindowModel(_BaseModel):
    """
    Per-split data.json windowing defaults. The app assumes dedicated train and
    validation input files, so both splits default to the whole file sliced into
    fixed-length windows.
    """

    start_seconds: float = 0.0
    stop_seconds: _Optional[float] = None
    ny: _Optional[int] = _DEFAULT_NY


class AudioSettingsModel(_BaseModel):
    """
    Devices are stored by name (not index): indices shift as devices come and go, so
    they are resolved by name each session. Channels are 1-based, matching how
    interfaces label their jacks.
    """

    host_api: _Optional[str] = None
    output_device: _Optional[str] = None
    input_device: _Optional[str] = None
    output_channel: int = 1
    input_channel: int = 1
    # Optional second I/O pair (on the same devices) wired as a direct loopback: a
    # clean copy of the timing blips is played on ``loopback_output_channel`` and
    # patched straight back into ``loopback_input_channel``. It stays undistorted no
    # matter how hard the amp is driven, so the delay is measured from it instead of the
    # (increasingly smeared) amp return. It also sees none of the amp's own knob-dependent
    # tone-stack group delay, which is real behaviour the model must learn rather than
    # have corrected away -- that is why it, and not the amp return, drives the
    # sub-sample correction in nam.capture.session.
    #
    # It only stands in for the amp path where it shares that path's conversion chain: it
    # tracks latency on the route it actually travels, so anything the capture route
    # crosses that the loopback does not -- an ADAT/optical link to a second interface
    # above all -- is invisible here, and can move the capture by several samples whenever
    # that link re-locks while this measurement does not budge. A mispatched loopback has
    # exactly this signature, and once cost a whole project: the loopback read one
    # constant across 93 captures while the amp path moved over 4.6 samples. Both delays
    # are now recorded per capture (QAModel.loopback_delay / amp_return_delay) and
    # compared at LOOPBACK_CROSSCHECK_SAMPLES, which is what makes that visible while it
    # is happening. Both must be set to enable the loopback.
    # Defaults to channel 2 (on both directions) so a fresh setup starts with the
    # loopback enabled without colliding with the channel-1 default of the main path.
    loopback_output_channel: _Optional[int] = 2
    loopback_input_channel: _Optional[int] = 2
    # Device buffer/block size in frames passed to the audio stream. 0 lets
    # PortAudio pick an optimal block size.
    blocksize: int = 0
    # Suggested stream latency, in seconds, or one of PortAudio's per-device presets
    # ("low"/"high"). This is the single biggest lever on the round trip: sounddevice
    # defaults to "high", which on an iD44 asks for a 0.1 s input buffer and lands the
    # measured delay near 8400 samples, where "low" or an explicit few milliseconds
    # reaches DAW territory. Small values raise the risk of a dropout mid-capture, which
    # is why SounddeviceRecorder fails the capture outright on an over/underflow rather
    # than saving corrupted audio.
    #
    # Changing it mid-project is fine: ``delay`` is measured and stored per capture, and
    # a buffer change moves the round trip by a whole number of samples, so it shifts the
    # blip response and its energy peak together and leaves the sub-sample timebase in
    # nam.capture.session untouched.
    latency: _Union[float, _Literal["low", "high"]] = "low"

    def stream_fingerprint(self) -> str:
        """
        Identifies the stream configuration a capture was recorded through. Captures
        made through different configurations have legitimately different round-trip
        delays, so QA compares each capture's delay only against others that share this.
        """
        return (
            f"{self.input_device}|{self.output_device}|{self.host_api}"
            f"|in{self.input_channel}|out{self.output_channel}"
            f"|lb{self.loopback_input_channel}>{self.loopback_output_channel}"
            f"|bs{self.blocksize}|lat{self.latency}"
        )

    @property
    def loopback_enabled(self) -> bool:
        return (
            self.loopback_output_channel is not None
            and self.loopback_input_channel is not None
        )


class QAModel(_BaseModel):
    # dBFS (0 = full scale, negative below). Projects saved before this was tracked in
    # dB store a linear 0-1 amplitude here instead; see migrate_legacy_peak_values.
    peak: _Optional[float] = None
    clipping: _Optional[bool] = None
    impulse_detected: _Optional[bool] = None
    delay_disagreement: _Optional[bool] = None
    # The clean loopback and the amp return disagreed about the delay by more than the
    # cross-check tolerance (a mispatched loopback or a routing change). ``None`` when
    # no loopback was used for this capture.
    loopback_disagreement: _Optional[bool] = None
    # The two measured delays themselves, not just the boolean above. A mispatched
    # loopback once went undetected across a whole project because the tolerance was
    # wider than the drift; a per-capture record of both numbers is what makes that
    # visible after the fact instead of only in aggregate.
    loopback_delay: _Optional[int] = None
    amp_return_delay: _Optional[int] = None
    # Shift, in samples, applied to the target before it was written, to put it on the
    # project's timebase (see CaptureSession._alignment_shift). ``None`` when no
    # correction was applied.
    subsample_shift: _Optional[float] = None
    messages: list[str] = _Field(default_factory=list)


class CaptureEntryModel(_BaseModel):
    index: int
    split: _Literal["train", "validation"]
    params: dict[str, float]
    y_path: str
    status: _Literal["pending", "captured"] = "pending"
    delay: _Optional[int] = None
    captured_at: _Optional[str] = None
    qa: _Optional[QAModel] = None
    # ``AudioSettingsModel.stream_fingerprint()`` at the time this was captured. Buffer
    # size and stream latency are the user's to change mid-project -- a session that
    # has to run at a bigger buffer to stay glitch-free should not be a broken project --
    # and each capture carries its own measured ``delay``, so a change is harmless. What
    # it would otherwise break is the delay-consistency check, which flags a capture whose
    # delay departs from the project's typical value; recording the configuration lets
    # that check compare like with like instead of firing on every capture after a
    # deliberate change. ``None`` for captures made before this was recorded.
    stream_config: _Optional[str] = None


class CaptureProject(_BaseModel):
    version: int = _PROJECT_VERSION
    name: str = ""
    knobs: list[KnobModel]
    seed: int = 0
    train_input: str = "input_train.wav"
    validation_input: str = "input_validation.wav"
    sample_rate: _Optional[int] = None
    train_window: WindowModel = _Field(default_factory=WindowModel)
    validation_window: WindowModel = _Field(
        default_factory=lambda: WindowModel(ny=317270)
    )
    audio: AudioSettingsModel = _Field(default_factory=AudioSettingsModel)
    entries: list[CaptureEntryModel] = _Field(default_factory=list)
    # The number of LHS training points requested at plan-generation time. Recorded
    # explicitly rather than derived from ``entries_for_split("train")`` because that
    # count grows when corner captures are added, which would otherwise make the
    # planned LHS size drift every time the project is reopened.
    n_train_lhs: int = 0
    include_initial_corners: bool = False
    # Timebase for this project: the offset, in samples, between the blip response's
    # energy peak and the integer delay reported alongside it. The first capture that
    # measures it sets it, and every later capture is shifted so its own offset matches --
    # which is what puts the whole project on one timebase. The value itself is arbitrary
    # (it carries the round trip's impulse-response shape); only holding it constant
    # matters. ``None`` until the first capture sets it, and for projects captured without
    # a loopback, where the correction is deliberately not applied.
    alignment_reference: _Optional[float] = None

    def knob_specs(self) -> tuple[KnobSpec, ...]:
        return tuple(knob.to_knob_spec() for knob in self.knobs)

    def entries_for_split(
        self, split: _Literal["train", "validation"]
    ) -> list[CaptureEntryModel]:
        return [entry for entry in self.entries if entry.split == split]

    def pending_entries(self) -> list[CaptureEntryModel]:
        return [entry for entry in self.entries if entry.status == "pending"]

    def captured_entries(self) -> list[CaptureEntryModel]:
        return [entry for entry in self.entries if entry.status == "captured"]

    def input_for_split(self, split: _Literal["train", "validation"]) -> str:
        return self.train_input if split == "train" else self.validation_input

    def window_for_split(self, split: _Literal["train", "validation"]) -> WindowModel:
        return self.train_window if split == "train" else self.validation_window


def new_project(
    knobs: _Sequence[KnobSpec],
    *,
    n_train: int,
    n_validation: int,
    seed: int = 0,
    name: str = "",
    train_input: str = "input_train.wav",
    validation_input: str = "input_validation.wav",
) -> CaptureProject:
    """
    Plan every capture up front and return a project ready to save. Nothing is
    recorded yet: all entries start pending.
    """
    train, validation = _plan_captures(
        knobs, n_train=n_train, n_validation=n_validation, seed=seed
    )
    entries = [
        CaptureEntryModel(
            index=planned.index,
            split=planned.split,
            params=planned.params,
            y_path=planned.y_path,
        )
        for planned in train + validation
    ]
    return CaptureProject(
        name=name,
        knobs=[KnobModel.from_knob_spec(knob) for knob in knobs],
        seed=seed,
        train_input=train_input,
        validation_input=validation_input,
        entries=entries,
        n_train_lhs=n_train,
    )


CORNER_Y_PATH_PREFIX = "corner_"


def _is_corner_entry(entry: CaptureEntryModel) -> bool:
    return _Path(entry.y_path).name.startswith(CORNER_Y_PATH_PREFIX)


def add_corner_captures(project: CaptureProject) -> tuple[list[CaptureEntryModel], int]:
    """
    Append the initial "corner" captures (knob-range extremes) to ``project`` as pending
    ``train`` entries, in addition to whatever is already planned. Corners whose setting
    already appears in the plan (an LHS point or an earlier corner) are skipped, so calling
    this again after adding LHS points -- or twice -- never duplicates a capture.

    Uses each knob's Gain/Drive marking (see :class:`~nam.capture.params.KnobSpec`) to shape
    the corner set. Mutates ``project.entries`` in place but does not save; the caller saves.
    Returns the appended entries and the count of distinct corners skipped as duplicates.
    """
    knobs = project.knob_specs()
    specs = tuple(knob.to_param_spec() for knob in knobs)
    existing_keys = {
        tuple(entry.params[spec.name] for spec in specs) for entry in project.entries
    }
    train_indices = [entry.index for entry in project.entries if entry.split == "train"]
    next_index = max(train_indices) + 1 if train_indices else 0
    corner_count = sum(1 for entry in project.entries if _is_corner_entry(entry))

    planned, skipped = _plan_corner_captures(
        knobs,
        exclude=existing_keys,
        index_offset=next_index,
        filename_start=corner_count,
    )
    appended = [
        CaptureEntryModel(
            index=capture.index,
            split=capture.split,
            params=capture.params,
            y_path=capture.y_path,
        )
        for capture in planned
    ]
    project.entries.extend(appended)
    return appended, skipped


def save_project(project: CaptureProject, project_dir: _Path) -> _Path:
    path = _Path(project_dir) / PROJECT_FILENAME
    atomic_write_json(path, project.model_dump(mode="json"))
    return path


def load_project(project_dir: _Path) -> CaptureProject:
    path = _Path(project_dir) / PROJECT_FILENAME
    with path.open() as fp:
        payload = _json.load(fp)
    version = payload.get("version")
    if version != _PROJECT_VERSION:
        raise ValueError(
            f"Unsupported capture project version {version!r}; expected {_PROJECT_VERSION}"
        )
    project = CaptureProject.model_validate(payload)
    # Surface bad knob definitions at load time rather than when planning/recapturing.
    project.knob_specs()
    return project


def reconcile_with_disk(project: CaptureProject, project_dir: _Path) -> list[str]:
    """
    Bring entry statuses back in line with the files actually on disk and return
    human-readable notes about anything that changed or looks off.

    A captured entry whose WAV has disappeared goes back to pending (its delay and QA
    are stale evidence about a file that no longer exists). A pending entry whose WAV
    already exists is left pending — the file's provenance is unknown, so the note
    warns that capturing will overwrite it.
    """
    project_dir = _Path(project_dir)
    notes: list[str] = []
    for entry in project.entries:
        wav_path = project_dir / entry.y_path
        exists = wav_path.is_file()
        if entry.status == "captured" and not exists:
            entry.status = "pending"
            entry.delay = None
            entry.captured_at = None
            entry.qa = None
            notes.append(
                f"{entry.y_path}: recorded as captured but the file is missing; "
                "reset to pending."
            )
        elif entry.status == "pending" and exists:
            notes.append(
                f"{entry.y_path}: file exists but is not recorded as captured; "
                "capturing this setting will overwrite it."
            )
    known = {str(_Path(entry.y_path)) for entry in project.entries}
    captures_dir = project_dir / CAPTURES_DIRNAME
    if captures_dir.is_dir():
        for wav_path in sorted(captures_dir.glob("*.wav")):
            relative = str(wav_path.relative_to(project_dir))
            if relative not in known:
                notes.append(f"{relative}: not part of this project's plan.")
    return notes


def migrate_legacy_peak_values(project: CaptureProject, project_dir: _Path) -> list[str]:
    """
    Recompute ``qa.peak`` in dBFS for captures saved back when it was a linear 0-1
    amplitude. dBFS is never positive, so a stored value greater than 0 is
    unambiguously the old format; those entries' capture WAVs are re-read from disk
    and the peak is recalculated (rather than just converting the stored number, in
    case the on-disk audio has since changed). Does not save; the caller saves.
    Returns a note per entry that was migrated.
    """
    from ..data import wav_to_np
    from .audio import peak_to_dbfs
    from .session import CLIPPING_THRESHOLD

    project_dir = _Path(project_dir)
    notes: list[str] = []
    for entry in project.entries:
        if entry.status != "captured" or entry.qa is None or entry.qa.peak is None:
            continue
        if entry.qa.peak <= 0:
            continue
        path = project_dir / entry.y_path
        try:
            y = _np.asarray(wav_to_np(path), dtype=_np.float32).squeeze()
        except Exception as exc:
            notes.append(
                f"{entry.y_path}: could not recompute peak from WAV ({exc})."
            )
            continue
        peak = float(_np.max(_np.abs(y))) if len(y) else 0.0
        entry.qa.peak = peak_to_dbfs(peak)
        entry.qa.clipping = peak >= CLIPPING_THRESHOLD
        notes.append(f"{entry.y_path}: peak migrated to {entry.qa.peak:.1f} dBFS.")
    return notes


def _data_json_delays(project_dir: _Path) -> dict[str, _Optional[int]]:
    """
    Map each capture's project-relative ``y_path`` to the ``delay`` recorded for it in
    an existing ``data.json``, if one is present and readable. Returns an empty mapping
    when there is no usable data.json.
    """
    path = _Path(project_dir) / DATA_FILENAME
    if not path.is_file():
        return {}
    try:
        with path.open() as fp:
            payload = _json.load(fp)
    except (OSError, ValueError):
        return {}
    delays: dict[str, _Optional[int]] = {}
    for split in ("train", "validation"):
        for item in payload.get(split) or []:
            y_path = item.get("y_path")
            if y_path is None:
                continue
            delays[str(_Path(y_path))] = item.get("delay")
    return delays


def find_recoverable_entries(
    project: CaptureProject, project_dir: _Path
) -> list[tuple[CaptureEntryModel, _Optional[int]]]:
    """
    Pending entries whose capture WAV already exists on disk *and* is recorded in an
    existing ``data.json``. Regenerating the plan (e.g. with the same seed) resets entry
    statuses to pending but leaves the WAV files and data.json in place, so these
    captures can be restored instead of recaptured.

    Returns ``(entry, delay)`` pairs, where ``delay`` is the value data.json recorded for
    that capture (``data.json`` does not record QA or ``captured_at``, so those are
    reconstructed when the entry is restored).
    """
    project_dir = _Path(project_dir)
    delays = _data_json_delays(project_dir)
    recoverable: list[tuple[CaptureEntryModel, _Optional[int]]] = []
    for entry in project.entries:
        if entry.status != "pending":
            continue
        key = str(_Path(entry.y_path))
        if key not in delays:
            continue
        if not (project_dir / entry.y_path).is_file():
            continue
        recoverable.append((entry, delays[key]))
    return recoverable


def mark_captured(
    entry: CaptureEntryModel,
    *,
    delay: _Optional[int],
    qa: QAModel,
    stream_config: _Optional[str] = None,
) -> None:
    entry.status = "captured"
    entry.delay = delay
    entry.qa = qa
    entry.stream_config = stream_config
    entry.captured_at = _datetime.now(_timezone.utc).isoformat(timespec="seconds")
