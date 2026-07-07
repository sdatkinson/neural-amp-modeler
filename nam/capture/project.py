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

from pydantic import BaseModel as _BaseModel
from pydantic import Field as _Field

from .params import DEFAULT_KNOB_STEP
from .params import KnobSpec
from .planner import CAPTURES_DIRNAME
from .planner import plan_captures as _plan_captures


PROJECT_FILENAME = "capture_project.json"
DATA_FILENAME = "data.json"
MODEL_CONFIG_FILENAME = "model.json"
LEARNING_CONFIG_FILENAME = "learning.json"

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

    def to_knob_spec(self) -> KnobSpec:
        return KnobSpec(
            name=self.name,
            min=self.min,
            max=self.max,
            step=self.step,
            default=self.default,
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


class QAModel(_BaseModel):
    peak: _Optional[float] = None
    clipping: _Optional[bool] = None
    impulse_detected: _Optional[bool] = None
    delay_disagreement: _Optional[bool] = None
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
        default_factory=lambda: WindowModel(ny=None)
    )
    audio: AudioSettingsModel = _Field(default_factory=AudioSettingsModel)
    entries: list[CaptureEntryModel] = _Field(default_factory=list)

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
    )


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


def mark_captured(
    entry: CaptureEntryModel,
    *,
    delay: _Optional[int],
    qa: QAModel,
) -> None:
    entry.status = "captured"
    entry.delay = delay
    entry.qa = qa
    entry.captured_at = _datetime.now(_timezone.utc).isoformat(timespec="seconds")
