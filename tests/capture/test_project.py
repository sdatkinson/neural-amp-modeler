import json as _json
from datetime import datetime as _datetime
from pathlib import Path as _Path

import pytest as _pytest

from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.planner import CAPTURES_DIRNAME as _CAPTURES_DIRNAME
from nam.capture.project import CaptureProject as _CaptureProject
from nam.capture.project import PROJECT_FILENAME as _PROJECT_FILENAME
from nam.capture.project import QAModel as _QAModel
from nam.capture.project import atomic_write_json as _atomic_write_json
from nam.capture.project import find_recoverable_entries as _find_recoverable_entries
from nam.capture.project import load_project as _load_project
from nam.capture.project import mark_captured as _mark_captured
from nam.capture.project import new_project as _new_project
from nam.capture.project import reconcile_with_disk as _reconcile_with_disk
from nam.capture.project import save_project as _save_project


def _knobs() -> list[_KnobSpec]:
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]


def _project(**kwargs) -> _CaptureProject:
    defaults = dict(n_train=4, n_validation=2, seed=0)
    defaults.update(kwargs)
    return _new_project(_knobs(), **defaults)


def test_new_project_plans_entries():
    project = _project(n_train=5, n_validation=3)

    assert len(project.entries) == 8
    assert all(entry.status == "pending" for entry in project.entries)

    y_paths = [entry.y_path for entry in project.entries]
    assert len(set(y_paths)) == len(y_paths)

    knob_names = {knob.name for knob in project.knobs}
    for entry in project.entries:
        assert set(entry.params.keys()) == knob_names


def test_save_and_load_project_round_trips(tmp_path: _Path):
    project = _project()

    path = _save_project(project, tmp_path)
    assert path == tmp_path / _PROJECT_FILENAME
    assert path.is_file()

    with path.open() as fp:
        payload = _json.load(fp)
    assert isinstance(payload, dict)

    loaded = _load_project(tmp_path)
    assert loaded == project


def test_load_project_rejects_wrong_version(tmp_path: _Path):
    path = tmp_path / _PROJECT_FILENAME
    with path.open("w") as fp:
        _json.dump({"version": 999}, fp)

    with _pytest.raises(ValueError):
        _load_project(tmp_path)


def test_atomic_write_json_writes_valid_json_and_no_tmp_files(tmp_path: _Path):
    path = tmp_path / "out.json"
    _atomic_write_json(path, {"a": 1})

    assert path.is_file()
    with path.open() as fp:
        assert _json.load(fp) == {"a": 1}
    assert list(tmp_path.glob("*.tmp")) == []

    # Overwrites existing file.
    _atomic_write_json(path, {"a": 2})
    with path.open() as fp:
        assert _json.load(fp) == {"a": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_reconcile_captured_entry_missing_wav_resets_to_pending(tmp_path: _Path):
    project = _project()
    entry = project.entries[0]
    _mark_captured(entry, delay=12, qa=_QAModel(peak=0.5))

    # No file was actually written to disk for this entry's y_path.
    notes = _reconcile_with_disk(project, tmp_path)

    assert entry.status == "pending"
    assert entry.delay is None
    assert entry.qa is None
    assert entry.captured_at is None
    assert any("missing" in note and entry.y_path in note for note in notes)


def test_reconcile_pending_entry_with_existing_wav_stays_pending_with_note(
    tmp_path: _Path,
):
    project = _project()
    entry = project.entries[0]
    wav_path = tmp_path / entry.y_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.touch()

    notes = _reconcile_with_disk(project, tmp_path)

    assert entry.status == "pending"
    assert any("overwrite" in note and entry.y_path in note for note in notes)


def test_reconcile_flags_stray_wav_not_in_plan(tmp_path: _Path):
    project = _project()
    captures_dir = tmp_path / _CAPTURES_DIRNAME
    captures_dir.mkdir(parents=True, exist_ok=True)
    stray = captures_dir / "not_in_plan.wav"
    stray.touch()

    notes = _reconcile_with_disk(project, tmp_path)

    relative = str(stray.relative_to(tmp_path))
    assert any(relative in note and "not part of this project's plan" in note for note in notes)


def test_reconcile_consistent_project_has_no_notes(tmp_path: _Path):
    project = _project()
    entry = project.entries[0]
    wav_path = tmp_path / entry.y_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.touch()
    _mark_captured(entry, delay=5, qa=_QAModel())

    notes = _reconcile_with_disk(project, tmp_path)

    assert notes == []


def _write_data_json(tmp_path: _Path, entries: list[tuple]) -> None:
    """entries: (split, y_path, delay) rows to record in a data.json."""
    payload: dict = {
        "type": "parametric",
        "common": {"delay": 0},
        "train": [],
        "validation": [],
    }
    for split, y_path, delay in entries:
        payload[split].append(
            {"x_path": "input.wav", "y_path": y_path, "delay": delay, "params": {}}
        )
    _atomic_write_json(tmp_path / "data.json", payload)


def _touch_wav(tmp_path: _Path, y_path: str) -> None:
    wav_path = tmp_path / y_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.touch()


def test_find_recoverable_entries_matches_disk_and_data_json(tmp_path: _Path):
    project = _project(n_train=3, n_validation=0)
    on_disk = project.entries[0]
    missing_wav = project.entries[1]  # in data.json but no WAV on disk
    _touch_wav(tmp_path, on_disk.y_path)
    _touch_wav(tmp_path, project.entries[2].y_path)  # WAV, but not in data.json
    _write_data_json(
        tmp_path,
        [("train", on_disk.y_path, 37), ("train", missing_wav.y_path, 5)],
    )

    recoverable = _find_recoverable_entries(project, tmp_path)

    assert recoverable == [(on_disk, 37)]


def test_find_recoverable_entries_ignores_already_captured(tmp_path: _Path):
    project = _project(n_train=2, n_validation=0)
    entry = project.entries[0]
    _touch_wav(tmp_path, entry.y_path)
    _write_data_json(tmp_path, [("train", entry.y_path, 9)])
    _mark_captured(entry, delay=9, qa=_QAModel())

    assert _find_recoverable_entries(project, tmp_path) == []


def test_find_recoverable_entries_empty_without_data_json(tmp_path: _Path):
    project = _project(n_train=2, n_validation=0)
    _touch_wav(tmp_path, project.entries[0].y_path)

    assert _find_recoverable_entries(project, tmp_path) == []


def test_mark_captured_sets_status_and_iso_utc_timestamp():
    project = _project()
    entry = project.entries[0]
    qa = _QAModel(peak=0.9, clipping=False)

    _mark_captured(entry, delay=42, qa=qa)

    assert entry.status == "captured"
    assert entry.delay == 42
    assert entry.qa == qa
    assert entry.captured_at is not None
    parsed = _datetime.fromisoformat(entry.captured_at)
    assert parsed.utcoffset().total_seconds() == 0


def test_capture_project_helper_methods():
    project = _project(n_train=3, n_validation=2)

    train_entries = project.entries_for_split("train")
    validation_entries = project.entries_for_split("validation")
    assert len(train_entries) == 3
    assert len(validation_entries) == 2
    assert all(entry.split == "train" for entry in train_entries)
    assert all(entry.split == "validation" for entry in validation_entries)

    assert project.pending_entries() == project.entries
    assert project.captured_entries() == []

    _mark_captured(train_entries[0], delay=1, qa=_QAModel())
    assert train_entries[0] in project.captured_entries()
    assert train_entries[0] not in project.pending_entries()

    assert project.input_for_split("train") == project.train_input
    assert project.input_for_split("validation") == project.validation_input

    assert project.window_for_split("train") == project.train_window
    assert project.window_for_split("validation") == project.validation_window


def test_default_windows_only_set_ny_for_train():
    project = _project()

    assert project.train_window.ny == 8192
    assert project.validation_window.ny is None
