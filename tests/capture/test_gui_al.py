import json as _json
import os as _os
from pathlib import Path as _Path

import pytest as _pytest

_pytest.importorskip("PySide6")

_os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication as _QApplication

from nam.capture import al_runner as _al_runner
from nam.capture.gui.main import MainWindow as _MainWindow
from nam.capture.params import KnobSpec as _KnobSpec
from nam.capture.project import PROJECT_FILENAME as _PROJECT_FILENAME
from nam.capture.project import load_project as _load_project
from nam.capture.project import new_project as _new_project
from nam.capture.project import save_project as _save_project


@_pytest.fixture(scope="module")
def _qapp():
    app = _QApplication.instance() or _QApplication([])
    yield app


def _knobs() -> list[_KnobSpec]:
    return [
        _KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5),
        _KnobSpec(name="Tone", min=0.0, max=10.0, step=1.0),
    ]


def _project(**kwargs):
    defaults = dict(n_train=2, n_validation=1, seed=0)
    defaults.update(kwargs)
    return _new_project(_knobs(), **defaults)


def _write_round(project_dir: _Path, round_idx: int, records: list[dict]) -> None:
    al_dir = project_dir / _al_runner.AL_DIRNAME
    al_dir.mkdir(parents=True, exist_ok=True)
    (al_dir / f"proposed_captures_round_{round_idx}.json").write_text(_json.dumps(records))


def test_al_tab_has_expected_widgets(_qapp):
    window = _MainWindow()
    assert window.al_start_button.text() == "Start round"
    assert window.al_cancel_button.text() == "Cancel round"
    assert window.al_import_button.text() == "Import proposals"
    assert window.al_export_button.text() == "Export remote runner files"
    assert window.al_cancel_button.isEnabled() is False
    assert window.al_next_round_label.text() == "No project open."
    window.close()


def test_refresh_al_tab_with_no_project_does_not_crash(_qapp):
    window = _MainWindow()
    window._refresh_al_tab()
    assert window.al_next_round_label.text() == "No project open."
    window.close()


def test_al_process_args_omits_max_workers_when_zero(_qapp, tmp_path: _Path):
    window = _MainWindow()
    window.project_dir = tmp_path
    args = window._al_process_args()
    assert args[:4] == ["-m", "nam.capture.al_runner", "--project-dir", str(tmp_path)]
    assert "--max-workers" not in args
    window.close()


def test_al_process_args_includes_max_workers_when_positive(_qapp, tmp_path: _Path):
    window = _MainWindow()
    window.project_dir = tmp_path
    window.al_max_workers_spin.setValue(3)
    args = window._al_process_args()
    assert "--max-workers" in args
    assert args[args.index("--max-workers") + 1] == "3"
    window.close()


def test_al_process_args_reflects_settings_spinboxes(_qapp, tmp_path: _Path):
    window = _MainWindow()
    window.project_dir = tmp_path
    window.al_max_per_round_spin.setValue(5)
    window.al_ensemble_size_spin.setValue(2)
    window.al_num_restarts_spin.setValue(3)
    window.al_num_steps_spin.setValue(50)
    args = window._al_process_args()
    assert args[args.index("--max-per-round") + 1] == "5"
    assert args[args.index("--ensemble-size") + 1] == "2"
    assert args[args.index("--num-restarts") + 1] == "3"
    assert args[args.index("--num-steps") + 1] == "50"
    window.close()


def test_import_al_proposals_adds_pending_entries_and_saves(_qapp, tmp_path: _Path):
    project = _project()
    _save_project(project, tmp_path)
    max_train_index = max(entry.index for entry in project.entries_for_split("train"))
    records = [
        {"params": {"Gain": 1.0, "Tone": 2.0}, "score": 0.9, "y_path": "captures/r0_a.wav"},
        {"params": {"Gain": 3.0, "Tone": 4.0}, "score": 0.8, "y_path": "captures/r0_b.wav"},
    ]
    _write_round(tmp_path, 0, records)

    window = _MainWindow()
    window.project = project
    window.project_dir = tmp_path

    window._on_al_import_proposals()

    train_entries = project.entries_for_split("train")
    new_entries = [e for e in train_entries if e.y_path in {"captures/r0_a.wav", "captures/r0_b.wav"}]
    assert len(new_entries) == 2
    assert [e.index for e in new_entries] == [max_train_index + 1, max_train_index + 2]
    assert all(e.status == "pending" for e in new_entries)

    reloaded = _load_project(tmp_path)
    reloaded_new = [
        e for e in reloaded.entries if e.y_path in {"captures/r0_a.wav", "captures/r0_b.wav"}
    ]
    assert len(reloaded_new) == 2
    assert (tmp_path / _PROJECT_FILENAME).exists()

    window._on_al_import_proposals()
    train_entries_after = project.entries_for_split("train")
    assert len(train_entries_after) == len(train_entries)

    window.close()
