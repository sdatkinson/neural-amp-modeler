"""
Desktop app for the parametric capture engine.

Wires the pure ``nam.capture`` primitives (planning, project persistence, audio
device enumeration, and the capture session) to a PySide6 UI. Every blocking audio
call runs on a :class:`~nam.capture.gui.workers.SessionWorker` thread; the GUI thread
only ever touches widgets and the in-memory project.
"""

from __future__ import annotations

import shutil as _shutil
import sys as _sys
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Sequence as _Sequence

from PySide6.QtWidgets import QApplication as _QApplication
from PySide6.QtWidgets import QComboBox as _QComboBox
from PySide6.QtWidgets import QDialog as _QDialog
from PySide6.QtWidgets import QDialogButtonBox as _QDialogButtonBox
from PySide6.QtWidgets import QFileDialog as _QFileDialog
from PySide6.QtWidgets import QFormLayout as _QFormLayout
from PySide6.QtWidgets import QLineEdit as _QLineEdit
from PySide6.QtWidgets import QHBoxLayout as _QHBoxLayout
from PySide6.QtWidgets import QLabel as _QLabel
from PySide6.QtWidgets import QMainWindow as _QMainWindow
from PySide6.QtWidgets import QMessageBox as _QMessageBox
from PySide6.QtWidgets import QPlainTextEdit as _QPlainTextEdit
from PySide6.QtWidgets import QProgressBar as _QProgressBar
from PySide6.QtWidgets import QPushButton as _QPushButton
from PySide6.QtWidgets import QSpinBox as _QSpinBox
from PySide6.QtWidgets import QTableWidget as _QTableWidget
from PySide6.QtWidgets import QTableWidgetItem as _QTableWidgetItem
from PySide6.QtWidgets import QTabWidget as _QTabWidget
from PySide6.QtWidgets import QVBoxLayout as _QVBoxLayout
from PySide6.QtWidgets import QWidget as _QWidget

from ..audio import DeviceInfo as _DeviceInfo
from ..audio import list_devices as _list_devices
from ..export import write_training_configs as _write_training_configs
from ..params import KnobSpec as _KnobSpec
from ..params import validate_knobs as _validate_knobs
from ..project import CaptureEntryModel as _CaptureEntryModel
from ..project import CaptureProject as _CaptureProject
from ..project import load_project as _load_project
from ..project import new_project as _new_project
from ..project import PROJECT_FILENAME as _PROJECT_FILENAME
from ..project import reconcile_with_disk as _reconcile_with_disk
from ..project import save_project as _save_project
from ..project import QAModel as _QAModel
from ..session import CaptureSession as _CaptureSession
from .workers import CancelToken as _CancelToken
from .workers import SessionWorker as _SessionWorker


_PLAN_COLUMNS = ("Split", "Index", "Params", "Filename", "Status", "Delay", "QA")


def format_number(value: float) -> str:
    """
    Render a knob value the way a user would dial it: "3.5", not "3.500000".
    """

    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_params(params: dict) -> str:
    return ", ".join(f"{name}={format_number(value)}" for name, value in params.items())


def format_qa_summary(qa: _Optional[_QAModel]) -> str:
    if qa is None:
        return ""
    parts = [f"peak={qa.peak:.3f}" if qa.peak is not None else "peak=?"]
    if qa.clipping:
        parts.append("CLIPPING")
    if qa.impulse_detected is False:
        parts.append("no impulse")
    if qa.delay_disagreement:
        parts.append("delay disagreement")
    return ", ".join(parts)


def format_entry_row(entry: _CaptureEntryModel) -> tuple:
    return (
        entry.split,
        str(entry.index),
        format_params(entry.params),
        entry.y_path,
        entry.status,
        "" if entry.delay is None else str(entry.delay),
        format_qa_summary(entry.qa),
    )


def knob_rows_to_specs(rows: _Sequence[tuple]) -> list[_KnobSpec]:
    """
    Convert raw (name, min, max, step) table rows into :class:`KnobSpec`\\ s.

    Values are passed through as-is; :class:`KnobSpec` parses and validates numeric
    fields itself and raises ``ValueError`` with a message fit to show the user.
    """

    return [
        _KnobSpec(name=name, min=min_value, max=max_value, step=step_value)
        for name, min_value, max_value, step_value in rows
    ]


def format_device_label(device: _DeviceInfo) -> str:
    return f"{device.name} ({device.host_api})"


def devices_for_direction(
    devices: _Sequence[_DeviceInfo], direction: str
) -> list[_DeviceInfo]:
    if direction == "output":
        return [device for device in devices if device.max_output_channels > 0]
    if direction == "input":
        return [device for device in devices if device.max_input_channels > 0]
    raise ValueError(f"direction must be 'output' or 'input'; got {direction!r}")


class InputWavDialog(_QDialog):
    """
    Collect the training and validation input WAVs with both fields labelled and
    visible at once. The native file picker does not show which file it is asking
    for, so each row has its own labelled "Browse..." button; leaving a field
    blank skips that input (it can be added later).
    """

    def __init__(self, parent: _Optional[_QWidget], project_dir: _Path) -> None:
        super().__init__(parent)
        self._project_dir = project_dir
        self.setWindowTitle("Input WAV files")

        layout = _QVBoxLayout(self)
        layout.addWidget(
            _QLabel(
                "Choose the input (DI / reamp) WAV files that get played through "
                "your rig during capture. Leave a field blank to add it later; "
                "files outside this folder are copied in."
            )
        )

        form = _QFormLayout()
        self.train_edit = _QLineEdit()
        self.train_edit.setPlaceholderText("(none -- add later)")
        self.validation_edit = _QLineEdit()
        self.validation_edit.setPlaceholderText("(none -- add later)")
        form.addRow("Training input:", self._browse_row(self.train_edit, "training"))
        form.addRow(
            "Validation input:", self._browse_row(self.validation_edit, "validation")
        )
        layout.addLayout(form)

        buttons = _QDialogButtonBox(
            _QDialogButtonBox.StandardButton.Ok
            | _QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_row(self, edit: _QLineEdit, which: str) -> _QWidget:
        container = _QWidget()
        row = _QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        browse = _QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse(edit, which))
        row.addWidget(browse)
        return container

    def _browse(self, edit: _QLineEdit, which: str) -> None:
        start = edit.text().strip() or str(self._project_dir)
        path, _filter = _QFileDialog.getOpenFileName(
            self, f"Select the {which} input WAV", start, "WAV files (*.wav)"
        )
        if path:
            edit.setText(path)

    def selected_paths(self) -> tuple[str, str]:
        """Return the chosen (training, validation) paths; either may be empty."""
        return self.train_edit.text().strip(), self.validation_edit.text().strip()


class MainWindow(_QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NAM Parametric Capture")

        self.project: _Optional[_CaptureProject] = None
        self.project_dir: _Optional[_Path] = None
        self.session: _Optional[_CaptureSession] = None
        # Project-relative filenames of the input WAVs, carried into the plan when
        # it is generated. Files copied in from outside use the canonical names;
        # files already inside the project folder keep their own name.
        self._train_input_name: str = "input_train.wav"
        self._validation_input_name: str = "input_validation.wav"
        self._devices: list[_DeviceInfo] = []
        self._worker: _Optional[_SessionWorker] = None
        self._cancel_token: _Optional[_CancelToken] = None

        self._build_ui()
        self._refresh_devices()
        self._refresh_all()

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        tabs = _QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_project_tab(), "Project")
        tabs.addTab(self._build_knobs_tab(), "Knobs")
        tabs.addTab(self._build_plan_tab(), "Plan")
        tabs.addTab(self._build_audio_tab(), "Audio I/O")
        tabs.addTab(self._build_capture_tab(), "Capture")
        self.status_bar = self.statusBar()

    def _build_project_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        self.project_label = _QLabel("No project open.")
        layout.addWidget(self.project_label)

        buttons = _QHBoxLayout()
        new_button = _QPushButton("New Project...")
        new_button.clicked.connect(self._on_new_project)
        open_button = _QPushButton("Open Project...")
        open_button.clicked.connect(self._on_open_project)
        buttons.addWidget(new_button)
        buttons.addWidget(open_button)
        layout.addLayout(buttons)

        self.project_log = _QPlainTextEdit()
        self.project_log.setReadOnly(True)
        layout.addWidget(self.project_log)
        return widget

    def _build_knobs_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        self.knob_table = _QTableWidget(0, 4)
        self.knob_table.setHorizontalHeaderLabels(["Name", "Min", "Max", "Step"])
        layout.addWidget(self.knob_table)

        row_buttons = _QHBoxLayout()
        add_button = _QPushButton("Add knob")
        add_button.clicked.connect(self._on_add_knob_row)
        remove_button = _QPushButton("Remove selected")
        remove_button.clicked.connect(self._on_remove_knob_row)
        row_buttons.addWidget(add_button)
        row_buttons.addWidget(remove_button)
        layout.addLayout(row_buttons)

        form = _QFormLayout()
        self.n_train_spin = _QSpinBox()
        self.n_train_spin.setRange(1, 100_000)
        self.n_train_spin.setValue(20)
        self.n_validation_spin = _QSpinBox()
        self.n_validation_spin.setRange(0, 100_000)
        self.n_validation_spin.setValue(5)
        self.seed_spin = _QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        form.addRow("Train captures", self.n_train_spin)
        form.addRow("Validation captures", self.n_validation_spin)
        form.addRow("Seed", self.seed_spin)
        layout.addLayout(form)

        generate_button = _QPushButton("Generate plan")
        generate_button.clicked.connect(self._on_generate_plan)
        layout.addWidget(generate_button)

        self._on_add_knob_row()
        return widget

    def _build_plan_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)
        self.plan_table = self._make_entry_table()
        layout.addWidget(self.plan_table)
        return widget

    def _build_audio_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        form = _QFormLayout()
        self.output_device_combo = _QComboBox()
        self.input_device_combo = _QComboBox()
        self.output_channel_spin = _QSpinBox()
        self.output_channel_spin.setRange(1, 1)
        self.input_channel_spin = _QSpinBox()
        self.input_channel_spin.setRange(1, 1)
        form.addRow("Output device", self.output_device_combo)
        form.addRow("Output channel", self.output_channel_spin)
        form.addRow("Input device", self.input_device_combo)
        form.addRow("Input channel", self.input_channel_spin)
        layout.addLayout(form)

        buttons = _QHBoxLayout()
        refresh_button = _QPushButton("Refresh devices")
        refresh_button.clicked.connect(self._refresh_devices)
        self.route_test_button = _QPushButton("Route test")
        self.route_test_button.clicked.connect(self._on_route_test)
        buttons.addWidget(refresh_button)
        buttons.addWidget(self.route_test_button)
        layout.addLayout(buttons)

        self.route_test_progress = _QProgressBar()
        self.route_test_progress.setRange(0, 100)
        layout.addWidget(self.route_test_progress)

        self.route_test_result_label = _QLabel("")
        layout.addWidget(self.route_test_result_label)

        self.output_device_combo.currentIndexChanged.connect(
            self._on_output_device_changed
        )
        self.input_device_combo.currentIndexChanged.connect(
            self._on_input_device_changed
        )
        self.output_channel_spin.valueChanged.connect(self._save_audio_settings)
        self.input_channel_spin.valueChanged.connect(self._save_audio_settings)
        return widget

    def _build_capture_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        self.next_entry_label = _QLabel("No project open.")
        layout.addWidget(self.next_entry_label)

        buttons = _QHBoxLayout()
        self.capture_next_button = _QPushButton("Capture next")
        self.capture_next_button.clicked.connect(self._on_capture_next)
        self.capture_selected_button = _QPushButton("Capture selected / recapture")
        self.capture_selected_button.clicked.connect(self._on_capture_selected)
        self.cancel_button = _QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_capture)
        buttons.addWidget(self.capture_next_button)
        buttons.addWidget(self.capture_selected_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self.capture_progress = _QProgressBar()
        self.capture_progress.setRange(0, 100)
        layout.addWidget(self.capture_progress)

        self.capture_table = self._make_entry_table()
        self.capture_table.setSelectionBehavior(
            _QTableWidget.SelectionBehavior.SelectRows
        )
        self.capture_table.setSelectionMode(_QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.capture_table)

        self.capture_log = _QPlainTextEdit()
        self.capture_log.setReadOnly(True)
        layout.addWidget(self.capture_log)

        export_button = _QPushButton("Export training configs")
        export_button.clicked.connect(self._on_export_configs)
        layout.addWidget(export_button)
        return widget

    @staticmethod
    def _make_entry_table() -> _QTableWidget:
        table = _QTableWidget(0, len(_PLAN_COLUMNS))
        table.setHorizontalHeaderLabels(list(_PLAN_COLUMNS))
        table.setEditTriggers(_QTableWidget.EditTrigger.NoEditTriggers)
        return table

    # -- project setup -----------------------------------------------------

    def _on_new_project(self) -> None:
        directory = _QFileDialog.getExistingDirectory(self, "Choose project folder")
        if not directory:
            return
        project_dir = _Path(directory)

        # Don't silently clobber an existing project in this folder.
        if (project_dir / _PROJECT_FILENAME).exists():
            confirm = _QMessageBox.question(
                self,
                "Project already exists",
                f"A project ({_PROJECT_FILENAME}) already exists in this folder. "
                "Creating a new project overwrites its plan and knob settings "
                "when you generate a plan (captured WAV files are left in place).\n\n"
                'To keep your existing work, choose No and use "Open Project..." '
                "instead.\n\nCreate a new project here anyway?",
                _QMessageBox.StandardButton.Yes | _QMessageBox.StandardButton.No,
                _QMessageBox.StandardButton.No,
            )
            if confirm != _QMessageBox.StandardButton.Yes:
                return

        # The input WAVs (DI/reamp files played during capture) are optional at
        # creation time -- you only need them once you start capturing. Cancelling
        # this dialog must NOT abort project creation.
        train_name = "input_train.wav"
        validation_name = "input_validation.wav"
        recorded: list[str] = []
        dialog = InputWavDialog(self, project_dir)
        if dialog.exec() == _QDialog.DialogCode.Accepted:
            train_selected, validation_selected = dialog.selected_paths()
            try:
                if train_selected:
                    train_name = self._resolve_input(
                        _Path(train_selected), project_dir, "input_train.wav"
                    )
                    recorded.append(train_name)
                if validation_selected:
                    validation_name = self._resolve_input(
                        _Path(validation_selected),
                        project_dir,
                        "input_validation.wav",
                    )
                    recorded.append(validation_name)
            except OSError as exc:
                _QMessageBox.critical(self, "Copy failed", str(exc))
                return

        self.project = None
        self.project_dir = project_dir
        self.session = None
        self._train_input_name = train_name
        self._validation_input_name = validation_name
        self.project_label.setText(
            f"Project folder: {project_dir} (no plan yet -- use the Knobs tab)"
        )
        self.project_log.appendPlainText(f"New project folder: {project_dir}")
        if recorded:
            self.project_log.appendPlainText(
                f"Using input WAVs: {', '.join(recorded)}"
            )
        else:
            self.project_log.appendPlainText(
                "No input WAVs set yet -- add them to the project folder before "
                "capturing."
            )
        self._refresh_all()

    def _resolve_input(
        self, source: _Path, project_dir: _Path, canonical_name: str
    ) -> str:
        """
        Return the project-relative filename to record for an input WAV.

        A file already inside the project folder keeps its own name and is left in
        place; a file from outside is copied in under ``canonical_name``.
        """
        project_resolved = project_dir.resolve()
        source_resolved = source.resolve()
        if source_resolved.is_relative_to(project_resolved):
            return source_resolved.relative_to(project_resolved).as_posix()
        self._copy_input(source, project_dir / canonical_name)
        return canonical_name

    @staticmethod
    def _copy_input(source: _Path, destination: _Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == destination.resolve():
            return
        _shutil.copy2(source, destination)

    def _on_open_project(self) -> None:
        directory = _QFileDialog.getExistingDirectory(self, "Choose project folder")
        if not directory:
            return
        project_dir = _Path(directory)
        try:
            project = _load_project(project_dir)
            notes = _reconcile_with_disk(project, project_dir)
            _save_project(project, project_dir)
        except Exception as exc:
            _QMessageBox.critical(self, "Failed to open project", str(exc))
            return

        self.project = project
        self.project_dir = project_dir
        self.session = _CaptureSession(self.project, self.project_dir)
        # Preserve the project's input filenames so regenerating the plan keeps
        # them rather than reverting to the canonical defaults.
        self._train_input_name = project.train_input
        self._validation_input_name = project.validation_input
        self.project_label.setText(
            f"Project: {project.name or project_dir.name} ({project_dir})"
        )
        self._load_knobs_into_ui()
        if notes:
            self.project_log.appendPlainText("\n".join(notes))
            _QMessageBox.information(self, "Project reconciled", "\n".join(notes))
        else:
            self.project_log.appendPlainText(f"Opened project: {project_dir}")
        self._refresh_devices()
        self._refresh_all()

    def _load_knobs_into_ui(self) -> None:
        if self.project is None:
            return
        self.knob_table.setRowCount(0)
        for knob in self.project.knobs:
            self._add_knob_row(knob.name, knob.min, knob.max, knob.step)
        self.n_train_spin.setValue(len(self.project.entries_for_split("train")) or 1)
        self.n_validation_spin.setValue(len(self.project.entries_for_split("validation")))
        self.seed_spin.setValue(self.project.seed)

    # -- knobs / plan --------------------------------------------------

    def _on_add_knob_row(self) -> None:
        self._add_knob_row("", 0.0, 10.0, 0.5)

    def _add_knob_row(self, name: _Any, minimum: _Any, maximum: _Any, step: _Any) -> None:
        row = self.knob_table.rowCount()
        self.knob_table.insertRow(row)
        for col, value in enumerate((name, minimum, maximum, step)):
            self.knob_table.setItem(row, col, _QTableWidgetItem(str(value)))

    def _on_remove_knob_row(self) -> None:
        row = self.knob_table.currentRow()
        if row >= 0:
            self.knob_table.removeRow(row)

    def _knob_table_rows(self) -> list[tuple]:
        rows = []
        for row in range(self.knob_table.rowCount()):
            values = []
            for col in range(4):
                item = self.knob_table.item(row, col)
                values.append(item.text() if item is not None else "")
            rows.append(tuple(values))
        return rows

    def _on_generate_plan(self) -> None:
        if self.project_dir is None:
            _QMessageBox.warning(
                self, "No project folder", "Create or open a project folder first."
            )
            return

        try:
            specs = knob_rows_to_specs(self._knob_table_rows())
            _validate_knobs(specs)
        except ValueError as exc:
            _QMessageBox.critical(self, "Invalid knob settings", str(exc))
            return

        if self.project is not None and self.project.captured_entries():
            confirm = _QMessageBox.question(
                self,
                "Discard captures?",
                "Regenerating the plan discards all existing capture progress. "
                "Continue?",
            )
            if confirm != _QMessageBox.StandardButton.Yes:
                return

        try:
            project = _new_project(
                list(specs),
                n_train=self.n_train_spin.value(),
                n_validation=self.n_validation_spin.value(),
                seed=self.seed_spin.value(),
                name=self.project_dir.name,
                train_input=self._train_input_name,
                validation_input=self._validation_input_name,
            )
        except ValueError as exc:
            _QMessageBox.critical(self, "Invalid plan", str(exc))
            return

        if self.project is not None:
            project.audio = self.project.audio

        _save_project(project, self.project_dir)
        self.project = project
        self.session = _CaptureSession(self.project, self.project_dir)
        self.project_label.setText(
            f"Project: {project.name or self.project_dir.name} ({self.project_dir})"
        )
        self.project_log.appendPlainText(
            f"Generated plan: {len(project.entries)} entries "
            f"({self.n_train_spin.value()} train / "
            f"{self.n_validation_spin.value()} validation)."
        )
        self._load_audio_settings_into_ui()
        self._refresh_all()

    # -- audio -----------------------------------------------------------

    def _refresh_devices(self) -> None:
        try:
            self._devices = _list_devices()
        except Exception as exc:
            self._devices = []
            self.project_log.appendPlainText(f"Could not list audio devices: {exc}")
        self._populate_device_combo(
            self.output_device_combo, devices_for_direction(self._devices, "output")
        )
        self._populate_device_combo(
            self.input_device_combo, devices_for_direction(self._devices, "input")
        )
        self._load_audio_settings_into_ui()

    @staticmethod
    def _populate_device_combo(
        combo: _QComboBox, devices: _Sequence[_DeviceInfo]
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        for device in devices:
            combo.addItem(format_device_label(device), device)
        combo.blockSignals(False)

    @staticmethod
    def _select_combo_by_name(combo: _QComboBox, name: _Optional[str]) -> None:
        if name is None:
            return
        for index in range(combo.count()):
            device = combo.itemData(index)
            if device is not None and device.name == name:
                combo.setCurrentIndex(index)
                return

    def _load_audio_settings_into_ui(self) -> None:
        if self.project is None:
            return
        audio = self.project.audio
        self._select_combo_by_name(self.output_device_combo, audio.output_device)
        self._select_combo_by_name(self.input_device_combo, audio.input_device)
        self._update_channel_ranges()
        self.output_channel_spin.blockSignals(True)
        self.output_channel_spin.setValue(audio.output_channel)
        self.output_channel_spin.blockSignals(False)
        self.input_channel_spin.blockSignals(True)
        self.input_channel_spin.setValue(audio.input_channel)
        self.input_channel_spin.blockSignals(False)

    def _on_output_device_changed(self) -> None:
        self._update_channel_ranges()
        self._save_audio_settings()

    def _on_input_device_changed(self) -> None:
        self._update_channel_ranges()
        self._save_audio_settings()

    def _update_channel_ranges(self) -> None:
        output_device = self.output_device_combo.currentData()
        input_device = self.input_device_combo.currentData()
        self.output_channel_spin.setRange(
            1, max(1, output_device.max_output_channels if output_device else 1)
        )
        self.input_channel_spin.setRange(
            1, max(1, input_device.max_input_channels if input_device else 1)
        )

    def _save_audio_settings(self) -> None:
        if self.project is None or self.project_dir is None:
            return
        output_device = self.output_device_combo.currentData()
        input_device = self.input_device_combo.currentData()
        self.project.audio.output_device = output_device.name if output_device else None
        self.project.audio.input_device = input_device.name if input_device else None
        # A device name is disambiguated by one shared host API; if the chosen output
        # and input devices live on different host APIs, that disambiguation can only
        # cover one of them (see the docstring on AudioSettingsModel in project.py).
        if output_device is not None:
            self.project.audio.host_api = output_device.host_api
        elif input_device is not None:
            self.project.audio.host_api = input_device.host_api
        self.project.audio.output_channel = self.output_channel_spin.value()
        self.project.audio.input_channel = self.input_channel_spin.value()
        _save_project(self.project, self.project_dir)

    def _on_route_test(self) -> None:
        if self.session is None:
            _QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        self._save_audio_settings()
        self.route_test_progress.setValue(0)
        self.route_test_result_label.setText("Running route test...")
        session = self.session
        self._run_worker(
            lambda progress, cancel: session.route_test(progress=progress, cancel=cancel),
            on_progress=lambda fraction: self.route_test_progress.setValue(
                int(fraction * 100)
            ),
            on_success=self._on_route_test_success,
            on_failure=lambda message: self.route_test_result_label.setText(
                f"Route test failed: {message}"
            ),
            on_cancelled=lambda: self.route_test_result_label.setText(
                "Route test cancelled."
            ),
            disable=[self.route_test_button],
        )

    def _on_route_test_success(self, result: _Any) -> None:
        self.route_test_progress.setValue(100)
        if result.ok:
            self.route_test_result_label.setText(
                f"Route OK: delay={result.latency.delay} samples, "
                f"peak={result.peak:.3f}"
            )
        else:
            self.route_test_result_label.setText(
                f"Route test did not detect the signal (peak={result.peak:.3f}). "
                "Check routing and levels."
            )

    # -- capture -----------------------------------------------------------

    def _on_capture_next(self) -> None:
        if self.session is None or self.project is None:
            _QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        pending = self.project.pending_entries()
        if not pending:
            _QMessageBox.information(self, "Nothing pending", "All captures are complete.")
            return
        self._begin_capture(pending[0])

    def _on_capture_selected(self) -> None:
        if self.session is None or self.project is None:
            _QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        row = self.capture_table.currentRow()
        if row < 0 or row >= len(self.project.entries):
            _QMessageBox.warning(self, "No selection", "Select a row to capture.")
            return
        self._begin_capture(self.project.entries[row])

    def _begin_capture(self, entry: _CaptureEntryModel) -> None:
        self.capture_progress.setValue(0)
        self.capture_log.appendPlainText(
            f"Capturing {entry.split} #{entry.index}: {format_params(entry.params)}"
        )
        self.cancel_button.setEnabled(True)
        session = self.session
        assert session is not None
        self._run_worker(
            lambda progress, cancel: session.capture_entry(
                entry, progress=progress, cancel=cancel
            ),
            on_progress=lambda fraction: self.capture_progress.setValue(
                int(fraction * 100)
            ),
            on_success=self._on_capture_success,
            on_failure=self._on_capture_failure,
            on_cancelled=self._on_capture_cancelled,
            disable=[self.capture_next_button, self.capture_selected_button],
            on_finished=lambda: self.cancel_button.setEnabled(False),
        )

    def _on_cancel_capture(self) -> None:
        if self._cancel_token is not None:
            self._cancel_token.cancel()
            self.capture_log.appendPlainText("Cancel requested...")

    def _on_capture_success(self, qa: _QAModel) -> None:
        self.capture_progress.setValue(100)
        self.capture_log.appendPlainText(
            "Capture complete."
            if qa.peak is None
            else f"Capture complete. peak={qa.peak:.3f}"
        )
        for message in qa.messages:
            self.capture_log.appendPlainText(f"QA: {message}")
        if qa.messages:
            _QMessageBox.warning(
                self, "Capture QA warnings", "\n".join(qa.messages)
            )
        self._refresh_all()

    def _on_capture_failure(self, message: str) -> None:
        self.capture_log.appendPlainText(f"Capture failed: {message}")
        _QMessageBox.critical(self, "Capture failed", message)
        self._refresh_all()

    def _on_capture_cancelled(self) -> None:
        self.capture_log.appendPlainText("Capture cancelled.")
        self._refresh_all()

    def _on_export_configs(self) -> None:
        if self.project is None or self.project_dir is None:
            _QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        try:
            paths = _write_training_configs(self.project, self.project_dir)
        except Exception as exc:
            _QMessageBox.critical(self, "Export failed", str(exc))
            return
        message = "\n".join(str(path) for path in paths)
        self.capture_log.appendPlainText(f"Wrote training configs:\n{message}")
        _QMessageBox.information(self, "Training configs written", message)

    # -- worker plumbing -----------------------------------------------

    def _run_worker(
        self,
        call: _Callable[[_Callable[[float], None], _Callable[[], bool]], _Any],
        *,
        on_success: _Callable[[_Any], None],
        on_progress: _Optional[_Callable[[float], None]] = None,
        on_failure: _Optional[_Callable[[str], None]] = None,
        on_cancelled: _Optional[_Callable[[], None]] = None,
        disable: _Sequence[_Any] = (),
        on_finished: _Optional[_Callable[[], None]] = None,
    ) -> None:
        if self._worker is not None and self._worker.isRunning():
            _QMessageBox.warning(
                self, "Busy", "Another audio operation is already running."
            )
            return

        cancel_token = _CancelToken()
        worker = _SessionWorker(call, cancel_token)
        self._worker = worker
        self._cancel_token = cancel_token

        for widget in disable:
            widget.setEnabled(False)

        def cleanup() -> None:
            for widget in disable:
                widget.setEnabled(True)
            self._worker = None
            self._cancel_token = None
            if on_finished is not None:
                on_finished()

        if on_progress is not None:
            worker.progress.connect(on_progress)

        def handle_success(result: _Any) -> None:
            cleanup()
            on_success(result)

        def handle_failure(message: str) -> None:
            cleanup()
            if on_failure is not None:
                on_failure(message)
            else:
                _QMessageBox.critical(self, "Operation failed", message)

        def handle_cancelled() -> None:
            cleanup()
            if on_cancelled is not None:
                on_cancelled()

        worker.succeeded.connect(handle_success)
        worker.failed.connect(handle_failure)
        worker.cancelled.connect(handle_cancelled)
        worker.start()

    # -- refresh -----------------------------------------------------------

    def _refresh_all(self) -> None:
        self._refresh_plan_tables()
        self._refresh_next_entry_label()
        self._refresh_status_bar()

    def _refresh_plan_tables(self) -> None:
        entries = self.project.entries if self.project is not None else []
        self._populate_entry_table(self.plan_table, entries)
        self._populate_entry_table(self.capture_table, entries)

    @staticmethod
    def _populate_entry_table(
        table: _QTableWidget, entries: _Sequence[_CaptureEntryModel]
    ) -> None:
        table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            for col, value in enumerate(format_entry_row(entry)):
                table.setItem(row, col, _QTableWidgetItem(value))

    def _refresh_next_entry_label(self) -> None:
        if self.project is None:
            self.next_entry_label.setText("No project open.")
            return
        pending = self.project.pending_entries()
        if not pending:
            self.next_entry_label.setText("All captures complete.")
            return
        entry = pending[0]
        self.next_entry_label.setText(
            f"Next ({entry.split} #{entry.index}): "
            f"Set knobs to: {format_params(entry.params)}"
        )

    def _refresh_status_bar(self) -> None:
        if self.project is None:
            self.status_bar.showMessage("No project open.")
            return
        train = self.project.entries_for_split("train")
        validation = self.project.entries_for_split("validation")
        train_done = len([entry for entry in train if entry.status == "captured"])
        validation_done = len(
            [entry for entry in validation if entry.status == "captured"]
        )
        self.status_bar.showMessage(
            f"Train: {train_done}/{len(train)} captured  |  "
            f"Validation: {validation_done}/{len(validation)} captured"
        )


def main() -> None:
    app = _QApplication.instance() or _QApplication(_sys.argv)
    window = MainWindow()
    window.show()
    _sys.exit(app.exec())


if __name__ == "__main__":
    main()
