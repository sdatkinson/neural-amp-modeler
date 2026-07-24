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
import time as _time
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Sequence as _Sequence

from PySide6.QtCore import QProcess as _QProcess
from PySide6.QtCore import QProcessEnvironment as _QProcessEnvironment
from PySide6.QtCore import Qt as _Qt
from PySide6.QtCore import QTimer as _QTimer
from PySide6.QtWidgets import QApplication as _QApplication
from PySide6.QtWidgets import QCheckBox as _QCheckBox
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

from .. import al_runner as _al_runner
from ..audio import current_device_sample_rates as _current_device_sample_rates
from ..audio import DeviceInfo as _DeviceInfo
from ..audio import list_devices as _list_devices
from ..export import write_training_configs as _write_training_configs
from ..params import KnobSpec as _KnobSpec
from ..params import validate_knobs as _validate_knobs
from ..project import add_corner_captures as _add_corner_captures
from ..project import AudioSettingsModel as _AudioSettingsModel
from ..project import CaptureEntryModel as _CaptureEntryModel
from ..project import CaptureProject as _CaptureProject
from ..project import find_recoverable_entries as _find_recoverable_entries
from ..project import KnobModel as _KnobModel
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

# Selectable audio device buffer sizes in frames; 0 lets PortAudio choose.
_BUFFER_SIZE_CHOICES = (0, 32, 64, 128, 256, 512, 1024, 2048, 4096)


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
    if qa.loopback_disagreement:
        parts.append("loopback mismatch")
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
    Convert raw (name, min, max, step, avoid_zero[, is_gain]) table rows into
    :class:`KnobSpec`\\ s. The trailing ``is_gain`` flag is optional and defaults to False.

    Values are passed through as-is; :class:`KnobSpec` parses and validates numeric
    fields itself and raises ``ValueError`` with a message fit to show the user.
    """

    specs = []
    for row in rows:
        name, min_value, max_value, step_value, avoid_zero = row[:5]
        is_gain = row[5] if len(row) > 5 else False
        specs.append(
            _KnobSpec(
                name=name,
                min=min_value,
                max=max_value,
                step=step_value,
                avoid_zero=avoid_zero,
                is_gain=is_gain,
            )
        )
    return specs


def format_device_label(device: _DeviceInfo) -> str:
    return f"{device.name} ({device.host_api})"


def read_wav_rate(path: _Path) -> _Optional[int]:
    """Return a WAV file's sample rate from its header, or ``None`` if unreadable."""
    import wave

    try:
        with wave.open(str(path), "rb") as fp:
            return fp.getframerate()
    except (wave.Error, EOFError, OSError):
        return None


def sample_rate_warnings(
    train_rate: _Optional[int],
    validation_rate: _Optional[int],
    output_device_rate: _Optional[int],
    input_device_rate: _Optional[int],
) -> list[str]:
    """
    Build warnings about sample-rate mismatches between the input WAVs and the
    selected audio devices. Captures play back and record at the input files' rate,
    so a device set to a different rate (or two inputs that disagree) forces silent
    resampling or an outright failure. Device rates are the current hardware rates.
    """
    warnings: list[str] = []
    if (
        train_rate is not None
        and validation_rate is not None
        and train_rate != validation_rate
    ):
        warnings.append(
            f"Training input ({train_rate} Hz) and validation input "
            f"({validation_rate} Hz) have different sample rates; they should match."
        )
    file_rates = {rate for rate in (train_rate, validation_rate) if rate is not None}
    # Only compare against devices when the inputs agree on a single rate; otherwise
    # the file mismatch above is the thing to fix first.
    if len(file_rates) == 1:
        file_rate = next(iter(file_rates))
        for label, device_rate in (
            ("Output", output_device_rate),
            ("Input", input_device_rate),
        ):
            if device_rate is not None and device_rate != file_rate:
                warnings.append(
                    f"{label} device rate ({device_rate} Hz) differs from the "
                    f"input files ({file_rate} Hz)."
                )
    return warnings


def duplex_devices(devices: _Sequence[_DeviceInfo]) -> list[_DeviceInfo]:
    """
    Devices usable as a single capture interface: they expose both input and output
    channels, so the same device drives playback and recording.
    """
    return [
        device
        for device in devices
        if device.max_input_channels > 0 and device.max_output_channels > 0
    ]


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
        # Audio routing is independent of any project: it lives here until a project
        # exists to own it, so the Audio I/O tab and route test work before a project
        # is opened or created. Creating a project seeds its audio from this; opening
        # one replaces this with the project's own saved settings (see
        # ``_current_audio_settings``).
        self._audio_settings: _AudioSettingsModel = _AudioSettingsModel()
        # Project-relative filenames of the input WAVs, carried into the plan when
        # it is generated. Files copied in from outside use the canonical names;
        # files already inside the project folder keep their own name.
        self._train_input_name: str = "input_train.wav"
        self._validation_input_name: str = "input_validation.wav"
        self._devices: list[_DeviceInfo] = []
        # Last known live device sample rates (name -> Hz), refreshed by the poll. Held
        # so non-macOS polls that skip the throttled PortAudio reinit reuse the last
        # reading instead of falling back and flickering.
        self._live_device_rates: dict[str, float] = {}
        self._last_rate_reinit: float = 0.0
        self._worker: _Optional[_SessionWorker] = None
        # Workers are retained here until their ``finished`` signal fires so the
        # underlying QThread is never garbage-collected (and destroyed) while its
        # ``run`` is still unwinding on the worker thread, which aborts the process.
        self._retired_workers: set[_SessionWorker] = set()
        self._cancel_token: _Optional[_CancelToken] = None
        self._al_process: _Optional[_QProcess] = None
        self._al_kill_timer: _Optional[_QTimer] = None
        self._al_cancel_requested = False

        self._build_ui()
        self._refresh_devices()
        self._refresh_all()

        # The OS can change a device's sample rate while we run; PortAudio would not
        # notice, so poll the live rates and keep the mismatch warning (and the capture
        # lock-out) current.
        self._rate_poll_timer = _QTimer(self)
        self._rate_poll_timer.setInterval(1500)
        self._rate_poll_timer.timeout.connect(self._refresh_sample_rate_warning)
        self._rate_poll_timer.start()

    # -- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        tabs = _QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_project_tab(), "Project")
        tabs.addTab(self._build_knobs_tab(), "Knobs")
        tabs.addTab(self._build_plan_tab(), "Plan")
        tabs.addTab(self._build_audio_tab(), "Audio I/O")
        tabs.addTab(self._build_capture_tab(), "Capture")
        tabs.addTab(self._build_al_tab(), "Active Learning")
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

        self.knob_table = _QTableWidget(0, 6)
        self.knob_table.setHorizontalHeaderLabels(
            ["Name", "Min", "Max", "Step", "Avoid zero", "Gain/Drive"]
        )
        self.knob_table.setToolTip(
            'Tick "Avoid zero" for gain/drive knobs so no capture sets them to zero '
            "(the nearest non-zero grid value is used instead). Tick \"Gain/Drive\" on "
            "the one gain/drive knob (optional) so the initial corner captures sweep it "
            "separately from the EQ knobs."
        )
        self.knob_table.itemChanged.connect(self._on_knob_item_changed)
        layout.addWidget(self.knob_table)

        row_buttons = _QHBoxLayout()
        add_button = _QPushButton("Add knob")
        add_button.clicked.connect(self._on_add_knob_row)
        remove_button = _QPushButton("Remove selected")
        remove_button.clicked.connect(self._on_remove_knob_row)
        row_buttons.addWidget(add_button)
        row_buttons.addWidget(remove_button)
        layout.addLayout(row_buttons)

        self._on_add_knob_row()
        return widget

    def _build_plan_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        form = _QFormLayout()
        self.n_train_spin = _QSpinBox()
        self.n_train_spin.setRange(1, 100_000)
        self.n_train_spin.setValue(10)
        self.n_validation_spin = _QSpinBox()
        self.n_validation_spin.setRange(0, 100_000)
        self.n_validation_spin.setValue(5)
        self.seed_spin = _QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        form.addRow("Initial LHS random training captures", self.n_train_spin)
        form.addRow("Validation captures", self.n_validation_spin)
        form.addRow("Seed", self.seed_spin)
        layout.addLayout(form)

        self.include_corners_check = _QCheckBox("Include initial corners")
        self.include_corners_check.setToolTip(
            "Add a small set of knob-range corner captures (all-min, all-max, and "
            "alternating extremes) on top of the LHS points, so the first active-learning "
            "round has a boundary picture of the amp. Deduplicated against the LHS points."
        )
        layout.addWidget(self.include_corners_check)

        self.generate_plan_button = _QPushButton("Generate plan")
        self.generate_plan_button.clicked.connect(self._on_generate_plan)
        layout.addWidget(self.generate_plan_button)

        self.add_corners_button = _QPushButton("Add corner captures to current plan")
        self.add_corners_button.setToolTip(
            "Append the corner captures to the existing plan without regenerating the "
            "LHS points or discarding captured progress. Corners that duplicate a setting "
            "already in the plan are skipped."
        )
        self.add_corners_button.clicked.connect(self._on_add_corner_captures)
        layout.addWidget(self.add_corners_button)

        self.plan_table = self._make_entry_table()
        layout.addWidget(self.plan_table)
        return widget

    def _build_audio_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        form = _QFormLayout()
        self.device_combo = _QComboBox()
        self.device_combo.setToolTip(
            "One audio interface used for both playback and recording. Capture and the "
            "loopback both run on this device's channels."
        )
        self.output_channel_spin = _QSpinBox()
        self.output_channel_spin.setRange(1, 1)
        self.input_channel_spin = _QSpinBox()
        self.input_channel_spin.setRange(1, 1)
        self.loopback_check = _QCheckBox(
            "Use a second I/O pair as a clean delay-detection loopback"
        )
        self.loopback_check.setToolTip(
            "Play the timing blips on a second output channel patched straight back "
            "into a second input channel. The delay is measured from that clean "
            "loopback instead of the amp return, so it stays consistent as gain and "
            "distortion increase (the buffer size must not change between captures)."
        )
        self.loopback_output_channel_spin = _QSpinBox()
        self.loopback_output_channel_spin.setRange(1, 1)
        self.loopback_input_channel_spin = _QSpinBox()
        self.loopback_input_channel_spin.setRange(1, 1)
        self.buffer_size_combo = _QComboBox()
        for frames in _BUFFER_SIZE_CHOICES:
            label = "Auto" if frames == 0 else f"{frames} frames"
            self.buffer_size_combo.addItem(label, frames)
        form.addRow("Audio device", self.device_combo)
        form.addRow("Output channel", self.output_channel_spin)
        form.addRow("Input channel", self.input_channel_spin)
        form.addRow(self.loopback_check)
        form.addRow("Loopback output channel", self.loopback_output_channel_spin)
        form.addRow("Loopback input channel", self.loopback_input_channel_spin)
        form.addRow("Buffer size", self.buffer_size_combo)
        layout.addLayout(form)

        self.sample_rate_warning_label = _QLabel("")
        self.sample_rate_warning_label.setWordWrap(True)
        self.sample_rate_warning_label.setStyleSheet("color: #b36b00;")
        self.sample_rate_warning_label.setVisible(False)
        layout.addWidget(self.sample_rate_warning_label)

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

        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.output_channel_spin.valueChanged.connect(self._save_audio_settings)
        self.input_channel_spin.valueChanged.connect(self._save_audio_settings)
        self.loopback_check.toggled.connect(self._on_loopback_toggled)
        self.loopback_output_channel_spin.valueChanged.connect(
            self._save_audio_settings
        )
        self.loopback_input_channel_spin.valueChanged.connect(
            self._save_audio_settings
        )
        self.buffer_size_combo.currentIndexChanged.connect(self._save_audio_settings)
        self._update_loopback_enabled_state()
        return widget

    def _build_capture_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        self.next_entry_label = _QLabel("No project open.")
        layout.addWidget(self.next_entry_label)

        self.capture_sample_rate_warning_label = _QLabel("")
        self.capture_sample_rate_warning_label.setWordWrap(True)
        self.capture_sample_rate_warning_label.setStyleSheet("color: #b36b00;")
        self.capture_sample_rate_warning_label.setVisible(False)
        layout.addWidget(self.capture_sample_rate_warning_label)

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

    def _build_al_tab(self) -> _QWidget:
        widget = _QWidget()
        layout = _QVBoxLayout(widget)

        self.al_next_round_label = _QLabel("No project open.")
        self.al_rounds_completed_label = _QLabel("")
        self.al_pending_proposals_label = _QLabel("")
        self.al_unimported_rounds_label = _QLabel("")
        for label in (
            self.al_next_round_label,
            self.al_rounds_completed_label,
            self.al_pending_proposals_label,
            self.al_unimported_rounds_label,
        ):
            layout.addWidget(label)

        form = _QFormLayout()
        self.al_max_per_round_spin = _QSpinBox()
        self.al_max_per_round_spin.setRange(1, 1000)
        self.al_max_per_round_spin.setValue(_al_runner.AL_MAX_PER_ROUND_DEFAULT)
        self.al_ensemble_size_spin = _QSpinBox()
        self.al_ensemble_size_spin.setRange(1, 64)
        self.al_ensemble_size_spin.setValue(_al_runner.AL_ENSEMBLE_SIZE_DEFAULT)
        self.al_num_restarts_spin = _QSpinBox()
        self.al_num_restarts_spin.setRange(1, 1000)
        self.al_num_restarts_spin.setValue(_al_runner.AL_NUM_RESTARTS_DEFAULT)
        self.al_max_workers_spin = _QSpinBox()
        self.al_max_workers_spin.setRange(0, 64)
        self.al_max_workers_spin.setValue(0)
        self.al_max_workers_spin.setSpecialValueText("Auto")
        form.addRow("Proposals per round", self.al_max_per_round_spin)
        form.addRow("Ensemble size", self.al_ensemble_size_spin)
        form.addRow("Restarts", self.al_num_restarts_spin)
        form.addRow("Max workers", self.al_max_workers_spin)
        layout.addLayout(form)

        buttons = _QHBoxLayout()
        self.al_start_button = _QPushButton("Start round")
        self.al_start_button.clicked.connect(self._on_al_start_round)
        self.al_cancel_button = _QPushButton("Cancel round")
        self.al_cancel_button.setEnabled(False)
        self.al_cancel_button.clicked.connect(self._on_al_cancel_round)
        self.al_import_button = _QPushButton("Import proposals")
        self.al_import_button.clicked.connect(self._on_al_import_proposals)
        self.al_export_button = _QPushButton("Export remote runner files")
        self.al_export_button.clicked.connect(self._on_al_export_runner_files)
        buttons.addWidget(self.al_start_button)
        buttons.addWidget(self.al_cancel_button)
        buttons.addWidget(self.al_import_button)
        buttons.addWidget(self.al_export_button)
        layout.addLayout(buttons)

        if getattr(_sys, "frozen", False):
            self.al_start_button.setEnabled(False)
            self.al_start_button.setToolTip(
                "Active learning needs the Python environment; the packaged app "
                "does not bundle torch."
            )

        self.al_log = _QPlainTextEdit()
        self.al_log.setReadOnly(True)
        layout.addWidget(self.al_log)

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

        # A project (and its audio-routing settings) exists independently of the
        # plan, so it -- and the session that lets the Audio I/O tab run a route
        # test -- are created here rather than deferred to "Generate plan". Carry
        # over whatever audio settings were already configured before this project
        # existed, rather than reverting to defaults.
        self.project = _CaptureProject(
            name=project_dir.name,
            knobs=[],
            train_input=train_name,
            validation_input=validation_name,
            audio=self._audio_settings.model_copy(deep=True),
        )
        self.project_dir = project_dir
        self.session = _CaptureSession(self.project, self.project_dir)
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
        self._refresh_devices()
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
        notes = self._recover_and_reconcile()
        if notes:
            self.project_log.appendPlainText("\n".join(notes))
            _QMessageBox.information(self, "Project reconciled", "\n".join(notes))
        else:
            self.project_log.appendPlainText(f"Opened project: {project_dir}")
        self._maybe_prompt_import_al_proposals()
        self._refresh_devices()
        self._refresh_all()

    def _maybe_prompt_import_al_proposals(self) -> None:
        if self.project is None or self.project_dir is None:
            return
        unimported = _al_runner.unimported_rounds(self.project, self.project_dir)
        if not unimported:
            return
        confirm = _QMessageBox.question(
            self,
            "Import active-learning proposals?",
            f"{len(unimported)} active-learning round(s) have proposals not yet "
            "imported as pending captures: "
            f"{', '.join(str(idx) for idx in unimported)}.\n\nImport them now?",
            _QMessageBox.StandardButton.Yes | _QMessageBox.StandardButton.No,
            _QMessageBox.StandardButton.Yes,
        )
        if confirm != _QMessageBox.StandardButton.Yes:
            return
        message = self._import_al_proposals()
        if message:
            self.project_log.appendPlainText(message)

    def _maybe_recover_from_disk(self) -> list[str]:
        """
        Offer to restore captures whose WAV files already exist on disk and are recorded
        in data.json but that the plan lists as pending -- the situation you land in
        after regenerating the plan with the same seed. Restoring them avoids
        recapturing settings that are already on disk.
        """
        if self.project is None or self.project_dir is None or self.session is None:
            return []
        recoverable = _find_recoverable_entries(self.project, self.project_dir)
        if not recoverable:
            return []
        confirm = _QMessageBox.question(
            self,
            "Restore existing captures?",
            f"{len(recoverable)} capture file(s) matching this plan already exist on "
            "disk and are recorded in data.json, but are not marked as captured. This "
            "happens when the plan is regenerated with the same seed.\n\n"
            "Mark them as captured (reusing the recorded delay; QA is reconstructed "
            "from the files) so you don't have to recapture them?",
            _QMessageBox.StandardButton.Yes | _QMessageBox.StandardButton.No,
            _QMessageBox.StandardButton.Yes,
        )
        if confirm != _QMessageBox.StandardButton.Yes:
            return ["Left existing capture files unmarked; they can be recaptured."]
        return self.session.recover_captured_from_disk(recoverable)

    def _recover_and_reconcile(self) -> list[str]:
        """
        Offer recovery before reconciling: recovered entries become captured, so
        reconcile won't then warn that capturing would overwrite their files. Shared by
        opening a project and regenerating its plan, so both land in the same state
        without requiring a close/reopen round-trip.
        """
        if self.project is None or self.project_dir is None:
            return []
        recover_notes = self._maybe_recover_from_disk()
        notes = recover_notes + _reconcile_with_disk(self.project, self.project_dir)
        _save_project(self.project, self.project_dir)
        return notes

    def _load_knobs_into_ui(self) -> None:
        if self.project is None:
            return
        self.knob_table.setRowCount(0)
        for knob in self.project.knobs:
            self._add_knob_row(
                knob.name, knob.min, knob.max, knob.step, knob.avoid_zero, knob.is_gain
            )
        self.n_train_spin.setValue(
            self.project.n_train_lhs
            or len(self.project.entries_for_split("train"))
            or 1
        )
        self.n_validation_spin.setValue(len(self.project.entries_for_split("validation")))
        self.seed_spin.setValue(self.project.seed)
        self.include_corners_check.setChecked(self.project.include_initial_corners)

    # -- knobs / plan --------------------------------------------------

    def _on_add_knob_row(self) -> None:
        self._add_knob_row("", 0.0, 10.0, 0.5, False, False)

    def _add_knob_row(
        self,
        name: _Any,
        minimum: _Any,
        maximum: _Any,
        step: _Any,
        avoid_zero: _Any = False,
        is_gain: _Any = False,
    ) -> None:
        # Filling checkbox cells fires itemChanged; block it so building a row never trips
        # the single-gain enforcement mid-construction.
        self.knob_table.blockSignals(True)
        try:
            row = self.knob_table.rowCount()
            self.knob_table.insertRow(row)
            for col, value in enumerate((name, minimum, maximum, step)):
                self.knob_table.setItem(row, col, _QTableWidgetItem(str(value)))
            self.knob_table.setItem(row, 4, self._make_check_item(avoid_zero))
            self.knob_table.setItem(row, 5, self._make_check_item(is_gain))
        finally:
            self.knob_table.blockSignals(False)

    @staticmethod
    def _make_check_item(checked: _Any) -> _QTableWidgetItem:
        item = _QTableWidgetItem()
        item.setFlags(_Qt.ItemFlag.ItemIsUserCheckable | _Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(
            _Qt.CheckState.Checked if checked else _Qt.CheckState.Unchecked
        )
        return item

    def _on_knob_item_changed(self, item: _QTableWidgetItem) -> None:
        # At most one knob is Gain/Drive: ticking one clears the rest.
        if item.column() != 5 or item.checkState() != _Qt.CheckState.Checked:
            return
        self.knob_table.blockSignals(True)
        try:
            for row in range(self.knob_table.rowCount()):
                if row == item.row():
                    continue
                other = self.knob_table.item(row, 5)
                if other is not None and other.checkState() == _Qt.CheckState.Checked:
                    other.setCheckState(_Qt.CheckState.Unchecked)
        finally:
            self.knob_table.blockSignals(False)

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
            for col in (4, 5):
                item = self.knob_table.item(row, col)
                values.append(
                    item is not None
                    and item.checkState() == _Qt.CheckState.Checked
                )
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
                "Regenerating the plan discards all existing capture progress. If a "
                "previously recorded capture matches an entry in the new plan (same "
                "WAV file on disk, recorded in data.json), it will automatically be "
                "re-imported.\n\nContinue?",
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

        project.include_initial_corners = self.include_corners_check.isChecked()
        corner_note = ""
        corner_skipped = 0
        if self.include_corners_check.isChecked():
            added, corner_skipped = _add_corner_captures(project)
            corner_note = self._corner_summary(len(added), corner_skipped)

        _save_project(project, self.project_dir)
        self.project = project
        self.session = _CaptureSession(self.project, self.project_dir)
        self.project_label.setText(
            f"Project: {project.name or self.project_dir.name} ({self.project_dir})"
        )
        self.project_log.appendPlainText(
            f"Generated plan: {len(project.entries)} entries "
            f"({self.n_train_spin.value()} train / "
            f"{self.n_validation_spin.value()} validation"
            + (f" + {corner_note}" if corner_note else "")
            + ")."
        )
        if corner_skipped:
            _QMessageBox.information(
                self,
                "Some corners already covered",
                f"{corner_skipped} corner setting(s) were already among the LHS points, "
                "so they were not added again (you are not missing captures).",
            )
        # Re-import captures already on disk that match the freshly regenerated plan
        # (e.g. a plan regenerated with the same seed) immediately, rather than making
        # the user close and reopen the project to trigger recovery.
        notes = self._recover_and_reconcile()
        if notes:
            self.project_log.appendPlainText("\n".join(notes))
            _QMessageBox.information(self, "Project reconciled", "\n".join(notes))
        self._load_audio_settings_into_ui()
        self._refresh_all()

    @staticmethod
    def _corner_summary(added: int, skipped: int) -> str:
        summary = f"{added} corner capture(s)"
        if skipped:
            summary += (
                f" ({skipped} corner setting(s) already in the plan were skipped)"
            )
        return summary

    def _on_add_corner_captures(self) -> None:
        if self.project is None or self.project_dir is None:
            _QMessageBox.warning(
                self, "No project", "Generate or open a plan before adding corners."
            )
            return
        # The plan may predate the current Gain/Drive marking in the table; re-sync the
        # project's knobs from the table so corners honor the latest gain choice without
        # regenerating (and discarding) the LHS points. The knob *set* (names, in order)
        # must still match the plan -- entry params are keyed by knob name -- so a genuinely
        # different knob set is rejected rather than silently mismatched.
        try:
            specs = knob_rows_to_specs(self._knob_table_rows())
            _validate_knobs(specs)
        except ValueError as exc:
            _QMessageBox.critical(self, "Invalid knob settings", str(exc))
            return
        if [spec.name for spec in specs] != [knob.name for knob in self.project.knobs]:
            _QMessageBox.warning(
                self,
                "Knobs changed",
                "The knobs in the table differ from the plan's knobs. Regenerate the "
                "plan first (the corner captures must match the planned knob set).",
            )
            return
        self.project.knobs = [_KnobModel.from_knob_spec(spec) for spec in specs]

        added, skipped = _add_corner_captures(self.project)
        _save_project(self.project, self.project_dir)
        summary = self._corner_summary(len(added), skipped)
        self.project_log.appendPlainText(f"Added {summary}.")
        if not added and skipped:
            _QMessageBox.information(
                self,
                "No new corners added",
                "Every corner setting is already in the plan, so nothing was added.",
            )
        else:
            _QMessageBox.information(self, "Corner captures added", f"Added {summary}.")
        self._refresh_all()

    # -- audio -----------------------------------------------------------

    def _refresh_devices(self) -> None:
        try:
            self._devices = _list_devices(refresh=True)
        except Exception as exc:
            self._devices = []
            self.project_log.appendPlainText(f"Could not list audio devices: {exc}")
        self._populate_device_combo(
            self.device_combo, duplex_devices(self._devices)
        )
        self._load_audio_settings_into_ui()
        self._refresh_sample_rate_warning()

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

    def _current_audio_settings(self) -> _AudioSettingsModel:
        """
        The audio settings backing the Audio I/O tab: the open project's, or the
        standalone settings used before any project exists.
        """
        return self.project.audio if self.project is not None else self._audio_settings

    def _load_audio_settings_into_ui(self) -> None:
        audio = self._current_audio_settings()
        # No device has actually been chosen yet (fresh app, no project): the combo
        # box still auto-selects whatever device sorts first, so its channel count is
        # arbitrary and shouldn't be allowed to clamp away the stored/default channel
        # numbers before the user has picked a real device.
        had_explicit_device = audio.output_device is not None
        self._select_combo_by_name(
            self.device_combo, audio.output_device or audio.input_device
        )
        self._update_channel_ranges()
        if not had_explicit_device:
            self.output_channel_spin.setRange(
                1, max(self.output_channel_spin.maximum(), audio.output_channel)
            )
            self.input_channel_spin.setRange(
                1, max(self.input_channel_spin.maximum(), audio.input_channel)
            )
            self.loopback_output_channel_spin.setRange(
                1,
                max(
                    self.loopback_output_channel_spin.maximum(),
                    audio.loopback_output_channel or 1,
                ),
            )
            self.loopback_input_channel_spin.setRange(
                1,
                max(
                    self.loopback_input_channel_spin.maximum(),
                    audio.loopback_input_channel or 1,
                ),
            )
        self.output_channel_spin.blockSignals(True)
        self.output_channel_spin.setValue(audio.output_channel)
        self.output_channel_spin.blockSignals(False)
        self.input_channel_spin.blockSignals(True)
        self.input_channel_spin.setValue(audio.input_channel)
        self.input_channel_spin.blockSignals(False)
        self.loopback_check.blockSignals(True)
        self.loopback_check.setChecked(audio.loopback_enabled)
        self.loopback_check.blockSignals(False)
        self.loopback_output_channel_spin.blockSignals(True)
        self.loopback_output_channel_spin.setValue(
            audio.loopback_output_channel or 1
        )
        self.loopback_output_channel_spin.blockSignals(False)
        self.loopback_input_channel_spin.blockSignals(True)
        self.loopback_input_channel_spin.setValue(audio.loopback_input_channel or 1)
        self.loopback_input_channel_spin.blockSignals(False)
        self._update_loopback_enabled_state()
        self.buffer_size_combo.blockSignals(True)
        buffer_index = self.buffer_size_combo.findData(audio.blocksize)
        self.buffer_size_combo.setCurrentIndex(max(0, buffer_index))
        self.buffer_size_combo.blockSignals(False)

    def _on_device_changed(self) -> None:
        self._update_channel_ranges()
        self._save_audio_settings()
        self._refresh_sample_rate_warning()

    def _device_rate(
        self, device: _Optional[_DeviceInfo], live: dict[str, float]
    ) -> _Optional[int]:
        """Current hardware rate for ``device``: live if known, else its PortAudio default."""
        if device is None:
            return None
        rate = live.get(device.name)
        if rate is None:
            rate = device.default_samplerate
        return int(round(rate))

    def _update_live_device_rates(self) -> None:
        now = _time.monotonic()
        # Off macOS a refresh reinitialises PortAudio (disruptive and noisy), so
        # throttle it and never run it while a capture or route test holds a stream.
        # macOS ignores the flag: its CoreAudio read is cheap and stream-safe.
        allow_reinit = self._worker is None and (now - self._last_rate_reinit) >= 2.5
        rates = _current_device_sample_rates(allow_reinit=allow_reinit)
        if allow_reinit:
            self._last_rate_reinit = now
        if rates:
            self._live_device_rates = rates

    def _current_sample_rate_warnings(self) -> list[str]:
        if self.project is None or self.project_dir is None:
            return []
        train_rate = read_wav_rate(self.project_dir / self.project.train_input)
        validation_rate = read_wav_rate(
            self.project_dir / self.project.validation_input
        )
        self._update_live_device_rates()
        live = self._live_device_rates
        device_rate = self._device_rate(self.device_combo.currentData(), live)
        return sample_rate_warnings(
            train_rate,
            validation_rate,
            device_rate,
            device_rate,
        )

    def _refresh_sample_rate_warning(self) -> None:
        warnings = self._current_sample_rate_warnings()
        text = "\n".join(f"⚠ {message}" for message in warnings)
        for label in (
            self.sample_rate_warning_label,
            self.capture_sample_rate_warning_label,
        ):
            label.setText(text)
            label.setVisible(bool(warnings))
        # Block captures while any rate disagrees, but never fight the worker's own
        # enable/disable of these buttons while a capture or route test is running.
        if self._worker is None:
            blocked = bool(warnings)
            self.capture_next_button.setEnabled(not blocked)
            self.capture_selected_button.setEnabled(not blocked)

    def _update_channel_ranges(self) -> None:
        device = self.device_combo.currentData()
        max_output = max(1, device.max_output_channels if device else 1)
        max_input = max(1, device.max_input_channels if device else 1)
        self.output_channel_spin.setRange(1, max_output)
        self.input_channel_spin.setRange(1, max_input)
        self.loopback_output_channel_spin.setRange(1, max_output)
        self.loopback_input_channel_spin.setRange(1, max_input)

    def _save_audio_settings(self) -> None:
        # One duplex device drives both playback and recording, so the same device
        # (and its host API) is stored for the output and input directions. Written
        # into the open project's settings, or the standalone settings when there is
        # no project yet -- see ``_current_audio_settings``.
        audio = self._current_audio_settings()
        device = self.device_combo.currentData()
        audio.output_device = device.name if device else None
        audio.input_device = device.name if device else None
        audio.host_api = device.host_api if device else None
        audio.output_channel = self.output_channel_spin.value()
        audio.input_channel = self.input_channel_spin.value()
        if self.loopback_check.isChecked():
            audio.loopback_output_channel = self.loopback_output_channel_spin.value()
            audio.loopback_input_channel = self.loopback_input_channel_spin.value()
        else:
            audio.loopback_output_channel = None
            audio.loopback_input_channel = None
        audio.blocksize = int(self.buffer_size_combo.currentData())
        if self.project is not None and self.project_dir is not None:
            _save_project(self.project, self.project_dir)

    def _on_loopback_toggled(self) -> None:
        self._update_loopback_enabled_state()
        self._save_audio_settings()

    def _update_loopback_enabled_state(self) -> None:
        enabled = self.loopback_check.isChecked()
        self.loopback_output_channel_spin.setEnabled(enabled)
        self.loopback_input_channel_spin.setEnabled(enabled)

    def _route_test_session_and_rate(
        self,
    ) -> tuple[_Optional[_CaptureSession], _Optional[int]]:
        """
        The session to route-test against, and an explicit sample rate to pass it.

        With a project open, use its session and let it infer the rate from the
        project (its recorded sample rate, or the input WAVs) as before. With no
        project, a route test still needs to work -- it only exercises the device
        routing, not the input files -- so build a throwaway session around the
        standalone audio settings and pin the rate to the device's current one
        (there is no project sample rate or input WAV to infer it from).
        """
        if self.session is not None:
            return self.session, None
        device = self.device_combo.currentData()
        if device is None:
            return None, None
        self._update_live_device_rates()
        rate = self._device_rate(device, self._live_device_rates)
        project = _CaptureProject(knobs=[], audio=self._audio_settings)
        session = _CaptureSession(project, self.project_dir or _Path("."))
        return session, rate

    def _on_route_test(self) -> None:
        self._save_audio_settings()
        session, sample_rate = self._route_test_session_and_rate()
        if session is None:
            _QMessageBox.warning(
                self, "No audio device", "Select an audio device first."
            )
            return
        self.route_test_progress.setValue(0)
        self.route_test_result_label.setText("Running route test...")
        self._run_worker(
            lambda progress, cancel: session.route_test(
                sample_rate=sample_rate, progress=progress, cancel=cancel
            ),
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
            source = "loopback" if result.loopback_used else "amp return"
            text = (
                f"Route OK: delay={result.latency.delay} samples (from {source}), "
                f"peak={result.peak:.3f}"
            )
            if result.loopback_used and result.crosscheck is not None:
                amp_delay = result.crosscheck.delay
                if amp_delay is None:
                    text += "; amp-return blip not detected (loopback only)"
                elif result.loopback_disagreement:
                    text += (
                        f"; ⚠ amp-return delay {amp_delay} disagrees with the loopback"
                    )
                else:
                    text += f"; amp-return delay {amp_delay} agrees"
            self.route_test_result_label.setText(text)
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
        warnings = self._current_sample_rate_warnings()
        if warnings:
            self._refresh_sample_rate_warning()
            body = "\n".join(f"• {message}" for message in warnings)
            _QMessageBox.critical(
                self,
                "Sample rate mismatch",
                f"{body}\n\nFix the sample rate mismatch before capturing.",
            )
            return
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

    # -- active learning -------------------------------------------------

    def _al_process_args(self) -> list[str]:
        """
        Argument list for the ``python -m nam.capture.al_runner`` subprocess (everything
        after ``sys.executable``), read from the tab's settings widgets. Split out from
        the launch method so it can be tested without spawning a process.
        """
        assert self.project_dir is not None
        args = [
            "-m",
            "nam.capture.al_runner",
            "--project-dir",
            str(self.project_dir),
            "--max-per-round",
            str(self.al_max_per_round_spin.value()),
            "--ensemble-size",
            str(self.al_ensemble_size_spin.value()),
            "--num-restarts",
            str(self.al_num_restarts_spin.value()),
        ]
        max_workers = self.al_max_workers_spin.value()
        if max_workers > 0:
            args += ["--max-workers", str(max_workers)]
        return args

    def _on_al_start_round(self) -> None:
        if self.project is None or self.project_dir is None:
            _QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        if self._worker is not None and self._worker.isRunning():
            _QMessageBox.warning(
                self, "Busy", "A capture is currently running; wait for it to finish."
            )
            return
        if self._al_process is not None:
            _QMessageBox.warning(
                self, "Busy", "An active-learning round is already running."
            )
            return

        env = _QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")

        process = _QProcess(self)
        process.setProcessChannelMode(_QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(self.project_dir))
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._on_al_output)
        process.finished.connect(self._on_al_finished)

        self._al_process = process
        self._al_cancel_requested = False
        self.al_start_button.setEnabled(False)
        self.generate_plan_button.setEnabled(False)
        self.add_corners_button.setEnabled(False)
        self.al_cancel_button.setEnabled(True)
        self.al_log.appendPlainText("Starting active-learning round...")

        process.start(_sys.executable, self._al_process_args())

    def _on_al_output(self) -> None:
        if self._al_process is None:
            return
        text = bytes(self._al_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if text:
            self.al_log.appendPlainText(text.rstrip("\n"))

    def _on_al_finished(self, exit_code: int, exit_status: _QProcess.ExitStatus) -> None:
        self._al_process = None
        self.al_start_button.setEnabled(not getattr(_sys, "frozen", False))
        self.generate_plan_button.setEnabled(True)
        self.add_corners_button.setEnabled(True)
        self.al_cancel_button.setEnabled(False)
        if self._al_kill_timer is not None:
            self._al_kill_timer.stop()
            self._al_kill_timer = None

        cancelled = self._al_cancel_requested
        self._al_cancel_requested = False

        success = exit_code == 0 and exit_status == _QProcess.ExitStatus.NormalExit
        if success:
            message = self._import_al_proposals()
            self.al_log.appendPlainText(
                message or "Active-learning round finished; no new proposals to import."
            )
        elif cancelled:
            self.al_log.appendPlainText("Active-learning round cancelled.")
        else:
            self.al_log.appendPlainText(
                f"Active-learning round exited with code {exit_code}."
            )

        self._refresh_all()

        if success:
            self.status_bar.showMessage(message or "Active-learning round finished.")
        elif cancelled:
            self.status_bar.showMessage("Active-learning round cancelled.")
        else:
            _QMessageBox.warning(
                self,
                "Active-learning round failed",
                f"The active-learning process exited with code {exit_code}. See the "
                "log pane on the Active Learning tab for details.",
            )

    def _on_al_cancel_round(self) -> None:
        if self._al_process is None:
            return
        self._al_cancel_requested = True
        self.al_log.appendPlainText("Cancelling active-learning round...")
        self._al_process.terminate()
        timer = _QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._on_al_kill_timeout)
        timer.start(5000)
        self._al_kill_timer = timer

    def _on_al_kill_timeout(self) -> None:
        self._al_kill_timer = None
        if self._al_process is not None:
            self._al_process.kill()

    def _import_al_proposals(self) -> str:
        """
        Import proposals from every unimported active-learning round as pending train
        entries and save the project. Returns a log line describing what was imported,
        or "" if there was nothing to do.
        """
        if self.project is None or self.project_dir is None:
            return ""
        rounds = _al_runner.unimported_rounds(self.project, self.project_dir)
        if not rounds:
            return ""
        added = 0
        for round_idx in rounds:
            proposals = _al_runner.load_round_proposals(self.project_dir, round_idx)
            added += len(_al_runner.import_round_proposals(self.project, proposals))
        _save_project(self.project, self.project_dir)
        return (
            f"Imported {added} proposed capture(s) from round(s) "
            f"{', '.join(str(idx) for idx in rounds)}."
        )

    def _on_al_import_proposals(self) -> None:
        if self.project is None or self.project_dir is None:
            _QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        message = self._import_al_proposals()
        self.al_log.appendPlainText(message or "No new active-learning proposals to import.")
        self._refresh_all()

    def _on_al_export_runner_files(self) -> None:
        if self.project is None or self.project_dir is None:
            _QMessageBox.warning(self, "No project", "Open or create a project first.")
            return
        if not any(entry.split == "train" for entry in self.project.captured_entries()):
            _QMessageBox.warning(
                self,
                "No captured train entries",
                "Capture at least one train entry before exporting active-learning "
                "runner files.",
            )
            return

        num_train_windows: _Optional[int] = None
        try:
            available_bytes = _al_runner.available_accelerator_memory_bytes()
            num_train_windows = _al_runner.count_train_windows(
                self.project, self.project_dir
            )
            batch_size, accumulate_grad_batches, _ = _al_runner.compute_al_batch_plan(
                available_bytes,
                _al_runner.AL_NY,
                num_train_windows,
            )
            drop_last = True
            val_batch_size = _al_runner.compute_al_val_batch_size(
                available_bytes, _al_runner.AL_NY
            )
        except Exception as exc:
            batch_size, accumulate_grad_batches, drop_last, val_batch_size = (
                32,
                16,
                True,
                None,
            )
            num_train_windows = None
            self.al_log.appendPlainText(
                f"Could not probe available memory ({exc}); using batch size 32."
            )

        max_workers = self.al_max_workers_spin.value()
        paths = _al_runner.write_al_configs(
            self.project,
            self.project_dir,
            batch_size=batch_size,
            drop_last=drop_last,
            accumulate_grad_batches=accumulate_grad_batches,
            auto_batch_size=True,
            val_batch_size=val_batch_size,
            num_train_windows=num_train_windows,
            max_per_round=self.al_max_per_round_spin.value(),
            ensemble_size=self.al_ensemble_size_spin.value(),
            num_restarts=self.al_num_restarts_spin.value(),
            max_workers=max_workers if max_workers > 0 else None,
        )
        message = "\n".join(str(path) for path in paths)
        self.al_log.appendPlainText(f"Wrote active-learning runner files:\n{message}")
        self.al_log.appendPlainText(
            "Copy the whole project folder to the training machine and run "
            "./run_active_learning.sh, then copy the active_learning/ folder back "
            "and reopen the project here."
        )
        _QMessageBox.information(self, "Active-learning runner files written", message)

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

        # Keep a reference to the thread until it has genuinely finished. The
        # success/failure/cancelled handlers below run (queued) on the GUI thread
        # while the worker's run() is still returning, so clearing self._worker
        # there would otherwise drop the last reference and delete the QThread on
        # its own thread -- a fatal "Destroyed while thread is still running".
        self._retired_workers.add(worker)

        def release_worker() -> None:
            self._retired_workers.discard(worker)
            worker.deleteLater()

        worker.finished.connect(release_worker)

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
        self._refresh_al_tab()
        self._refresh_sample_rate_warning()

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

    def _refresh_al_tab(self) -> None:
        if self.project is None or self.project_dir is None:
            self.al_next_round_label.setText("No project open.")
            self.al_rounds_completed_label.setText("")
            self.al_pending_proposals_label.setText("")
            self.al_unimported_rounds_label.setText("")
            return
        next_round = _al_runner.next_round_idx(self.project_dir)
        outstanding = _al_runner.outstanding_proposal_y_paths(
            self.project, self.project_dir
        )
        unimported = _al_runner.unimported_rounds(self.project, self.project_dir)
        self.al_next_round_label.setText(f"Next round: {next_round}")
        self.al_rounds_completed_label.setText(f"Rounds completed: {next_round}")
        self.al_pending_proposals_label.setText(
            f"AL-proposed captures pending: {len(outstanding)}"
        )
        self.al_unimported_rounds_label.setText(f"Unimported rounds: {len(unimported)}")


def main() -> None:
    app = _QApplication.instance() or _QApplication(_sys.argv)
    window = MainWindow()
    window.show()
    _sys.exit(app.exec())


if __name__ == "__main__":
    main()
