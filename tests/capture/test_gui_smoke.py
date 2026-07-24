import os as _os

import pytest as _pytest

_pytest.importorskip("PySide6")

_os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication as _QApplication

from nam.capture.audio import DeviceInfo as _DeviceInfo
from nam.capture.gui import main as _MainWindow_module
from nam.capture.gui.main import InputWavDialog as _InputWavDialog
from nam.capture.gui.main import MainWindow as _MainWindow
from nam.capture.gui.main import duplex_devices as _duplex_devices
from nam.capture.gui.main import format_device_label as _format_device_label
from nam.capture.gui.main import format_entry_row as _format_entry_row
from nam.capture.gui.main import format_params as _format_params
from nam.capture.gui.main import format_qa_summary as _format_qa_summary
from nam.capture.gui.main import knob_rows_to_specs as _knob_rows_to_specs
from nam.capture.project import CaptureEntryModel as _CaptureEntryModel
from nam.capture.project import load_project as _load_project
from nam.capture.project import QAModel as _QAModel


@_pytest.fixture(scope="module")
def _qapp():
    app = _QApplication.instance() or _QApplication([])
    yield app


def test_format_params_formats_like_a_human_would_dial_it():
    assert _format_params({"Gain": 3.5, "Tone": 7.0}) == "Gain=3.5, Tone=7"


def test_format_qa_summary_flags_problems():
    qa = _QAModel(peak=0.5, clipping=False, impulse_detected=True, delay_disagreement=False)
    assert _format_qa_summary(qa) == "peak=0.500"

    bad = _QAModel(peak=1.0, clipping=True, impulse_detected=False, delay_disagreement=True)
    summary = _format_qa_summary(bad)
    assert "CLIPPING" in summary
    assert "no impulse" in summary
    assert "delay disagreement" in summary

    loopback = _QAModel(peak=0.5, impulse_detected=True, loopback_disagreement=True)
    assert "loopback mismatch" in _format_qa_summary(loopback)


def test_format_qa_summary_handles_missing_qa():
    assert _format_qa_summary(None) == ""


def test_format_entry_row_matches_plan_columns():
    entry = _CaptureEntryModel(
        index=0,
        split="train",
        params={"Gain": 3.5, "Tone": 7.0},
        y_path="captures/train_000_gain_3.5_tone_7.0.wav",
        status="captured",
        delay=183,
        qa=_QAModel(peak=0.4),
    )
    row = _format_entry_row(entry)
    assert row == (
        "train",
        "0",
        "Gain=3.5, Tone=7",
        "captures/train_000_gain_3.5_tone_7.0.wav",
        "captured",
        "183",
        "peak=0.400",
    )


def test_knob_rows_to_specs_builds_valid_specs():
    specs = _knob_rows_to_specs(
        [("Gain", "0", "10", "0.5", True), ("Tone", "0.0", "10.0", "1.0", False)]
    )
    assert [spec.name for spec in specs] == ["Gain", "Tone"]
    assert specs[0].min == 0.0
    assert specs[0].max == 10.0
    assert specs[0].step == 0.5
    assert specs[0].avoid_zero is True
    assert specs[1].avoid_zero is False


def test_knob_rows_to_specs_raises_on_bad_row():
    with _pytest.raises(ValueError):
        _knob_rows_to_specs([("Gain", "10", "0", "0.5", False)])


def test_knob_rows_to_specs_reads_is_gain_and_defaults_it_off():
    specs = _knob_rows_to_specs(
        [
            ("Gain", "0", "10", "0.5", False, True),
            ("Tone", "0", "10", "1.0", False),  # 5-tuple -> is_gain defaults off
        ]
    )
    assert specs[0].is_gain is True
    assert specs[1].is_gain is False


def test_duplex_devices_keeps_only_full_io_devices():
    devices = [
        _DeviceInfo(
            index=0,
            name="Input Only",
            host_api="Core Audio",
            max_input_channels=8,
            max_output_channels=0,
            default_samplerate=48000.0,
        ),
        _DeviceInfo(
            index=1,
            name="Output Only",
            host_api="Core Audio",
            max_input_channels=0,
            max_output_channels=8,
            default_samplerate=48000.0,
        ),
        _DeviceInfo(
            index=2,
            name="Duplex Interface",
            host_api="Core Audio",
            max_input_channels=4,
            max_output_channels=4,
            default_samplerate=48000.0,
        ),
    ]
    assert [d.name for d in _duplex_devices(devices)] == ["Duplex Interface"]


def test_format_device_label_includes_host_api():
    device = _DeviceInfo(
        index=0,
        name="Audient iD44",
        host_api="Core Audio",
        max_input_channels=4,
        max_output_channels=4,
        default_samplerate=48000.0,
    )
    assert _format_device_label(device) == "Audient iD44 (Core Audio)"


def test_main_window_builds_with_no_project(_qapp):
    window = _MainWindow()
    assert window.centralWidget() is not None
    assert window.project is None
    assert window.knob_table.rowCount() == 1
    window.close()


def test_loopback_defaults_to_enabled(_qapp):
    window = _MainWindow()
    assert window.loopback_check.isChecked()
    assert window.loopback_output_channel_spin.value() == 2
    assert window.loopback_input_channel_spin.value() == 2
    window.close()


def test_main_window_add_remove_knob_rows(_qapp):
    window = _MainWindow()
    window._on_add_knob_row()
    assert window.knob_table.rowCount() == 2
    window.knob_table.setCurrentCell(0, 0)
    window._on_remove_knob_row()
    assert window.knob_table.rowCount() == 1
    window.close()


def test_main_window_refresh_all_without_project_does_not_crash(_qapp):
    window = _MainWindow()
    window._refresh_all()
    assert window.plan_table.rowCount() == 0
    assert window.next_entry_label.text() == "No project open."
    window.close()


def _fill_knob_rows(window, rows):
    window.knob_table.setRowCount(0)
    for name, minimum, maximum, step, avoid_zero, is_gain in rows:
        window._add_knob_row(name, minimum, maximum, step, avoid_zero, is_gain)


_GAIN_KNOB_ROWS = [
    ("Gain", 0.0, 10.0, 1.0, False, True),
    ("Tone", 0.0, 10.0, 1.0, False, False),
    ("Bass", 0.0, 10.0, 1.0, False, False),
]


def test_gain_column_is_single_select(_qapp):
    from PySide6.QtCore import Qt as _Qt

    window = _MainWindow()
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    # Ticking a second knob's Gain/Drive box clears the first.
    window.knob_table.item(1, 5).setCheckState(_Qt.CheckState.Checked)
    assert window.knob_table.item(0, 5).checkState() == _Qt.CheckState.Unchecked
    assert window.knob_table.item(1, 5).checkState() == _Qt.CheckState.Checked
    window.close()


def test_generate_plan_includes_corners_when_checked(_qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(_MainWindow_module._QMessageBox, "information", lambda *a, **k: None)
    window = _MainWindow()
    window.project_dir = tmp_path
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(4)
    window.n_validation_spin.setValue(2)
    window.include_corners_check.setChecked(True)
    window._on_generate_plan()

    corner_entries = [e for e in window.project.entries if "/corner_" in e.y_path]
    lhs_entries = [e for e in window.project.entries if "/lhs_" in e.y_path]
    assert len(corner_entries) == 8  # 3 knobs, one gain -> 8 corners
    assert len(lhs_entries) == 4
    window.close()


def test_initial_device_gets_correct_channel_ranges_without_reselecting(
    _qapp, monkeypatch
):
    # The channel spinboxes start locked to (1, 1) until _update_channel_ranges runs
    # for the selected device; it used to only run on a manual device change or once
    # a project existed, so the device the combo box defaults to on startup never got
    # its real channel range applied.
    devices = [
        _DeviceInfo(
            index=0,
            name="Duplex Interface",
            host_api="Core Audio",
            max_input_channels=6,
            max_output_channels=2,
            default_samplerate=48000.0,
        ),
    ]
    monkeypatch.setattr(
        _MainWindow_module, "_list_devices", lambda refresh=False: devices
    )
    window = _MainWindow()
    assert window.output_channel_spin.maximum() == 2
    assert window.input_channel_spin.maximum() == 6
    window.close()


def test_route_test_session_available_without_a_project(_qapp, monkeypatch):
    # A route test only exercises device routing, not any project file, so it must
    # not require a project to be open first.
    devices = [
        _DeviceInfo(
            index=0,
            name="Duplex Interface",
            host_api="Core Audio",
            max_input_channels=4,
            max_output_channels=4,
            default_samplerate=48000.0,
        ),
    ]
    monkeypatch.setattr(
        _MainWindow_module, "_list_devices", lambda refresh=False: devices
    )
    window = _MainWindow()
    assert window.project is None
    assert window.session is None

    session, sample_rate = window._route_test_session_and_rate()
    assert session is not None
    # With no project (and so no captured sample rate or input WAV to read one
    # from), the route test falls back to the device's own current rate.
    assert sample_rate == 48000
    window.close()


def test_new_project_gets_a_session_and_keeps_audio_settings_through_generate_plan(
    _qapp, tmp_path, monkeypatch
):
    # A route test needs a session; new projects used to leave window.session as
    # None until a plan was generated, blocking the route test until then.
    monkeypatch.setattr(
        _MainWindow_module._QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: str(tmp_path),
    )
    monkeypatch.setattr(
        _InputWavDialog, "exec", lambda self: _MainWindow_module._QDialog.DialogCode.Rejected
    )
    window = _MainWindow()
    window._on_new_project()

    assert window.project is not None
    assert window.session is not None

    # Adjusting audio settings before any plan exists must persist to disk, not
    # get silently dropped because there was nowhere to save them.
    index = window.buffer_size_combo.findData(256)
    window.buffer_size_combo.setCurrentIndex(index)
    assert window.project.audio.blocksize == 256
    assert _load_project(tmp_path).audio.blocksize == 256

    # Generating the plan must not reset the audio settings chosen beforehand.
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(2)
    window.n_validation_spin.setValue(1)
    window._on_generate_plan()
    assert window.project.audio.blocksize == 256
    assert _load_project(tmp_path).audio.blocksize == 256
    window.close()


def test_add_corner_captures_button_appends_without_regenerating(
    _qapp, tmp_path, monkeypatch
):
    monkeypatch.setattr(_MainWindow_module._QMessageBox, "information", lambda *a, **k: None)
    window = _MainWindow()
    window.project_dir = tmp_path
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(4)
    window.n_validation_spin.setValue(2)
    window.include_corners_check.setChecked(False)
    window._on_generate_plan()
    base = len(window.project.entries)

    window._on_add_corner_captures()
    corner_entries = [e for e in window.project.entries if "/corner_" in e.y_path]
    assert len(corner_entries) == 8
    assert len(window.project.entries) == base + 8
    window.close()
