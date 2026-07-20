import os as _os

import pytest as _pytest

_pytest.importorskip("PySide6")

_os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication as _QApplication

from nam.capture.audio import DeviceInfo as _DeviceInfo
from nam.capture.gui.main import MainWindow as _MainWindow
from nam.capture.gui.main import duplex_devices as _duplex_devices
from nam.capture.gui.main import format_device_label as _format_device_label
from nam.capture.gui.main import format_entry_row as _format_entry_row
from nam.capture.gui.main import format_params as _format_params
from nam.capture.gui.main import format_qa_summary as _format_qa_summary
from nam.capture.gui.main import knob_rows_to_specs as _knob_rows_to_specs
from nam.capture.project import CaptureEntryModel as _CaptureEntryModel
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
