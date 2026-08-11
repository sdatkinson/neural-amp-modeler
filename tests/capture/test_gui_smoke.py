import os as _os
import wave as _wave

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
from nam.capture.gui.main import no_duplex_devices_message as _no_duplex_devices_message
from nam.capture.gui.main import sort_entries as _sort_entries
from nam.capture.gui.main import SORT_MODES as _SORT_MODES
from nam.capture.gui.main import _latency_index
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
    qa = _QAModel(peak=-6.0, clipping=False, impulse_detected=True, delay_disagreement=False)
    assert _format_qa_summary(qa) == "peak=-6.0 dBFS"

    bad = _QAModel(peak=0.0, clipping=True, impulse_detected=False, delay_disagreement=True)
    summary = _format_qa_summary(bad)
    assert "CLIPPING" in summary
    assert "no impulse" in summary
    assert "delay disagreement" in summary

    loopback = _QAModel(peak=-6.0, impulse_detected=True, loopback_disagreement=True)
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
        qa=_QAModel(peak=-8.0),
    )
    row = _format_entry_row(entry)
    assert row == (
        "train",
        "0",
        "Gain=3.5, Tone=7",
        "captures/train_000_gain_3.5_tone_7.0.wav",
        "captured",
        "183",
        "peak=-8.0 dBFS",
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


def _device(name, host_api, max_input_channels, max_output_channels):
    return _DeviceInfo(
        index=0,
        name=name,
        host_api=host_api,
        max_input_channels=max_input_channels,
        max_output_channels=max_output_channels,
        default_samplerate=48000.0,
    )


_ASIO_DEVICE = _device("Audient USB Audio ASIO Driver", "ASIO", 20, 24)
# What Windows looks like with an interface connected but no ASIO driver: every backend
# splits it into a separate capture and render device.
_SPLIT_WINDOWS_DEVICES = [
    _device("Input 1/2 (Audient iD44)", "MME", 2, 0),
    _device("Output 1/2 (Audient iD44)", "MME", 0, 2),
    _device("Input 1/2 (Audient iD44)", "Windows WASAPI", 2, 0),
    _device("Output 1/2 (Audient iD44)", "Windows WASAPI", 0, 2),
]


def test_no_message_when_a_duplex_device_exists():
    """One duplex device among the split ones is enough to keep the label hidden."""
    assert _no_duplex_devices_message(_SPLIT_WINDOWS_DEVICES + [_ASIO_DEVICE], "win32") == ""
    assert _no_duplex_devices_message([_device("Mic", "Core Audio", 4, 4)], "darwin") == ""


def test_each_empty_picker_case_gets_its_own_advice():
    """
    The three ways the picker comes up empty are three different faults -- no ASIO
    driver, no hardware at all, and nothing plugged in off Windows -- so each has to
    reach its own branch. What those branches actually say is copy, not behaviour.
    """
    no_asio = _no_duplex_devices_message(_SPLIT_WINDOWS_DEVICES, "win32")
    no_hardware = _no_duplex_devices_message([], "win32")
    off_windows = _no_duplex_devices_message(
        [_device("Mic", "Core Audio", 1, 0)], "darwin"
    )
    assert all([no_asio, no_hardware, off_windows])
    assert len({no_asio, no_hardware, off_windows}) == 3


def test_asio_device_rate_is_unknown_rather_than_wrong(_qapp):
    """
    An ASIO device's reported rate is not the rate it is running at, so it must come
    back as None: a wrong one warns forever and leaves the capture buttons disabled.
    Everything else still falls back to the PortAudio default.
    """
    window = _MainWindow()
    asio = _device("Audient USB Audio ASIO Driver", "ASIO", 20, 24)
    assert window._device_rate(asio, {}) is None
    # A live reading, if one ever existed, is still believed over the default.
    assert window._device_rate(asio, {asio.name: 48000.0}) == 48000
    assert window._device_rate(_device("Audient iD44", "Core Audio", 4, 4), {}) == 48000
    window.close()


def test_asio_rate_mismatch_does_not_block_capture(_qapp):
    """
    The regression this guards: a 48 kHz input file against an ASIO device reporting
    44100 produced a permanent 'device rate differs' warning, and the warning disables
    the capture buttons.
    """
    asio = _device("Audient USB Audio ASIO Driver", "ASIO", 20, 24)
    window = _MainWindow()
    rate = window._device_rate(asio, {})
    window.close()
    assert _MainWindow_module.sample_rate_warnings(48000, 48000, rate, rate) == []


def test_route_test_without_a_project_still_gets_a_usable_rate(_qapp, monkeypatch):
    """
    With no project there is no WAV to infer a rate from, so the route test is handed
    one explicitly. ASIO reports no meaningful *current* rate, but default_samplerate is
    still a rate the driver accepts -- and None makes the session hunt for input WAVs
    that do not exist.
    """
    asio = _device("Audient USB Audio ASIO Driver", "ASIO", 20, 24)
    window = _MainWindow()
    monkeypatch.setattr(_MainWindow_module, "_list_devices", lambda refresh=False: [asio])
    window._refresh_devices()
    window.project = None
    window.session = None

    session, rate = window._route_test_session_and_rate()
    window.close()
    assert session is not None
    assert rate == 48000  # the DeviceInfo's default_samplerate, not None


def test_refreshing_devices_cancels_an_operation_instead_of_killing_it(
    _qapp, monkeypatch
):
    """
    Reinitialising PortAudio under an open stream killed the route test with
    "PortAudio not initialized". The refresh has to cancel first, so the operation
    closes its stream and reports itself cancelled.
    """
    import threading
    import time

    from nam.capture.audio import CaptureCancelled

    window = _MainWindow()
    # No hardware in the test: only the cancellation handshake is under test here.
    monkeypatch.setattr(_MainWindow_module, "_list_devices", lambda refresh=False: [])

    running = threading.Event()
    outcome = []

    def slow_operation(progress, cancel):
        running.set()
        while not cancel():
            time.sleep(0.01)
        raise CaptureCancelled()

    window._run_worker(
        slow_operation,
        on_success=lambda result: outcome.append("succeeded"),
        on_failure=lambda message: outcome.append(f"failed: {message}"),
        on_cancelled=lambda: outcome.append("cancelled"),
    )
    assert running.wait(5.0), "worker never started"

    assert window._stop_worker_before_reinit() is True
    assert not window._worker.isRunning()

    # The cancelled signal is queued; deliver it the way the event loop would.
    _qapp.processEvents()
    assert outcome == ["cancelled"]
    window.close()


def test_refresh_reinitialises_when_nothing_is_running(_qapp, monkeypatch):
    """The safe case must keep doing the full re-enumeration, or the button stops
    doing the one job it exists for -- picking up a rate changed in the OS."""
    window = _MainWindow()
    calls = []
    monkeypatch.setattr(
        _MainWindow_module,
        "_list_devices",
        lambda refresh=False: calls.append(refresh) or [],
    )
    window._refresh_devices()
    assert calls == [True]
    window.close()


def test_session_worker_runs_the_capture_inside_a_com_apartment(_qapp):
    """
    The apartment has to be entered on the worker thread itself, and be open for the
    whole engine call -- the stream is opened partway through it, not before it.
    """
    import contextlib

    from nam.capture.gui import workers as _workers

    events = []

    @contextlib.contextmanager
    def _tracking_apartment():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch = _pytest.MonkeyPatch()
    monkeypatch.setattr(_workers, "_asio_com_apartment", _tracking_apartment)
    try:
        worker = _workers.SessionWorker(
            lambda progress, cancel: events.append("call") or "done",
            _workers.CancelToken(),
        )
        worker.start()
        worker.wait()
    finally:
        monkeypatch.undo()

    assert events == ["enter", "call", "exit"]


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

    # The latency choices mix strings and floats, so the combo has to round-trip both.
    window.latency_combo.setCurrentIndex(_latency_index(0.002))
    assert window.project.audio.latency == 0.002
    assert _load_project(tmp_path).audio.latency == 0.002

    # Generating the plan must not reset the audio settings chosen beforehand.
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(2)
    window.n_validation_spin.setValue(1)
    window._on_generate_plan()
    assert window.project.audio.blocksize == 256
    assert _load_project(tmp_path).audio.blocksize == 256
    assert window.project.audio.latency == 0.002
    window.close()


def test_audio_settings_load_the_stored_latency_into_the_combo(
    _qapp, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        _MainWindow_module._QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: str(tmp_path),
    )
    monkeypatch.setattr(
        _InputWavDialog,
        "exec",
        lambda self: _MainWindow_module._QDialog.DialogCode.Rejected,
    )
    window = _MainWindow()
    window._on_new_project()
    window.project.audio.latency = "high"
    window._load_audio_settings_into_ui()
    assert window.latency_combo.currentData() == "high"
    window.close()


def test_reopening_project_does_not_recompute_train_captures_from_corner_entries(
    _qapp, tmp_path, monkeypatch
):
    # Corners are appended as train-split entries, so recomputing "train captures"
    # from entries_for_split("train") on reopen would inflate it past the originally
    # planned LHS count. The count -- and the checkbox that produced the corners --
    # must instead survive the round-trip unchanged.
    monkeypatch.setattr(_MainWindow_module._QMessageBox, "information", lambda *a, **k: None)
    window = _MainWindow()
    window.project_dir = tmp_path
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(4)
    window.n_validation_spin.setValue(2)
    window.include_corners_check.setChecked(True)
    window._on_generate_plan()
    assert len([e for e in window.project.entries if "/corner_" in e.y_path]) == 8

    monkeypatch.setattr(
        _MainWindow_module._QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: str(tmp_path),
    )
    reloaded = _MainWindow()
    reloaded._on_open_project()

    assert reloaded.n_train_spin.value() == 4
    assert reloaded.include_corners_check.isChecked() is True
    window.close()
    reloaded.close()


def test_regenerate_plan_reimports_matching_captures_immediately(
    _qapp, tmp_path, monkeypatch
):
    # Regenerating a plan with the same seed reproduces the same y_paths. If a WAV
    # for one of them is already on disk and recorded in data.json, it must be
    # re-imported as part of regeneration itself -- not only after a close/reopen.
    monkeypatch.setattr(_MainWindow_module._QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(
        _MainWindow_module._QMessageBox,
        "question",
        lambda *a, **k: _MainWindow_module._QMessageBox.StandardButton.Yes,
    )
    window = _MainWindow()
    window.project_dir = tmp_path
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(4)
    window.n_validation_spin.setValue(2)
    window.seed_spin.setValue(7)
    window.include_corners_check.setChecked(False)
    window._on_generate_plan()

    entry = window.project.entries[0]
    wav_path = tmp_path / entry.y_path
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with _wave.open(str(wav_path), "wb") as fp:
        fp.setnchannels(1)
        fp.setsampwidth(2)
        fp.setframerate(48000)
        fp.writeframes(b"\x00\x00" * 100)
    from nam.capture.project import atomic_write_json as _atomic_write_json

    _atomic_write_json(
        tmp_path / "data.json",
        {
            "type": "parametric",
            "common": {"delay": 0},
            "train": [
                {
                    "x_path": "input.wav",
                    "y_path": entry.y_path,
                    "delay": 12,
                    "params": {},
                }
            ],
            "validation": [],
        },
    )

    window._on_generate_plan()

    reimported = next(e for e in window.project.entries if e.y_path == entry.y_path)
    assert reimported.status == "captured"
    assert reimported.delay == 12
    window.close()


def test_regenerate_plan_keeps_the_projects_recorded_timebase(
    _qapp, tmp_path, monkeypatch
):
    # Regenerating rebuilds the project file from the plan, but the capture WAVs survive
    # and are re-imported straight after, so the timebase they were written against must
    # survive too. Dropping it left them on one lead and later captures on the constant.
    monkeypatch.setattr(
        _MainWindow_module._QMessageBox, "information", lambda *a, **k: None
    )
    monkeypatch.setattr(
        _MainWindow_module._QMessageBox, "question",
        lambda *a, **k: _MainWindow_module._QMessageBox.StandardButton.Yes,
    )
    window = _MainWindow()
    window.project_dir = tmp_path
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(4)
    window.n_validation_spin.setValue(2)
    window.seed_spin.setValue(7)
    window.include_corners_check.setChecked(False)
    window._on_generate_plan()

    window.project.alignment_reference = 4.4297

    window._on_generate_plan()

    assert window.project.alignment_reference == 4.4297
    assert _load_project(tmp_path).alignment_reference == 4.4297
    window.close()


def _entry(index, gain, tone, status="pending"):
    return _CaptureEntryModel(
        index=index,
        split="train",
        params={"Gain": gain, "Tone": tone},
        y_path=f"captures/lhs_{index:03d}.wav",
        status=status,
    )


def test_sort_entries_off_keeps_generation_order():
    entries = [_entry(0, 5.0, 1.0), _entry(1, 0.0, 3.0), _entry(2, 5.0, 0.0)]
    assert _sort_entries(entries, ["Gain", "Tone"], "off") == entries


def test_sort_entries_ascending_and_descending_sweep_the_last_knob_fastest():
    entries = [_entry(0, 5.0, 1.0), _entry(1, 0.0, 3.0), _entry(2, 5.0, 0.0)]

    ascending = _sort_entries(entries, ["Gain", "Tone"], "ascending")
    assert [(e.params["Gain"], e.params["Tone"]) for e in ascending] == [
        (0.0, 3.0),
        (5.0, 0.0),
        (5.0, 1.0),
    ]
    descending = _sort_entries(entries, ["Gain", "Tone"], "descending")
    assert descending == list(reversed(ascending))


def test_sort_entries_rejects_an_unknown_mode():
    with _pytest.raises(ValueError):
        _sort_entries([], ["Gain"], "sideways")


def test_sort_button_cycles_the_three_modes_and_reorders_both_tables(
    _qapp, tmp_path, monkeypatch
):
    monkeypatch.setattr(_MainWindow_module._QMessageBox, "information", lambda *a, **k: None)
    window = _MainWindow()
    window.project_dir = tmp_path
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(8)
    window.n_validation_spin.setValue(2)
    window.include_corners_check.setChecked(True)
    window._on_generate_plan()

    names = [knob.name for knob in window.project.knobs]

    def shown_params(table):
        return [
            table.item(row, _MainWindow_module._PLAN_COLUMNS.index("Params")).text()
            for row in range(table.rowCount())
        ]

    def expected(mode):
        return [
            _format_params(entry.params)
            for entry in _sort_entries(window.project.entries, names, mode)
        ]

    # One button per capture list; they show the mode both lists are in, so they must
    # always read the same.
    assert len(window._sort_buttons) == 2
    for mode in list(_SORT_MODES[1:]) + [_SORT_MODES[0]]:
        window._on_cycle_sort_mode()
        assert window._sort_mode == mode
        labels = {button.text() for button in window._sort_buttons}
        assert len(labels) == 1 and mode.capitalize() in labels.pop()
        assert shown_params(window.plan_table) == expected(mode)
        assert shown_params(window.capture_table) == expected(mode)
        # The plan itself is untouched: sorting is display-only.
        assert [e.y_path for e in window.project.entries] == [
            e.y_path for e in _load_project(tmp_path).entries
        ]
    window.close()


def test_sorting_remaps_capture_selected_and_capture_next(_qapp, tmp_path, monkeypatch):
    # The tables are sorted but the project keeps plan order, so a row index must be read
    # through the displayed order -- otherwise "capture selected" records the wrong knobs.
    monkeypatch.setattr(_MainWindow_module._QMessageBox, "information", lambda *a, **k: None)
    window = _MainWindow()
    window.project_dir = tmp_path
    _fill_knob_rows(window, _GAIN_KNOB_ROWS)
    window.n_train_spin.setValue(8)
    window.n_validation_spin.setValue(2)
    window._on_generate_plan()

    captured = []
    monkeypatch.setattr(_MainWindow, "_begin_capture", lambda self, entry: captured.append(entry))

    window._sort_mode = "descending"
    window._refresh_plan_tables()
    window._refresh_next_entry_label()
    window.capture_table.setCurrentCell(3, 0)
    window._on_capture_selected()
    assert captured[-1] is window._displayed_entries[3]
    assert captured[-1] is not window.project.entries[3]

    # "Capture next" follows the list the user is reading, not the plan order.
    window._on_capture_next()
    assert captured[-1] is window._displayed_entries[0]
    assert f"Set knobs to: {_format_params(captured[-1].params)}" in (
        window.next_entry_label.text()
    )
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


def _project_with_captures(tmp_path, _qapp):
    """A window on a project with one real capture in it."""
    from nam.capture.gui.main import MainWindow as _MW
    from nam.capture.params import KnobSpec
    from nam.capture.project import new_project, save_project
    from nam.capture.session import CaptureSession
    from nam.data import np_to_wav
    import numpy as np

    from tests.capture.test_session import _DriftingRecorder, _enable_loopback

    rng = np.random.default_rng(42)
    for name, n in (("input_train.wav", 48_000), ("input_validation.wav", 24_000)):
        np_to_wav((0.1 * rng.standard_normal(n)).astype(np.float32),
                  tmp_path / name, rate=48_000)
    project = new_project([KnobSpec(name="Gain", min=0.0, max=10.0, step=0.5)],
                          n_train=2, n_validation=1, seed=0)
    save_project(project, tmp_path)
    project = _load_project(tmp_path)
    _enable_loopback(project)
    session = CaptureSession(project, tmp_path, recorder=_DriftingRecorder(480))
    entry = project.pending_entries()[0]
    session.capture_entry(entry)

    window = _MW()
    window.project = project
    window.project_dir = tmp_path
    window.session = session
    return window, project, entry


def test_a_bad_timebase_blocks_capturing_before_the_rig_is_driven(_qapp, tmp_path, monkeypatch):
    window, project, _ = _project_with_captures(tmp_path, _qapp)
    project.alignment_reference = 129.0
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        _MainWindow_module._QMessageBox, "critical",
        lambda parent, title, text, *a, **k: shown.append((title, text)),
    )
    started: list[object] = []
    monkeypatch.setattr(window, "_run_worker", lambda *a, **k: started.append(a))

    window._begin_capture(project.pending_entries()[0])
    window.close()

    assert started == []  # refused before any capture was started
    (title, text), = shown
    assert title == "Captures can't be timed together"
    assert "Clear captures" in text


def test_clear_captures_button_resets_the_project(_qapp, tmp_path, monkeypatch):
    window, project, entry = _project_with_captures(tmp_path, _qapp)
    project.alignment_reference = 129.0
    monkeypatch.setattr(
        _MainWindow_module._QMessageBox, "question",
        lambda *a, **k: _MainWindow_module._QMessageBox.StandardButton.Yes,
    )

    window._on_clear_captures()
    window.close()

    assert entry.status == "pending"
    assert not (tmp_path / entry.y_path).exists()
    assert project.alignment_reference is None
    # Persisted, so reopening does not bring the old state back.
    assert _load_project(tmp_path).alignment_reference is None
    assert _load_project(tmp_path).captured_entries() == []


def test_clear_captures_button_reaches_a_pending_entry_whose_file_survives(
    _qapp, tmp_path, monkeypatch
):
    # The deadlock this guards: regenerating the plan (or declining the restore offer)
    # leaves an entry pending with its WAV on disk. The refusal names this button as the
    # fix, so it must not answer "Nothing to clear" and leave the project stuck.
    window, project, entry = _project_with_captures(tmp_path, _qapp)
    project.alignment_reference = 129.0
    entry.status = "pending"
    assert not project.captured_entries()
    assert (tmp_path / entry.y_path).is_file()

    informed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        _MainWindow_module._QMessageBox, "information",
        lambda parent, title, text, *a, **k: informed.append((title, text)),
    )
    monkeypatch.setattr(
        _MainWindow_module._QMessageBox, "question",
        lambda *a, **k: _MainWindow_module._QMessageBox.StandardButton.Yes,
    )

    window._on_clear_captures()
    window.close()

    assert not any(title == "Nothing to clear" for title, _ in informed)
    assert not (tmp_path / entry.y_path).exists()
    assert project.alignment_reference is None


def test_the_running_version_is_shown_at_the_bottom_right(_qapp):
    from nam.capture import CAPTURE_APP_VERSION

    window = _MainWindow()

    assert window.version_label.text() == f"v{CAPTURE_APP_VERSION}"
    # A permanent widget, so the constant rewriting of the message area leaves it alone.
    window.status_bar.showMessage("Capturing 3 of 12...")
    assert window.version_label.text() == f"v{CAPTURE_APP_VERSION}"
    assert window.version_label.isVisible() or not window.isVisible()
    # Right of the message area: permanent widgets are laid out from the right edge, so
    # it must start beyond where a status message is drawn.
    assert window.version_label.x() > 0
    window.close()
