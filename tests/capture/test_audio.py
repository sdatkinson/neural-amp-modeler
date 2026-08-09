import os as _os
import sys as _sys
from types import ModuleType as _ModuleType

import pytest as _pytest

from nam.capture.audio import _enable_asio_on_windows
from nam.capture.audio import _raise_on_dropout
from nam.capture.audio import asio_com_apartment as _asio_com_apartment
from nam.capture.audio import current_device_sample_rates as _current_device_sample_rates
from nam.capture.audio import AudioDeviceError as _AudioDeviceError
from nam.capture.audio import AudioDropoutError as _AudioDropoutError
from nam.capture.audio import DeviceInfo as _DeviceInfo
from nam.capture.audio import LATENCY_CHOICES as _LATENCY_CHOICES
from nam.capture.audio import reports_current_sample_rate as _reports_current_sample_rate


class _Status:
    """Stands in for PortAudio's ``CallbackFlags``."""

    def __init__(self, **flags):
        self.input_underflow = flags.get("input_underflow", False)
        self.input_overflow = flags.get("input_overflow", False)
        self.output_underflow = flags.get("output_underflow", False)
        self.output_overflow = flags.get("output_overflow", False)
        self.priming_output = flags.get("priming_output", False)


def test_clean_stream_status_is_not_a_dropout():
    _raise_on_dropout(_Status(), latency="low", blocksize=0)


def test_blocking_api_flags_are_not_dropouts():
    """
    ``input_underflow``/``output_overflow`` belong to PortAudio's blocking read/write
    API and never mean lost audio on the callback stream the recorder uses; treating
    them as failures would refuse good captures.
    """
    _raise_on_dropout(
        _Status(input_underflow=True, output_overflow=True, priming_output=True),
        latency="low",
        blocksize=0,
    )


@_pytest.mark.parametrize("flag", ["input_overflow", "output_underflow"])
def test_lost_audio_raises(flag):
    with _pytest.raises(_AudioDropoutError) as excinfo:
        _raise_on_dropout(_Status(**{flag: True}), latency=0.002, blocksize=64)
    message = str(excinfo.value)
    assert "was not saved" in message
    # The message has to name the two settings that fix it, since the whole point of
    # the low-latency option is that the user can back off when it does not hold.
    assert "Stream latency" in message
    assert "0.002" in message
    assert "64" in message


def test_dropout_is_an_audio_device_error():
    # SessionWorker catches AudioDeviceError to surface engine failures in the GUI.
    assert issubclass(_AudioDropoutError, _AudioDeviceError)


def test_asio_is_enabled_on_windows(monkeypatch):
    monkeypatch.setattr(_sys, "platform", "win32")
    monkeypatch.delenv("SD_ENABLE_ASIO", raising=False)
    _enable_asio_on_windows()
    assert _os.environ["SD_ENABLE_ASIO"] == "1"


def test_asio_is_not_enabled_off_windows(monkeypatch):
    """
    ASIO is Windows-only; setting this anywhere else sends ``sounddevice`` looking for
    a DLL variant that does not exist there.
    """
    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.delenv("SD_ENABLE_ASIO", raising=False)
    _enable_asio_on_windows()
    assert "SD_ENABLE_ASIO" not in _os.environ


def test_sample_rate_poll_never_reinitialises_portaudio(monkeypatch):
    """
    The rate poll runs on a timer, and off macOS a PortAudio reinit loads every
    installed ASIO driver.

    The stand-in avoids importing the real ``sounddevice``, which needs a PortAudio
    library CI has no reason to have; anything reaching for PortAudio gets this.
    """
    fake_sounddevice = _ModuleType("sounddevice")

    def _explode(*args, **kwargs):
        raise AssertionError("PortAudio was reinitialised on the rate poll path")

    fake_sounddevice._terminate = _explode
    fake_sounddevice._initialize = _explode
    monkeypatch.setitem(_sys.modules, "sounddevice", fake_sounddevice)
    monkeypatch.setattr(_sys, "platform", "win32")

    # Empty, so callers fall back to the cached DeviceInfo.default_samplerate.
    assert _current_device_sample_rates() == {}


def test_darwin_still_reads_coreaudio(monkeypatch):
    """
    macOS must keep its live CoreAudio read, which is what makes the sample-rate
    warning work there.
    """
    import nam.capture.audio as _audio

    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.setattr(
        _audio, "_coreaudio_sample_rates", lambda: {"Audient iD44": 96000.0}
    )
    assert _current_device_sample_rates() == {"Audient iD44": 96000.0}


def test_darwin_falls_back_to_empty_when_coreaudio_fails(monkeypatch):
    import nam.capture.audio as _audio

    monkeypatch.setattr(_sys, "platform", "darwin")

    def _boom():
        raise OSError("CoreAudio unavailable")

    monkeypatch.setattr(_audio, "_coreaudio_sample_rates", _boom)
    assert _current_device_sample_rates() == {}


class _FakeOle32:
    """Stands in for ``ctypes.windll.ole32`` so both branches run on any platform."""

    # ctypes' default restype is signed, so an HRESULT arrives the way it does here.
    S_OK = 0
    S_FALSE = 1
    RPC_E_CHANGED_MODE = -2147417850

    def __init__(self, result=S_OK):
        self._result = result
        self.calls = []

    def CoInitializeEx(self, reserved, flags):
        self.calls.append(("CoInitializeEx", reserved, flags))
        return self._result

    def CoUninitialize(self):
        self.calls.append(("CoUninitialize",))


class _ExplodingWindll:
    def __getattr__(self, name):
        raise AssertionError(f"COM was touched off Windows (windll.{name})")


def _install_fake_windll(monkeypatch, ole32):
    import ctypes

    class _Windll:
        pass

    windll = _Windll()
    windll.ole32 = ole32
    # raising=False: there is no ctypes.windll at all on macOS or Linux.
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)


@_pytest.mark.parametrize("result", [_FakeOle32.S_OK, _FakeOle32.S_FALSE])
def test_worker_thread_enters_a_single_threaded_apartment_on_windows(
    monkeypatch, result
):
    """
    ASIO drivers are in-process COM servers loaded on whichever thread opens the
    stream, so the worker must be in an apartment or the open fails.
    """
    ole32 = _FakeOle32(result)
    monkeypatch.setattr(_sys, "platform", "win32")
    _install_fake_windll(monkeypatch, ole32)

    with _asio_com_apartment():
        assert ole32.calls == [("CoInitializeEx", None, 0x2)]

    # Whatever we entered, we leave -- a leaked apartment outlives the QThread.
    assert ole32.calls[-1] == ("CoUninitialize",)


def test_apartment_is_left_even_if_the_capture_raises(monkeypatch):
    ole32 = _FakeOle32()
    monkeypatch.setattr(_sys, "platform", "win32")
    _install_fake_windll(monkeypatch, ole32)

    with _pytest.raises(RuntimeError):
        with _asio_com_apartment():
            raise RuntimeError("capture blew up")

    assert ole32.calls[-1] == ("CoUninitialize",)


def test_an_existing_mta_is_not_torn_down(monkeypatch):
    """
    RPC_E_CHANGED_MODE means someone else put this thread in a different apartment and
    still owns it; uninitialising one we never entered unbalances their refcount.
    """
    ole32 = _FakeOle32(_FakeOle32.RPC_E_CHANGED_MODE)
    monkeypatch.setattr(_sys, "platform", "win32")
    _install_fake_windll(monkeypatch, ole32)

    with _asio_com_apartment():
        pass

    assert ("CoUninitialize",) not in ole32.calls


def test_no_com_anywhere_but_windows(monkeypatch):
    """
    macOS has no COM and no ASIO; this must stay a plain pass-through there.
    """
    import ctypes

    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.setattr(ctypes, "windll", _ExplodingWindll(), raising=False)

    ran = False
    with _asio_com_apartment():
        ran = True
    assert ran


class _BrokenOle32(_FakeOle32):
    def CoInitializeEx(self, reserved, flags):
        raise OSError("ole32 is unhappy")


@_pytest.mark.parametrize("broken", ["no ole32", "CoInitializeEx raises"])
def test_com_trouble_does_not_take_down_a_capture(monkeypatch, broken):
    """
    The apartment is a precondition, not the job: whether COM is missing entirely or
    just refuses, the capture should still be attempted.
    """
    import ctypes

    monkeypatch.setattr(_sys, "platform", "win32")
    if broken == "no ole32":
        # A bare object has no ``.ole32`` -- the AttributeError the code guards on.
        monkeypatch.setattr(ctypes, "windll", object(), raising=False)
    else:
        _install_fake_windll(monkeypatch, _BrokenOle32())

    ran = False
    with _asio_com_apartment():
        ran = True
    assert ran


def _rate_device(host_api):
    return _DeviceInfo(
        index=0,
        name="Interface",
        host_api=host_api,
        max_input_channels=20,
        max_output_channels=24,
        default_samplerate=44100.0,
    )


def test_only_asio_fails_to_report_the_rate_the_hardware_is_running_at():
    """
    PortAudio reports the first rate from a fixed search order for ASIO, not the rate
    the hardware is on. Treating that as live raises a permanent false mismatch against
    a 48 kHz input file, which blocks capture. Every other host API means what it says.
    """
    assert not _reports_current_sample_rate(_rate_device("ASIO"))
    assert _reports_current_sample_rate(_rate_device("Core Audio"))


def test_latency_choices_run_from_safest_to_tightest():
    assert _LATENCY_CHOICES[0][1] == "high"
    assert _LATENCY_CHOICES[1][1] == "low"
    seconds = [value for _, value in _LATENCY_CHOICES if isinstance(value, float)]
    assert seconds == sorted(seconds, reverse=True)
