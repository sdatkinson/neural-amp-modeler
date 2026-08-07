import pytest as _pytest

from nam.capture.audio import _raise_on_dropout
from nam.capture.audio import AudioDeviceError as _AudioDeviceError
from nam.capture.audio import AudioDropoutError as _AudioDropoutError
from nam.capture.audio import LATENCY_CHOICES as _LATENCY_CHOICES


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


def test_latency_choices_run_from_safest_to_tightest():
    assert _LATENCY_CHOICES[0][1] == "high"
    assert _LATENCY_CHOICES[1][1] == "low"
    seconds = [value for _, value in _LATENCY_CHOICES if isinstance(value, float)]
    assert seconds == sorted(seconds, reverse=True)
