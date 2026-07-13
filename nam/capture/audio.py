"""
Audio device enumeration and simultaneous playback/recording.

Everything hardware-facing hides behind the :class:`PlaybackRecorder` protocol so the
capture session (and its tests) can run against a fake recorder without opening a
stream. ``sounddevice`` is imported lazily: enumerating or streaming only happens on
user action, and the GUI must be able to start even if PortAudio is unhappy.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass as _dataclass
from typing import Callable as _Callable
from typing import Optional as _Optional
from typing import Protocol as _Protocol

import numpy as _np


class CaptureCancelled(Exception):
    """
    Raised when the user cancels an in-flight capture.
    """

    pass


class AudioDeviceError(RuntimeError):
    pass


@_dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_devices() -> list[DeviceInfo]:
    import sounddevice as sd

    host_apis = sd.query_hostapis()
    devices = []
    for index, device in enumerate(sd.query_devices()):
        devices.append(
            DeviceInfo(
                index=index,
                name=device["name"],
                host_api=host_apis[device["hostapi"]]["name"],
                max_input_channels=device["max_input_channels"],
                max_output_channels=device["max_output_channels"],
                default_samplerate=device["default_samplerate"],
            )
        )
    return devices


def find_device(
    name: str,
    *,
    kind: str,
    host_api: _Optional[str] = None,
) -> DeviceInfo:
    """
    Resolve a stored device name to today's device table. Names are stored instead of
    indices because indices shift as hardware comes and goes.

    :param kind: "input" or "output" — the direction the device must support.
    """
    if kind not in ("input", "output"):
        raise ValueError(f"kind must be 'input' or 'output'; got {kind!r}")
    candidates = [
        device
        for device in list_devices()
        if device.name == name
        and (host_api is None or device.host_api == host_api)
        and (
            device.max_input_channels > 0
            if kind == "input"
            else device.max_output_channels > 0
        )
    ]
    if len(candidates) == 0:
        available = ", ".join(
            sorted(
                {
                    device.name
                    for device in list_devices()
                    if (
                        device.max_input_channels > 0
                        if kind == "input"
                        else device.max_output_channels > 0
                    )
                }
            )
        )
        raise AudioDeviceError(
            f"No {kind} device named {name!r}"
            + (f" on host API {host_api!r}" if host_api else "")
            + f". Available: {available}"
        )
    return candidates[0]


class PlaybackRecorder(_Protocol):
    def playrec(
        self,
        playback: _np.ndarray,
        sample_rate: int,
        *,
        output_device: _Optional[int] = None,
        input_device: _Optional[int] = None,
        output_channel: int = 1,
        input_channel: int = 1,
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> _np.ndarray:
        """
        Play ``playback`` (mono, float32 in [-1, 1]) on ``output_channel`` (1-based)
        of the output device while recording ``input_channel`` of the input device.
        Returns a mono recording of the same length, time-aligned with the playback.

        ``progress`` is called with a fraction in [0, 1]; ``cancel`` is polled and a
        truthy return aborts the stream by raising :class:`CaptureCancelled`.
        """
        ...


def _device_channels(index: _Optional[int], *, kind: str) -> int:
    """
    Number of channels the device exposes in ``kind`` ("input" or "output"). ``None``
    means PortAudio's default device for that direction.
    """
    import sounddevice as sd

    info = sd.query_devices(kind=kind) if index is None else sd.query_devices(index)
    return int(info[f"max_{kind}_channels"])


class SounddeviceRecorder:
    _POLL_MS = 50

    def playrec(
        self,
        playback: _np.ndarray,
        sample_rate: int,
        *,
        output_device: _Optional[int] = None,
        input_device: _Optional[int] = None,
        output_channel: int = 1,
        input_channel: int = 1,
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> _np.ndarray:
        import sounddevice as sd

        playback = _np.asarray(playback, dtype=_np.float32)
        if playback.ndim != 1:
            raise ValueError(f"Expected mono playback; got shape {playback.shape}")

        # Open the device at its full channel width and place the signal on the exact
        # channel index, the way a DAW does. sounddevice's channel *mapping* would
        # instead open only max(mapping) channels; a 1-channel stream on a
        # multichannel interface is routed by CoreAudio to the device's default pair
        # rather than physical output 1, so "output on channel 1" would land
        # elsewhere. Addressing full-width buffers keeps channel numbers literal.
        output_channels = _device_channels(output_device, kind="output")
        input_channels = _device_channels(input_device, kind="input")
        if not 1 <= output_channel <= output_channels:
            raise AudioDeviceError(
                f"Output channel {output_channel} is out of range for a device with "
                f"{output_channels} output channels."
            )
        if not 1 <= input_channel <= input_channels:
            raise AudioDeviceError(
                f"Input channel {input_channel} is out of range for a device with "
                f"{input_channels} input channels."
            )

        playback_frame = _np.zeros(
            (len(playback), output_channels), dtype=_np.float32
        )
        playback_frame[:, output_channel - 1] = playback

        recording = sd.playrec(
            playback_frame,
            samplerate=sample_rate,
            device=(input_device, output_device),
            channels=input_channels,
            dtype="float32",
            blocking=False,
        )
        duration = len(playback) / sample_rate
        started = _time.monotonic()
        try:
            stream = sd.get_stream()
            while stream.active:
                if cancel is not None and cancel():
                    raise CaptureCancelled()
                if progress is not None:
                    elapsed = _time.monotonic() - started
                    progress(min(elapsed / duration, 1.0))
                sd.sleep(self._POLL_MS)
            sd.wait()
        except BaseException:
            sd.stop()
            raise
        if progress is not None:
            progress(1.0)
        return recording[:, input_channel - 1].copy()
