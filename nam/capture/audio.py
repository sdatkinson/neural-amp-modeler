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
from typing import Literal
from typing import Optional as _Optional
from typing import Protocol as _Protocol
from typing import Tuple as _Tuple
from typing import Union as _Union

import numpy as _np


# Suggested stream latency: seconds, or one of PortAudio's per-device presets.
_Latency = _Union[float, Literal["low", "high"]]

# Offered latency settings, coarsest first, as ``(label, value)``. sounddevice's own
# default is "high", which is why the capture app measured a ~150 ms round trip before
# this was settable; PortAudio's high-latency suggestion for an interface can be an
# order of magnitude above what the same hardware does in a DAW (an iD44 asks for 0.1 s
# on the input alone). A tighter setting costs dropout headroom, not accuracy -- and a
# dropout is caught, see :class:`AudioDropoutError`.
#
# Note that these interact with the block size rather than adding to it: once the
# latency hint is small, a large ``blocksize`` puts it back, because PortAudio inserts
# its own ring buffer when the requested block does not match the device's. A tight
# latency wants "Auto" or a small block size.
LATENCY_CHOICES: _Tuple[_Tuple[str, _Latency], ...] = (
    ("System default (safest)", "high"),
    ("Low (device default)", "low"),
    ("5 ms", 0.005),
    ("2 ms", 0.002),
)


class CaptureCancelled(Exception):
    """
    Raised when the user cancels an in-flight capture.
    """

    pass


class AudioDeviceError(RuntimeError):
    pass


class AudioDropoutError(AudioDeviceError):
    """
    The stream dropped samples: PortAudio reported an input overflow (recorded audio
    was lost) or an output underflow (a gap in what was played).

    A capture this happens to is silently wrong -- a hole somewhere in the middle that
    no delay measurement or QA level check would notice -- so it is raised rather than
    returned, and the capture is refused. It is the counterweight to a low
    ``latency`` setting: small buffers are what make dropouts possible in the first
    place, and without this check lowering the latency would trade a known, measurable
    delay for silent data corruption.
    """

    pass


# A dB value low enough to stand in for silence without producing -inf, which would
# break JSON round-tripping and numeric formatting.
DBFS_FLOOR = -120.0


def peak_to_dbfs(peak: float) -> float:
    """
    Convert a linear peak amplitude (0-1 full scale) to dBFS, where 0 dBFS is full
    scale and level drops negative from there. Floored at ``DBFS_FLOOR`` so silence
    doesn't produce -inf.
    """
    if peak <= 0:
        return DBFS_FLOOR
    return max(20.0 * float(_np.log10(peak)), DBFS_FLOOR)


@_dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


def list_devices(refresh: bool = False) -> list[DeviceInfo]:
    """
    Enumerate the audio devices PortAudio currently sees.

    PortAudio reads each device's ``default_samplerate`` (and the rest of the table)
    once when it initialises and caches it, so a sample rate changed in the OS while
    the app is running is not reflected until PortAudio is reinitialised. Pass
    ``refresh=True`` to force that reinitialisation and pick up such changes.
    """
    import sounddevice as sd

    if refresh:
        # No streams are open on the refresh path, so terminating and reinitialising
        # PortAudio is safe and is the only way to re-read the current device table.
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass

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


def current_device_sample_rates(allow_reinit: bool = False) -> dict[str, float]:
    """
    Map device name -> its *current* nominal sample rate in Hz, read live.

    PortAudio latches each device's ``default_samplerate`` when it initialises, so it
    cannot see a rate changed in the OS while the app is running.

    - On macOS this reads CoreAudio's nominal sample rate, which always reflects the
      current hardware setting and is cheap enough to poll.
    - On other platforms there is no comparably cheap always-live query, so the only
      way to re-read the current rates is to reinitialise PortAudio. That must never
      happen while a stream is open, so it is done only when ``allow_reinit`` is True;
      the refreshed rates still come straight from PortAudio, so they are never wrong,
      at worst merely not refreshed.

    Returns an empty dict when it cannot produce live values; callers then fall back to
    :attr:`DeviceInfo.default_samplerate`.
    """
    import sys as _sys

    if _sys.platform == "darwin":
        try:
            return _coreaudio_sample_rates()
        except Exception:
            return {}

    if not allow_reinit:
        return {}
    import sounddevice as sd

    try:
        sd._terminate()
        sd._initialize()
        return {device.name: device.default_samplerate for device in list_devices()}
    except Exception:
        return {}


def _coreaudio_sample_rates() -> dict[str, float]:
    import ctypes
    import ctypes.util

    def fourcc(code: str) -> int:
        return (
            (ord(code[0]) << 24)
            | (ord(code[1]) << 16)
            | (ord(code[2]) << 8)
            | ord(code[3])
        )

    class _Addr(ctypes.Structure):
        _fields_ = [
            ("mSelector", ctypes.c_uint32),
            ("mScope", ctypes.c_uint32),
            ("mElement", ctypes.c_uint32),
        ]

    system_object = 1
    scope_global = fourcc("glob")
    element_main = 0
    prop_devices = fourcc("dev#")
    prop_name = fourcc("lnam")
    prop_nominal_rate = fourcc("nsrt")
    utf8 = 0x08000100

    core_audio = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
    core_foundation = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    core_foundation.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    core_foundation.CFStringGetCString.restype = ctypes.c_bool

    addr = _Addr(prop_devices, scope_global, element_main)
    size = ctypes.c_uint32(0)
    core_audio.AudioObjectGetPropertyDataSize(
        system_object, ctypes.byref(addr), 0, None, ctypes.byref(size)
    )
    count = size.value // ctypes.sizeof(ctypes.c_uint32)
    device_ids = (ctypes.c_uint32 * count)()
    core_audio.AudioObjectGetPropertyData(
        system_object, ctypes.byref(addr), 0, None, ctypes.byref(size), device_ids
    )

    rates: dict[str, float] = {}
    for device_id in device_ids:
        name_addr = _Addr(prop_name, scope_global, element_main)
        cfstr = ctypes.c_void_p()
        name_size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
        if (
            core_audio.AudioObjectGetPropertyData(
                device_id,
                ctypes.byref(name_addr),
                0,
                None,
                ctypes.byref(name_size),
                ctypes.byref(cfstr),
            )
            != 0
            or not cfstr.value
        ):
            continue
        buffer = ctypes.create_string_buffer(256)
        ok = core_foundation.CFStringGetCString(cfstr, buffer, 256, utf8)
        core_foundation.CFRelease(cfstr)
        if not ok:
            continue
        name = buffer.value.decode("utf-8", "replace")

        rate_addr = _Addr(prop_nominal_rate, scope_global, element_main)
        rate = ctypes.c_double(0)
        rate_size = ctypes.c_uint32(ctypes.sizeof(ctypes.c_double))
        if (
            core_audio.AudioObjectGetPropertyData(
                device_id,
                ctypes.byref(rate_addr),
                0,
                None,
                ctypes.byref(rate_size),
                ctypes.byref(rate),
            )
            == 0
            and rate.value > 0
        ):
            rates[name] = rate.value
    return rates


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
        loopback_output_channel: _Optional[int] = None,
        loopback_input_channel: _Optional[int] = None,
        loopback_playback: _Optional[_np.ndarray] = None,
        blocksize: int = 0,
        latency: _Latency = "low",
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> _Tuple[_np.ndarray, _Optional[_np.ndarray]]:
        """
        Play ``playback`` (mono, float32 in [-1, 1]) on ``output_channel`` (1-based)
        of the output device while recording ``input_channel`` of the input device.

        Returns ``(recording, loopback_recording)``, both mono and time-aligned with
        the playback. When ``loopback_output_channel``/``loopback_input_channel`` are
        given, ``loopback_playback`` (mono, same length as ``playback``) is played on
        the loopback output channel and that input channel is returned as the second
        element; otherwise ``loopback_recording`` is ``None``. The loopback channels
        must differ from the primary channels and share the same devices.

        ``blocksize`` and ``latency`` set the stream's block size in frames (0 = let
        PortAudio choose) and its suggested latency in seconds, or one of PortAudio's
        per-device presets ("low"/"high"). Together they set the round-trip delay; see
        :data:`LATENCY_CHOICES`.

        ``progress`` is called with a fraction in [0, 1]; ``cancel`` is polled and a
        truthy return aborts the stream by raising :class:`CaptureCancelled`.

        Raises :class:`AudioDropoutError` if the stream over- or underflowed, since the
        recording is then silently missing samples.
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


def _raise_on_dropout(status, *, latency: _Latency, blocksize: int) -> None:
    """
    Turn PortAudio's accumulated callback status flags into an
    :class:`AudioDropoutError`, or return quietly if the stream ran clean.

    Only the two flags that mean lost audio on a callback duplex stream are fatal:
    ``input_overflow`` (recorded samples discarded) and ``output_underflow`` (silence
    inserted into the playback). ``input_underflow``/``output_overflow`` are artefacts
    of PortAudio's blocking read/write API and never indicate a problem here, and
    ``priming_output`` is normal at stream start.
    """
    lost_input = bool(getattr(status, "input_overflow", False))
    lost_output = bool(getattr(status, "output_underflow", False))
    if not (lost_input or lost_output):
        return
    what = []
    if lost_input:
        what.append("recorded samples were dropped")
    if lost_output:
        what.append("gaps were played into the output")
    raise AudioDropoutError(
        f"The audio stream could not keep up: {' and '.join(what)}. The capture is "
        "missing audio and was not saved. Raise 'Stream latency' (currently "
        f"{latency!r}) or the buffer size (currently "
        f"{'Auto' if blocksize == 0 else blocksize}) in Audio settings, close other "
        "audio applications, and capture again."
    )


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
        loopback_output_channel: _Optional[int] = None,
        loopback_input_channel: _Optional[int] = None,
        loopback_playback: _Optional[_np.ndarray] = None,
        blocksize: int = 0,
        latency: _Latency = "low",
        progress: _Optional[_Callable[[float], None]] = None,
        cancel: _Optional[_Callable[[], bool]] = None,
    ) -> _Tuple[_np.ndarray, _Optional[_np.ndarray]]:
        import sounddevice as sd

        playback = _np.asarray(playback, dtype=_np.float32)
        if playback.ndim != 1:
            raise ValueError(f"Expected mono playback; got shape {playback.shape}")

        use_loopback = (
            loopback_output_channel is not None
            and loopback_input_channel is not None
        )
        if use_loopback:
            if loopback_playback is None:
                raise ValueError(
                    "loopback_playback is required when loopback channels are set."
                )
            loopback_playback = _np.asarray(loopback_playback, dtype=_np.float32)
            if loopback_playback.shape != playback.shape:
                raise ValueError(
                    f"loopback_playback shape {loopback_playback.shape} must match the "
                    f"playback shape {playback.shape}."
                )

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
        if use_loopback:
            if not 1 <= loopback_output_channel <= output_channels:
                raise AudioDeviceError(
                    f"Loopback output channel {loopback_output_channel} is out of range "
                    f"for a device with {output_channels} output channels."
                )
            if not 1 <= loopback_input_channel <= input_channels:
                raise AudioDeviceError(
                    f"Loopback input channel {loopback_input_channel} is out of range "
                    f"for a device with {input_channels} input channels."
                )
            if loopback_output_channel == output_channel:
                raise AudioDeviceError(
                    "Loopback output channel must differ from the capture output "
                    f"channel (both {output_channel})."
                )
            if loopback_input_channel == input_channel:
                raise AudioDeviceError(
                    "Loopback input channel must differ from the capture input "
                    f"channel (both {input_channel})."
                )

        playback_frame = _np.zeros(
            (len(playback), output_channels), dtype=_np.float32
        )
        playback_frame[:, output_channel - 1] = playback
        if use_loopback:
            playback_frame[:, loopback_output_channel - 1] = loopback_playback

        recording = sd.playrec(
            playback_frame,
            samplerate=sample_rate,
            device=(input_device, output_device),
            channels=input_channels,
            dtype="float32",
            blocksize=blocksize,
            latency=latency,
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
        _raise_on_dropout(sd.get_status(), latency=latency, blocksize=blocksize)
        if progress is not None:
            progress(1.0)
        main = recording[:, input_channel - 1].copy()
        loopback = (
            recording[:, loopback_input_channel - 1].copy() if use_loopback else None
        )
        return main, loopback
