"""
Background execution for audio operations.

``CaptureSession.route_test`` and ``CaptureSession.capture_entry`` block for as long as
the underlying playback/recording takes, so the GUI must never call them on its own
thread. :class:`SessionWorker` runs one such call on a :class:`QThread`, forwards its
progress callback through a Qt signal, and reports the outcome (success, a known
engine error, or a user cancellation) through separate signals so the GUI thread never
has to catch exceptions itself.
"""

from __future__ import annotations

import threading as _threading
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Optional as _Optional

from PySide6.QtCore import QObject as _QObject
from PySide6.QtCore import QThread as _QThread
from PySide6.QtCore import Signal as _Signal

from ..audio import asio_com_apartment as _asio_com_apartment
from ..audio import AudioDeviceError as _AudioDeviceError
from ..audio import CaptureCancelled as _CaptureCancelled
from ..session import CaptureSessionError as _CaptureSessionError


class CancelToken:
    """
    A cancel flag safe to set from the GUI thread and poll from the worker thread.
    """

    def __init__(self) -> None:
        self._event = _threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class SessionWorker(_QThread):
    """
    Runs ``call(progress, cancel)`` on a background thread and reports the result.

    ``call`` is expected to be a closure over the actual engine call, e.g.
    ``lambda progress, cancel: session.route_test(progress=progress, cancel=cancel)``.
    """

    progress = _Signal(float)
    succeeded = _Signal(object)
    failed = _Signal(str)
    cancelled = _Signal()

    def __init__(
        self,
        call: _Callable[[_Callable[[float], None], _Callable[[], bool]], _Any],
        cancel_token: CancelToken,
        parent: _Optional[_QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._call = call
        self._cancel_token = cancel_token

    def run(self) -> None:
        # This thread opens the PortAudio stream, and on Windows an ASIO driver only
        # loads from a thread in a COM apartment -- see ``asio_com_apartment``. It wraps
        # the whole call because the apartment has to outlive the stream.
        try:
            with _asio_com_apartment():
                result = self._call(
                    self.progress.emit, self._cancel_token.is_cancelled
                )
        except _CaptureCancelled:
            self.cancelled.emit()
            return
        except (_CaptureSessionError, _AudioDeviceError) as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            # Anything unexpected still gets surfaced as a message instead of
            # crashing the application.
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)
