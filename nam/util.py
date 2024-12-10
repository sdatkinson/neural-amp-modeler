# File: util.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Helpful utilities
"""

import warnings as _warnings
from datetime import datetime as _datetime


def timestamp() -> str:
    t = _datetime.now()
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}-{t.second:02d}"


class _FilterWarnings(object):
    """
    Context manager.

    Kinda hacky since it doesn't restore to what it was before, but to what the
    global default is.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __enter__(self):
        _warnings.filterwarnings(*self._args, **self._kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        _warnings.resetwarnings()


def filter_warnings(*args, **kwargs):
    """
    Simple-but-kinda-hacky context manager that allows you to use
    `warnings.filterwarnings()` / `warnings.resetwarnings()` as if it were a
    context manager.
    """
    return _FilterWarnings(*args, **kwargs)
