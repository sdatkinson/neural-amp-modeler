# File: util.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Helpful utilities
"""

from datetime import datetime


def timestamp() -> str:
    t = datetime.now()
    return f"{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}-{t.second:02d}"
