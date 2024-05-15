# File: settings.py
# Created Date: Tuesday May 14th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import json
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

__all__ = ["PathKey", "get_last_path", "set_last_path"]

_THIS_DIR = Path(__file__).parent.resolve()
_SETTINGS_JSON_PATH = Path(_THIS_DIR, "settings.json")
_LAST_PATHS_KEY = "last_paths"


class PathKey(Enum):
    INPUT_FILE = "input_file"
    OUTPUT_FILE = "output_file"
    TRAINING_DESTINATION = "training_destination"


def get_last_path(path_key: PathKey) -> Optional[Path]:
    s = _get_settings()
    if _LAST_PATHS_KEY not in s:
        return None
    last_path = s[_LAST_PATHS_KEY].get(path_key.value)
    if last_path is None:
        return None
    assert isinstance(last_path, str)
    return Path(last_path)


def set_last_path(path_key: PathKey, path: Path):
    s = _get_settings()
    if _LAST_PATHS_KEY not in s:
        s[_LAST_PATHS_KEY] = {}
    s[_LAST_PATHS_KEY][path_key.value] = str(path)
    _write_settings(s)


def _get_settings() -> dict:
    """
    Make sure that ./settings.json exists; if it does, then read it. If not, empty dict.
    """

    if not _SETTINGS_JSON_PATH.exists():
        _write_settings({})
    with open(_SETTINGS_JSON_PATH, "r") as fp:
        return json.load(fp)


def _write_settings(obj: dict):
    with open(_SETTINGS_JSON_PATH, "w") as fp:
        json.dump(obj, fp, indent=4)
