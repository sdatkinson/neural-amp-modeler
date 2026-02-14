# File: settings.py
# Created Date: Tuesday May 14th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import json
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).parent.resolve()
_SETTINGS_JSON_PATH = Path(_THIS_DIR, "settings.json")
_LAST_PATHS_KEY = "last_paths"
_UPDATE_KEY = "update"
_NEWEST_AVAILABLE_VERSION_KEY = "newest_available_version"
_NEVER_SHOW_AGAIN_KEY = "never_show_again"


class PathKey(Enum):
    INPUT_FILE = "input_file"
    OUTPUT_FILE = "output_file"
    TRAINING_DESTINATION = "training_destination"


def get_last_path(
    path_key: PathKey, *, settings_path: Path = _SETTINGS_JSON_PATH
) -> Optional[Path]:
    s = _get_settings(settings_path)
    if _LAST_PATHS_KEY not in s:
        return None
    last_path = s[_LAST_PATHS_KEY].get(path_key.value)
    if last_path is None:
        return None
    assert isinstance(last_path, str)
    return Path(last_path)


def set_last_path(
    path_key: PathKey, path: Path, *, settings_path: Path = _SETTINGS_JSON_PATH
):
    s = _get_settings(settings_path)
    if _LAST_PATHS_KEY not in s:
        s[_LAST_PATHS_KEY] = {}
    s[_LAST_PATHS_KEY][path_key.value] = str(path)
    _write_settings(s, settings_path=settings_path)


def get_update_settings(*, settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    """
    Return update-related settings: newest_available_version (str or None),
    never_show_again (bool).
    """
    s = _get_settings(settings_path)
    update = s.get(_UPDATE_KEY) or {}
    return {
        _NEWEST_AVAILABLE_VERSION_KEY: update.get(_NEWEST_AVAILABLE_VERSION_KEY),
        _NEVER_SHOW_AGAIN_KEY: bool(update.get(_NEVER_SHOW_AGAIN_KEY, False)),
    }


def set_update_settings(
    newest_available_version: Optional[str] = None,
    never_show_again: Optional[bool] = None,
    *,
    settings_path: Path = _SETTINGS_JSON_PATH,
):
    """
    Update one or more update settings. Pass None for a key to leave it unchanged.
    """
    s = _get_settings(settings_path)
    if _UPDATE_KEY not in s:
        s[_UPDATE_KEY] = {}
    if newest_available_version is not None:
        s[_UPDATE_KEY][_NEWEST_AVAILABLE_VERSION_KEY] = newest_available_version
    if never_show_again is not None:
        s[_UPDATE_KEY][_NEVER_SHOW_AGAIN_KEY] = never_show_again
    _write_settings(s, settings_path=settings_path)


def _get_settings(settings_path: Path = _SETTINGS_JSON_PATH) -> dict:
    """
    Make sure that ./settings.json exists; if it does, then read it. If not, empty dict.
    """
    if not settings_path.exists():
        return dict()
    else:
        with open(settings_path, "r") as fp:
            return json.load(fp)


class _WriteSettings(object):
    def __init__(self):
        self._oserror = False

    def __call__(self, *args, **kwargs):
        if self._oserror:
            return
        # Try-catch for Issue 448
        try:
            return _write_settings_unsafe(*args, **kwargs)
        except OSError as e:
            if "Read-only filesystem" in str(e):
                print(
                    "Failed to write settings--NAM appears to be installed to a "
                    "read-only filesystem. This is discouraged; consider installing to "
                    "a location with user-level access."
                )
                self._oserror = True
            else:
                raise e


_write_settings = _WriteSettings()


def _write_settings_unsafe(obj: dict, settings_path: Path = _SETTINGS_JSON_PATH):
    with open(settings_path, "w") as fp:
        json.dump(obj, fp, indent=4)
