# File: test_resources.py
# Created Date: Tuesday September 17th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

from contextlib import contextmanager
from pathlib import Path

import pytest

from nam.train.gui._resources import settings

_NEWEST_AVAILABLE_VERSION_KEY = settings._NEWEST_AVAILABLE_VERSION_KEY
_NEVER_SHOW_AGAIN_KEY = settings._NEVER_SHOW_AGAIN_KEY


class TestLastPaths(object):
    """get_last_path / set_last_path round-trip and custom settings_path."""

    @pytest.mark.parametrize("path_key", tuple(pk for pk in settings.PathKey))
    def test_set_then_get_last_path_round_trip(
        self, path_key: settings.PathKey, tmp_path: Path
    ):
        settings_path = tmp_path / "settings.json"
        value = tmp_path / "some" / "file.wav"
        value.parent.mkdir(parents=True, exist_ok=True)
        settings.set_last_path(path_key, value, settings_path=settings_path)
        assert settings.get_last_path(path_key, settings_path=settings_path) == value

    def test_custom_settings_path_does_not_affect_default(self, tmp_path: Path):
        """Writing with a custom settings_path leaves the default file unchanged."""
        custom_path = tmp_path / "custom_settings.json"
        key = settings.PathKey.INPUT_FILE
        value = tmp_path / "input.wav"
        settings.set_last_path(key, value, settings_path=custom_path)
        # Default path (package dir) should be unchanged: either missing or no key
        default_result = settings.get_last_path(key)
        assert default_result is None or default_result != value


class TestUpdateSettings(object):
    """get_update_settings / set_update_settings and custom settings_path."""

    def test_get_update_settings_empty_file_returns_defaults(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")
        out = settings.get_update_settings(settings_path=settings_path)
        assert out[_NEWEST_AVAILABLE_VERSION_KEY] is None
        assert out[_NEVER_SHOW_AGAIN_KEY] is False

    def test_get_update_settings_missing_file_returns_defaults(self, tmp_path: Path):
        settings_path = tmp_path / "nonexistent.json"
        assert not settings_path.exists()
        out = settings.get_update_settings(settings_path=settings_path)
        assert out[_NEWEST_AVAILABLE_VERSION_KEY] is None
        assert out[_NEVER_SHOW_AGAIN_KEY] is False

    def test_set_then_get_update_settings_round_trip(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        settings.set_update_settings(
            newest_available_version="1.2.3",
            never_show_again=True,
            settings_path=settings_path,
        )
        out = settings.get_update_settings(settings_path=settings_path)
        assert out[_NEWEST_AVAILABLE_VERSION_KEY] == "1.2.3"
        assert out[_NEVER_SHOW_AGAIN_KEY] is True

    def test_set_update_settings_partial_update_preserves_other(self, tmp_path: Path):
        settings_path = tmp_path / "settings.json"
        settings.set_update_settings(
            newest_available_version="2.0.0",
            never_show_again=False,
            settings_path=settings_path,
        )
        settings.set_update_settings(
            never_show_again=True,
            settings_path=settings_path,
        )
        out = settings.get_update_settings(settings_path=settings_path)
        assert out[_NEWEST_AVAILABLE_VERSION_KEY] == "2.0.0"
        assert out[_NEVER_SHOW_AGAIN_KEY] is True

    def test_custom_settings_path_isolates_update_settings(self, tmp_path: Path):
        """Update settings written to custom path do not affect default."""
        custom_path = tmp_path / "custom.json"
        # CAREFUL: this assumes that the newest available version is not in fact 9.9.9!
        settings.set_update_settings(
            newest_available_version="9.9.9",
            never_show_again=True,
            settings_path=custom_path,
        )
        default_out = settings.get_update_settings()
        # Default (package) settings should not have our test values
        assert default_out[_NEWEST_AVAILABLE_VERSION_KEY] != "9.9.9" or (
            default_out[_NEWEST_AVAILABLE_VERSION_KEY] is None
        )
        # never_show_again could be True if user had it set; we only assert custom has it
        custom_out = settings.get_update_settings(settings_path=custom_path)
        assert custom_out[_NEWEST_AVAILABLE_VERSION_KEY] == "9.9.9"
        assert custom_out[_NEVER_SHOW_AGAIN_KEY] is True


class TestReadOnly(object):
    """
    Issue 448
    """

    @pytest.mark.parametrize("path_key", tuple(pk for pk in settings.PathKey))
    def test_get_last_path(self, path_key: settings.PathKey):
        with self._mock_read_only():
            last_path = settings.get_last_path(path_key)
        assert last_path is None or isinstance(last_path, Path)

    @pytest.mark.parametrize("path_key", tuple(pk for pk in settings.PathKey))
    def test_set_last_path(self, path_key: settings.PathKey):
        path = Path(__file__).parent / Path("dummy.txt")
        with self._mock_read_only():
            settings.set_last_path(path_key, path)

    @contextmanager
    def _mock_read_only(self):
        def write_settings(*args, **kwargs):
            raise OSError("Read-only filesystem")

        try:
            tmp = settings._write_settings_unsafe
            settings._write_settings_unsafe = write_settings
            yield
        finally:
            settings._write_settings_unsafe = tmp


if __name__ == "__main__":
    pytest.main()
