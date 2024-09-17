# File: test_resources.py
# Created Date: Tuesday September 17th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

from contextlib import contextmanager
from pathlib import Path

import pytest

from nam.train.gui._resources import settings


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
