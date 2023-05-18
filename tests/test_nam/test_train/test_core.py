# File: test_core.py
# Created Date: Thursday May 18th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from nam.data import np_to_wav, wav_to_np
from nam.train import core

from ...resources import requires_v1_0_0, requires_v1_1_1, requires_v2_0_0


class TestDetectInputVersion(object):
    @requires_v1_0_0
    def test_detect_input_version_v1_0_0_strong(self):
        self._t_detect_input_version_strong(core.Version(1, 0, 0))

    @requires_v1_1_1
    def test_detect_input_version_v1_1_1_strong(self):
        self._t_detect_input_version_strong(core.Version(1, 1, 1))

    @requires_v2_0_0
    def test_detect_input_version_v2_0_0_strong(self):
        self._t_detect_input_version_strong(core.Version(2, 0, 0))

    @requires_v1_0_0
    def test_detect_input_version_v1_0_0_weak(self):
        self._t_detect_input_version_weak(core.Version(1, 0, 0))

    @requires_v1_1_1
    def test_detect_input_version_v1_1_1_weak(self):
        self._t_detect_input_version_weak(core.Version(1, 1, 1))

    @requires_v2_0_0
    def test_detect_input_version_v2_0_0_weak(self):
        self._t_detect_input_version_weak(core.Version(2, 0, 0))

    @classmethod
    def _customize_resource(cls, path_in, path_out):
        x, info = wav_to_np(path_in, info=True)
        # Should be safe...
        i = info.rate * 60
        y = np.concatenate([x[:i], np.zeros((1,)), x[i:]])
        np_to_wav(y, path_out)

    @classmethod
    def _t_detect_input_version(
        cls,
        path: Path,
        expected_input_version: core.Version,
        expected_strong_match: bool,
    ):
        input_version, strong_match = core._detect_input_version(path)
        assert input_version == expected_input_version
        assert strong_match == expected_strong_match

    @classmethod
    def _t_detect_input_version_strong(cls, version: core.Version):
        cls._t_detect_input_version(cls._resource_path(version), version, True)

    @classmethod
    def _t_detect_input_version_weak(cls, version: core.Version):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "temp.wav")
            cls._customize_resource(cls._resource_path(version), path)
            cls._t_detect_input_version(path, version, False)

    @classmethod
    def _resource_path(cls, version: core.Version) -> Path:
        return Path(__file__).absolute().parent.parent.parent / Path(
            "resources", f'v{str(version).replace(".", "_")}.wav'
        )


if __name__ == "__main__":
    pytest.main()
