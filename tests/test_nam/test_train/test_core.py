# File: test_core.py
# Created Date: Thursday May 18th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from nam.data import (
    Dataset,
    np_to_wav,
    wav_to_np,
    wav_to_tensor,
    _DEFAULT_REQUIRE_INPUT_PRE_SILENCE,
)
from nam.train import core
from nam.train._version import Version

from ...resources import (
    requires_v1_0_0,
    requires_v1_1_1,
    requires_v2_0_0,
    resource_path,
)

__all__ = []


def _resource_path(version: Version) -> Path:
    if version == Version(1, 0, 0):
        name = "v1.wav"
    else:
        name = f'v{str(version).replace(".", "_")}.wav'
    return resource_path(name)


class TestDetectInputVersion(object):
    @requires_v1_0_0
    def test_detect_input_version_v1_0_0_strong(self):
        self._t_detect_input_version_strong(Version(1, 0, 0))

    @requires_v1_1_1
    def test_detect_input_version_v1_1_1_strong(self):
        self._t_detect_input_version_strong(Version(1, 1, 1))

    @requires_v2_0_0
    def test_detect_input_version_v2_0_0_strong(self):
        self._t_detect_input_version_strong(Version(2, 0, 0))

    @requires_v1_0_0
    def test_detect_input_version_v1_0_0_weak(self):
        self._t_detect_input_version_weak(Version(1, 0, 0))

    @requires_v1_1_1
    def test_detect_input_version_v1_1_1_weak(self):
        self._t_detect_input_version_weak(Version(1, 1, 1))

    @requires_v2_0_0
    def test_detect_input_version_v2_0_0_weak(self):
        self._t_detect_input_version_weak(Version(2, 0, 0))

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
        expected_input_version: Version,
        expected_strong_match: bool,
    ):
        input_version, strong_match = core._detect_input_version(path)
        assert input_version == expected_input_version
        assert strong_match == expected_strong_match

    @classmethod
    def _t_detect_input_version_strong(cls, version: Version):
        cls._t_detect_input_version(_resource_path(version), version, True)

    @classmethod
    def _t_detect_input_version_weak(cls, version: Version):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "temp.wav")
            cls._customize_resource(_resource_path(version), path)
            cls._t_detect_input_version(path, version, False)


class _TCalibrateDelay(object):
    _calibrate_delay = None
    _data_info: core._DataInfo = None

    @pytest.mark.parametrize("expected_delay", (-10, 0, 5, 100))
    def test_calibrate_delay(self, expected_delay: int):
        x = np.zeros((self._data_info.t_blips))
        for i in self._data_info.start_blip_locations:
            x[i + expected_delay] = 1.0
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "output.wav")
            np_to_wav(x, path)
            delay = self._calibrate_delay(None, path)
            assert delay == expected_delay - core._DELAY_CALIBRATION_SAFETY_FACTOR


class TestCalibrateDelayV1(_TCalibrateDelay):
    _calibrate_delay = core._calibrate_delay_v1
    _data_info = core._V1_DATA_INFO


class TestCalibrateDelayV2(_TCalibrateDelay):
    _calibrate_delay = core._calibrate_delay_v2
    _data_info = core._V2_DATA_INFO


def _make_t_validation_dataset_class(
    version: Version, decorator, data_info: core._DataInfo
):
    class C(object):
        @decorator
        def test_validation_preceded_by_silence(self):
            """
            Validate that the datasets that we've made are valid
            """
            x = wav_to_tensor(_resource_path(version))
            Dataset._validate_preceding_silence(
                x,
                data_info.validation_start,
                int(_DEFAULT_REQUIRE_INPUT_PRE_SILENCE * data_info.rate),
            )

    return C


TestValidationDatasetV1_0_0 = _make_t_validation_dataset_class(
    Version(1, 0, 0), requires_v1_0_0, core._V1_DATA_INFO
)


TestValidationDatasetV1_1_1 = _make_t_validation_dataset_class(
    Version(1, 1, 1), requires_v1_1_1, core._V1_DATA_INFO
)


TestValidationDatasetV2_0_0 = _make_t_validation_dataset_class(
    Version(2, 0, 0), requires_v2_0_0, core._V2_DATA_INFO
)

if __name__ == "__main__":
    pytest.main()
