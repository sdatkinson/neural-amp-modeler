# File: test_core.py
# Created Date: Thursday May 18th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import inspect
import json
import sys
from copy import deepcopy
from importlib import resources
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from nam.data import (
    _DEFAULT_REQUIRE_INPUT_PRE_SILENCE,
    Dataset,
    np_to_wav,
    wav_to_np,
    wav_to_tensor,
)
from nam.train import core
from nam.train import metadata as _metadata
from nam.train._version import Version
from nam.train.lightning_module import PackedLightningModule

from ...resources import (
    requires_proteus,
    requires_v1_0_0,
    requires_v1_1_1,
    requires_v2_0_0,
    requires_v3_0_0,
    resource_path,
)

__all__ = []


_REMOVED_SIMPLIFIED_TRAINER_KWARGS = {
    "model_type",
    "architecture",
    "lr",
    "lr_decay",
    "fit_mrstft",
}


def _load_packaged_packed_model_config():
    resource = resources.files("nam.train._resources").joinpath(
        "config_model_packed.json"
    )
    with resource.open("r") as fp:
        return json.load(fp)


def _core_removed_kwargs_present():
    functions = (core.train, core._get_configs)
    return {
        name
        for function in functions
        for name in inspect.signature(function).parameters
        if name in _REMOVED_SIMPLIFIED_TRAINER_KWARGS
    }


def _call_get_configs_for_current_core():
    values = {
        "input_version": Version(3, 0, 0),
        "input_path": "input.wav",
        "output_path": "output.wav",
        "latency": 0,
        "epochs": 5,
        "ny": 16,
        "batch_size": 2,
    }
    signature = inspect.signature(core._get_configs)
    kwargs = {
        name: deepcopy(values[name])
        for name in signature.parameters
        if name in values
    }
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    assert not missing, f"Unhandled _get_configs parameters: {missing}"
    return core._get_configs(**kwargs)


def _resource_path(version: Version) -> Path:
    if version == Version(1, 0, 0):
        name = "v1.wav"
    elif version == Version(4, 0, 0):
        name = "Proteus_Capture.wav"
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

    @requires_v3_0_0
    def test_detect_input_version_v3_0_0_strong(self):
        self._t_detect_input_version_strong(Version(3, 0, 0))

    @requires_v1_0_0
    def test_detect_input_version_v1_0_0_weak(self):
        self._t_detect_input_version_weak(Version(1, 0, 0))

    @requires_v1_1_1
    def test_detect_input_version_v1_1_1_weak(self):
        self._t_detect_input_version_weak(Version(1, 1, 1))

    @requires_v2_0_0
    def test_detect_input_version_v2_0_0_weak(self):
        self._t_detect_input_version_weak(Version(2, 0, 0))

    @requires_v3_0_0
    def test_detect_input_version_v3_0_0_weak(self):
        self._t_detect_input_version_weak(Version(3, 0, 0))

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
        x = np.zeros((self._data_info.first_blips_start + self._data_info.t_blips,))
        # This test only works with the first set of blip locations. Any other set of
        # blip locations is used to check the data, not to calibrate the delay.
        for i in self._data_info.blip_locations[0]:
            # The blip locations are absolute in the file, not relative to the start of
            # the blip section, so `first_blips_start` isn't used.
            x[i + expected_delay] = 1.0

        delay_calibration = self._calibrate_delay(
            x, manual_available=False, show_plots=False
        )
        actual_recommended = delay_calibration.recommended
        assert (
            actual_recommended == expected_delay - core._DELAY_CALIBRATION_SAFETY_FACTOR
        )

    def test_lookahead_warning(self):
        """
        If the delay is equal to the (negative) lookahead, then something is probably wrong.
        Assert that we're warned.

        See: https://github.com/sdatkinson/neural-amp-modeler/issues/304
        """

        # Make the response loud enough to trigger the threshold everywhere.
        # Use the absolute threshold since the relative will be zero (The signal will be
        # zeroed next so it's silent where the thresholds are calibrated.)
        y = np.full(
            (self._data_info.first_blips_start + self._data_info.t_blips,),
            core._DELAY_CALIBRATION_ABS_THRESHOLD + 0.01,
        )
        # Make the signal silent where the threshold is calibrated so the absolute
        # threshold is used.
        y[self._data_info.noise_interval[0] : self._data_info.noise_interval[1]] = 0.0

        # Prepare to capture the output and look for a warning.
        class Capturing(list):
            def __enter__(self):
                self._stdout = sys.stdout
                sys.stdout = self._stringio = StringIO()
                return self

            def __exit__(self, *args):
                self.extend(self._stringio.getvalue().splitlines())
                del self._stringio
                sys.stdout = self._stdout

        with Capturing() as output:
            self._calibrate_delay(y, manual_available=False, show_plots=False)
        # `[0]` -- Only look in the first set of blip locations
        # With #485, we average them all together so there's only one index.
        # TODO clean this up.
        expected_warning = core._warn_lookaheads([1])  # "Blip 1"
        assert any(o == expected_warning for o in output), output


class TestCalibrateDelayV1(_TCalibrateDelay):
    _calibrate_delay = core._calibrate_latency_v1
    _data_info = core._V1_DATA_INFO


class TestCalibrateDelayV2(_TCalibrateDelay):
    _calibrate_delay = core._calibrate_latency_v2
    _data_info = core._V2_DATA_INFO


class TestCalibrateDelayV3(_TCalibrateDelay):
    _calibrate_delay = core._calibrate_latency_v3
    _data_info = core._V3_DATA_INFO


class TestCalibrateDelayV4(_TCalibrateDelay):
    _calibrate_delay = core._calibrate_latency_v4
    _data_info = core._V4_DATA_INFO


def _make_t_validation_dataset_class(
    version: Version, decorator, data_info: core._DataInfo
):
    class C(object):
        pass

    # Proteus has a bad validation split; don't define the silence test for it.
    if version == Version(4, 0, 0):
        return C
    else:

        class C2(C):
            @decorator
            def test_validation_preceded_by_silence(self):
                """
                Validate that the datasets that we've made are valid
                """
                x = wav_to_tensor(_resource_path(version))
                Dataset._validate_preceding_silence(
                    x,
                    data_info.validation_start,
                    _DEFAULT_REQUIRE_INPUT_PRE_SILENCE,
                    data_info.rate,
                )

        return C2


TestValidationDatasetV1_0_0 = _make_t_validation_dataset_class(
    Version(1, 0, 0), requires_v1_0_0, core._V1_DATA_INFO
)


TestValidationDatasetV1_1_1 = _make_t_validation_dataset_class(
    Version(1, 1, 1), requires_v1_1_1, core._V1_DATA_INFO
)


TestValidationDatasetV2_0_0 = _make_t_validation_dataset_class(
    Version(2, 0, 0), requires_v2_0_0, core._V2_DATA_INFO
)


TestValidationDatasetV3_0_0 = _make_t_validation_dataset_class(
    Version(3, 0, 0), requires_v3_0_0, core._V3_DATA_INFO
)


# Aka Proteus
TestValidationDatasetV4_0_0 = _make_t_validation_dataset_class(
    Version(4, 0, 0), requires_proteus, core._V4_DATA_INFO
)


def test_v3_check_doesnt_make_figure_if_silent(mocker):
    """
    Issue 337

    :param mocker: Provided by pytest-mock
    """
    import matplotlib.pyplot

    class MadeFigureError(RuntimeError):
        """
        For this test, detect if a figure was made, and raise an exception if so
        """

        pass

    def figure_mock(*args, **kwargs):
        raise MadeFigureError("The test tried to make a figure")

    mocker.patch("matplotlib.pyplot.figure", figure_mock)

    # Make some data that's totally going to biff it
    # [:-1] won't match [1:]
    x = np.random.rand(core._V3_DATA_INFO.t_validate + 1) - 0.5

    with TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir, "input.wav")
        output_path = Path(tmpdir, "output.wav")
        np_to_wav(x, input_path)  # Doesn't need to be the actual thing for now
        np_to_wav(x, output_path)
        # If this makes a figure, then it wasn't silent!
        core._check_v3(input_path, output_path, silent=True)


def test_simplified_trainer_removed_knobs_are_absent_from_core_api():
    present = _core_removed_kwargs_present()
    assert not present


def test_get_configs_uses_packaged_packed_model_config():
    _, model_config, _ = _call_get_configs_for_current_core()
    expected_model_config = _load_packaged_packed_model_config()

    assert model_config["net"]["name"] == "PackedWaveNet"
    assert model_config["net"] == expected_model_config["net"]
    assert model_config["loss"] == expected_model_config["loss"]
    assert model_config["optimizer"] == expected_model_config["optimizer"]
    assert model_config["lr_scheduler"] == expected_model_config["lr_scheduler"]


def test_plot_reports_and_plots_each_packed_prediction(mocker, capsys):
    target = torch.tensor([1.0, -1.0, 2.0, -2.0])
    predictions = torch.stack([target, 0.5 * target])
    plot_calls = []

    class FakeDataset:
        x = torch.zeros_like(target)
        y = target

    class FakeModel:
        class net:
            num_submodels = 2
            submodel_names = ("small", "large")

        def __call__(self, x):
            assert torch.equal(x, FakeDataset.x)
            return predictions

    def capture_plot(*args, **kwargs):
        plot_calls.append((args, kwargs))

    times = iter((0.0, 0.5))
    mocker.patch.object(core, "_time", lambda: next(times))
    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.plot", capture_plot)
    mocker.patch("matplotlib.pyplot.title")
    mocker.patch("matplotlib.pyplot.legend")
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.show")

    core._plot(FakeModel(), FakeDataset, silent=True)

    stdout = capsys.readouterr().out
    assert stdout.count("Error-signal ratio") == 2

    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    prediction_plots = [
        as_numpy(args[0])
        for args, kwargs in plot_calls
        if "Prediction" in kwargs.get("label", "")
    ]
    target_plots = [
        as_numpy(args[0])
        for args, kwargs in plot_calls
        if kwargs.get("label") == "Target"
    ]
    assert len(prediction_plots) == 2
    assert len(target_plots) == 1
    np.testing.assert_allclose(prediction_plots[0], predictions[0].numpy())
    np.testing.assert_allclose(prediction_plots[1], predictions[1].numpy())
    for plotted_target in target_plots:
        np.testing.assert_allclose(plotted_target, target.numpy())


@requires_v3_0_0
def test_end_to_end():
    """
    Run a training using core.train()
    """
    with TemporaryDirectory() as tmpdir:
        basename = "v3_0_0"
        input_path = resource_path(basename + ".wav")
        output_path = input_path  # Identity mapping!
        train_path = Path(tmpdir)
        train_output = core.train(
            input_path,
            output_path,
            train_path,
            silent=True,
            fast_dev_run=True,
        )
        # Assertions...
        assert isinstance(train_output.model, PackedLightningModule)


def test_get_callbacks():
    """
    Sanity check for get_callbacks with a custom extension callback and threshold_esr
    """
    threshold_esr = 0.01
    callbacks = core.get_callbacks(threshold_esr=threshold_esr)

    # dumb example of a user-extended custom callback
    class CustomCallback:
        pass

    extended_callbacks = callbacks + [CustomCallback()]

    # sanity default callbacks
    assert any(
        isinstance(cb, core._ModelCheckpoint) for cb in extended_callbacks
    ), "Expected _ModelCheckpoint to be part of the default callbacks."

    # custom callback
    assert any(
        isinstance(cb, CustomCallback) for cb in extended_callbacks
    ), "Expected CustomCallback to be added to the extended callbacks."

    # _ValidationStopping cb when threshold_esr is prvided
    assert any(
        isinstance(cb, core._ValidationStopping) for cb in extended_callbacks
    ), "_ValidationStopping should still be present after adding a custom callback."


class TestAnalyzeLatency:
    """
    Assertions about the behavior of _analyze_latency()
    """

    @requires_v3_0_0
    def test_analyze_latency_doesnt_fail_if_user_provides(self):
        """
        Assert that the latency analysis succeeds whenever the user provides the
        latency that should be used, i.e. doesn't fail whenever because automatic
        detection fails.
        """
        with TemporaryDirectory() as tmpdir:
            input_path = resource_path("v3_0_0.wav")
            output_path = Path(tmpdir, "output.wav")
            # This output is silent, so the calibration will fail
            np_to_wav(np.zeros_like(wav_to_np(input_path)), output_path)
            analysis = core._analyze_latency(
                user_latency=100,
                input_version=Version(3, 0, 0),
                input_path=input_path,
                output_path=output_path,
                silent=True,
            )
        assert analysis.manual == 100
        assert analysis.calibration.recommended is None

    @requires_v3_0_0
    def test_no_fail_if_no_user_and_automatic_calibration_fails(self):
        """
        Even if the automatic calibration fails, the function should not raise
        an exception. That should happen only when getting the final latency.
        """
        with TemporaryDirectory() as tmpdir:
            input_path = resource_path("v3_0_0.wav")
            output_path = Path(tmpdir, "output.wav")
            # This output is silent, so the calibration will fail
            np_to_wav(np.zeros_like(wav_to_np(input_path)), output_path)
            analysis = core._analyze_latency(
                user_latency=None,  # No fallback; should fail
                input_version=Version(3, 0, 0),
                input_path=input_path,
                output_path=output_path,
                silent=True,
                _override_suppress_plots=True,
            )
        assert analysis.manual is None
        assert analysis.calibration.recommended is None


def test_get_final_latency_fails_if_no_user_and_automatic_calibration_fails():
    """
    If the automatic and manual calibration aren't available, then the function
    should raise an exception.
    """
    latency_analysis = _metadata.Latency(
        manual=None,
        calibration=_metadata.LatencyCalibration(
            algorithm_version=1,
            delays=[],
            safety_factor=1,
            recommended=None,
            warnings=_metadata.LatencyCalibrationWarnings(
                matches_lookahead=False,
                disagreement_too_high=False,
                not_detected=True,
            ),
        ),
    )
    with pytest.raises(core._FinalLatencyError):
        core._get_final_latency(latency_analysis)


if __name__ == "__main__":
    pytest.main()
