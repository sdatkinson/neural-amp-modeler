# File: test_data.py
# Created Date: Friday May 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import math
import os
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import torch

from nam import data
from nam.models.base import BaseNet

_SAMPLE_RATES = (44_100.0, 48_000.0, 88_200.0, 96_000.0)
_DEFAULT_SAMPLE_RATE = 48_000.0


class _XYMethod(Enum):
    ARANGE = "arange"
    RAND = "rand"
    STEP = "step"


class TestDataset(object):
    """
    Assertions about nam.data.Dataset
    """

    def test_apply_delay_zero(self):
        """
        Assert proper function of Dataset._apply_delay() when zero delay is given, i.e.
        no change.
        """
        x, y = self._create_xy()
        x_out, y_out = data.Dataset._apply_delay(x, y, 0)
        assert torch.all(x == x_out)
        assert torch.all(y == y_out)

    def test_apply_delay_int_negative(self):
        """
        Assert proper function of Dataset._apply_delay() when a positive integer delay
        is given.
        """
        n = 7
        delay = -3
        x_out, y_out = self._t_apply_delay_int(n, delay)

        assert torch.all(x_out == torch.Tensor([3, 4, 5, 6]))
        assert torch.all(y_out == torch.Tensor([0, 1, 2, 3]))

    def test_apply_delay_int_positive(self):
        """
        Assert proper function of Dataset._apply_delay() when a positive integer delay
        is given.
        """
        n = 7
        delay = 3
        x_out, y_out = self._t_apply_delay_int(n, delay)

        assert torch.all(x_out == torch.Tensor([0, 1, 2, 3]))
        assert torch.all(y_out == torch.Tensor([3, 4, 5, 6]))

    def test_init(self):
        x, y = self._create_xy()
        data.Dataset(x, y, 3, None, sample_rate=_DEFAULT_SAMPLE_RATE)

    def test_init_sample_rate(self):
        x, y = self._create_xy()
        sample_rate = _DEFAULT_SAMPLE_RATE
        d = data.Dataset(x, y, 3, None, sample_rate=sample_rate)
        assert hasattr(d, "sample_rate")
        assert isinstance(d.sample_rate, float)
        assert d.sample_rate == sample_rate

    def test_init_zero_delay(self):
        """
        Assert https://github.com/sdatkinson/neural-amp-modeler/issues/15 fixed
        """
        x, y = self._create_xy()
        data.Dataset(x, y, 3, None, delay=0, sample_rate=_DEFAULT_SAMPLE_RATE)

    def test_input_gain(self):
        """
        Checks correctness of input gain parameter
        """
        x_scale = 2.0
        input_gain = 20.0 * math.log10(x_scale)
        x, y = self._create_xy()
        nx = 3
        ny = None
        args = (x, y, nx, ny)
        d1 = data.Dataset(*args, sample_rate=_DEFAULT_SAMPLE_RATE)
        d2 = data.Dataset(
            *args, sample_rate=_DEFAULT_SAMPLE_RATE, input_gain=input_gain
        )

        sample_x1 = d1[0][0]
        sample_x2 = d2[0][0]
        assert torch.allclose(sample_x1 * x_scale, sample_x2)

    def test_scale_output_handshake_compensates_export_head_scale(self):
        x, y = self._create_xy(n=8)
        dataset = data.Dataset(x, y, 1, None, sample_rate=_DEFAULT_SAMPLE_RATE)
        dataset.scale_output(2.0)
        dataset.scale_output(4.0)
        model = _DummyWaveNet()

        dataset.handshake(model)
        dataset.handshake(model)
        model_dict = {
            "architecture": "WaveNet",
            "config": {"head_scale": 0.5},
            "weights": [0.125],
        }

        assert len(model.export_model_dict_post_hooks) == 1
        assert model._apply_export_model_dict_post_hooks(model_dict) == {
            "architecture": "WaveNet",
            "config": {"head_scale": pytest.approx(0.5 / 8.0)},
            "weights": [pytest.approx(0.125 / 8.0)],
        }

    @pytest.mark.parametrize("sample_rate", _SAMPLE_RATES)
    def test_sample_rates(self, sample_rate: int):
        """
        Test that datasets with various sample rates can be made
        """
        x = np.random.rand(16) - 0.5
        y = x
        with TemporaryDirectory() as tmpdir:
            x_path = Path(tmpdir, "input.wav")
            y_path = Path(tmpdir, "output.wav")
            data.np_to_wav(x, x_path, rate=sample_rate)
            data.np_to_wav(y, y_path, rate=sample_rate)
            config = {"x_path": str(x_path), "y_path": str(y_path), "nx": 4, "ny": 2}
            parsed_config = data.Dataset.parse_config(config)
        assert parsed_config["sample_rate"] == sample_rate

    @pytest.mark.parametrize(
        "n,start,valid",
        (
            (13, None, True),  # No start restrictions; nothing wrong
            (13, 2, True),  # Starts before the end; fine.
            (13, 12, True),  # Starts w/ one to go--ok
            (13, 13, False),  # Starts after the end
            (13, -5, True),  # Starts counting back from the end, fine
            (13, -13, True),  # Starts at the beginning of the array--ok
            (13, -14, False),  # Starts before the beginning of the array--invalid
        ),
    )
    def test_validate_start(self, n: int, start: int, valid: bool):
        """
        Assert that a data set can be successfully instantiated when valid args are
        given, including `start`.
        Assert that `StartError` is raised if invalid start is provided
        """

        def init():
            data.Dataset(x, y, nx, ny, start=start, sample_rate=_DEFAULT_SAMPLE_RATE)

        nx = 1
        ny = None
        x, y = self._create_xy(n=n)
        if start is not None:
            x[:start] = 0.0  # Ensure silent input before the start
        if valid:
            init()
            assert True  # No problem!
        else:
            with pytest.raises(data.StartError):
                init()

    @pytest.mark.parametrize(
        "start,start_samples,start_seconds,stop,stop_samples,stop_seconds,sample_rate,raises",
        (
            # Nones across the board (valid)
            (None, None, None, None, None, None, None, None),
            # start and stop (valid)
            (1, None, None, -1, None, None, None, None),
            # start_samples and stop_samples (valid)
            (None, 1, None, None, -1, None, None, None),
            # start_seconds and stop_seconds with sample_rate (valid)
            (None, None, 0.5, None, None, -0.5, 2, None),
            # Multiple start-like, even if they agree (invalid)
            (1, 1, None, None, None, None, None, ValueError),
            # Multiple stop-like, even if they agree (invalid)
            (None, None, None, -1, -1, None, None, ValueError),
            # seconds w/o sample rate (invalid)
            (None, None, 1.0, None, None, None, None, ValueError),
        ),
    )
    def test_validate_start_stop(
        self,
        start: Optional[int],
        start_samples: Optional[int],
        start_seconds: Optional[Union[int, float]],
        stop: Optional[int],
        stop_samples: Optional[int],
        stop_seconds: Optional[Union[int, float]],
        sample_rate: Optional[int],
        raises: Optional[Exception],
    ):
        """
        Assert correct behavior of `._validate_start_stop()` class method.
        """

        def f():
            # Don't provide start/stop that are too large for the fake data plz.
            x, y = torch.zeros((2, 32))
            data.Dataset._validate_start_stop(
                x,
                y,
                start,
                stop,
                start_samples,
                stop_samples,
                start_seconds,
                stop_seconds,
                sample_rate,
            )
            assert True

        if raises is None:
            f()
        else:
            with pytest.raises(raises):
                f()

    @pytest.mark.parametrize(
        "n,stop,valid",
        (
            (13, None, True),  # No stop restrictions; nothing wrong
            (13, 2, True),  # Stops before the end; fine.
            (13, 13, True),  # Stops at the end--ok
            (13, 14, False),  # Stops after the end--not ok
            (13, -5, True),  # Stops counting back from the end, fine
            (13, -12, True),  # Stops w/ one sample--ok
            (13, -13, False),  # Stops w/ no samples--not ok
        ),
    )
    def test_validate_stop(self, n: int, stop: int, valid: bool):
        def init():
            data.Dataset(x, y, nx, ny, stop=stop, sample_rate=_DEFAULT_SAMPLE_RATE)

        nx = 1
        ny = None
        x, y = self._create_xy(n=n)
        if valid:
            init()
            assert True  # No problem!
        else:
            with pytest.raises(data.StopError):
                init()

    @pytest.mark.parametrize(
        "lenx,leny,valid",
        ((3, 3, True), (3, 4, False), (0, 0, False)),  # Lenght mismatch  # Empty!
    )
    def test_validate_x_y(self, lenx: int, leny: int, valid: bool):
        def init():
            data.Dataset(x, y, nx, ny, sample_rate=_DEFAULT_SAMPLE_RATE)

        x, y = self._create_xy()
        assert len(x) >= lenx, "Invalid test!"
        assert len(y) >= leny, "Invalid test!"
        x = x[:lenx]
        y = y[:leny]
        nx = 1
        ny = None
        if valid:
            init()
            assert True  # It worked!
        else:
            with pytest.raises(data.XYError):
                init()

    def _create_xy(
        self,
        n: int = 7,
        method: _XYMethod = _XYMethod.RAND,
        must_be_in_valid_range: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (n,), (n,)
        """
        if method == _XYMethod.ARANGE:
            # note: this isn't "valid" data in the sense that it's beyond (-1, 1).
            # But it is useful for the delay code.
            assert not must_be_in_valid_range
            return tuple(
                torch.tile(torch.arange(n, dtype=torch.float)[None, :], (2, 1))
            )
        elif method == _XYMethod.RAND:
            return tuple(0.99 * (2.0 * torch.rand((2, n)) - 1.0))  # Don't clip
        elif method == _XYMethod.STEP:
            return tuple(
                torch.tile((torch.linspace(0.0, 1.0, n) > 0.5)[None, :], (2, 1))
            )

    def _t_apply_delay_int(self, n: int, delay: int):
        x, y = self._create_xy(
            n=n, method=_XYMethod.ARANGE, must_be_in_valid_range=False
        )

        x_out, y_out = data.Dataset._apply_delay(x, y, delay)
        n_out = n - np.abs(delay)
        assert len(x_out) == n_out
        assert len(y_out) == n_out

        return x_out, y_out


class TestScaleOutputHookLoudnessCompensation:
    """
    `_ScaleOutputHook` undoes a dataset y_scale on export by scaling
    `head_scale` (and the duplicated `weights[-1]`). `metadata.loudness`
    must move along with it so it describes the compensated model that the
    plugin actually loads — not the pre-compensation snapshot taken inside
    `_get_export_dict()`.

    Output of WaveNet (no top-level head) and SlimmableContainer is linear
    in `head_scale`, so the dB adjustment is exact:

        loudness_new = loudness_old + 20 * log10(scale)
    """

    @staticmethod
    def _hook(scale: float) -> data.Dataset._ScaleOutputHook:
        return data.Dataset._ScaleOutputHook(scale=scale)

    def test_wavenet_loudness_shifts_in_db_by_scale(self):
        scale = 2.0
        model_dict = {
            "architecture": "WaveNet",
            "config": {"head_scale": 0.01},
            "metadata": {"loudness": -20.0, "gain": 0.5},
            "weights": [0.01],
        }
        self._hook(scale).apply(model_dict)
        assert model_dict["metadata"]["loudness"] == pytest.approx(
            -20.0 + 20.0 * math.log10(scale)
        )
        # gain is invariant under uniform output scaling
        assert model_dict["metadata"]["gain"] == 0.5

    def test_slimmable_container_shifts_container_and_submodels(self):
        scale = 0.5
        container = {
            "architecture": "SlimmableContainer",
            "metadata": {"loudness": -17.0, "gain": 0.5},
            "config": {
                "submodels": [
                    {
                        "max_value": 0.5,
                        "model": {
                            "architecture": "WaveNet",
                            "config": {"head_scale": 0.01},
                            "metadata": {"loudness": -19.0, "gain": 0.3},
                            "weights": [0.01],
                        },
                    },
                    {
                        "max_value": 1.0,
                        "model": {
                            "architecture": "WaveNet",
                            "config": {"head_scale": 0.01},
                            "metadata": {"loudness": -17.0, "gain": 0.5},
                            "weights": [0.01],
                        },
                    },
                ]
            },
        }
        self._hook(scale).apply(container)
        offset = 20.0 * math.log10(scale)
        assert container["config"]["submodels"][0]["model"]["metadata"][
            "loudness"
        ] == pytest.approx(-19.0 + offset)
        assert container["config"]["submodels"][1]["model"]["metadata"][
            "loudness"
        ] == pytest.approx(-17.0 + offset)
        assert container["metadata"]["loudness"] == pytest.approx(
            container["config"]["submodels"][1]["model"]["metadata"]["loudness"]
        )
        assert (
            container["metadata"]["gain"]
            == container["config"]["submodels"][1]["model"]["metadata"]["gain"]
        )

    def test_no_op_when_loudness_metadata_absent(self):
        """Hook is robust when called on a dict without loudness metadata."""
        model_dict = {
            "architecture": "WaveNet",
            "config": {"head_scale": 0.01},
            "weights": [0.01],
        }
        self._hook(2.0).apply(model_dict)
        assert "metadata" not in model_dict
        assert model_dict["config"]["head_scale"] == pytest.approx(0.02)


class TestWav(object):
    tolerance = 1e-6

    @pytest.fixture(scope="class")
    def tmpdir(self):
        with TemporaryDirectory() as tmp:
            yield tmp

    def test_np_to_wav_to_np(self, tmpdir):
        # Create random numpy array
        x = np.random.rand(1000)
        # Save numpy array as WAV file
        filename = os.path.join(tmpdir, "test.wav")
        data.np_to_wav(x, filename)
        # Load WAV file
        y = data.wav_to_np(filename)
        # Check if the two arrays are equal
        assert y == pytest.approx(x, abs=self.tolerance)

    @pytest.mark.parametrize("sample_rate", _SAMPLE_RATES)
    def test_np_to_wav_to_np_sample_rates(self, sample_rate: int):
        with TemporaryDirectory() as tmpdir:
            # Create random numpy array
            x = np.random.rand(8)
            # Save numpy array as WAV file with sampling rate of 44 kHz
            filename = Path(tmpdir, "x.wav")
            data.np_to_wav(x, filename, rate=sample_rate)
            # Load WAV file with sampling rate of 44 kHz
            y = data.wav_to_np(filename, rate=sample_rate)
            # Check if the two arrays are equal
            assert y == pytest.approx(x, abs=self.tolerance)

    def test_np_to_wav_to_np_scale_arg(self, tmpdir):
        # Create random numpy array
        x = np.random.rand(100)
        # Save numpy array as WAV file with scaling
        filename = os.path.join(tmpdir, "test.wav")
        data.np_to_wav(x, filename, scale=None)
        # Load WAV file
        y = data.wav_to_np(filename)
        # Check if the two arrays are equal
        assert y == pytest.approx(x, abs=self.tolerance)

    @pytest.mark.parametrize("sample_width", (2, 3))
    def test_sample_widths(self, sample_width: int):
        """
        Test that datasets with various sample widths can be made
        """
        x = np.random.rand(16) - 0.5
        with TemporaryDirectory() as tmpdir:
            x_path = Path(tmpdir, "x.wav")
            data.np_to_wav(x, x_path, sampwidth=sample_width)
            _, info = data.wav_to_np(x_path, info=True)
        assert info.sampwidth == sample_width


def test_wav_to_np_fp32():
    """
    Issue 598
    """
    path = Path(__file__).parents[1] / "resources" / "fp32.wav"
    # Assert that this works:
    data.wav_to_np(path)


class TestConcatDataset(object):
    @pytest.mark.parametrize("attrname", ("nx", "ny", "sample_rate"))
    def test_valiation_sample_rate_fail(self, attrname: str):
        """
        Assert failed validation for datasets with different nx, ny, sample rates
        """
        nx, ny, sample_rate = 1, 2, 48_000.0

        n1 = 16
        ds1_kwargs = dict(
            x=torch.zeros((n1,)),
            y=torch.zeros((n1,)),
            nx=nx,
            ny=ny,
            sample_rate=sample_rate,
        )
        ds1 = data.Dataset(**ds1_kwargs)
        n2 = 7
        ds2_kwargs = dict(
            x=torch.zeros((n2,)),
            y=torch.zeros((n2,)),
            nx=nx,
            ny=ny,
            sample_rate=sample_rate,
        )
        # Cause the error by moving the named attr:
        ds2_kwargs[attrname] += 1
        ds2 = data.Dataset(**ds2_kwargs)
        with pytest.raises(data.ConcatDatasetValidationError):
            data.ConcatDataset([ds1, ds2])


class _DummyWaveNet(BaseNet):
    @property
    def pad_start_default(self) -> bool:
        return False

    @property
    def receptive_field(self) -> int:
        return 1

    def _forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x

    def _export_config(self):
        return {"head_scale": 0.5}

    def _export_weights(self) -> np.ndarray:
        return np.array([], dtype=np.float32)

    def _get_export_architecture(self) -> str:
        return "WaveNet"

    def _get_non_user_metadata(self):
        return {}


def _dataset_with_constant_y(value: float, n: int = 8) -> data.Dataset:
    return data.Dataset(
        x=torch.zeros((n,)),
        y=torch.full((n,), value),
        nx=1,
        ny=None,
        sample_rate=_DEFAULT_SAMPLE_RATE,
    )


def _rms(x: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(torch.square(x))).item()


def test_get_joint_dataset_hooks_instantiates_normalizer_alias():
    hooks = data.get_joint_dataset_hooks(
        [
            {
                "name": "nam.data.normalize_joint_dataset_output",
                "kwargs": {"level_rms_dbfs": -18.0},
            }
        ]
    )

    assert len(hooks) == 2
    assert isinstance(hooks[1], data.NormalizeJointDatasetOutput)


def test_apply_joint_dataset_hooks_normalizes_outputs_from_training_rms():
    target_dbfs = -12.0
    target_rms = 10 ** (target_dbfs / 20.0)
    dataset_train = _dataset_with_constant_y(0.25)
    dataset_validation = _dataset_with_constant_y(0.125)

    data.apply_joint_dataset_hooks(
        dataset_train=dataset_train,
        dataset_validation=dataset_validation,
        hooks=data.get_joint_dataset_hooks(
            [
                {
                    "name": "nam.data.normalize_joint_dataset_output",
                    "kwargs": {"level_rms_dbfs": target_dbfs},
                }
            ]
        ),
    )

    assert _rms(dataset_train.y) == pytest.approx(target_rms)
    assert _rms(dataset_validation.y) == pytest.approx(target_rms * 0.5)


def test_normalize_joint_dataset_output_uses_concat_training_rms():
    target_dbfs = -6.0
    target_rms = 10 ** (target_dbfs / 20.0)
    train_a = _dataset_with_constant_y(0.25, n=4)
    train_b = _dataset_with_constant_y(0.5, n=4)
    dataset_train = data.ConcatDataset([train_a, train_b])
    dataset_validation = _dataset_with_constant_y(0.125, n=4)
    expected_scale = target_rms / math.sqrt((4 * 0.25**2 + 4 * 0.5**2) / 8)

    data.NormalizeJointDatasetOutput(target_dbfs).apply(
        dataset_train=dataset_train,
        dataset_validation=dataset_validation,
    )

    assert torch.allclose(train_a.y, torch.full((4,), 0.25 * expected_scale))
    assert torch.allclose(train_b.y, torch.full((4,), 0.5 * expected_scale))
    assert torch.allclose(
        dataset_validation.y, torch.full((4,), 0.125 * expected_scale)
    )


def test_normalize_joint_dataset_output_rejects_zero_training_output():
    with pytest.raises(data.JointDatasetValidationError, match="all zeroes"):
        data.NormalizeJointDatasetOutput(-18.0).apply(
            dataset_train=_dataset_with_constant_y(0.0),
            dataset_validation=_dataset_with_constant_y(0.125),
        )


def test_audio_mismatch_shapes_in_order():
    """
    https://github.com/sdatkinson/neural-amp-modeler/issues/257
    """
    x_samples, y_samples = 5, 7
    num_channels = 1

    x, y = [np.zeros((n, num_channels)) for n in (x_samples, y_samples)]

    with TemporaryDirectory() as tmpdir:
        y_path = Path(tmpdir, "y.wav")
        data.np_to_wav(y, y_path)
        f = lambda: data.wav_to_np(y_path, required_shape=x.shape)

        with pytest.raises(data.AudioShapeMismatchError) as e:
            f()

        try:
            f()
            assert False, "Shouldn't have succeeded!"
        except data.AudioShapeMismatchError as e:
            # x is loaded first; we expect that y matches.
            assert e.shape_expected == (x_samples, num_channels)
            assert e.shape_actual == (y_samples, num_channels)


def test_register_dataset_initializer():
    """
    Assert that you can add and use new data sets
    """

    class MyDataset(data.Dataset):
        pass

    name = "my_dataset"

    data.register_dataset_initializer(name, MyDataset.init_from_config)

    x = np.random.rand(32) - 0.5
    y = x
    split = data.Split.TRAIN

    with TemporaryDirectory() as tmpdir:
        x_path = Path(tmpdir, "x.wav")
        y_path = Path(tmpdir, "y.wav")
        data.np_to_wav(x, x_path)
        data.np_to_wav(y, y_path)
        config = {
            "type": name,
            split.value: {
                "x_path": str(x_path),
                "y_path": str(y_path),
                "nx": 3,
                "ny": 2,
            },
        }
        dataset = data.init_dataset(config, split)
    assert isinstance(dataset, MyDataset)


if __name__ == "__main__":
    pytest.main()
