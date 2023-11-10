# File: data.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
import logging
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import wavio
from scipy.interpolate import interp1d
from torch.utils.data import Dataset as _Dataset
from tqdm import tqdm

from ._core import InitializableFromConfig

logger = logging.getLogger(__name__)

REQUIRED_RATE = 48_000  # FIXME not "required" anymore!
_DEFAULT_RATE = REQUIRED_RATE  # There we go :)
_REQUIRED_CHANNELS = 1  # Mono


class Split(Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class WavInfo:
    sampwidth: int
    rate: int


class AudioShapeMismatchError(ValueError):
    """
    Exception where the shape (number of samples, number of channels) of two audio files
    don't match but were supposed to.
    """

    def __init__(self, shape_expected, shape_actual, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape_expected = shape_expected
        self._shape_actual = shape_actual

    @property
    def shape_expected(self):
        return self._shape_expected

    @property
    def shape_actual(self):
        return self._shape_actual


def wav_to_np(
    filename: Union[str, Path],
    rate: Optional[int] = _DEFAULT_RATE,
    require_match: Optional[Union[str, Path]] = None,
    required_shape: Optional[Tuple[int]] = None,
    required_wavinfo: Optional[WavInfo] = None,
    preroll: Optional[int] = None,
    info: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, WavInfo]]:
    """
    :param preroll: Drop this many samples off the front
    """
    x_wav = wavio.read(str(filename))
    assert x_wav.data.shape[1] == _REQUIRED_CHANNELS, "Mono"
    if rate is not None and x_wav.rate != rate:
        raise RuntimeError(
            f"Explicitly expected sample rate of {rate}, but found {x_wav.rate} in "
            f"file {filename}!"
        )

    if require_match is not None:
        assert required_shape is None
        assert required_wavinfo is None
        y_wav = wavio.read(str(require_match))
        required_shape = y_wav.data.shape
        required_wavinfo = WavInfo(y_wav.sampwidth, y_wav.rate)
    if required_wavinfo is not None:
        if x_wav.rate != required_wavinfo.rate:
            raise ValueError(
                f"Mismatched rates {x_wav.rate} versus {required_wavinfo.rate}"
            )
    arr_premono = x_wav.data[preroll:] / (2.0 ** (8 * x_wav.sampwidth - 1))
    if required_shape is not None:
        if arr_premono.shape != required_shape:
            raise AudioShapeMismatchError(
                required_shape,  # Expected
                arr_premono.shape,  # Actual
                f"Mismatched shapes. Expected {required_shape}, but this is "
                f"{arr_premono.shape}!",
            )
        # sampwidth fine--we're just casting to 32-bit float anyways
    arr = arr_premono[:, 0]
    return arr if not info else (arr, WavInfo(x_wav.sampwidth, x_wav.rate))


def wav_to_tensor(
    *args, info: bool = False, **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, WavInfo]]:
    out = wav_to_np(*args, info=info, **kwargs)
    if info:
        arr, info = out
        return torch.Tensor(arr), info
    else:
        arr = out
        return torch.Tensor(arr)


def tensor_to_wav(x: torch.Tensor, *args, **kwargs):
    np_to_wav(x.detach().cpu().numpy(), *args, **kwargs)


def np_to_wav(
    x: np.ndarray,
    filename: Union[str, Path],
    rate: int = 48_000,
    sampwidth: int = 3,
    scale=None,
    **kwargs,
):
    if wavio.__version__ <= "0.0.4" and scale is None:
        scale = "none"
    wavio.write(
        str(filename),
        (np.clip(x, -1.0, 1.0) * (2 ** (8 * sampwidth - 1))).astype(np.int32),
        rate,
        scale=scale,
        sampwidth=sampwidth,
        **kwargs,
    )


class AbstractDataset(_Dataset, abc.ABC):
    @abc.abstractmethod
    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        :return:
            Case 1: Input (N1,), Output (N2,)
            Case 2: Parameters (D,), Input (N1,), Output (N2,)
        """
        pass


class _DelayInterpolationMethod(Enum):
    """
    :param LINEAR: Linear interpolation
    :param CUBIC: Cubic spline interpolation
    """

    # Note: these match scipy.interpolate.interp1d kwarg "kind"
    LINEAR = "linear"
    CUBIC = "cubic"


def _interpolate_delay(
    x: torch.Tensor, delay: float, method: _DelayInterpolationMethod
) -> np.ndarray:
    """
    NOTE: This breaks the gradient tape!
    """
    if delay == 0.0:
        return x
    t_in = np.arange(len(x))
    n_out = len(x) - int(np.ceil(np.abs(delay)))
    if delay > 0:
        t_out = np.arange(n_out) + delay
    elif delay < 0:
        t_out = np.arange(len(x) - n_out, len(x)) - np.abs(delay)

    return torch.Tensor(
        interp1d(t_in, x.detach().cpu().numpy(), kind=method.value)(t_out)
    )


class XYError(ValueError):
    """
    Exceptions related to invalid x and y provided for data sets
    """

    pass


class StartStopError(ValueError):
    """
    Exceptions related to invalid start and stop arguments
    """

    pass


class StartError(StartStopError):
    pass


class StopError(StartStopError):
    pass


# In seconds. Can't be 0.5 or else v1.wav is invalid! Oops!
_DEFAULT_REQUIRE_INPUT_PRE_SILENCE = 0.4


class Dataset(AbstractDataset, InitializableFromConfig):
    """
    Take a pair of matched audio files and serve input + output pairs.

    No conditioning parameters associated w/ the data.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        nx: int,
        ny: Optional[int],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        delay: Optional[Union[int, float]] = None,
        delay_interpolation_method: Union[
            str, _DelayInterpolationMethod
        ] = _DelayInterpolationMethod.CUBIC,
        y_scale: float = 1.0,
        x_path: Optional[Union[str, Path]] = None,
        y_path: Optional[Union[str, Path]] = None,
        input_gain: float = 0.0,
        sample_rate: Optional[int] = None,
        rate: Optional[int] = None,
        require_input_pre_silence: Optional[float] = _DEFAULT_REQUIRE_INPUT_PRE_SILENCE,
    ):
        """
        :param x: The input signal. A 1D array.
        :param y: The associated output from the model. A 1D array.
        :param nx: The number of samples required as input for the model. For example,
            for a ConvNet, this would be the receptive field.
        :param ny: How many samples to provide as the output array for a single "datum".
            It's usually more computationally-efficient to provide a larger `ny` than 1
            so that the forward pass can process more audio all at once. However, this
            shouldn't be too large or else you won't be able to provide a large batch
            size (where each input-output pair could be something substantially
            different and improve batch diversity).
        :param start: In samples; clip x and y up to this point.
        :param stop: In samples; clip x and y past this point.
        :param delay: In samples. Positive means we get rid of the start of x, end of y
            (i.e. we are correcting for an alignment error in which y is delayed behind
            x). If a non-integer delay is provided, then y is interpolated, with
            the extra sample removed.
        :param y_scale: Multiplies the output signal by a factor (e.g. if the data are
            too quiet).
        :param input_gain: In dB. If the input signal wasn't fed to the amp at unity
            gain, you can indicate the gain here. The data set will multipy the raw
            audio file by the specified gain so that the true input signal amplitude
            experienced by the signal chain will be provided as input to the model. If
            you are using a reamping setup, you can estimate this by reamping a
            completely dry signal (i.e. connecting the interface output directly back
            into the input with which the guitar was originally recorded.)
        :param sample_rate: Sample rate for the data
        :param rate: Sample rate for the data (deprecated)
        :param require_input_pre_silence: If provided, require that this much time (in
            seconds) preceding the start of the data set (`start`) have a silent input.
            If it's not, then raise an exception because the output due to it will leak
            into the data set that we're trying to use. If `None`, don't assert.
        """
        self._validate_x_y(x, y)
        self._validate_start_stop(x, y, start, stop)
        self._sample_rate = self._validate_sample_rate(sample_rate, rate)
        if not isinstance(delay_interpolation_method, _DelayInterpolationMethod):
            delay_interpolation_method = _DelayInterpolationMethod(
                delay_interpolation_method
            )
        if require_input_pre_silence is not None:
            self._validate_preceding_silence(
                x, start, int(require_input_pre_silence * self._sample_rate)
            )
        x, y = [z[start:stop] for z in (x, y)]
        if delay is not None and delay != 0:
            x, y = self._apply_delay(x, y, delay, delay_interpolation_method)
        x_scale = 10.0 ** (input_gain / 20.0)
        x = x * x_scale
        y = y * y_scale
        self._x_path = x_path
        self._y_path = y_path
        self._validate_inputs_after_processing(x, y, nx, ny)
        self._x = x
        self._y = y
        self._nx = nx
        self._ny = ny if ny is not None else len(x) - nx + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return:
            Input (NX+NY-1,)
            Output (NY,)
        """
        if idx >= len(self):
            raise IndexError(f"Attempted to access datum {idx}, but len is {len(self)}")
        i = idx * self._ny
        j = i + self.y_offset
        return self.x[i : i + self._nx + self._ny - 1], self.y[j : j + self._ny]

    def __len__(self) -> int:
        n = len(self.x)
        # If ny were 1
        single_pairs = n - self._nx + 1
        return single_pairs // self._ny

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate

    @property
    def x(self) -> torch.Tensor:
        """
        The input audio data

        :return: (N,)
        """
        return self._x

    @property
    def y(self) -> torch.Tensor:
        """
        The output audio data

        :return: (N,)
        """
        return self._y

    @property
    def y_offset(self) -> int:
        return self._nx - 1

    @classmethod
    def parse_config(cls, config):
        x, x_wavinfo = wav_to_tensor(
            config["x_path"], info=True, rate=config.get("rate")
        )
        rate = x_wavinfo.rate
        try:
            y = wav_to_tensor(
                config["y_path"],
                rate=rate,
                preroll=config.get("y_preroll"),
                required_shape=(len(x), 1),
                required_wavinfo=x_wavinfo,
            )
        except AudioShapeMismatchError as e:
            # Really verbose message since users see this.
            x_samples, x_channels = e.shape_expected
            y_samples, y_channels = e.shape_actual
            msg = "Your audio files aren't the same shape as each other!"
            if x_channels != y_channels:
                ctosm = {1: "mono", 2: "stereo"}
                msg += f"\n * The input is {ctosm[x_channels]}, but the output is {ctosm[y_channels]}!"
            if x_samples != y_samples:

                def sample_to_time(s, rate):
                    seconds = s // rate
                    remainder = s % rate
                    hours, minutes = 0, 0
                    seconds_per_hour = 3600
                    while seconds >= seconds_per_hour:
                        hours += 1
                        seconds -= seconds_per_hour
                    seconds_per_minute = 60
                    while seconds >= seconds_per_minute:
                        minutes += 1
                        seconds -= seconds_per_minute
                    return (
                        f"{hours}:{minutes:02d}:{seconds:02d} and {remainder} samples"
                    )

                msg += f"\n * The input is {sample_to_time(x_samples, rate)} long"
                msg += f"\n * The output is {sample_to_time(y_samples, rate)} long"
            raise ValueError(msg)
        return {
            "x": x,
            "y": y,
            "nx": config["nx"],
            "ny": config["ny"],
            "start": config.get("start"),
            "stop": config.get("stop"),
            "delay": config.get("delay"),
            "delay_interpolation_method": config.get(
                "delay_interpolation_method", _DelayInterpolationMethod.CUBIC.value
            ),
            "y_scale": config.get("y_scale", 1.0),
            "x_path": config["x_path"],
            "y_path": config["y_path"],
            "sample_rate": rate,
            "require_input_pre_silence": config.get(
                "require_input_pre_silence", _DEFAULT_REQUIRE_INPUT_PRE_SILENCE
            ),
        }

    @classmethod
    def _apply_delay(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        delay: Union[int, float],
        method: _DelayInterpolationMethod,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check for floats that could be treated like ints (simpler algorithm)
        if isinstance(delay, float) and int(delay) == delay:
            delay = int(delay)
        if isinstance(delay, int):
            return cls._apply_delay_int(x, y, delay)
        elif isinstance(delay, float):
            return cls._apply_delay_float(x, y, delay, method)

    @classmethod
    def _apply_delay_int(
        cls, x: torch.Tensor, y: torch.Tensor, delay: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if delay > 0:
            x = x[:-delay]
            y = y[delay:]
        elif delay < 0:
            x = x[-delay:]
            y = y[:delay]
        return x, y

    @classmethod
    def _apply_delay_float(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        delay: float,
        method: _DelayInterpolationMethod,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_out = len(y) - int(np.ceil(np.abs(delay)))
        if delay > 0:
            x = x[:n_out]
        elif delay < 0:
            x = x[-n_out:]
        y = _interpolate_delay(y, delay, method)
        return x, y

    @classmethod
    def _validate_sample_rate(
        cls, sample_rate: Optional[float], rate: Optional[int]
    ) -> float:
        if sample_rate is None and rate is None:  # Default value
            return _DEFAULT_RATE
        if rate is not None:
            if sample_rate is not None:
                raise ValueError(
                    "Provided both sample_rate and rate. Provide only sample_rate!"
                )
            else:
                logger.warning(
                    "Use of 'rate' is deprecated and will be removed. Use sample_rate instead"
                )
                return float(rate)
        else:
            return sample_rate

    @classmethod
    def _validate_start_stop(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        """
        Check for potential input errors.

        These may be valid indices in Python, but probably point to invalid usage, so
        we will raise an exception if something fishy is going on (e.g. starting after
        the end of the file, etc)
        """
        # We could do this whole thing with `if len(x[start: stop]==0`, but being more
        # explicit makes the error messages better for users.
        if start is None and stop is None:
            return
        if len(x) != len(y):
            raise ValueError(
                f"Input and output are different length. Input has {len(x)} samples, "
                f"and output has {len(y)}"
            )
        n = len(x)
        if start is not None:
            # Start after the files' end?
            if start >= n:
                raise StartError(
                    f"Arrays are only {n} samples long, but start was provided as {start}, "
                    "which is beyond the end of the array!"
                )
            # Start before the files' beginning?
            if start < -n:
                raise StartError(
                    f"Arrays are only {n} samples long, but start was provided as {start}, "
                    "which is before the beginning of the array!"
                )
        if stop is not None:
            # Stop after the files' end?
            if stop > n:
                raise StopError(
                    f"Arrays are only {n} samples long, but stop was provided as {stop}, "
                    "which is beyond the end of the array!"
                )
            # Start before the files' beginning?
            if stop <= -n:
                raise StopError(
                    f"Arrays are only {n} samples long, but stop was provided as {stop}, "
                    "which is before the beginning of the array!"
                )
        # Just in case...
        if len(x[start:stop]) == 0:
            raise StartStopError(
                f"Array length {n} with start={start} and stop={stop} would get "
                "rid of all of the data!"
            )

    @classmethod
    def _validate_x_y(self, x, y):
        if len(x) != len(y):
            raise XYError(
                f"Input and output aren't the same lengths! ({len(x)} vs {len(y)})"
            )
        # TODO channels
        n = len(x)
        if n == 0:
            raise XYError("Input and output are empty!")

    def _validate_inputs_after_processing(self, x, y, nx, ny):
        assert x.ndim == 1
        assert y.ndim == 1
        assert len(x) == len(y)
        if nx > len(x):
            raise RuntimeError(  # TODO XYError?
                f"Input of length {len(x)}, but receptive field is {nx}."
            )
        if ny is not None:
            assert ny <= len(y) - nx + 1
        if torch.abs(y).max() >= 1.0:
            msg = "Output clipped."
            if self._y_path is not None:
                msg += f"Source is {self._y_path}"
            raise ValueError(msg)

    @classmethod
    def _validate_preceding_silence(
        cls, x: torch.Tensor, start: Optional[int], silent_samples: int
    ):
        """
        Make sure that the input is silent before the starting index.
        If it's not, then the output from that non-silent input will leak into the data
        set and couldn't be predicted!

        See: Issue #252

        :param x: Input
        :param start: Where the data starts
        :param silent_samples: How many are expected to be silent
        """
        if start is None:
            return
        raw_check_start = start - silent_samples
        check_start = max(raw_check_start, 0) if start >= 0 else min(raw_check_start, 0)
        check_end = start
        if not torch.all(x[check_start:check_end] == 0.0):
            raise XYError(
                f"Input provided isn't silent for at least {silent_samples} samples "
                "before the starting index. Responses to this non-silent input may "
                "leak into the dataset!"
            )


class ParametricDataset(Dataset):
    """
    Additionally tracks some conditioning parameters
    """

    def __init__(self, params: Dict[str, Union[bool, float, int]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = sorted(tuple(k for k in params.keys()))
        self._vals = torch.Tensor([float(params[k]) for k in self._keys])

    @classmethod
    def init_from_config(cls, config):
        if "slices" not in config:
            return super().init_from_config(config)
        else:
            return cls.init_from_config_with_slices(config)

    @classmethod
    def init_from_config_with_slices(cls, config):
        config, x, y, slices = cls.parse_config_with_slices(config)
        datasets = []
        for s in tqdm(slices, desc="Slices..."):
            c = deepcopy(config)
            start, stop, params = [s[k] for k in ("start", "stop", "params")]
            c.update(x=x[start:stop], y=y[start:stop], params=params)
            if "delay" in s:
                c["delay"] = s["delay"]
            datasets.append(ParametricDataset(**c))
        return ConcatDataset(datasets)

    @classmethod
    def parse_config(cls, config):
        assert "slices" not in config
        params = config["params"]
        return {
            "params": params,
            "id": config.get("id"),
            "common_params": config.get("common_params"),
            "param_map": config.get("param_map"),
            **super().parse_config(config),
        }

    @classmethod
    def parse_config_with_slices(cls, config):
        slices = config["slices"]
        config = super().parse_config(config)
        x, y = [config.pop(k) for k in "xy"]
        return config, x, y, slices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :return:
            Parameter values (D,)
            Input (NX+NY-1,)
            Output (NY,)
        """
        # FIXME don't override signature
        x, y = super().__getitem__(idx)
        return self.vals, x, y

    @property
    def keys(self) -> Tuple[str]:
        return self._keys

    @property
    def vals(self):
        return self._vals


class ConcatDataset(AbstractDataset, InitializableFromConfig):
    def __init__(self, datasets: Sequence[Dataset], flatten=True):
        if flatten:
            datasets = self._flatten_datasets(datasets)
        self._validate_datasets(datasets)
        self._datasets = datasets
        self._lookup = self._make_lookup()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j = self._lookup[idx]
        return self.datasets[i][j]

    def __len__(self) -> int:
        """
        How many data sets are in this data set
        """
        return sum(len(d) for d in self._datasets)

    @property
    def datasets(self):
        return self._datasets

    @classmethod
    def parse_config(cls, config):
        init = _dataset_init_registry[config.get("type", "dataset")]
        return {
            "datasets": tuple(
                init(c) for c in tqdm(config["dataset_configs"], desc="Loading data")
            )
        }

    def _flatten_datasets(self, datasets):
        """
        If any dataset is a ConcatDataset, pull it out
        """
        flattened = []
        for d in datasets:
            if isinstance(d, ConcatDataset):
                flattened.extend(d.datasets)
            else:
                flattened.append(d)
        return flattened

    def _make_lookup(self):
        """
        For faster __getitem__
        """
        lookup = {}
        offset = 0
        j = 0  # Dataset index
        for i in range(len(self)):
            if offset == len(self.datasets[j]):
                offset -= len(self.datasets[j])
                j += 1
            lookup[i] = (j, offset)
            offset += 1
        # Assert that we got to the last data set
        if j != len(self.datasets) - 1:
            raise RuntimeError(
                f"During lookup population, didn't get to the last dataset (index "
                f"{len(self.datasets)-1}). Instead index ended at {j}."
            )
        if offset != len(self.datasets[-1]):
            raise RuntimeError(
                "During lookup population, didn't end at the index of the last datum "
                f"in the last dataset. Expected index {len(self.datasets[-1])}, got "
                f"{offset} instead."
            )
        return lookup

    @classmethod
    def _validate_datasets(cls, datasets: Sequence[Dataset]):
        Reference = namedtuple("Reference", ("index", "val"))
        ref_keys, ref_ny = None, None
        for i, d in enumerate(datasets):
            ref_ny = Reference(i, d.ny) if ref_ny is None else ref_ny
            if d.ny != ref_ny.val:
                raise ValueError(
                    f"Mismatch between ny of datasets {ref_ny.index} ({ref_ny.val}) and {i} ({d.ny})"
                )
            if isinstance(d, ParametricDataset):
                val = d.keys
                if ref_keys is None:
                    ref_keys = Reference(i, val)
                if val != ref_keys.val:
                    raise ValueError(
                        f"Mismatch between keys of datasets {ref_keys.index} "
                        f"({ref_keys.val}) and {i} ({val})"
                    )


_dataset_init_registry = {
    "dataset": Dataset.init_from_config,
    "parametric": ParametricDataset.init_from_config,  # To be removed in v0.8
}


def register_dataset_initializer(
    name: str, constructor: Callable[[Any], AbstractDataset]
):
    """
    If you have otehr data set types, you can register their initializer by name using
    this.

    For example, the basic NAM is registered by default under the name "default", but if
    it weren't, you could register it like this:

    >>> from nam import data
    >>> data.register_dataset_initializer("parametric", data.Dataset.init_from_config)

    :param name: The name that'll be used in the config to ask for the data set type
    :param constructor: The constructor that'll be fed the config.
    """
    if name in _dataset_init_registry:
        raise KeyError(
            f"A constructor for dataset name '{name}' is already registered!"
        )
    _dataset_init_registry[name] = constructor


def init_dataset(config, split: Split) -> AbstractDataset:
    if "parametric" in config:
        logger.warning(
            "Using the 'parametric' keyword is deprecated and will be removed in next "
            "version. Instead, register the parametric dataset type using "
            "`nam.data.register_dataset_initializer()` and then specify "
            '`"type": "name"` in the config, using the name you registered.'
        )
        name = "parametric" if config["parametric"] else "dataset"
    else:
        name = config.get("type", "dataset")
    base_config = config[split.value]
    common = config.get("common", {})
    if isinstance(base_config, dict):
        init = _dataset_init_registry[name]
        return init({**common, **base_config})
    elif isinstance(base_config, list):
        return ConcatDataset.init_from_config(
            {
                "type": name,
                "dataset_configs": [{**common, **c} for c in base_config],
            }
        )
