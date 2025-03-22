# File: data.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Functions and classes for working with audio data with NAM
"""

import abc as _abc
import logging as _logging
from collections import namedtuple as _namedtuple
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from pathlib import Path as _Path
from typing import (
    Any as _Any,
    Callable as _Callable,
    Dict as _Dict,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import numpy as _np
import torch as _torch
import wavio as _wavio
from torch.utils.data import Dataset as _Dataset
from tqdm import tqdm as _tqdm

from ._core import InitializableFromConfig as _InitializableFromConfig

logger = _logging.getLogger(__name__)

_REQUIRED_CHANNELS = 1  # Mono


class Split(_Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@_dataclass
class WavInfo:
    sampwidth: int
    rate: int


class DataError(Exception):
    """
    Parent class for all special exceptions raised by NAM data sets
    """

    pass


class AudioShapeMismatchError(ValueError, DataError):
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
    filename: _Union[str, _Path],
    rate: _Optional[int] = None,
    require_match: _Optional[_Union[str, _Path]] = None,
    required_shape: _Optional[_Tuple[int, ...]] = None,
    required_wavinfo: _Optional[WavInfo] = None,
    preroll: _Optional[int] = None,
    info: bool = False,
) -> _Union[_np.ndarray, _Tuple[_np.ndarray, WavInfo]]:
    """
    :param filename: Where to load from
    :param rate: Expected sample rate. `None` allows for anything.
    :param require_match: If not `None`, assert that the data you get matches the shape
        and other characteristics of another audio file at the provided location
    :param required_shape: If not `None`, assert that the audio loaded is of shape
        `(num_samples, num_channels)`.
    :param required_wavinfo: If not `None`, assert that the WAV info of the laoded audio
        matches that provided.
    :param preroll: Drop this many samples off the front
    :param info: If `True`, also return the WAV info of this file.
    """
    x_wav = _wavio.read(str(filename))
    assert x_wav.data.shape[1] == _REQUIRED_CHANNELS, "Mono"
    if rate is not None and x_wav.rate != rate:
        raise RuntimeError(
            f"Explicitly expected sample rate of {rate}, but found {x_wav.rate} in "
            f"file {filename}!"
        )

    if require_match is not None:
        assert required_shape is None
        assert required_wavinfo is None
        y_wav = _wavio.read(str(require_match))
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
) -> _Union[_torch.Tensor, _Tuple[_torch.Tensor, WavInfo]]:
    out = wav_to_np(*args, info=info, **kwargs)
    if info:
        arr, info = out
        return _torch.Tensor(arr), info
    else:
        arr = out
        return _torch.Tensor(arr)


def tensor_to_wav(x: _torch.Tensor, *args, **kwargs):
    np_to_wav(x.detach().cpu().numpy(), *args, **kwargs)


def np_to_wav(
    x: _np.ndarray,
    filename: _Union[str, _Path],
    rate: int = 48_000,
    sampwidth: int = 3,
    scale=None,
    **kwargs,
):
    if _wavio.__version__ <= "0.0.4" and scale is None:
        scale = "none"
    _wavio.write(
        str(filename),
        (_np.clip(x, -1.0, 1.0) * (2 ** (8 * sampwidth - 1))).astype(_np.int32),
        rate,
        scale=scale,
        sampwidth=sampwidth,
        **kwargs,
    )


class AbstractDataset(_Dataset, _abc.ABC):
    @_abc.abstractmethod
    def __getitem__(self, idx: int):
        """
        Get input and output audio segment for training / evaluation.
        :return:
        """
        pass


class XYError(ValueError, DataError):
    """
    Exceptions related to invalid x and y provided for data sets
    """

    pass


class StartStopError(ValueError, DataError):
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


def _sample_to_time(s, rate):
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
    return f"{hours}:{minutes:02d}:{seconds:02d} and {remainder} samples"


class Dataset(AbstractDataset, _InitializableFromConfig):
    """
    Take a pair of matched audio files and serve input + output pairs.
    """

    def __init__(
        self,
        x: _torch.Tensor,
        y: _torch.Tensor,
        nx: int,
        ny: _Optional[int],
        start: _Optional[int] = None,
        stop: _Optional[int] = None,
        start_samples: _Optional[int] = None,
        stop_samples: _Optional[int] = None,
        start_seconds: _Optional[_Union[int, float]] = None,
        stop_seconds: _Optional[_Union[int, float]] = None,
        delay: _Optional[_Union[int, float]] = None,
        y_scale: float = 1.0,
        x_path: _Optional[_Union[str, _Path]] = None,
        y_path: _Optional[_Union[str, _Path]] = None,
        input_gain: float = 0.0,
        sample_rate: _Optional[float] = None,
        require_input_pre_silence: _Optional[
            float
        ] = _DEFAULT_REQUIRE_INPUT_PRE_SILENCE,
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
        :param start: [DEPRECATED; use start_samples instead.] In samples; clip x and y
            at this point. Negative values are taken from the end of the audio.
        :param stop: [DEPRECATED; use stop_samples instead.] In samples; clip x and y at
            this point. Negative values are taken from the end of the audio.
        :param start_samples: Clip x and y at this point. Negative values are taken from
            the end of the audio.
        :param stop: Clip x and y at this point. Negative values are taken from the end
            of the audio.
        :param start_seconds: Clip x and y at this point. Negative values are taken from
            the end of the audio. Requires providing `sample_rate`.
        :param stop_seconds: Clip x and y at this point. Negative values are taken from
            the end of the audio. Requires providing `sample_rate`.
        :param delay: In samples. Positive means we get rid of the start of x, end of y
            (i.e. we are correcting for an alignment error in which y is delayed behind
            x). Only integer delays are supported.
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
        :param require_input_pre_silence: If provided, require that this much time (in
            seconds) preceding the start of the data set (`start`) have a silent input.
            If it's not, then raise an exception because the output due to it will leak
            into the data set that we're trying to use. If `None`, don't assert.
        """
        self._validate_x_y(x, y)
        self._sample_rate = sample_rate
        start, stop = self._validate_start_stop(
            x,
            y,
            start,
            stop,
            start_samples,
            stop_samples,
            start_seconds,
            stop_seconds,
            self.sample_rate,
        )
        if require_input_pre_silence is not None:
            self._validate_preceding_silence(
                x, start, require_input_pre_silence, self.sample_rate
            )
        x, y = [z[start:stop] for z in (x, y)]
        if delay is not None and delay != 0:
            x, y = self._apply_delay(x, y, delay)
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

    def __getitem__(self, idx: int) -> _Tuple[_torch.Tensor, _torch.Tensor]:
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
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def sample_rate(self) -> _Optional[float]:
        return self._sample_rate

    @property
    def x(self) -> _torch.Tensor:
        """
        The input audio data

        :return: (N,)
        """
        return self._x

    @property
    def y(self) -> _torch.Tensor:
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
        """
        :param config:
            Must contain:
                x_path (path-like)
                y_path (path-like)
            May contain:
                sample_rate (int)
                y_preroll (int)
                allow_unequal_lengths (bool)
            Must NOT contain:
                x (torch.Tensor) - loaded from x_path
                y (torch.Tensor) - loaded from y_path
            Everything else is passed on to __init__
        """
        config = _deepcopy(config)
        sample_rate = config.pop("sample_rate", None)
        x, x_wavinfo = wav_to_tensor(config.pop("x_path"), info=True, rate=sample_rate)
        sample_rate = x_wavinfo.rate
        if config.pop("allow_unequal_lengths", False):
            y = wav_to_tensor(
                config.pop("y_path"),
                rate=sample_rate,
                preroll=config.pop("y_preroll", None),
                required_wavinfo=x_wavinfo,
            )
            # Truncate to the shorter of the two
            if len(x) == 0:
                raise DataError("Input is zero-length!")
            if len(y) == 0:
                raise DataError("Output is zero-length!")
            n = min(len(x), len(y))
            if n < len(x):
                print(f"Truncating input to {_sample_to_time(n, sample_rate)}")
            if n < len(y):
                print(f"Truncating output to {_sample_to_time(n, sample_rate)}")
            x, y = [z[:n] for z in (x, y)]
        else:
            try:
                y = wav_to_tensor(
                    config.pop("y_path"),
                    rate=sample_rate,
                    preroll=config.pop("y_preroll", None),
                    required_shape=(len(x), 1),
                    required_wavinfo=x_wavinfo,
                )
            except AudioShapeMismatchError as e:
                # Really verbose message since users see this.
                x_samples, x_channels = e.shape_expected
                y_samples, y_channels = e.shape_actual
                msg = "Your audio files aren't the same shape as each other!"
                if x_channels != y_channels:
                    channels_to_stereo_mono = {1: "mono", 2: "stereo"}
                    msg += f"\n * The input is {channels_to_stereo_mono[x_channels]}, but the output is {channels_to_stereo_mono[y_channels]}!"
                if x_samples != y_samples:
                    msg += f"\n * The input is {_sample_to_time(x_samples, sample_rate)} long"
                    msg += f"\n * The output is {_sample_to_time(y_samples, sample_rate)} long"
                    msg += f"\n\nOriginal exception:\n{e}"
                raise DataError(msg)
        return {"x": x, "y": y, "sample_rate": sample_rate, **config}

    @classmethod
    def _apply_delay(
        cls,
        x: _torch.Tensor,
        y: _torch.Tensor,
        delay: _Union[int, float],
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        # Check for floats that could be treated like ints (simpler algorithm)
        if isinstance(delay, float) and int(delay) == delay:
            delay = int(delay)
        if isinstance(delay, int):
            return cls._apply_delay_int(x, y, delay)
        else:
            raise TypeError(type(delay))

    @classmethod
    def _apply_delay_int(
        cls, x: _torch.Tensor, y: _torch.Tensor, delay: int
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        if delay > 0:
            x = x[:-delay]
            y = y[delay:]
        elif delay < 0:
            x = x[-delay:]
            y = y[:delay]
        return x, y

    @classmethod
    def _validate_start_stop(
        cls,
        x: _torch.Tensor,
        y: _torch.Tensor,
        start: _Optional[int] = None,
        stop: _Optional[int] = None,
        start_samples: _Optional[int] = None,
        stop_samples: _Optional[int] = None,
        start_seconds: _Optional[_Union[int, float]] = None,
        stop_seconds: _Optional[_Union[int, float]] = None,
        sample_rate: _Optional[int] = None,
    ) -> _Tuple[_Optional[int], _Optional[int]]:
        """
        Parse the requested start and stop trim points.

        These may be valid indices in Python, but probably point to invalid usage, so
        we will raise an exception if something fishy is going on (e.g. starting after
        the end of the file, etc)

        :return: parsed start/stop (if valid).
        """

        def parse_start_stop(s, samples, seconds, rate):
            # Assumes validated inputs
            if s is not None:
                return s
            if samples is not None:
                return samples
            if seconds is not None:
                return int(seconds * rate)
            # else
            return None

        # Resolve different ways of asking for start/stop...
        if start is not None:
            logger.warning("Using `start` is deprecated; use `start_samples` instead.")
        if start is not None:
            logger.warning("Using `stop` is deprecated; use `start_samples` instead.")
        if (
            int(start is not None)
            + int(start_samples is not None)
            + int(start_seconds is not None)
            >= 2
        ):
            raise ValueError(
                "More than one start provided. Use only one of `start`, `start_samples`, or `start_seconds`!"
            )
        if (
            int(stop is not None)
            + int(stop_samples is not None)
            + int(stop_seconds is not None)
            >= 2
        ):
            raise ValueError(
                "More than one stop provided. Use only one of `stop`, `stop_samples`, or `stop_seconds`!"
            )
        if start_seconds is not None and sample_rate is None:
            raise ValueError(
                "Provided `start_seconds` without sample rate; cannot resolve into samples!"
            )
        if stop_seconds is not None and sample_rate is None:
            raise ValueError(
                "Provided `stop_seconds` without sample rate; cannot resolve into samples!"
            )

        # By this point, we should have a valid, unambiguous way of asking.
        start = parse_start_stop(start, start_samples, start_seconds, sample_rate)
        stop = parse_start_stop(stop, stop_samples, stop_seconds, sample_rate)
        # And only use start/stop from this point.

        # We could do this whole thing with `if len(x[start: stop]==0`, but being more
        # explicit makes the error messages better for users.
        if start is None and stop is None:
            return start, stop
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
        return start, stop

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
        if _torch.abs(y).max() >= 1.0:
            msg = "Output clipped."
            if self._y_path is not None:
                msg += f"Source is {self._y_path}"
            raise ValueError(msg)

    @classmethod
    def _validate_preceding_silence(
        cls,
        x: _torch.Tensor,
        start: _Optional[int],
        silent_seconds: float,
        sample_rate: _Optional[float],
    ):
        """
        Make sure that the input is silent before the starting index.
        If it's not, then the output from that non-silent input will leak into the data
        set and couldn't be predicted!

        This assumes that silence is indeed required. If it's not, then don't call this!

        See: Issue #252

        :param x: Input
        :param start: Where the data starts
        :param silent_samples: How many are expected to be silent
        """
        if sample_rate is None:
            raise ValueError(
                f"Pre-silence was required for {silent_seconds} seconds, but no sample "
                "rate was provided!"
            )
        silent_samples = int(silent_seconds * sample_rate)
        if start is None:
            return
        raw_check_start = start - silent_samples
        check_start = max(raw_check_start, 0) if start >= 0 else min(raw_check_start, 0)
        check_end = start
        if not _torch.all(x[check_start:check_end] == 0.0):
            raise XYError(
                f"Input provided isn't silent for at least {silent_samples} samples "
                "before the starting index. Responses to this non-silent input may "
                "leak into the dataset!"
            )


class ConcatDatasetValidationError(ValueError):
    """
    Error raised when a ConcatDataset fails validation
    """

    pass


class ConcatDataset(AbstractDataset, _InitializableFromConfig):
    def __init__(self, datasets: _Sequence[Dataset], flatten=True):
        if flatten:
            datasets = self._flatten_datasets(datasets)
        self._validate_datasets(datasets)
        self._datasets = datasets
        self._lookup = self._make_lookup()

    def __getitem__(self, idx: int) -> _Tuple[_torch.Tensor, _torch.Tensor]:
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

    @property
    def nx(self) -> int:
        # Validated at initialization
        return self.datasets[0].nx

    @property
    def ny(self) -> int:
        # Validated at initialization
        return self.datasets[0].ny

    @property
    def sample_rate(self) -> _Optional[float]:
        # This is validated to be consistent across datasets during initialization
        return self.datasets[0].sample_rate

    @classmethod
    def parse_config(cls, config):
        init = _dataset_init_registry[config.get("type", "dataset")]
        return {
            "datasets": tuple(
                init(c) for c in _tqdm(config["dataset_configs"], desc="Loading data")
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
    def _validate_datasets(cls, datasets: _Sequence[Dataset]):
        # Ensure that a couple attrs are consistent across the sub-datasets.
        Reference = _namedtuple("Reference", ("index", "val"))
        references = {name: None for name in ("nx", "ny", "sample_rate")}
        for i, d in enumerate(datasets):
            for name in references.keys():
                this_val = getattr(d, name)
                if references[name] is None:
                    references[name] = Reference(i, this_val)

                if this_val != references[name].val:
                    raise ConcatDatasetValidationError(
                        f"Mismatch between {name} of datasets {references[name].index} "
                        f"({references[name].val}) and {i} ({this_val})"
                    )


_dataset_init_registry = {"dataset": Dataset.init_from_config}


def register_dataset_initializer(
    name: str, constructor: _Callable[[_Any], AbstractDataset], overwrite=False
):
    """
    If you have other data set types, you can register their initializer by name using
    this.

    For example, the basic NAM is registered by default under the name "default", but if
    it weren't, you could register it like this:

    >>> from nam import data
    >>> data.register_dataset_initializer("parametric", MyParametricDataset.init_from_config)

    :param name: The name that'll be used in the config to ask for the data set type
    :param constructor: The constructor that'll be fed the config.
    """
    if name in _dataset_init_registry and not overwrite:
        raise KeyError(
            f"A constructor for dataset name '{name}' is already registered!"
        )
    _dataset_init_registry[name] = constructor


def init_dataset(config, split: Split) -> AbstractDataset:
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
