# File: data.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import wavio
from torch.utils.data import Dataset as _Dataset
from tqdm import tqdm

from ._core import InitializableFromConfig

_REQUIRED_SAMPWIDTH = 3
REQUIRED_RATE = 48_000
_REQUIRED_CHANNELS = 1  # Mono


class Split(Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class WavInfo:
    sampwidth: int
    rate: int


def wav_to_np(
    filename: Union[str, Path],
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
    assert x_wav.sampwidth == _REQUIRED_SAMPWIDTH, "24-bit"
    assert x_wav.rate == REQUIRED_RATE, "48 kHz"

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
            raise ValueError(
                f"Mismatched shapes {arr_premono.shape} versus {required_shape}"
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
    scale="none",
):
    wavio.write(
        str(filename),
        (np.clip(x, -1.0, 1.0) * (2 ** (8 * sampwidth - 1))).astype(np.int32),
        rate,
        scale=scale,
        sampwidth=sampwidth,
    )


class AbstractDataset(_Dataset, abc.ABC):
    @abc.abstractmethod
    def __getitem__(
        self, idx
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        pass


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
        delay: Optional[int] = None,
        y_scale: float = 1.0,
        x_path: Optional[Union[str, Path]] = None,
        y_path: Optional[Union[str, Path]] = None,
    ):
        """
        :param start: In samples
        :param stop: In samples
        :param delay: In samples. Positive means we get rid of the start of x, end of y.
        """
        x, y = [z[start:stop] for z in (x, y)]
        if delay is not None:
            if delay > 0:
                x = x[:-delay]
                y = y[delay:]
            elif delay < 0:
                x = x[-delay:]
                y = y[:delay]
        y = y * y_scale
        self._x_path = x_path
        self._y_path = y_path
        self._validate_inputs(x, y, nx, ny)
        self._x = x
        self._y = y
        self._nx = nx
        self._ny = ny if ny is not None else len(x) - nx + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def y_offset(self) -> int:
        return self._nx - 1

    @classmethod
    def parse_config(cls, config):
        x, x_wavinfo = wav_to_tensor(config["x_path"], info=True)
        y = wav_to_tensor(
            config["y_path"],
            preroll=config.get("y_preroll"),
            required_shape=(len(x), 1),
            required_wavinfo=x_wavinfo,
        )
        return {
            "x": x,
            "y": y,
            "nx": config["nx"],
            "ny": config["ny"],
            "start": config.get("start"),
            "stop": config.get("stop"),
            "delay": config.get("delay"),
            "y_scale": config.get("y_scale", 1.0),
            "x_path": config["x_path"],
            "y_path": config["y_path"],
        }

    def _validate_inputs(self, x, y, nx, ny):
        assert x.ndim == 1
        assert y.ndim == 1
        assert len(x) == len(y)
        if nx > len(x):
            raise RuntimeError(f"Input of length {len(x)}, but receptive field is {nx}.")
        if ny is not None:
            assert ny <= len(y) - nx + 1
        if torch.abs(y).max() >= 1.0:
            msg = "Output clipped."
            if self._y_path is not None:
                msg += f"Source is {self._y_path}"
            raise ValueError(msg)


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
        return sum(len(d) for d in self._datasets)

    @property
    def datasets(self):
        return self._datasets

    @classmethod
    def parse_config(cls, config):
        init = (
            ParametricDataset.init_from_config
            if config["parametric"]
            else Dataset.init_from_config
        )
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
        for i in len(self):
            if offset == len(self.datasets[j]):
                offset -= len(self.datasets[j])
                j += 1
            lookup[i] = (j, offset)
            offset += 1
        assert j == len(self.datasets)
        assert offset == 1
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


def init_dataset(config, split: Split) -> AbstractDataset:
    parametric = config.get("parametric", False)
    base_config = config[split.value]
    common = config.get("common", {})
    if isinstance(base_config, dict):
        init = (
            ParametricDataset.init_from_config
            if parametric
            else Dataset.init_from_config
        )
        return init({**common, **base_config})
    elif isinstance(base_config, list):
        return ConcatDataset.init_from_config(
            {
                "parametric": parametric,
                "dataset_configs": [{**common, **c} for c in base_config],
            }
        )
