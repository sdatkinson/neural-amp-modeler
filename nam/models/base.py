# File: _base.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
The foundation of the model without the PyTorch Lightning attributes (losses, training
steps)
"""

import abc as _abc
import importlib as _importlib
import math as _math
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import numpy as _np
import torch as _torch
import torch.nn as _nn
from scipy.signal import sosfilt as _sosfilt

from .._core import InitializableFromConfig as _InitializableFromConfig
from .._handshake import HandshakeError as _HandshakeError
from ..data import wav_to_tensor as _wav_to_tensor
from .exportable import Exportable as _Exportable

# `metadata.loudness` stimulus and measurement
# ---------------------------------------------
# The stimulus (`_resources/loudness_input.wav`) is a 1.75 s "level-sweep burst
# train": 7 bursts of 0.25 s of instrument-shaped pink noise (pink x 2nd-order
# high-pass at 70 Hz x 2nd-order low-pass at 1 kHz) at -39, -35, -31, -27, -23,
# -19 and -15 dBFS RMS, with 10 ms raised-cosine edges, at 48 kHz.
#
# The model's response is K-weighted (ITU-R BS.1770-4 pre-filter), the level of
# each burst is measured in dB with 30 ms trimmed from each edge, and the burst
# levels are averaged *in dB*.  Averaging in dB over a range of input levels is
# what makes the number insensitive to how hard the player happens to hit the
# model; K-weighting makes it track perceived loudness rather than flat RMS.
# The constant offset keeps the median value over a large catalogue of models
# where the previous (1 s pink noise at -15 dBFS, flat RMS) calculation had it,
# so existing players that do `gain = target - loudness` keep working.
#
# Background and evaluation: https://github.com/tone-3000/loudness-research
_LOUDNESS_SAMPLE_RATE = 48_000
_LOUDNESS_NUM_BURSTS = 7
_LOUDNESS_BURST_EDGE_TRIM_SAMPLES = int(0.030 * _LOUDNESS_SAMPLE_RATE)
_LOUDNESS_CALIBRATION_DB = 3.0
# ITU-R BS.1770-4 K-weighting (48 kHz): high-shelf stage, then high-pass stage.
_LOUDNESS_K_WEIGHTING_SOS = _np.array(
    [
        [
            1.53512485958697,
            -2.69169618940638,
            1.19839281085285,
            1.0,
            -1.69065929318241,
            0.73248077421585,
        ],
        [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
    ]
)


def _burst_train_loudness_db(y: _np.ndarray, num_input_samples: int) -> float:
    """
    K-weighted, per-burst, mean-of-dB level of a model's response to the
    loudness stimulus.

    :param y: The model's output (1D).  Its last sample is taken to correspond to
        the last sample of the stimulus.
    :param num_input_samples: Length of the stimulus, used to lay out the bursts.
    :return: Level in dB (-inf if any burst is silent).
    """
    y = _sosfilt(_LOUDNESS_K_WEIGHTING_SOS, y.astype(_np.float64))
    burst_len = num_input_samples // _LOUDNESS_NUM_BURSTS
    # Keep the trim sensible for short (e.g. mocked) stimuli.
    trim = min(_LOUDNESS_BURST_EDGE_TRIM_SAMPLES, burst_len // 4)
    offset = len(y) - num_input_samples
    mean_squares = []
    for b in range(_LOUDNESS_NUM_BURSTS):
        start = max(b * burst_len + trim + offset, 0)
        stop = max((b + 1) * burst_len - trim + offset, 0)
        segment = y[start:stop]
        mean_squares.append(
            float(_np.mean(_np.square(segment))) if len(segment) else 0.0
        )
    mean_squares = _np.array(mean_squares)
    if _np.any(mean_squares <= 0.0):
        return -_math.inf
    burst_levels_db = -0.691 + 10.0 * _np.log10(mean_squares)
    return float(_np.mean(burst_levels_db)) + _LOUDNESS_CALIBRATION_DB


class ModelDatasetHandshakeError(_HandshakeError):
    """
    Raised if a handshake fails from model to dataset
    """

    pass


class _Base(_nn.Module, _InitializableFromConfig, _Exportable):
    def __init__(self, sample_rate: _Optional[float] = None):
        super().__init__()
        self.register_buffer(
            "_has_sample_rate",
            _torch.tensor(sample_rate is not None, dtype=_torch.bool),
        )
        self.register_buffer(
            "_sample_rate", _torch.tensor(0.0 if sample_rate is None else sample_rate)
        )

    @property
    @_abc.abstractmethod
    def pad_start_default(self) -> bool:
        pass

    @property
    @_abc.abstractmethod
    def receptive_field(self) -> int:
        """
        Receptive field of the model
        """
        pass

    @_abc.abstractmethod
    def forward(self, *args, **kwargs) -> _torch.Tensor:
        pass

    def handshake(self, dataset: "nam.data.AbstractDataset"):  # noqa: F821
        """
        Perform a handshake with the dataset to ensure that it's compatible.
        Raise a ModelDatasetHandshakeError if the handshake fails.

        :param dataset: The dataset to handshake with.
        """
        from ..data import AbstractDataset

        if not isinstance(dataset, AbstractDataset):
            raise ModelDatasetHandshakeError(
                f"Dataset is not a NAM dataset: {type(dataset)}"
            )

    @classmethod
    def _metadata_loudness_x(cls) -> _torch.Tensor:
        return _wav_to_tensor(
            _importlib.resources.files("nam").joinpath(
                "models/_resources/loudness_input.wav"
            )
        )

    @property
    def device(self) -> _Optional[_torch.device]:
        """
        Helpful property, where the parameters of the model live.
        """
        # We can do this because the models are tiny and I don't expect a NAM to be on
        # multiple devices
        try:
            return next(self.parameters()).device
        except StopIteration:
            return None

    @property
    def sample_rate(self) -> _Optional[float]:
        return self._sample_rate.item() if self._has_sample_rate else None

    @sample_rate.setter
    def sample_rate(self, val: _Optional[float]):
        self._has_sample_rate = _torch.tensor(val is not None, dtype=_torch.bool)
        self._sample_rate = _torch.tensor(0.0 if val is None else val)

    def _get_export_dict(self):
        d = super()._get_export_dict()
        sample_rate_key = "sample_rate"
        if sample_rate_key in d:
            raise RuntimeError(
                "Model wants to put 'sample_rate' into model export dict, but the key "
                "is already taken!"
            )
        # Only write sample_rate when set; C++ reader expects a number or missing key (-1.0).
        if self.sample_rate is not None:
            d[sample_rate_key] = self.sample_rate
        return d

    def _metadata_loudness(self, gain: float = 1.0, db: bool = True) -> float:
        """
        How loud is this model when given a standardized input?
        In dB

        The input is a 1.75 s train of 7 instrument-shaped noise bursts spanning
        -39 to -15 dBFS; the output is K-weighted (BS.1770), measured per burst,
        and the burst levels are averaged in dB, plus a fixed calibration offset
        (see the module-level notes).  Players normalize with
        `gain = target - loudness` (target typically -18 dB).

        :param gain: Multiplies input signal
        :param db: If `False`, return the equivalent linear amplitude instead.
        """
        x = self._metadata_loudness_x().to(self.device)
        y = self._at_nominal_settings(gain * x)
        loudness = _burst_train_loudness_db(
            y.detach().cpu().numpy().reshape(-1), num_input_samples=x.numel()
        )
        if db:
            return loudness
        return 0.0 if loudness == -_math.inf else 10.0 ** (loudness / 20.0)

    def _metadata_gain(self) -> float:
        """
        Between 0 and 1, how much gain / compression does the model seem to have?
        """
        x = _np.linspace(0.0, 1.0, 11)
        y = _np.array([self._metadata_loudness(gain=gain, db=False) for gain in x])
        #
        # O ^ o o o o o o
        # u | o       x   +-------------------------------------+
        # t | o     x     | x: Minimum gain (no compression)    |
        # p | o   x       | o: Max gain     (100% compression)  |
        # u | o x         +-------------------------------------+
        # t | o
        #   +------------->
        #       Input
        #
        max_gain = y[-1] * len(x)  # "Square"
        min_gain = 0.5 * max_gain  # "Triangle"
        gain_range = max_gain - min_gain
        this_gain = y.sum()
        normalized_gain = (this_gain - min_gain) / gain_range
        return _np.clip(normalized_gain, 0.0, 1.0)

    def _at_nominal_settings(self, x: _torch.Tensor) -> _torch.Tensor:
        # parametric?...
        raise NotImplementedError()

    @_abc.abstractmethod
    def _forward(self, *args) -> _torch.Tensor:
        """
        The true forward method.

        :param x: (N,L1)
        :return: (N,L1-RF+1)
        """
        pass

    def _export_input_output_args(self) -> _Tuple[_Any]:
        """
        Create any other args necessesary (e.g. params to eval at)
        """
        return ()

    def _export_input_output(self) -> _Tuple[_np.ndarray, _np.ndarray]:
        args = self._export_input_output_args()
        rate = self.sample_rate
        if rate is None:
            raise RuntimeError(
                "Cannot export model's input and output without a sample rate."
            )
        x = _torch.cat(
            [
                _torch.zeros((rate,)),
                0.5
                * _torch.sin(
                    2.0 * _math.pi * 220.0 * _torch.linspace(0.0, 1.0, rate + 1)[:-1]
                ),
                _torch.zeros((rate,)),
            ]
        )
        # Use pad start to ensure same length as requested by ._export_input_output()
        return (
            x.detach().cpu().numpy(),
            self(*args, x, pad_start=True).detach().cpu().numpy(),
        )


def _get_torch_version() -> str:
    return _torch.__version__


class BaseNet(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mps_65536_fallback = False

    @property
    def _mps_fallback_cat_dim(self) -> int:
        return 1

    def forward(self, x: _torch.Tensor, pad_start: _Optional[bool] = None, **kwargs):
        pad_start = self.pad_start_default if pad_start is None else pad_start
        scalar = x.ndim == 1
        if scalar:
            x = x[None]
        if pad_start:
            x = _torch.cat(
                (_torch.zeros((len(x), self.receptive_field - 1)).to(x.device), x),
                dim=1,
            )
        if x.shape[1] < self.receptive_field:
            raise ValueError(
                f"Input has {x.shape[1]} samples, which is too few for this model with "
                f"receptive field {self.receptive_field}!"
            )
        y = self._forward_mps_safe(x, **kwargs)
        if scalar:
            y = y[0]
        return y

    def _at_nominal_settings(self, x: _torch.Tensor) -> _torch.Tensor:
        return self(x)

    def _forward_mps_safe(self, x: _torch.Tensor, **kwargs) -> _torch.Tensor:
        """
        Wrap `._forward()` to protect against MPS-unsupported input lengths
        beyond 65,536 samples.

        Check this again when PyTorch 2.5.2 is released--hopefully it's fixed
        then.
        """
        if not self._mps_65536_fallback:
            try:
                return self._forward(x, **kwargs)
            except NotImplementedError as e:
                if "Output channels > 65536 not supported at the MPS device." in str(e):
                    msg = (
                        "Warning: NAM encountered a bug in PyTorch's MPS backend and "
                        "will switch to a fallback."
                    )
                    known_bad_versions = {"2.5.0", "2.5.1"}
                    torch_version = _get_torch_version()
                    if torch_version not in known_bad_versions:
                        msg += (
                            "\n"
                            f"Your version of PyTorch is {torch_version}, which "
                            "wasn't known to have this problem.\n"
                            "Please open an Issue at:\n"
                            "https://github.com/sdatkinson/neural-amp-modeler/issues/507"
                            "\n"
                            f"and report your PyTorch version ({torch_version}) "
                            "so that we can keep track of versions of PyTorch that "
                            "might be avoided."
                        )
                    print(msg)
                    self._mps_65536_fallback = True
                    return self._forward_mps_safe(x, **kwargs)
                else:
                    raise e
        else:
            # Stitch together the output one piece at a time to avoid the MPS error
            stride = 65_536 - (self.receptive_field - 1)
            # We need to make sure that the last segment is big enough that we have the required history for the receptive field.
            out_list = []
            for i in range(0, x.shape[1], stride):
                j = min(i + 65_536, x.shape[1])
                xi = x[:, i:j]
                out_list.append(self._forward(xi, **kwargs))
                # Bit hacky, but correct.
                if j == x.shape[1]:
                    break
            return _torch.cat(out_list, dim=self._mps_fallback_cat_dim)

    @_abc.abstractmethod
    def _forward(self, x: _torch.Tensor, **kwargs) -> _torch.Tensor:
        """
        The true forward method.

        :param x: (N,L1)
        :return: (N,L1-RF+1)
        """
        pass

    def _get_non_user_metadata(self) -> _Dict[str, _Union[str, int, float]]:
        d = super()._get_non_user_metadata()
        d["loudness"] = self._metadata_loudness()
        d["gain"] = self._metadata_gain()
        return d
