# File: core.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Functions used by the GUI trainer.
"""

import hashlib
import tkinter as tk
from copy import deepcopy
from enum import Enum
from functools import partial
from time import time
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import BaseModel
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

from ..data import REQUIRED_RATE, Split, init_dataset, wav_to_np, wav_to_tensor
from ..models import Model
from ..models.losses import esr
from ..util import filter_warnings
from ._version import Version

__all__ = ["train"]


class Architecture(Enum):
    STANDARD = "standard"
    LITE = "lite"
    FEATHER = "feather"
    NANO = "nano"


def _detect_input_version(input_path) -> Tuple[Version, bool]:
    """
    Check to see if the input matches any of the known inputs

    :return: version, strong match
    """

    def detect_strong(input_path) -> Optional[Version]:
        def assign_hash(path):
            # Use this to create hashes for new files
            md5 = hashlib.md5()
            buffer_size = 65536
            with open(path, "rb") as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    md5.update(data)
            file_hash = md5.hexdigest()
            return file_hash

        file_hash = assign_hash(input_path)
        print(f"Strong hash: {file_hash}")

        version = {
            "4d54a958861bf720ec4637f43d44a7ef": Version(1, 0, 0),
            "7c3b6119c74465f79d96c761a0e27370": Version(1, 1, 1),
            "ede3b9d82135ce10c7ace3bb27469422": Version(2, 0, 0),
            "36cd1af62985c2fac3e654333e36431e": Version(3, 0, 0),
        }.get(file_hash)
        if version is None:
            print(
                f"Provided input file {input_path} does not strong-match any known "
                "standard input files."
            )
        return version

    def detect_weak(input_path) -> Optional[Version]:
        def assign_hash(path):
            Hashes = Tuple[Optional[str], Optional[str]]

            def _hash(x: np.ndarray) -> str:
                return hashlib.md5(x).hexdigest()

            def assign_hashes_v1(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V1_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                t_blips = _V1_DATA_INFO.t_blips
                t_sweep = 3 * rate
                t_white = 3 * rate
                t_validation = _V1_DATA_INFO.t_validate
                # v1 and v2 start with 1 blips, sine sweeps, and white noise
                start_hash = _hash(x[: t_blips + t_sweep + t_white])
                # v1 ends with validation signal
                end_hash = _hash(x[-t_validation:])
                return start_hash, end_hash

            def assign_hashes_v2(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V2_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                t_blips = _V2_DATA_INFO.t_blips
                t_sweep = 3 * rate
                t_white = 3 * rate
                t_validation = _V1_DATA_INFO.t_validate
                # v1 and v2 start with 1 blips, sine sweeps, and white noise
                start_hash = _hash(x[: (t_blips + t_sweep + t_white)])
                # v2 ends with 2x validation & blips
                end_hash = _hash(x[-(2 * t_validation + t_blips) :])
                return start_hash, end_hash

            def assign_hashes_v3(path) -> Hashes:
                # Use this to create recognized hashes for new files
                x, info = wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V3_DATA_INFO.rate:
                    return None, None
                # Times of intervals, in seconds
                # See below.
                end_of_start_interval = 17 * rate  # Start at 0
                start_of_end_interval = -9 * rate
                start_hash = _hash(x[:end_of_start_interval])
                end_hash = _hash(x[start_of_end_interval:])
                return start_hash, end_hash

            start_hash_v1, end_hash_v1 = assign_hashes_v1(path)
            start_hash_v2, end_hash_v2 = assign_hashes_v2(path)
            start_hash_v3, end_hash_v3 = assign_hashes_v3(path)
            return (
                start_hash_v1,
                end_hash_v1,
                start_hash_v2,
                end_hash_v2,
                start_hash_v3,
                end_hash_v3,
            )

        (
            start_hash_v1,
            end_hash_v1,
            start_hash_v2,
            end_hash_v2,
            start_hash_v3,
            end_hash_v3,
        ) = assign_hash(input_path)
        print(
            "Weak hashes:\n"
            f" Start (v1) : {start_hash_v1}\n"
            f" End (v1)   : {end_hash_v1}\n"
            f" Start (v2) : {start_hash_v2}\n"
            f" End (v2)   : {end_hash_v2}\n"
            f" Start (v3) : {start_hash_v3}\n"
            f" End (v3)   : {end_hash_v3}\n"
        )

        # Check for matches, starting with most recent
        version = {
            (
                "dadb5d62f6c3973a59bf01439799809b",
                "8458126969a3f9d8e19a53554eb1fd52",
            ): Version(3, 0, 0)
        }.get((start_hash_v3, end_hash_v3))
        if version is not None:
            return version
        version = {
            (
                "1c4d94fbcb47e4d820bef611c1d4ae65",
                "28694e7bf9ab3f8ae6ef86e9545d4663",
            ): Version(2, 0, 0)
        }.get((start_hash_v2, end_hash_v2))
        if version is not None:
            return version
        version = {
            (
                "bb4e140c9299bae67560d280917eb52b",
                "9b2468fcb6e9460a399fc5f64389d353",
            ): Version(
                1, 0, 0
            ),  # FIXME!
            (
                "9f20c6b5f7fef68dd88307625a573a14",
                "8458126969a3f9d8e19a53554eb1fd52",
            ): Version(1, 1, 1),
        }.get((start_hash_v1, end_hash_v1))
        return version

    version = detect_strong(input_path)
    if version is not None:
        strong_match = True
        return version, strong_match
    print("Falling back to weak-matching...")
    version = detect_weak(input_path)
    if version is None:
        raise ValueError(
            f"Input file at {input_path} cannot be recognized as any known version!"
        )
    strong_match = False
    return version, strong_match


class _DataInfo(BaseModel):
    """
    :param major_version: Data major version
    :param rate: Sample rate, in Hz
    :param t_blips: How long the blips are, in samples
    :param first_blips_start: When the first blips section starts, in samples
    :param t_validate: Validation signal length, in samples
    :param train_start: Where training signal starts, in samples.
    :param validation_start: Where validation signal starts, in samples. Less than zero
        (from the end of the array).
    :param noise_interval: Inside which we quantify the noise level
    :param blip_locations: In samples, absolute location in the file. Negative values
        mean from the end instead of from the start (typical "Python" negastive
        indexing).
    """

    major_version: int
    rate: Optional[int]
    t_blips: int
    first_blips_start: int
    t_validate: int
    train_start: int
    validation_start: int
    noise_interval: Tuple[int, int]
    blip_locations: Sequence[Sequence[int]]


_V1_DATA_INFO = _DataInfo(
    major_version=1,
    rate=REQUIRED_RATE,
    t_blips=48_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-432_000,
    noise_interval=(0, 6000),
    blip_locations=((12_000, 36_000),),
)
# V2:
# (0:00-0:02) Blips at 0:00.5 and 0:01.5
# (0:02-0:05) Chirps
# (0:05-0:07) Noise
# (0:07-2:50.5) General training data
# (2:50.5-2:51) Silence
# (2:51-3:00) Validation 1
# (3:00-3:09) Validation 2
# (3:09-3:11) Blips at 3:09.5 and 3:10.5
_V2_DATA_INFO = _DataInfo(
    major_version=2,
    rate=REQUIRED_RATE,
    t_blips=96_000,
    first_blips_start=0,
    t_validate=432_000,
    train_start=0,
    validation_start=-960_000,  # 96_000 + 2 * 432_000
    noise_interval=(12_000, 18_000),
    blip_locations=((24_000, 72_000), (-72_000, -24_000)),
)
# V3:
# (0:00-0:09) Validation 1
# (0:09-0:10) Silence
# (0:10-0:12) Blips at 0:10.5 and 0:11.5
# (0:12-0:15) Chirps
# (0:15-0:17) Noise
# (0:17-3:00.5) General training data
# (3:00.5-3:01) Silence
# (3:01-3:10) Validation 2
_V3_DATA_INFO = _DataInfo(
    major_version=3,
    rate=REQUIRED_RATE,
    t_blips=96_000,
    first_blips_start=480_000,
    t_validate=432_000,
    train_start=480_000,
    validation_start=-432_000,
    noise_interval=(492_000, 498_000),
    blip_locations=((504_000, 552_000),),
)

_DELAY_CALIBRATION_ABS_THRESHOLD = 0.0003
_DELAY_CALIBRATION_REL_THRESHOLD = 0.001
_DELAY_CALIBRATION_SAFETY_FACTOR = 4


def _warn_lookaheads(indices: Sequence[int]) -> str:
    return (
        f"WARNING: delays from some blips ({','.join([str(i) for i in indices])}) are "
        "at the minimum value possible. This usually means that something is "
        "wrong with your data. Check if trianing ends with a poor result!"
    )


def _calibrate_delay_v_all(
    data_info: _DataInfo,
    y,
    abs_threshold=_DELAY_CALIBRATION_ABS_THRESHOLD,
    rel_threshold=_DELAY_CALIBRATION_REL_THRESHOLD,
    safety_factor=_DELAY_CALIBRATION_SAFETY_FACTOR,
) -> int:
    """
    Calibrate the delay in teh input-output pair based on blips.
    This only uses the blips in the first set of blip locations!

    :param y: The output audio, in complete.
    """

    def report_any_delay_warnings(delays: Sequence[int]):
        # Warnings associated with any single delay:

        lookahead_warnings = [i for i, d in enumerate(delays, 1) if d == -lookahead]
        if len(lookahead_warnings) > 0:
            print(_warn_lookaheads(lookahead_warnings))

        # Ensemble warnings

        # If they're _really_ different, then something might be wrong.
        if np.max(delays) - np.min(delays) >= 20:
            print(
                "WARNING: Delays are anomalously different from each other. If this model "
                "turns out badly, then you might need to provide the delay manually."
            )

    lookahead = 1_000
    lookback = 10_000
    # Calibrate the trigger:
    y = y[data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips]
    background_level = np.max(
        np.abs(
            y[
                data_info.noise_interval[0]
                - data_info.first_blips_start : data_info.noise_interval[1]
                - data_info.first_blips_start
            ]
        )
    )
    trigger_threshold = max(
        background_level + abs_threshold,
        (1.0 + rel_threshold) * background_level,
    )

    delays = []
    for blip_index, i_abs in enumerate(data_info.blip_locations[0], 1):
        # Relative to start of the data
        i_rel = i_abs - data_info.first_blips_start
        start_looking = i_rel - lookahead
        stop_looking = i_rel + lookback
        y_scan = y[start_looking:stop_looking]
        triggered = np.where(np.abs(y_scan) > trigger_threshold)[0]
        if len(triggered) == 0:
            msg = (
                f"No response activated the trigger in response to blip "
                f"{blip_index}. Is something wrong with the reamp?"
            )
            print(msg)
            print("SHARE THIS PLOT IF YOU ASK FOR HELP")
            plt.figure()
            plt.plot(np.arange(-lookahead, lookback), y_scan, label="Signal")
            plt.axvline(x=0, color="C1", linestyle="--", label="Trigger")
            plt.axhline(
                y=-trigger_threshold, color="k", linestyle="--", label="Threshold"
            )
            plt.axhline(y=trigger_threshold, color="k", linestyle="--")
            plt.xlim((-lookahead, lookback))
            plt.xlabel("Samples")
            plt.ylabel("Response")
            plt.legend()
            plt.show()
            raise RuntimeError(msg)
        else:
            j = triggered[0]
            delays.append(j + start_looking - i_rel)

    print("Delays:")
    for i_rel, d in enumerate(delays, 1):
        print(f" Blip {i_rel:2d}: {d}")
    report_any_delay_warnings(delays)

    delay = int(np.min(delays)) - safety_factor
    print(f"After aplying safety factor of {safety_factor}, the final delay is {delay}")
    return delay


_calibrate_delay_v1 = partial(_calibrate_delay_v_all, _V1_DATA_INFO)
_calibrate_delay_v2 = partial(_calibrate_delay_v_all, _V2_DATA_INFO)
_calibrate_delay_v3 = partial(_calibrate_delay_v_all, _V3_DATA_INFO)


def _plot_delay_v_all(
    data_info: _DataInfo, delay: int, input_path: str, output_path: str, _nofail=True
):
    print("Plotting the delay for manual inspection...")
    x = wav_to_np(input_path)[
        data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips
    ]
    y = wav_to_np(output_path)[
        data_info.first_blips_start : data_info.first_blips_start + data_info.t_blips
    ]
    # Only get the blips we really want.
    i = np.where(np.abs(x) > 0.5 * np.abs(x).max())[0]
    if len(i) == 0:
        print("Failed to find the spike in the input file.")
        print(
            "Plotting the input and output; there should be spikes at around the "
            "marked locations."
        )
        t = np.arange(
            data_info.first_blips_start, data_info.first_blips_start + data_info.t_blips
        )
        expected_spikes = data_info.blip_locations[0]  # For v1 specifically
        fig, axs = plt.subplots(len((x, y)), 1)
        for ax, curve in zip(axs, (x, y)):
            ax.plot(t, curve)
            [ax.axvline(x=es, color="C1", linestyle="--") for es in expected_spikes]
        plt.show()
        if _nofail:
            raise RuntimeError("Failed to plot delay")
    else:
        plt.figure()
        di = 20
        # V1's got not a spike but a longer plateau; take the front of it.
        if data_info.major_version == 1:
            i = [i[0]]
        for e, ii in enumerate(i, 1):
            plt.plot(
                np.arange(-di, di),
                y[ii - di + delay : ii + di + delay],
                ".-",
                label=f"Output {e}",
            )
        plt.axvline(x=0, linestyle="--", color="k")
        plt.legend()
        plt.show()  # This doesn't freeze the notebook


_plot_delay_v1 = partial(_plot_delay_v_all, _V1_DATA_INFO)
_plot_delay_v2 = partial(_plot_delay_v_all, _V2_DATA_INFO)
_plot_delay_v3 = partial(_plot_delay_v_all, _V3_DATA_INFO)


def _calibrate_delay(
    delay: Optional[int],
    input_version: Version,
    input_path: str,
    output_path: str,
    silent: bool = False,
) -> int:
    if input_version.major == 1:
        calibrate, plot = _calibrate_delay_v1, _plot_delay_v1
    elif input_version.major == 2:
        calibrate, plot = _calibrate_delay_v2, _plot_delay_v2
    elif input_version.major == 3:
        calibrate, plot = _calibrate_delay_v3, _plot_delay_v3
    else:
        raise NotImplementedError(
            f"Input calibration not implemented for input version {input_version}"
        )
    if delay is not None:
        print(f"Delay is specified as {delay}")
    else:
        print("Delay wasn't provided; attempting to calibrate automatically...")
        delay = calibrate(wav_to_np(output_path))
    if not silent:
        plot(delay, input_path, output_path)
    return delay


def _get_lstm_config(architecture):
    return {
        Architecture.STANDARD: {
            "num_layers": 1,
            "hidden_size": 24,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.LITE: {
            "num_layers": 2,
            "hidden_size": 8,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.FEATHER: {
            "num_layers": 1,
            "hidden_size": 16,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
        Architecture.NANO: {
            "num_layers": 1,
            "hidden_size": 12,
            "train_burn_in": 4096,
            "train_truncate": 512,
        },
    }[architecture]


def _check_v1(*args, **kwargs):
    return True


def _esr_validation_replicate_msg(threshold: float) -> str:
    return (
        f"Validation replicates have a self-ESR of over {threshold}. "
        "Your gear doesn't sound like itself when played twice!\n\n"
        "Possible causes:\n"
        " * Your signal chain is too noisy.\n"
        " * There's a time-based effect (chorus, delay, reverb) turned on.\n"
        " * Some knob got moved while reamping.\n"
        " * You started reamping before the amp had time to warm up fully."
    )


def _check_v2(input_path, output_path, delay: int, silent: bool) -> bool:
    with torch.no_grad():
        print("V2 checks...")
        rate = _V2_DATA_INFO.rate
        y = wav_to_tensor(output_path, rate=rate)
        t_blips = _V2_DATA_INFO.t_blips
        t_validate = _V2_DATA_INFO.t_validate
        y_val_1 = y[-(t_blips + 2 * t_validate) : -(t_blips + t_validate)]
        y_val_2 = y[-(t_blips + t_validate) : -t_blips]
        esr_replicate = esr(y_val_1, y_val_2).item()
        print(f"Replicate ESR is {esr_replicate:.8f}.")
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))

        # Do the blips line up?
        # If the ESR is too bad, then flag it.
        print("Checking blips...")

        def get_blips(y):
            """
            :return: [start/end,replicate]
            """
            i0, i1 = _V2_DATA_INFO.blip_locations[0]
            j0, j1 = _V2_DATA_INFO.blip_locations[1]

            i0, i1, j0, j1 = [i + delay for i in (i0, i1, j0, j1)]
            start = -10
            end = 1000
            blips = torch.stack(
                [
                    torch.stack([y[i0 + start : i0 + end], y[i1 + start : i1 + end]]),
                    torch.stack([y[j0 + start : j0 + end], y[j1 + start : j1 + end]]),
                ]
            )
            return blips

        blips = get_blips(y)
        esr_0 = esr(blips[0][0], blips[0][1]).item()  # Within start
        esr_1 = esr(blips[1][0], blips[1][1]).item()  # Within end
        esr_cross_0 = esr(blips[0][0], blips[1][0]).item()  # 1st repeat, start vs end
        esr_cross_1 = esr(blips[0][1], blips[1][1]).item()  # 2nd repeat, start vs end

        print("  ESRs:")
        print(f"    Start     : {esr_0}")
        print(f"    End       : {esr_1}")
        print(f"    Cross (1) : {esr_cross_0}")
        print(f"    Cross (2) : {esr_cross_1}")

        esr_threshold = 1.0e-2

        def plot_esr_blip_error(
            show_plot: bool,
            msg: str,
            arrays: Sequence[Sequence[float]],
            labels: Sequence[str],
        ):
            """
            :param silent: Whether to make and show a plot about it
            """
            if show_plot:
                plt.figure()
                [plt.plot(array, label=label) for array, label in zip(arrays, labels)]
                plt.xlabel("Sample")
                plt.ylabel("Output")
                plt.legend()
                plt.grid()
            print(msg)
            if show_plot:
                plt.show()
            print(
                "This is known to be a very sensitive test, so training will continue. "
                "If the model doesn't look good, then this may be why!"
            )

        # Check consecutive blips
        show_blip_plots = False
        for e, blip_pair, when in zip((esr_0, esr_1), blips, ("start", "end")):
            if e >= esr_threshold:
                plot_esr_blip_error(
                    show_blip_plots,
                    f"Failed consecutive blip check at {when} of training signal. The "
                    "target tone doesn't seem to be replicable over short timespans."
                    "\n\n"
                    "  Possible causes:\n\n"
                    "    * Your recording setup is really noisy.\n"
                    "    * There's a noise gate that's messing things up.\n"
                    "    * There's a time-based effect (chorus, delay, reverb) in "
                    "the signal chain",
                    blip_pair,
                    ("Replicate 1", "Replicate 2"),
                )
                # return False  # Stop bothering me! :(
        # Check blips between start & end of train signal
        for e, blip_pair, replicate in zip(
            (esr_cross_0, esr_cross_1), blips.permute(1, 0, 2), (1, 2)
        ):
            if e >= esr_threshold:
                plot_esr_blip_error(
                    show_blip_plots,
                    f"Failed start-to-end blip check for blip replicate {replicate}. "
                    "The target tone doesn't seem to be same at the end of the reamp "
                    "as it was at the start. Did some setting change during reamping?",
                    blip_pair,
                    (f"Start, replicate {replicate}", f"End, replicate {replicate}"),
                )
                # return False  # Stop bothering me! :(
        return True


def _check_v3(input_path, output_path, silent: bool, *args, **kwargs) -> bool:
    with torch.no_grad():
        print("V3 checks...")
        rate = _V3_DATA_INFO.rate
        y = wav_to_tensor(output_path, rate=rate)
        y_val_1 = y[: _V3_DATA_INFO.t_validate]
        y_val_2 = y[-_V3_DATA_INFO.t_validate :]
        esr_replicate = esr(y_val_1, y_val_2).item()
        print(f"Replicate ESR is {esr_replicate:.8f}.")
        esr_replicate_threshold = 0.01
        if esr_replicate > esr_replicate_threshold:
            print(_esr_validation_replicate_msg(esr_replicate_threshold))
            if not silent:
                plt.figure()
                t = np.arange(len(y_val_1)) / rate
                plt.plot(t, y_val_1, label="Validation 1")
                plt.plot(t, y_val_2, label="Validation 2")
                plt.xlabel("Time (sec)")
                plt.legend()
                plt.title("V3 check: Validation replicate FAILURE")
                plt.show()
            return False
    return True


def _check(
    input_path: str, output_path: str, input_version: Version, delay: int, silent: bool
) -> bool:
    """
    Ensure that everything should go smoothly

    :return: True if looks good
    """
    if input_version.major == 1:
        f = _check_v1
    elif input_version.major == 2:
        f = _check_v2
    elif input_version.major == 3:
        f = _check_v3
    else:
        print(f"Checks not implemented for input version {input_version}; skip")
        return True
    return f(input_path, output_path, delay, silent)


def _get_wavenet_config(architecture):
    return {
        Architecture.STANDARD: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 16,
                    "head_size": 8,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 16,
                    "channels": 8,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.LITE: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 12,
                    "head_size": 6,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 12,
                    "channels": 6,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.FEATHER: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 8,
                    "head_size": 4,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 8,
                    "channels": 4,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        Architecture.NANO: {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "channels": 4,
                    "head_size": 2,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8, 16, 32, 64],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "condition_size": 1,
                    "input_size": 4,
                    "channels": 2,
                    "head_size": 1,
                    "kernel_size": 3,
                    "dilations": [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
    }[architecture]


_CAB_MRSTFT_PRE_EMPH_WEIGHT = 2.0e-4
_CAB_MRSTFT_PRE_EMPH_COEF = 0.85


def _get_configs(
    input_version: Version,
    input_path: str,
    output_path: str,
    delay: int,
    epochs: int,
    model_type: str,
    architecture: Architecture,
    ny: int,
    lr: float,
    lr_decay: float,
    batch_size: int,
    fit_cab: bool,
):
    def get_kwargs(data_info: _DataInfo):
        if data_info.major_version == 1:
            train_val_split = data_info.validation_start
            train_kwargs = {"stop": train_val_split}
            validation_kwargs = {"start": train_val_split}
        elif data_info.major_version == 2:
            validation_start = data_info.validation_start
            train_stop = validation_start
            validation_stop = validation_start + data_info.t_validate
            train_kwargs = {"stop": train_stop}
            validation_kwargs = {"start": validation_start, "stop": validation_stop}
        elif data_info.major_version == 3:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {"start": 480_000, "stop": train_stop}
            validation_kwargs = {"start": validation_start}
        else:
            raise NotImplementedError(f"kwargs for input version {input_version}")
        return train_kwargs, validation_kwargs

    data_info = {1: _V1_DATA_INFO, 2: _V2_DATA_INFO, 3: _V3_DATA_INFO}[
        input_version.major
    ]
    train_kwargs, validation_kwargs = get_kwargs(data_info)
    data_config = {
        "train": {"ny": ny, **train_kwargs},
        "validation": {"ny": None, **validation_kwargs},
        "common": {
            "x_path": input_path,
            "y_path": output_path,
            "delay": delay,
        },
    }

    if model_type == "WaveNet":
        model_config = {
            "net": {
                "name": "WaveNet",
                # This should do decently. If you really want a nice model, try turning up
                # "channels" in the first block and "input_size" in the second from 12 to 16.
                "config": _get_wavenet_config(architecture),
            },
            "loss": {"val_loss": "esr"},
            "optimizer": {"lr": lr},
            "lr_scheduler": {
                "class": "ExponentialLR",
                "kwargs": {"gamma": 1.0 - lr_decay},
            },
        }
    else:
        model_config = {
            "net": {
                "name": "LSTM",
                "config": _get_lstm_config(architecture),
            },
            "loss": {
                "val_loss": "mse",
                "mask_first": 4096,
                "pre_emph_weight": 1.0,
                "pre_emph_coef": 0.85,
            },
            "optimizer": {"lr": 0.01},
            "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.995}},
        }
    if fit_cab:
        model_config["loss"]["pre_emph_mrstft_weight"] = _CAB_MRSTFT_PRE_EMPH_WEIGHT
        model_config["loss"]["pre_emph_mrstft_coef"] = _CAB_MRSTFT_PRE_EMPH_COEF

    if torch.cuda.is_available():
        device_config = {"accelerator": "gpu", "devices": 1}
    elif torch.backends.mps.is_available():
        device_config = {"accelerator": "mps", "devices": 1}
    else:
        print("WARNING: No GPU was found. Training will be very slow!")
        device_config = {}
    learning_config = {
        "train_dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {},
        "trainer": {"max_epochs": epochs, **device_config},
    }
    return data_config, model_config, learning_config


def _get_dataloaders(
    data_config: Dict, learning_config: Dict, model: Model
) -> Tuple[DataLoader, DataLoader]:
    data_config, learning_config = [deepcopy(c) for c in (data_config, learning_config)]
    data_config["common"]["nx"] = model.net.receptive_field
    dataset_train = init_dataset(data_config, Split.TRAIN)
    dataset_validation = init_dataset(data_config, Split.VALIDATION)
    train_dataloader = DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = DataLoader(dataset_validation, **learning_config["val_dataloader"])
    return train_dataloader, val_dataloader


def _esr(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (
        torch.mean(torch.square(pred - target)).item()
        / torch.mean(torch.square(target)).item()
    )


def _plot(
    model,
    ds,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
    filepath: Optional[str] = None,
    silent: bool = False,
):
    print("Plotting a comparison of your model with the target output...")
    with torch.no_grad():
        tx = len(ds.x) / 48_000
        print(f"Run (t={tx:.2f} sec)")
        t0 = time()
        output = model(ds.x).flatten().cpu().numpy()
        t1 = time()
        print(f"Took {t1 - t0:.2f} sec ({tx / (t1 - t0):.2f}x)")

    esr = _esr(torch.Tensor(output), ds.y)
    # Trying my best to put numbers to it...
    if esr < 0.01:
        esr_comment = "Great!"
    elif esr < 0.035:
        esr_comment = "Not bad!"
    elif esr < 0.1:
        esr_comment = "...This *might* sound ok!"
    elif esr < 0.3:
        esr_comment = "...This probably won't sound great :("
    else:
        esr_comment = "...Something seems to have gone wrong."
    print(f"Error-signal ratio = {esr:.4g}")
    print(esr_comment)

    plt.figure(figsize=(16, 5))
    plt.plot(output[window_start:window_end], label="Prediction")
    plt.plot(ds.y[window_start:window_end], linestyle="--", label="Target")
    plt.title(f"ESR={esr:.4g}")
    plt.legend()
    if filepath is not None:
        plt.savefig(filepath + ".png")
    if not silent:
        plt.show()


def _print_nasty_checks_warning():
    """
    "ffs" -Dom
    """
    print(
        "\n"
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
        "X                                                                          X\n"
        "X                                WARNING:                                  X\n"
        "X                                                                          X\n"
        "X       You are ignoring the checks! Your model might turn out bad!        X\n"
        "X                                                                          X\n"
        "X                              I warned you!                               X\n"
        "X                                                                          X\n"
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
    )


def _nasty_checks_modal():
    msg = "You are ignoring the checks!\nYour model might turn out bad!"

    root = tk.Tk()
    root.withdraw()  # hide the root window
    modal = tk.Toplevel(root)
    modal.geometry("300x100")
    modal.title("Warning!")
    label = tk.Label(modal, text=msg)
    label.pack(pady=10)
    ok_button = tk.Button(
        modal,
        text="I can only blame myself!",
        command=lambda: [modal.destroy(), root.quit()],
    )
    ok_button.pack()
    modal.grab_set()  # disable interaction with root window while modal is open
    modal.mainloop()


# Example usage:
# show_modal("Hello, World!")


def train(
    input_path: str,
    output_path: str,
    train_path: str,
    input_version: Optional[Version] = None,
    epochs=100,
    delay=None,
    model_type: str = "WaveNet",
    architecture: Union[Architecture, str] = Architecture.STANDARD,
    batch_size: int = 16,
    ny: int = 8192,
    lr=0.004,
    lr_decay=0.007,
    seed: Optional[int] = 0,
    save_plot: bool = False,
    silent: bool = False,
    modelname: str = "model",
    ignore_checks: bool = False,
    local: bool = False,
    fit_cab: bool = False,
) -> Optional[Model]:
    if seed is not None:
        torch.manual_seed(seed)

    if input_version is None:
        input_version, strong_match = _detect_input_version(input_path)

    if delay is None:
        delay = _calibrate_delay(
            delay, input_version, input_path, output_path, silent=silent
        )
    else:
        print(f"Delay provided as {delay}; skip calibration")

    if _check(input_path, output_path, input_version, delay, silent):
        print("-Checks passed")
    else:
        print("Failed checks!")
        if ignore_checks:
            if local and not silent:
                _nasty_checks_modal()
            else:
                _print_nasty_checks_warning()
        elif not local:  # And not ignore_checks
            print(
                "(To disable this check, run AT YOUR OWN RISK with "
                "`ignore_checks=True`.)"
            )
        if not ignore_checks:
            print("Exiting core training...")
            return

    data_config, model_config, learning_config = _get_configs(
        input_version,
        input_path,
        output_path,
        delay,
        epochs,
        model_type,
        Architecture(architecture),
        ny,
        lr,
        lr_decay,
        batch_size,
        fit_cab,
    )

    print("Starting training. It's time to kick ass and chew bubblegum!")
    # Issue:
    # * Model needs sample rate from data, but data set needs nx from model.
    # * Model is re-instantiated after training anyways.
    # (Hacky) solution: set sample rate in model from dataloader after second
    # instantiation from final checkpoint.
    model = Model.init_from_config(model_config)
    train_dataloader, val_dataloader = _get_dataloaders(
        data_config, learning_config, model
    )
    if train_dataloader.dataset.sample_rate != val_dataloader.dataset.sample_rate:
        raise RuntimeError(
            "Train and validation data loaders have different data set sample rates: "
            f"{train_dataloader.dataset.sample_rate}, "
            f"{val_dataloader.dataset.sample_rate}"
        )

    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                filename="checkpoint_best_{epoch:04d}_{step}_{ESR:.4g}_{MSE:.3e}",
                save_top_k=3,
                monitor="val_loss",
                every_n_epochs=1,
            ),
            pl.callbacks.model_checkpoint.ModelCheckpoint(
                filename="checkpoint_last_{epoch:04d}_{step}", every_n_epochs=1
            ),
        ],
        default_root_dir=train_path,
        **learning_config["trainer"],
    )
    # Suppress the PossibleUserWarning about num_workers (Issue 345)
    with filter_warnings("ignore", category=PossibleUserWarning):
        trainer.fit(model, train_dataloader, val_dataloader)

    # Go to best checkpoint
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    if best_checkpoint != "":
        model = Model.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            **Model.parse_config(model_config),
        )
    model.cpu()
    model.eval()
    model.net.sample_rate = train_dataloader.dataset.sample_rate

    def window_kwargs(version: Version):
        if version.major == 1:
            return dict(
                window_start=100_000,  # Start of the plotting window, in samples
                window_end=101_000,  # End of the plotting window, in samples
            )
        elif version.major == 2:
            # Same validation set even though it's a different spot in the reamp file
            return dict(
                window_start=100_000,  # Start of the plotting window, in samples
                window_end=101_000,  # End of the plotting window, in samples
            )
        # Fallback:
        return dict(
            window_start=100_000,  # Start of the plotting window, in samples
            window_end=101_000,  # End of the plotting window, in samples
        )

    _plot(
        model,
        val_dataloader.dataset,
        filepath=train_path + "/" + modelname if save_plot else None,
        silent=silent,
        **window_kwargs(input_version),
    )
    return model
