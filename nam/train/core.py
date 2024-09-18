# File: core.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
The core of the "simplified trainer"

Used by the GUI and Colab trainers.
"""

import hashlib
import tkinter as tk
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from time import time
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import BaseModel
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

from ..data import DataError, Split, init_dataset, wav_to_np, wav_to_tensor
from ..models import Model
from ..models.exportable import Exportable
from ..models.losses import esr
from ..models.metadata import UserMetadata
from ..util import filter_warnings
from ._version import PROTEUS_VERSION, Version
from . import metadata

# Training using the simplified trainers in NAM is done at 48k.
STANDARD_SAMPLE_RATE = 48_000.0
# Default number of output samples per datum.
_NY_DEFAULT = 8192


class Architecture(Enum):
    STANDARD = "standard"
    LITE = "lite"
    FEATHER = "feather"
    NANO = "nano"


class _InputValidationError(ValueError):
    pass


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
            "80e224bd5622fd6153ff1fd9f34cb3bd": PROTEUS_VERSION,
        }.get(file_hash)
        if version is None:
            print(
                f"Provided input file {input_path} does not strong-match any known "
                "standard input files."
            )
        return version

    def detect_weak(input_path) -> Optional[Version]:
        def assign_hash(path):
            Hash = Optional[str]
            Hashes = Tuple[Hash, Hash]

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

            def assign_hash_v4(path) -> Hash:
                # Use this to create recognized hashes for new files
                x, info = wav_to_np(path, info=True)
                rate = info.rate
                if rate != _V4_DATA_INFO.rate:
                    return None
                # I don't care about anything in the file except the starting blip and
                start_hash = _hash(x[: int(1 * _V4_DATA_INFO.rate)])
                return start_hash

            start_hash_v1, end_hash_v1 = assign_hashes_v1(path)
            start_hash_v2, end_hash_v2 = assign_hashes_v2(path)
            start_hash_v3, end_hash_v3 = assign_hashes_v3(path)
            hash_v4 = assign_hash_v4(path)
            return (
                start_hash_v1,
                end_hash_v1,
                start_hash_v2,
                end_hash_v2,
                start_hash_v3,
                end_hash_v3,
                hash_v4,
            )

        (
            start_hash_v1,
            end_hash_v1,
            start_hash_v2,
            end_hash_v2,
            start_hash_v3,
            end_hash_v3,
            hash_v4,
        ) = assign_hash(input_path)
        print(
            "Weak hashes:\n"
            f" Start (v1) : {start_hash_v1}\n"
            f" End (v1)   : {end_hash_v1}\n"
            f" Start (v2) : {start_hash_v2}\n"
            f" End (v2)   : {end_hash_v2}\n"
            f" Start (v3) : {start_hash_v3}\n"
            f" End (v3)   : {end_hash_v3}\n"
            f" Proteus    : {hash_v4}\n"
        )

        # Check for matches, starting with most recent. Proteus last since its match is
        # the most permissive.
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
        if version is not None:
            return version
        version = {"46151c8030798081acc00a725325a07d": PROTEUS_VERSION}.get(hash_v4)
        return version

    version = detect_strong(input_path)
    if version is not None:
        strong_match = True
        return version, strong_match
    print("Falling back to weak-matching...")
    version = detect_weak(input_path)
    if version is None:
        raise _InputValidationError(
            f"Input file at {input_path} cannot be recognized as any known version!"
        )
    strong_match = False

    return version, strong_match


class _DataInfo(BaseModel):
    """
    :param major_version: Data major version
    """

    major_version: int
    rate: Optional[float]
    t_blips: int
    first_blips_start: int
    t_validate: int
    train_start: int
    validation_start: int
    noise_interval: Tuple[int, int]
    blip_locations: Sequence[Sequence[int]]


_V1_DATA_INFO = _DataInfo(
    major_version=1,
    rate=STANDARD_SAMPLE_RATE,
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
    rate=STANDARD_SAMPLE_RATE,
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
    rate=STANDARD_SAMPLE_RATE,
    t_blips=96_000,
    first_blips_start=480_000,
    t_validate=432_000,
    train_start=480_000,
    validation_start=-432_000,
    noise_interval=(492_000, 498_000),
    blip_locations=((504_000, 552_000),),
)
# V4 (aka GuitarML Proteus)
# https://github.com/GuitarML/Releases/releases/download/v1.0.0/Proteus_Capture_Utility.zip
# * 44.1k
# * Odd length...
# * There's a blip on sample zero. This has to be ignored or else over-compensated
#   latencies will come out wrong!
# (0:00-0:01) Blips at 0:00.0 and 0:00.5
# (0:01-0:09) Sine sweeps
# (0:09-0:17) White noise
# (0:17:0.20) Rising white noise (to 0:20.333 appx)
# (0:20-3:30.858) General training data (ends on sample 9,298,872)
# I'm arbitrarily assigning the last 10 seconds as validation data.
_V4_DATA_INFO = _DataInfo(
    major_version=4,
    rate=44_100.0,
    t_blips=44_099,  # Need to ignore the first blip!
    first_blips_start=1,  # Need to ignore the first blip!
    t_validate=441_000,
    # Blips are problematic for training because they don't have preceding silence
    train_start=44_100,
    validation_start=-441_000,
    noise_interval=(6_000, 12_000),
    blip_locations=((22_050,),),
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


def _calibrate_latency_v_all(
    data_info: _DataInfo,
    y,
    abs_threshold=_DELAY_CALIBRATION_ABS_THRESHOLD,
    rel_threshold=_DELAY_CALIBRATION_REL_THRESHOLD,
    safety_factor=_DELAY_CALIBRATION_SAFETY_FACTOR,
) -> metadata.LatencyCalibration:
    """
    Calibrate the delay in teh input-output pair based on blips.
    This only uses the blips in the first set of blip locations!

    :param y: The output audio, in complete.
    """

    def report_any_latency_warnings(
        delays: Sequence[int],
    ) -> metadata.LatencyCalibrationWarnings:
        # Warnings associated with any single delay:

        # "Lookahead warning": if the delay is equal to the lookahead, then it's
        # probably an error.
        lookahead_warnings = [i for i, d in enumerate(delays, 1) if d == -lookahead]
        matches_lookahead = len(lookahead_warnings) > 0
        if matches_lookahead:
            print(_warn_lookaheads(lookahead_warnings))

        # Ensemble warnings

        # If they're _really_ different, then something might be wrong.
        max_disagreement_threshold = 20
        max_disagreement_too_high = (
            np.max(delays) - np.min(delays) >= max_disagreement_threshold
        )
        if max_disagreement_too_high:
            print(
                "WARNING: Latencies are anomalously different from each other (more "
                f"than {max_disagreement_threshold} samples). If this model turns out "
                "badly, then you might need to provide the latency manually."
            )

        return metadata.LatencyCalibrationWarnings(
            matches_lookahead=matches_lookahead,
            disagreement_too_high=max_disagreement_too_high,
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
    warnings = report_any_latency_warnings(delays)

    delay_post_safety_factor = int(np.min(delays)) - safety_factor
    print(
        f"After aplying safety factor of {safety_factor}, the final delay is "
        f"{delay_post_safety_factor}"
    )
    return metadata.LatencyCalibration(
        algorithm_version=1,
        delays=delays,
        safety_factor=safety_factor,
        recommended=delay_post_safety_factor,
        warnings=warnings,
    )


_calibrate_latency_v1 = partial(_calibrate_latency_v_all, _V1_DATA_INFO)
_calibrate_latency_v2 = partial(_calibrate_latency_v_all, _V2_DATA_INFO)
_calibrate_latency_v3 = partial(_calibrate_latency_v_all, _V3_DATA_INFO)
_calibrate_latency_v4 = partial(_calibrate_latency_v_all, _V4_DATA_INFO)


def _plot_latency_v_all(
    data_info: _DataInfo, latency: int, input_path: str, output_path: str, _nofail=True
):
    print("Plotting the latency for manual inspection...")
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
                y[ii - di + latency : ii + di + latency],
                ".-",
                label=f"Output {e}",
            )
        plt.axvline(x=0, linestyle="--", color="k")
        plt.legend()
        plt.show()  # This doesn't freeze the notebook


_plot_latency_v1 = partial(_plot_latency_v_all, _V1_DATA_INFO)
_plot_latency_v2 = partial(_plot_latency_v_all, _V2_DATA_INFO)
_plot_latency_v3 = partial(_plot_latency_v_all, _V3_DATA_INFO)
_plot_latency_v4 = partial(_plot_latency_v_all, _V4_DATA_INFO)


def _analyze_latency(
    user_latency: Optional[int],
    input_version: Version,
    input_path: str,
    output_path: str,
    silent: bool = False,
) -> metadata.Latency:
    """
    :param is_proteus: Forget the version; d
    """
    if input_version.major == 1:
        calibrate, plot = _calibrate_latency_v1, _plot_latency_v1
    elif input_version.major == 2:
        calibrate, plot = _calibrate_latency_v2, _plot_latency_v2
    elif input_version.major == 3:
        calibrate, plot = _calibrate_latency_v3, _plot_latency_v3
    elif input_version.major == 4:
        calibrate, plot = _calibrate_latency_v4, _plot_latency_v4
    else:
        raise NotImplementedError(
            f"Input calibration not implemented for input version {input_version}"
        )
    if user_latency is not None:
        print(f"Delay is specified as {user_latency}")
    calibration_output = calibrate(wav_to_np(output_path))
    latency = (
        user_latency if user_latency is not None else calibration_output.recommended
    )
    if not silent:
        plot(latency, input_path, output_path)

    return metadata.Latency(manual=user_latency, calibration=calibration_output)


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


def _check_v1(*args, **kwargs) -> metadata.DataChecks:
    return metadata.DataChecks(version=1, passed=True)


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


def _check_v2(input_path, output_path, delay: int, silent: bool) -> metadata.DataChecks:
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
                return metadata.DataChecks(version=2, passed=False)
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
                return metadata.DataChecks(version=2, passed=False)
        return metadata.DataChecks(version=2, passed=True)


def _check_v3(
    input_path, output_path, silent: bool, *args, **kwargs
) -> metadata.DataChecks:
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
            return metadata.DataChecks(version=3, passed=False)
    return metadata.DataChecks(version=3, passed=True)


def _check_v4(
    input_path, output_path, silent: bool, *args, **kwargs
) -> metadata.DataChecks:
    # Things we can't check:
    # Latency compensation agreement
    # Data replicability
    print("Using Proteus audio file. Standard data checks aren't possible!")
    signal, info = wav_to_np(output_path, info=True)
    passed = True
    if info.rate != _V4_DATA_INFO.rate:
        print(
            f"Output signal has sample rate {info.rate}; expected {_V4_DATA_INFO.rate}!"
        )
        passed = False
    # I don't care what's in the files except that they're long enough to hold the blip
    # and the last 10 seconds I decided to use as validation
    required_length = int((1.0 + 10.0) * _V4_DATA_INFO.rate)
    if len(signal) < required_length:
        print(
            "File doesn't meet the minimum length requirements for latency compensation and validation signal!"
        )
        passed = False
    return metadata.DataChecks(version=4, passed=passed)


def _check_data(
    input_path: str, output_path: str, input_version: Version, delay: int, silent: bool
) -> Optional[metadata.DataChecks]:
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
    elif input_version.major == 4:
        f = _check_v4
    else:
        print(f"Checks not implemented for input version {input_version}; skip")
        return None
    out = f(input_path, output_path, delay, silent)
    # Issue 442: Deprecate inputs
    if input_version.major != 3:
        print(
            f"Input version {input_version} is deprecated and will be removed in "
            "version 0.11 of the trainer. To continue using it, you must ignore checks."
        )
        out.passed = False
    return out


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


def _get_data_config(
    input_version: Version, input_path: Path, output_path: Path, ny: int, latency: int
) -> dict:
    def get_kwargs(data_info: _DataInfo):
        if data_info.major_version == 1:
            train_val_split = data_info.validation_start
            train_kwargs = {"stop_samples": train_val_split}
            validation_kwargs = {"start_samples": train_val_split}
        elif data_info.major_version == 2:
            validation_start = data_info.validation_start
            train_stop = validation_start
            validation_stop = validation_start + data_info.t_validate
            train_kwargs = {"stop_samples": train_stop}
            validation_kwargs = {
                "start_samples": validation_start,
                "stop_samples": validation_stop,
            }
        elif data_info.major_version == 3:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {"start_samples": 480_000, "stop_samples": train_stop}
            validation_kwargs = {"start_samples": validation_start}
        elif data_info.major_version == 4:
            validation_start = data_info.validation_start
            train_stop = validation_start
            train_kwargs = {"stop_samples": train_stop}
            # Proteus doesn't have silence to get a clean split. Bite the bullet.
            print(
                "Using Proteus files:\n"
                " * There isn't a silent point to split the validation set, so some of "
                "your gear's response from the train set will leak into the start of "
                "the validation set and impact validation accuracy (Bypassing data "
                "quality check)\n"
                " * Since the validation set is different, the ESRs reported for this "
                "model aren't comparable to those from the other 'NAM' training files."
            )
            validation_kwargs = {
                "start_samples": validation_start,
                "require_input_pre_silence": False,
            }
        else:
            raise NotImplementedError(f"kwargs for input version {input_version}")
        return train_kwargs, validation_kwargs

    data_info = {
        1: _V1_DATA_INFO,
        2: _V2_DATA_INFO,
        3: _V3_DATA_INFO,
        4: _V4_DATA_INFO,
    }[input_version.major]
    train_kwargs, validation_kwargs = get_kwargs(data_info)
    data_config = {
        "train": {"ny": ny, **train_kwargs},
        "validation": {"ny": None, **validation_kwargs},
        "common": {
            "x_path": input_path,
            "y_path": output_path,
            "delay": latency,
        },
    }
    return data_config


def _get_configs(
    input_version: Version,
    input_path: str,
    output_path: str,
    latency: int,
    epochs: int,
    model_type: str,
    architecture: Architecture,
    ny: int,
    lr: float,
    lr_decay: float,
    batch_size: int,
    fit_mrstft: bool,
):
    data_config = _get_data_config(
        input_version=input_version,
        input_path=input_path,
        output_path=output_path,
        ny=ny,
        latency=latency,
    )

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
    if fit_mrstft:
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
) -> float:
    """
    :return: The ESR
    """
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
    return esr


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


class _ValidationStopping(pl.callbacks.EarlyStopping):
    """
    Callback to indicate to stop training if the validation metric is good enough,
    without the other conditions that EarlyStopping usually forces like patience.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = np.inf


class _ModelCheckpoint(pl.callbacks.model_checkpoint.ModelCheckpoint):
    """
    Extension to model checkpoint to save a .nam file as well as the .ckpt file.
    """

    def __init__(
        self,
        *args,
        user_metadata: Optional[UserMetadata] = None,
        settings_metadata: Optional[metadata.Settings] = None,
        data_metadata: Optional[metadata.Data] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._user_metadata = user_metadata
        self._settings_metadata = settings_metadata
        self._data_metadata = data_metadata

    _NAM_FILE_EXTENSION = Exportable.FILE_EXTENSION

    @classmethod
    def _get_nam_filepath(cls, filepath: str) -> Path:
        """
        Given a .ckpt filepath, figure out a .nam for it.
        """
        if not filepath.endswith(cls.FILE_EXTENSION):
            raise ValueError(
                f"Checkpoint filepath {filepath} doesn't end in expected extension "
                f"{cls.FILE_EXTENSION}"
            )
        return Path(filepath[: -len(cls.FILE_EXTENSION)] + cls._NAM_FILE_EXTENSION)

    @property
    def _include_other_metadata(self) -> bool:
        return self._settings_metadata is not None and self._data_metadata is not None

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str):
        # Save the .ckpt:
        super()._save_checkpoint(trainer, filepath)
        # Save the .nam:
        nam_filepath = self._get_nam_filepath(filepath)
        pl_model: Model = trainer.model
        nam_model = pl_model.net
        outdir = nam_filepath.parent
        # HACK: Assume the extension
        basename = nam_filepath.name[: -len(self._NAM_FILE_EXTENSION)]
        other_metadata = (
            None
            if not self._include_other_metadata
            else {
                metadata.TRAINING_KEY: metadata.TrainingMetadata(
                    settings=self._settings_metadata,
                    data=self._data_metadata,
                    validation_esr=None,  # TODO how to get this?
                ).model_dump()
            }
        )
        nam_model.export(
            outdir,
            basename=basename,
            user_metadata=self._user_metadata,
            other_metadata=other_metadata,
        )

    def _remove_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        nam_path = self._get_nam_filepath(filepath)
        if nam_path.exists():
            nam_path.unlink()


def _get_callbacks(
    threshold_esr: Optional[float],
    user_metadata: Optional[UserMetadata] = None,
    settings_metadata: Optional[metadata.Settings] = None,
    data_metadata: Optional[metadata.Data] = None,
):
    callbacks = [
        _ModelCheckpoint(
            filename="checkpoint_best_{epoch:04d}_{step}_{ESR:.4g}_{MSE:.3e}",
            save_top_k=3,
            monitor="val_loss",
            every_n_epochs=1,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
        ),
        _ModelCheckpoint(
            filename="checkpoint_last_{epoch:04d}_{step}",
            every_n_epochs=1,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
        ),
    ]
    if threshold_esr is not None:
        callbacks.append(
            _ValidationStopping(monitor="ESR", stopping_threshold=threshold_esr)
        )
    return callbacks


class TrainOutput(NamedTuple):
    """
    :param model: The trained model
    :param simpliifed_trianer_metadata: The metadata summarizing training with the
        simplified trainer.
    """

    model: Optional[Model]
    metadata: metadata.TrainingMetadata


def _get_final_latency(latency_analysis: metadata.Latency) -> int:
    if latency_analysis.manual is not None:
        latency = latency_analysis.manual
        print(f"Latency provided as {latency_analysis.manual}; override calibration")
    else:
        latency = latency_analysis.calibration.recommended
        print(f"Set latency to recommended {latency_analysis.calibration.recommended}")
    return latency


def train(
    input_path: str,
    output_path: str,
    train_path: str,
    input_version: Optional[Version] = None,  # Deprecate?
    epochs=100,
    delay: Optional[int] = None,
    latency: Optional[int] = None,
    model_type: str = "WaveNet",
    architecture: Union[Architecture, str] = Architecture.STANDARD,
    batch_size: int = 16,
    ny: int = _NY_DEFAULT,
    lr=0.004,
    lr_decay=0.007,
    seed: Optional[int] = 0,
    save_plot: bool = False,
    silent: bool = False,
    modelname: str = "model",
    ignore_checks: bool = False,
    local: bool = False,
    fit_mrstft: bool = True,
    threshold_esr: Optional[bool] = None,
    user_metadata: Optional[UserMetadata] = None,
    fast_dev_run: Union[bool, int] = False,
) -> Optional[TrainOutput]:
    """
    :param lr_decay: =1-gamma for Exponential learning rate decay.
    :param threshold_esr: Stop training if ESR is better than this. Ignore if `None`.
    :param fast_dev_run: One-step training, used for tests.
    """

    def parse_user_latency(
        delay: Optional[int], latency: Optional[int]
    ) -> Optional[int]:
        if delay is not None:
            if latency is not None:
                raise ValueError("Both delay and latency are provided; use latency!")
            print("WARNING: use of `delay` is deprecated; use `latency` instead")
            return delay
        return latency

    if seed is not None:
        torch.manual_seed(seed)

    if input_version is None:
        input_version, strong_match = _detect_input_version(input_path)

    user_latency = parse_user_latency(delay, latency)
    latency_analysis = _analyze_latency(
        user_latency, input_version, input_path, output_path, silent=silent
    )
    final_latency = _get_final_latency(latency_analysis)

    data_check_output = _check_data(
        input_path, output_path, input_version, final_latency, silent
    )
    if data_check_output is not None:
        if data_check_output.passed:
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
                return TrainOutput(
                    model=None,
                    metadata=metadata.TrainingMetadata(
                        settings=metadata.Settings(ignore_checks=ignore_checks),
                        data=metadata.Data(
                            latency=latency_analysis, checks=data_check_output
                        ),
                        validation_esr=None,
                    ),
                )

    data_config, model_config, learning_config = _get_configs(
        input_version,
        input_path,
        output_path,
        final_latency,
        epochs,
        model_type,
        Architecture(architecture),
        ny,
        lr,
        lr_decay,
        batch_size,
        fit_mrstft,
    )
    assert (
        "fast_dev_run" not in learning_config
    ), "fast_dev_run is set as a kwarg to train()"

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
    sample_rate = train_dataloader.dataset.sample_rate
    model.net.sample_rate = sample_rate

    # Put together the metadata that's needed in checkpoints:
    settings_metadata = metadata.Settings(ignore_checks=ignore_checks)
    data_metadata = metadata.Data(latency=latency_analysis, checks=data_check_output)

    trainer = pl.Trainer(
        callbacks=_get_callbacks(
            threshold_esr,
            user_metadata=user_metadata,
            settings_metadata=settings_metadata,
            data_metadata=data_metadata,
        ),
        default_root_dir=train_path,
        fast_dev_run=fast_dev_run,
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
    model.net.sample_rate = sample_rate  # Hack, part 2

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

    validation_esr = _plot(
        model,
        val_dataloader.dataset,
        filepath=train_path + "/" + modelname if save_plot else None,
        silent=silent,
        **window_kwargs(input_version),
    )
    return TrainOutput(
        model=model,
        metadata=metadata.TrainingMetadata(
            settings=settings_metadata,
            data=data_metadata,
            validation_esr=validation_esr,
        ),
    )


class DataInputValidation(BaseModel):
    passed: bool


def validate_input(input_path) -> DataInputValidation:
    """
    :return: Could it be validated?
    """
    try:
        _detect_input_version(input_path)
        # succeeded...
        return DataInputValidation(passed=True)
    except _InputValidationError as e:
        print(f"Input validation failed!\n\n{e}")
        return DataInputValidation(passed=False)


class _PyTorchDataSplitValidation(BaseModel):
    """
    :param msg: On exception, catch and assign. Otherwise None
    """

    passed: bool
    msg: Optional[str]


class _PyTorchDataValidation(BaseModel):
    passed: bool
    train: _PyTorchDataSplitValidation  # cf Split.TRAIN
    validation: _PyTorchDataSplitValidation  # Split.VALIDATION


class DataValidationOutput(BaseModel):
    passed: bool
    input_version: str
    latency: metadata.Latency
    checks: metadata.DataChecks
    pytorch: _PyTorchDataValidation


def validate_data(
    input_path: Path,
    output_path: Path,
    user_latency: Optional[int],
    num_output_samples_per_datum: int = _NY_DEFAULT,
):
    """
    Just do the checks to make sure that the data are ok.

    * Version identification
    * Latency calibration
    * Other checks
    """
    passed = True  # Until proven otherwise

    # Data version ID
    input_version, strong_match = _detect_input_version(input_path)

    # Latency analysis
    latency_analysis = _analyze_latency(
        user_latency, input_version, input_path, output_path, silent=True
    )
    if latency_analysis.manual is None and any(
        val for val in latency_analysis.calibration.warnings.model_dump().values()
    ):
        passed = False
    final_latency = _get_final_latency(latency_analysis)

    # Other data checks based on input file version
    data_checks = _check_data(
        input_path,
        output_path,
        input_version,
        latency_analysis.calibration.recommended,
        silent=True,
    )
    passed = passed and data_checks.passed

    # Finally, try to make the PyTorch Dataset objects and note any failures:
    data_config = _get_data_config(
        input_version=input_version,
        input_path=input_path,
        output_path=output_path,
        ny=num_output_samples_per_datum,
        latency=final_latency,
    )
    # HACK this should depend on the model that's going to be used, but I think it will
    # be unlikely to make a difference. Still, would be nice to fix.
    data_config["common"]["nx"] = 4096

    pytorch_data_split_validation_dict: Dict[str, _PyTorchDataSplitValidation] = {}
    for split in Split:
        try:
            init_dataset(data_config, split)
            pytorch_data_split_validation_dict[
                split.value
            ] = _PyTorchDataSplitValidation(passed=True, msg=None)
        except DataError as e:
            pytorch_data_split_validation_dict[
                split.value
            ] = _PyTorchDataSplitValidation(passed=False, msg=str(e))
    pytorch_data_validation = _PyTorchDataValidation(
        passed=all(v.passed for v in pytorch_data_split_validation_dict.values()),
        **pytorch_data_split_validation_dict,
    )
    passed = passed and pytorch_data_validation.passed

    return DataValidationOutput(
        passed=passed,
        input_version=str(input_version),
        latency=latency_analysis,
        checks=data_checks,
        pytorch=pytorch_data_validation,
    )
