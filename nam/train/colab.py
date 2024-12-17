# File: colab.py
# Created Date: Sunday December 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Hide the mess in Colab to make things look pretty for users.
"""

from pathlib import Path as _Path
from typing import Optional as _Optional, Tuple as _Tuple

from ..models.metadata import UserMetadata as _UserMetadata
from ._names import (
    INPUT_BASENAMES as _INPUT_BASENAMES,
    LATEST_VERSION as _LATEST_VERSION,
    Version as _Version,
)
from ._version import PROTEUS_VERSION as _PROTEUS_VERSION, Version as _Version
from .core import TrainOutput as _TrainOutput, train as _train
from .metadata import TRAINING_KEY as _TRAINING_KEY

_BUGGY_INPUT_BASENAMES = {
    # 1.1.0 has the spikes at the wrong spots.
    "v1_1_0.wav"
}
_OUTPUT_BASENAME = "output.wav"
_TRAIN_PATH = "."


def _check_for_files() -> _Tuple[_Version, str]:
    # TODO use hash logic as in GUI trainer!
    print("Checking that we have all of the required audio files...")
    for name in _BUGGY_INPUT_BASENAMES:
        if _Path(name).exists():
            raise RuntimeError(
                f"Detected input signal {name} that has known bugs. Please download the latest input signal, {_LATEST_VERSION[1]}"
            )
    for input_version, input_basename, other_names in _INPUT_BASENAMES:
        if _Path(input_basename).exists():
            if input_version == _PROTEUS_VERSION:
                print(f"Using Proteus input file...")
            elif input_version != _LATEST_VERSION.version:
                print(
                    f"WARNING: Using out-of-date input file {input_basename}. "
                    "Recommend downloading and using the latest version, "
                    f"{_LATEST_VERSION.name}."
                )
            break
        if other_names is not None:
            for other_name in other_names:
                if _Path(other_name).exists():
                    raise RuntimeError(
                        f"Found out-of-date input file {other_name}. Rename it to {input_basename} and re-run."
                    )
    else:
        raise FileNotFoundError(
            f"Didn't find NAM's input audio file. Please upload {_LATEST_VERSION.name}"
        )
    # We found it
    if not _Path(_OUTPUT_BASENAME).exists():
        raise FileNotFoundError(
            f"Didn't find your reamped output audio file. Please upload {_OUTPUT_BASENAME}."
        )
    if input_version != _PROTEUS_VERSION:
        print(f"Found {input_basename}, version {input_version}")
    else:
        print(f"Found Proteus input {input_basename}.")
    return input_version, input_basename


def _get_valid_export_directory():
    def get_path(version):
        return _Path("exported_models", f"version_{version}")

    version = 0
    while get_path(version).exists():
        version += 1
    return get_path(version)


def run(
    epochs: int = 100,
    delay: _Optional[int] = None,
    model_type: str = "WaveNet",
    architecture: str = "standard",
    lr: float = 0.004,
    lr_decay: float = 0.007,
    seed: _Optional[int] = 0,
    user_metadata: _Optional[_UserMetadata] = None,
    ignore_checks: bool = False,
    fit_mrstft: bool = True,
):
    """
    :param epochs: How many epochs we'll train for.
    :param delay: How far the output algs the input due to round-trip latency during
        reamping, in samples.
    :param stage_1_channels: The number of channels in the WaveNet's first stage.
    :param stage_2_channels: The number of channels in the WaveNet's second stage.
    :param lr: The initial learning rate
    :param lr_decay: The amount by which the learning rate decays each epoch
    :param seed: RNG seed for reproducibility.
    :param user_metadata: User-specified metadata to include in the .nam file.
    :param ignore_checks: Ignores the data quality checks and YOLOs it
    """

    input_version, input_basename = _check_for_files()

    train_output: _TrainOutput = _train(
        input_basename,
        _OUTPUT_BASENAME,
        _TRAIN_PATH,
        input_version=input_version,
        epochs=epochs,
        latency=delay,
        model_type=model_type,
        architecture=architecture,
        lr=lr,
        lr_decay=lr_decay,
        seed=seed,
        local=False,
        ignore_checks=ignore_checks,
        fit_mrstft=fit_mrstft,
    )
    model = train_output.model
    training_metadata = train_output.metadata

    if model is None:
        print("No model returned; skip exporting!")
    else:
        print("Exporting your model...")
        model_export_outdir = _get_valid_export_directory()
        model_export_outdir.mkdir(parents=True, exist_ok=False)
        model.net.export(
            model_export_outdir,
            user_metadata=user_metadata,
            other_metadata={_TRAINING_KEY: training_metadata.model_dump()},
        )
        print(f"Model exported to {model_export_outdir}. Enjoy!")
