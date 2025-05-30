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


def _check_for_files() -> str:
    """
    :return: The basename of the input file to be used (output is always the same)
    """
    # TODO use hash logic as in GUI trainer!
    print("Checking that we have all of the required audio files...")

    # First, look to see if we've got a buggy input file. If we do, then complain.
    for name in _BUGGY_INPUT_BASENAMES:
        if _Path(name).exists():
            raise RuntimeError(
                f"Detected input signal {name} that has known bugs. Please download the latest input signal, {_LATEST_VERSION[1]}"
            )
    # Find valid input file by checking for recognized names:
    for input_version, input_basename, other_names in _INPUT_BASENAMES:
        if _Path(input_basename).exists():
            # We found it. We're done here, but maybe print some things before breaking.
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
    if input_version != _PROTEUS_VERSION:
        print(f"Found {input_basename}, presumed version {input_version}")
    else:
        print(f"Found Proteus input {input_basename}.")
    
    # We found the input. Now check for the output and we'll be good.
    if not _Path(_OUTPUT_BASENAME).exists():
        raise FileNotFoundError(
            f"Didn't find your reamped output audio file. Please upload {_OUTPUT_BASENAME}."
        )
    
    return input_basename


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
    :param model_type: The type of model to train.
    :param architecture: The architecture hyperparameters to use
    :param lr: The initial learning rate
    :param lr_decay: The amount by which the learning rate decays each epoch
    :param seed: RNG seed for reproducibility.
    :param user_metadata: User-specified metadata to include in the .nam file.
    :param ignore_checks: Ignores the data quality checks and YOLOs it
    """

    input_basename = _check_for_files()

    train_output: _TrainOutput = _train(
        input_path=input_basename,
        output_path=_OUTPUT_BASENAME,
        train_path=_TRAIN_PATH,
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
