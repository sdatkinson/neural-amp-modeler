# File: colab.py
# Created Date: Sunday December 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Hide the mess in Colab to make things look pretty for users.
"""


from pathlib import Path
from typing import NamedTuple, Optional, Tuple

from ..models.metadata import UserMetadata
from ._names import INPUT_BASENAMES, LATEST_VERSION, Version
from ._version import Version
from .core import train


_BUGGY_INPUT_BASENAMES = {
    # 1.1.0 has the spikes at the wrong spots.
    "v1_1_0.wav"
}
_OUTPUT_BASENAME = "output.wav"
_TRAIN_PATH = "."


def _check_for_files() -> Tuple[Version, str]:
    # TODO use hash logic as in GUI trainer!
    print("Checking that we have all of the required audio files...")
    for name in _BUGGY_INPUT_BASENAMES:
        if Path(name).exists():
            raise RuntimeError(
                f"Detected input signal {name} that has known bugs. Please download the latest input signal, {LATEST_VERSION[1]}"
            )
    for input_version, input_basename in INPUT_BASENAMES:
        if Path(input_basename).exists():
            if input_version != LATEST_VERSION.version:
                print(
                    f"WARNING: Using out-of-date input file {input_basename}. "
                    "Recommend downloading and using the latest version, "
                    f"{LATEST_VERSION.name}."
                )
            break
    else:
        raise FileNotFoundError(
            f"Didn't find NAM's input audio file. Please upload {LATEST_VERSION.name}"
        )
    if not Path(_OUTPUT_BASENAME).exists():
        raise FileNotFoundError(
            f"Didn't find your reamped output audio file. Please upload {_OUTPUT_BASENAME}."
        )
    print(f"Found {input_basename}, version {input_version}")
    return input_version, input_basename


def _get_valid_export_directory():
    def get_path(version):
        return Path("exported_models", f"version_{version}")

    version = 0
    while get_path(version).exists():
        version += 1
    return get_path(version)


def run(
    epochs: int = 100,
    delay: Optional[int] = None,
    model_type: str = "WaveNet",
    architecture: str = "standard",
    lr: float = 0.004,
    lr_decay: float = 0.007,
    seed: Optional[int] = 0,
    user_metadata: Optional[UserMetadata] = None,
    ignore_checks: bool = False,
    fit_cab: bool = False,
):
    """
    :param epochs: How amny epochs we'll train for.
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

    model = train(
        input_basename,
        _OUTPUT_BASENAME,
        _TRAIN_PATH,
        input_version=input_version,
        epochs=epochs,
        delay=delay,
        model_type=model_type,
        architecture=architecture,
        lr=lr,
        lr_decay=lr_decay,
        seed=seed,
        local=False,
        ignore_checks=ignore_checks,
        fit_cab=fit_cab,
    )

    if model is None:
        print("No model returned; skip exporting!")
    else:
        print("Exporting your model...")
        model_export_outdir = _get_valid_export_directory()
        model_export_outdir.mkdir(parents=True, exist_ok=False)
        model.net.export(model_export_outdir, user_metadata=user_metadata)
        print(f"Model exported to {model_export_outdir}. Enjoy!")
