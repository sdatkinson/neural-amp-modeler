# File: colab.py
# Created Date: Sunday December 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Hide the mess in Colab to make things look pretty for users.
"""


from pathlib import Path
from typing import Optional, Tuple

from ..models.metadata import UserMetadata
from ._version import Version
from .core import train


_INPUT_BASENAMES = ((Version(1, 1, 1), "v1_1_1.wav"), (Version(1, 0, 0), "v1.wav"))
_BUGGY_INPUT_BASENAMES = {
    # 1.1.0 has the spikes at the wrong spots.
    "v1_1_0.wav"
}
_OUTPUT_BASENAME = "output.wav"
_TRAIN_PATH = "."


def _check_for_files() -> Tuple[Version, str]:
    print("Checking that we have all of the required audio files...")
    for name in _BUGGY_INPUT_BASENAMES:
        if Path(name).exists():
            raise RuntimeError(
                f"Detected input signal {name} that has known bugs. Please download the latest input signal, {_INPUT_BASENAMES[0][1]}"
            )
    for i, (input_version, input_basename) in enumerate(_INPUT_BASENAMES):
        if Path(input_basename).exists():
            if i > 0:
                print(
                    f"WARNING: Using out-of-date input file {input_basename}. "
                    "Recommend downloading and using the latest version."
                )
            break
    else:
        raise FileNotFoundError(
            f"Didn't find NAM's input audio file. Please upload {_INPUT_BASENAMES[0][1]}"
        )
    if not Path(_OUTPUT_BASENAME).exists():
        raise FileNotFoundError(
            f"Didn't find your reamped output audio file. Please upload {_OUTPUT_BASENAME}."
        )
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
    architecture: str = "standard",
    lr: float = 0.004,
    lr_decay: float = 0.007,
    seed: Optional[int] = 0,
    user_metadata: Optional[UserMetadata] = None,
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
    """

    input_version, input_basename = _check_for_files()

    model = train(
        input_basename,
        _OUTPUT_BASENAME,
        _TRAIN_PATH,
        input_version=input_version,
        epochs=epochs,
        delay=delay,
        architecture=architecture,
        lr=lr,
        lr_decay=lr_decay,
        seed=seed,
    )

    print("Exporting your model...")
    model_export_outdir = _get_valid_export_directory()
    model_export_outdir.mkdir(parents=True, exist_ok=False)
    model.net.export(model_export_outdir, user_metadata=user_metadata)
    print(f"Model exported to {model_export_outdir}. Enjoy!")
