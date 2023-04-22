# File: colab.py
# Created Date: Sunday December 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Hide the mess in Colab to make things look pretty for users.
"""


from pathlib import Path
from typing import Optional, Tuple

from nam.models.metadata import UserMetadata
from nam.train._version import Version
from nam.train.core import train
from nam.util import find_files


_INPUT_BASENAMES = ((Version(1, 1, 1), "v1_1_1.wav"), (Version(1, 0, 0), "v1.wav"))
_BUGGY_INPUT_BASENAMES = {
    # 1.1.0 has the spikes at the wrong spots.
    "v1_1_0.wav"
}
_OUTPUT_BASENAME = "output.wav"
_TRAIN_PATH = "."


def _check_for_input_file() -> Tuple[Version, str]:
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
    return input_version, input_basename

def _check_for_default_output_file():
    if not Path(_OUTPUT_BASENAME).exists():
        raise FileNotFoundError(
            f"Didn't find your reamped output audio file. Please upload {_OUTPUT_BASENAME}."
    )

def _get_valid_export_directory(reuse=False):
    def get_path(version):
        return Path("exported_models", f"version_{version}")

    version = 0
    while get_path(version).exists():
        version += 1
    version = version if version == 0 or not reuse else version - 1
    return get_path(version)


def run(
    epochs: int = 100,
    delay: Optional[int] = None,
    architecture: str = "standard",
    lr: float = 0.004,
    lr_decay: float = 0.007,
    seed: Optional[int] = 0,
    user_metadata: Optional[UserMetadata] = None,
    training_directory: str = None,
    include_files: str = None,
    exclude_files: str = None,
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
    :param training_directory: training files directory
    :param include_files: A regex or sequence of regex patterns separated by comma for files to include in the training.
                           Defaults to None, which includes all files.
    :param exclude_files: A regex or sequence of regex patterns separated by comma for files to exclude from the training.
                           Defaults to None, which excludes no files.
    """
    save_plot = False
    silent = False

    training_list = []
    # keep current behavior
    if not training_directory:
        _check_for_default_output_file()
        training_list = [_OUTPUT_BASENAME]
    else:
        default_exclude =  [input_basename for (_, input_basename) in _INPUT_BASENAMES]
        exclude_files = default_exclude if exclude_files is None else default_exclude + exclude_files.split(",")
        training_list = find_files(training_directory, 'wav', include_files, exclude_files)
        save_plot = True # always save plot when batch training
        silent = True # always silent when batch training

    if not training_list:
        raise FileNotFoundError(
            f"Didn't find any training files in directory '{training_directory}'. Please upload some training files."
        )

    print(f"The following files are going to be trained: {training_list}")
    for index, file in enumerate(training_list):
        print(f"\n* Training file ({index + 1}/{len(training_list)}): {file}")
        try:
            _run_single_file(
                epochs,
                delay,
                architecture,
                lr,
                lr_decay,
                seed,
                user_metadata,
                file,
                training_directory,
                save_plot,
                silent,
                index
            )
        except Exception as e:
            print(f"Error training file {file}: {e}")
            continue

def _get_model_name(file):
    if file == _OUTPUT_BASENAME:
        return 'model'
    return Path(file).stem

def _run_single_file(
    epochs: int = 100,
    delay: Optional[int] = None,
    architecture: str = "standard",
    lr: float = 0.004,
    lr_decay: float = 0.007,
    seed: Optional[int] = 0,
    user_metadata: Optional[UserMetadata] = None,
    output_file: str = _OUTPUT_BASENAME,
    train_path: str = _TRAIN_PATH,
    save_plot: bool = False,
    silent: bool = False,
    index: int = 0,
):

    input_version, input_basename = _check_for_input_file()

    modelname = _get_model_name(output_file)
    model = train(
        input_basename,
        output_file,
        train_path,
        input_version=input_version,
        epochs=epochs,
        delay=delay,
        architecture=architecture,
        lr=lr,
        lr_decay=lr_decay,
        seed=seed,
        save_plot=save_plot,
        silent=silent,
        modelname=modelname
    )

    print("Exporting your model...")
    model_export_outdir = _get_valid_export_directory(reuse=(index > 0))
    model_export_outdir.mkdir(parents=True, exist_ok=(index > 0))
    model.net.export(outdir=model_export_outdir, basename=modelname, user_metadata=user_metadata)
    if save_plot:
        _move_plot_to_export_dir(model_export_outdir, output_file)
    print(f"* Model exported to {model_export_outdir}/{modelname}.nam. Enjoy!\n")

def _move_plot_to_export_dir(model_export_outdir, output_file):
    plot_path = Path(output_file[:output_file.rindex('.')] + ".png")
    if plot_path.exists():
        plot_path.rename(Path(model_export_outdir, plot_path.name))
