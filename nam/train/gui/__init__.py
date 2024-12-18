# File: gui.py
# Created Date: Saturday February 25th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
GUI for training

Usage:
>>> from nam.train.gui import run
>>> run()
"""

import abc as _abc
import re as _re
import requests as _requests
import tkinter as _tk
import subprocess as _subprocess
import sys as _sys
import webbrowser as _webbrowser
from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from functools import partial as _partial

try:  # Not supported in Colab
    from idlelib.tooltip import Hovertip
except ModuleNotFoundError:
    # Hovertips won't work
    class Hovertip(object):
        """
        Shell class
        """

        def __init__(self, *args, **kwargs):
            pass


from pathlib import Path as _Path
from tkinter import filedialog as _filedialog
from typing import (
    Any as _Any,
    Callable as _Callable,
    Dict as _Dict,
    NamedTuple as _NamedTuple,
    Optional as _Optional,
    Sequence as _Sequence,
)

try:  # 3rd-party and 1st-party imports
    import torch as _torch

    from nam import __version__
    from nam.data import Split as _Split
    from nam.train import core as _core
    from nam.train.gui._resources import settings as _settings
    from nam.models.metadata import (
        GearType as _GearType,
        UserMetadata as _UserMetadata,
        ToneType as _ToneType,
    )

    # Ok private access here--this is technically allowed access
    from nam.train import metadata as _metadata
    from nam.train._names import (
        INPUT_BASENAMES as _INPUT_BASENAMES,
        LATEST_VERSION as _LATEST_VERSION,
    )
    from nam.train._version import (
        Version as _Version,
        get_current_version as _get_current_version,
    )

    _install_is_valid = True
    _HAVE_ACCELERATOR = _torch.cuda.is_available() or _torch.backends.mps.is_available()
except ImportError:
    _install_is_valid = False
    _HAVE_ACCELERATOR = False

if _HAVE_ACCELERATOR:
    _DEFAULT_NUM_EPOCHS = 100
    _DEFAULT_BATCH_SIZE = 16
    _DEFAULT_LR_DECAY = 0.007
else:
    _DEFAULT_NUM_EPOCHS = 20
    _DEFAULT_BATCH_SIZE = 1
    _DEFAULT_LR_DECAY = 0.05
_BUTTON_WIDTH = 20
_BUTTON_HEIGHT = 2
_TEXT_WIDTH = 70

_DEFAULT_DELAY = None
_DEFAULT_IGNORE_CHECKS = False
_DEFAULT_THRESHOLD_ESR = None

_ADVANCED_OPTIONS_LEFT_WIDTH = 12
_ADVANCED_OPTIONS_RIGHT_WIDTH = 12
_METADATA_LEFT_WIDTH = 19
_METADATA_RIGHT_WIDTH = 60


def _is_mac() -> bool:
    return _sys.platform == "darwin"


_SYSTEM_TEXT_COLOR = "systemTextColor" if _is_mac() else "black"


@_dataclass
class AdvancedOptions(object):
    """
    :param architecture: Which architecture to use.
    :param num_epochs: How many epochs to train for.
    :param latency: Latency between the input and output audio, in samples.
        None means we don't know and it has to be calibrated.
    :param ignore_checks: Keep going even if a check says that something is wrong.
    :param threshold_esr: Stop training if the ESR gets better than this. If None, don't
        stop.
    """

    architecture: _core.Architecture
    num_epochs: int
    latency: _Optional[int]
    ignore_checks: bool
    threshold_esr: _Optional[float]


class _PathType(_Enum):
    FILE = "file"
    DIRECTORY = "directory"
    MULTIFILE = "multifile"


class _PathButton(object):
    """
    Button and the path
    """

    def __init__(
        self,
        frame: _tk.Frame,
        button_text: str,
        info_str: str,
        path_type: _PathType,
        path_key: _settings.PathKey,
        hooks: _Optional[_Sequence[_Callable[[], None]]] = None,
        color_when_not_set: str = "#EF0000",  # Darker red
        color_when_set: str = _SYSTEM_TEXT_COLOR,
        default: _Optional[_Path] = None,
    ):
        """
        :param hooks: Callables run at the end of setting the value.
        """
        self._button_text = button_text
        self._info_str = info_str
        self._path: _Optional[_Path] = default
        self._path_type = path_type
        self._path_key = path_key
        self._frame = frame
        self._widgets = {}
        self._widgets["button"] = _tk.Button(
            self._frame,
            text=button_text,
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._set_val,
        )
        self._widgets["button"].pack(side=_tk.LEFT)
        self._widgets["label"] = _tk.Label(
            self._frame,
            width=_TEXT_WIDTH,
            height=_BUTTON_HEIGHT,
            bg=None,
            anchor="w",
        )
        self._widgets["label"].pack(side=_tk.LEFT)
        self._hooks = hooks
        self._color_when_not_set = color_when_not_set
        self._color_when_set = color_when_set
        self._set_text()

    def __setitem__(self, key, val):
        """
        Implement tk-style setter for state
        """
        if key == "state":
            for widget in self._widgets.values():
                widget["state"] = val
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} instance does not support item assignment for non-state key {key}!"
            )

    @property
    def val(self) -> _Optional[_Path]:
        return self._path

    def _set_text(self):
        if self._path is None:
            self._widgets["label"]["fg"] = self._color_when_not_set
            self._widgets["label"]["text"] = self._info_str
        else:
            val = self.val
            val = val[0] if isinstance(val, tuple) and len(val) == 1 else val
            self._widgets["label"]["fg"] = self._color_when_set
            self._widgets["label"][
                "text"
            ] = f"{self._button_text.capitalize()} set to {val}"

    def _set_val(self):
        last_path = _settings.get_last_path(self._path_key)
        if last_path is None:
            initial_dir = None
        elif not last_path.is_dir():
            initial_dir = last_path.parent
        else:
            initial_dir = last_path
        result = {
            _PathType.FILE: _filedialog.askopenfilename,
            _PathType.DIRECTORY: _filedialog.askdirectory,
            _PathType.MULTIFILE: _filedialog.askopenfilenames,
        }[self._path_type](initialdir=str(initial_dir))
        if result != "":
            self._path = result
            _settings.set_last_path(
                self._path_key,
                _Path(result[0] if self._path_type == _PathType.MULTIFILE else result),
            )
        self._set_text()

        if self._hooks is not None:
            for h in self._hooks:
                h()


class _InputPathButton(_PathButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Download the training file!
        self._widgets["button_download_input"] = _tk.Button(
            self._frame,
            text="Download input file",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._download_input_file,
        )
        self._widgets["button_download_input"].pack(side=_tk.RIGHT)

    @classmethod
    def _download_input_file(cls):
        file_urls = {
            "input.wav": "https://drive.google.com/file/d/1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G/view?usp=drive_link",
            "v3_0_0.wav": "https://drive.google.com/file/d/1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9/view?usp=drive_link",
            "v2_0_0.wav": "https://drive.google.com/file/d/1xnyJP_IZ7NuyDSTJfn-Jmc5lw0IE7nfu/view?usp=drive_link",
            "v1_1_1.wav": "",
            "v1.wav": "",
        }
        # Pick the most recent file.
        for input_basename in _INPUT_BASENAMES:
            name = input_basename.name
            url = file_urls.get(name)
            if url:
                if name != _LATEST_VERSION.name:
                    print(
                        f"WARNING: File {name} is out of date. "
                        "This needs to be updated!"
                    )
                _webbrowser.open(url)
                return


class _CheckboxKeys(_Enum):
    """
    Keys for checkboxes
    """

    SILENT_TRAINING = "silent_training"
    SAVE_PLOT = "save_plot"


class _TopLevelWithOk(_tk.Toplevel):
    """
    Toplevel with an Ok button (provide yourself!)
    """

    def __init__(
        self, on_ok: _Callable[[None], None], resume_main: _Callable[[None], None]
    ):
        """
        :param on_ok: What to do when "Ok" button is pressed
        """
        super().__init__()
        self._on_ok = on_ok
        self._resume_main = resume_main

    def destroy(self, pressed_ok: bool = False):
        if pressed_ok:
            self._on_ok()
        self._resume_main()
        super().destroy()


class _TopLevelWithYesNo(_tk.Toplevel):
    """
    Toplevel holding functions for yes/no buttons to close
    """

    def __init__(
        self,
        on_yes: _Callable[[None], None],
        on_no: _Callable[[None], None],
        on_close: _Optional[_Callable[[None], None]],
        resume_main: _Callable[[None], None],
    ):
        """
        :param on_yes: What to do when "Yes" button is pressed.
        :param on_no: What to do when "No" button is pressed.
        :param on_close: Do this regardless when closing (via yes/no/x) before
            resuming.
        """
        super().__init__()
        self._on_yes = on_yes
        self._on_no = on_no
        self._on_close = on_close
        self._resume_main = resume_main

    def destroy(self, pressed_yes: bool = False, pressed_no: bool = False):
        if pressed_yes:
            self._on_yes()
        if pressed_no:
            self._on_no()
        if self._on_close is not None:
            self._on_close()
        self._resume_main()
        super().destroy()


class _OkModal(object):
    """
    Message and OK button
    """

    def __init__(self, resume_main, msg: str, label_kwargs: _Optional[dict] = None):
        label_kwargs = {} if label_kwargs is None else label_kwargs

        self._root = _TopLevelWithOk((lambda: None), resume_main)
        self._text = _tk.Label(self._root, text=msg, **label_kwargs)
        self._text.pack()
        self._ok = _tk.Button(
            self._root,
            text="Ok",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_ok=True),
        )
        self._ok.pack()


class _YesNoModal(object):
    """
    Modal w/ yes/no buttons
    """

    def __init__(
        self,
        on_yes: _Callable[[None], None],
        on_no: _Callable[[None], None],
        resume_main,
        msg: str,
        on_close: _Optional[_Callable[[None], None]] = None,
        label_kwargs: _Optional[dict] = None,
    ):
        label_kwargs = {} if label_kwargs is None else label_kwargs
        self._root = _TopLevelWithYesNo(on_yes, on_no, on_close, resume_main)
        self._text = _tk.Label(self._root, text=msg, **label_kwargs)
        self._text.pack()
        self._buttons_frame = _tk.Frame(self._root)
        self._buttons_frame.pack()
        self._yes = _tk.Button(
            self._buttons_frame,
            text="Yes",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_yes=True),
        )
        self._yes.pack(side=_tk.LEFT)
        self._no = _tk.Button(
            self._buttons_frame,
            text="No",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_no=True),
        )
        self._no.pack(side=_tk.RIGHT)


class _GUIWidgets(_Enum):
    INPUT_PATH = "input_path"
    OUTPUT_PATH = "output_path"
    TRAINING_DESTINATION = "training_destination"
    METADATA = "metadata"
    ADVANCED_OPTIONS = "advanced_options"
    TRAIN = "train"
    UPDATE = "update"


@_dataclass
class Checkbox(object):
    variable: _tk.BooleanVar
    check_button: _tk.Checkbutton


class GUI(object):
    def __init__(self):
        self._root = _tk.Tk()
        self._root.title(f"NAM Trainer - v{__version__}")
        self._widgets = {}

        # Buttons for paths:
        self._frame_input = _tk.Frame(self._root)
        self._frame_input.pack(anchor="w")
        self._widgets[_GUIWidgets.INPUT_PATH] = _InputPathButton(
            self._frame_input,
            "Input Audio",
            f"Select input (DI) file (e.g. {_LATEST_VERSION.name})",
            _PathType.FILE,
            _settings.PathKey.INPUT_FILE,
            hooks=[self._check_button_states],
        )

        self._frame_output_path = _tk.Frame(self._root)
        self._frame_output_path.pack(anchor="w")
        self._widgets[_GUIWidgets.OUTPUT_PATH] = _PathButton(
            self._frame_output_path,
            "Output Audio",
            "Select output (reamped) file - (Choose MULTIPLE FILES to enable BATCH TRAINING)",
            _PathType.MULTIFILE,
            _settings.PathKey.OUTPUT_FILE,
            hooks=[self._check_button_states],
        )

        self._frame_train_destination = _tk.Frame(self._root)
        self._frame_train_destination.pack(anchor="w")
        self._widgets[_GUIWidgets.TRAINING_DESTINATION] = _PathButton(
            self._frame_train_destination,
            "Train Destination",
            "Select training output directory",
            _PathType.DIRECTORY,
            _settings.PathKey.TRAINING_DESTINATION,
            hooks=[self._check_button_states],
        )

        # Metadata
        self.user_metadata = _UserMetadata()
        self._frame_metadata = _tk.Frame(self._root)
        self._frame_metadata.pack(anchor="w")
        self._widgets["metadata"] = _tk.Button(
            self._frame_metadata,
            text="Metadata...",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._open_metadata,
        )
        self._widgets["metadata"].pack()
        self.user_metadata_flag = False

        # This should probably be to the right somewhere
        self._get_additional_options_frame()

        # Last frames: avdanced options & train in the SE corner:
        self._frame_advanced_options = _tk.Frame(self._root)
        self._frame_train = _tk.Frame(self._root)
        self._frame_update = _tk.Frame(self._root)
        # Pack must be in reverse order
        self._frame_update.pack(side=_tk.BOTTOM, anchor="e")
        self._frame_train.pack(side=_tk.BOTTOM, anchor="e")
        self._frame_advanced_options.pack(side=_tk.BOTTOM, anchor="e")

        # Advanced options for training
        default_architecture = _core.Architecture.STANDARD
        self.advanced_options = AdvancedOptions(
            default_architecture,
            _DEFAULT_NUM_EPOCHS,
            _DEFAULT_DELAY,
            _DEFAULT_IGNORE_CHECKS,
            _DEFAULT_THRESHOLD_ESR,
        )
        # Window to edit them:

        self._widgets[_GUIWidgets.ADVANCED_OPTIONS] = _tk.Button(
            self._frame_advanced_options,
            text="Advanced options...",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._open_advanced_options,
        )
        self._widgets[_GUIWidgets.ADVANCED_OPTIONS].pack()

        # Train button

        self._widgets[_GUIWidgets.TRAIN] = _tk.Button(
            self._frame_train,
            text="Train",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._train,
        )
        self._widgets[_GUIWidgets.TRAIN].pack()

        self._pack_update_button_if_update_is_available()

        self._check_button_states()

    def core_train_kwargs(self) -> _Dict[str, _Any]:
        """
        Get any additional kwargs to provide to `core.train`
        """
        return {
            "lr": 0.004,
            "lr_decay": _DEFAULT_LR_DECAY,
            "batch_size": _DEFAULT_BATCH_SIZE,
            "seed": 0,
        }

    def get_mrstft_fit(self) -> bool:
        """
        Use a pre-emphasized multi-resolution shot-time Fourier transform loss during
        training.

        This improves agreement in the high frequencies, usually with a minimial loss in
        ESR.
        """
        # Leave this as a public method to anticipate an extension to make it
        # changeable.
        return True

    def _check_button_states(self):
        """
        Determine if any buttons should be disabled
        """
        # Train button is disabled unless all paths are set
        if any(
            pb.val is None
            for pb in (
                self._widgets[_GUIWidgets.INPUT_PATH],
                self._widgets[_GUIWidgets.OUTPUT_PATH],
                self._widgets[_GUIWidgets.TRAINING_DESTINATION],
            )
        ):
            self._widgets[_GUIWidgets.TRAIN]["state"] = _tk.DISABLED
            return
        self._widgets[_GUIWidgets.TRAIN]["state"] = _tk.NORMAL

    def _get_additional_options_frame(self):
        # Checkboxes
        # TODO get these definitions into __init__()
        self._frame_checkboxes = _tk.Frame(self._root)
        self._frame_checkboxes.pack(side=_tk.LEFT)
        row = 1

        def make_checkbox(
            key: _CheckboxKeys, text: str, default_value: bool
        ) -> Checkbox:
            variable = _tk.BooleanVar()
            variable.set(default_value)
            check_button = _tk.Checkbutton(
                self._frame_checkboxes, text=text, variable=variable
            )
            self._checkboxes[key] = Checkbox(variable, check_button)
            self._widgets[key] = check_button  # For tracking in set-all-widgets ops

        self._checkboxes: _Dict[_CheckboxKeys, Checkbox] = dict()
        make_checkbox(
            _CheckboxKeys.SILENT_TRAINING,
            "Silent run (suggested for batch training)",
            False,
        )
        make_checkbox(_CheckboxKeys.SAVE_PLOT, "Save ESR plot automatically", True)

        # Grid them:
        row = 1
        for v in self._checkboxes.values():
            v.check_button.grid(row=row, column=1, sticky="W")
            row += 1

    def mainloop(self):
        self._root.mainloop()

    def _disable(self):
        self._set_all_widget_states_to(_tk.DISABLED)

    def _open_advanced_options(self):
        """
        Open window for advanced options
        """

        self._wait_while_func(lambda resume: AdvancedOptionsGUI(resume, self))

    def _open_metadata(self):
        """
        Open window for metadata
        """

        self._wait_while_func(lambda resume: UserMetadataGUI(resume, self))

    def _pack_update_button(self, version_from: _Version, version_to: _Version):
        """
        Pack a button that a user can click to update
        """

        def update_nam():
            result = _subprocess.run(
                [
                    f"{_sys.executable}",
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "neural-amp-modeler",
                ]
            )
            if result.returncode == 0:
                self._wait_while_func(
                    (lambda resume, *args, **kwargs: _OkModal(resume, *args, **kwargs)),
                    "Update complete! Restart NAM for changes to take effect.",
                )
            else:
                self._wait_while_func(
                    (lambda resume, *args, **kwargs: _OkModal(resume, *args, **kwargs)),
                    "Update failed! See logs.",
                )

        self._widgets[_GUIWidgets.UPDATE] = _tk.Button(
            self._frame_update,
            text=f"Update ({str(version_from)} -> {str(version_to)})",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=update_nam,
        )
        self._widgets[_GUIWidgets.UPDATE].pack()

    def _pack_update_button_if_update_is_available(self):
        class UpdateInfo(_NamedTuple):
            available: bool
            current_version: _Version
            new_version: _Optional[_Version]

        def get_info() -> UpdateInfo:
            # TODO error handling
            url = f"https://api.github.com/repos/sdatkinson/neural-amp-modeler/releases"
            current_version = _get_current_version()
            try:
                response = _requests.get(url)
            except _requests.exceptions.ConnectionError:
                print("WARNING: Failed to reach the server to check for updates")
                return UpdateInfo(
                    available=False, current_version=current_version, new_version=None
                )
            if response.status_code != 200:
                print(f"Failed to fetch releases. Status code: {response.status_code}")
                return UpdateInfo(
                    available=False, current_version=current_version, new_version=None
                )
            else:
                releases = response.json()
                latest_version = None
                if releases:
                    for release in releases:
                        tag = release["tag_name"]
                        if not tag.startswith("v"):
                            print(f"Found invalid version {tag}")
                        else:
                            this_version = _Version.from_string(tag[1:])
                            if latest_version is None or this_version > latest_version:
                                latest_version = this_version
                else:
                    print("No releases found for this repository.")
            update_available = (
                latest_version is not None and latest_version > current_version
            )
            return UpdateInfo(
                available=update_available,
                current_version=current_version,
                new_version=latest_version,
            )

        update_info = get_info()
        if update_info.available:
            self._pack_update_button(
                update_info.current_version, update_info.new_version
            )

    def _resume(self):
        self._set_all_widget_states_to(_tk.NORMAL)
        self._check_button_states()

    def _set_all_widget_states_to(self, state):
        for widget in self._widgets.values():
            widget["state"] = state

    def _train(self):
        input_path = self._widgets[_GUIWidgets.INPUT_PATH].val
        output_paths = self._widgets[_GUIWidgets.OUTPUT_PATH].val
        # Validate all files before running:
        success = self._validate_all_data(input_path, output_paths)
        if success:
            self._train2()

    def _train2(self, ignore_checks=False):
        input_path = self._widgets[_GUIWidgets.INPUT_PATH].val

        # Advanced options:
        num_epochs = self.advanced_options.num_epochs
        architecture = self.advanced_options.architecture
        user_latency = self.advanced_options.latency
        file_list = self._widgets[_GUIWidgets.OUTPUT_PATH].val
        threshold_esr = self.advanced_options.threshold_esr

        # Run it
        for file in file_list:
            print(f"Now training {file}")
            basename = _re.sub(r"\.wav$", "", file.split("/")[-1])
            user_metadata = (
                self.user_metadata if self.user_metadata_flag else _UserMetadata()
            )

            train_output = _core.train(
                input_path,
                file,
                self._widgets[_GUIWidgets.TRAINING_DESTINATION].val,
                epochs=num_epochs,
                latency=user_latency,
                architecture=architecture,
                silent=self._checkboxes[_CheckboxKeys.SILENT_TRAINING].variable.get(),
                save_plot=self._checkboxes[_CheckboxKeys.SAVE_PLOT].variable.get(),
                modelname=basename,
                ignore_checks=ignore_checks,
                local=True,
                fit_mrstft=self.get_mrstft_fit(),
                threshold_esr=threshold_esr,
                user_metadata=user_metadata,
                **self.core_train_kwargs(),
            )

            if train_output.model is None:
                print("Model training failed! Skip exporting...")
                continue
            print("Model training complete!")
            print("Exporting...")
            outdir = self._widgets[_GUIWidgets.TRAINING_DESTINATION].val
            print(f"Exporting trained model to {outdir}...")
            train_output.model.net.export(
                outdir,
                basename=basename,
                user_metadata=user_metadata,
                other_metadata={
                    _metadata.TRAINING_KEY: train_output.metadata.model_dump()
                },
            )
            print("Done!")

        # Metadata was only valid for 1 run (possibly a batch), so make sure it's not
        # used again unless the user re-visits the window and clicks "ok".
        self.user_metadata_flag = False

    def _validate_all_data(
        self, input_path: _Path, output_paths: _Sequence[_Path]
    ) -> bool:
        """
        Validate all the data.
        If something doesn't pass, then alert the user and ask them whether they
        want to continue.

        :return: whether we passed (NOTE: Training in spite of failure is
            triggered by a modal that is produced on failure.)
        """

        def make_message_for_file(
            output_path: str, validation_output: _core.DataValidationOutput
        ) -> str:
            """
            State the file and explain what's wrong with it.
            """
            # TODO put this closer to what it looks at, i.e. core.DataValidationOutput
            msg = (
                f"\t{_Path(output_path).name}:\n"  # They all have the same directory so
            )
            if not validation_output.sample_rate.passed:
                msg += (
                    "\t\t There are different sample rates for the input ("
                    f"{validation_output.sample_rate.input}) and output ("
                    f"{validation_output.sample_rate.output}).\n"
                )
            if not validation_output.length.passed:
                msg += (
                    "\t\t* The input and output audio files are too different in length"
                )
                if validation_output.length.delta_seconds > 0:
                    msg += (
                        f" (the output is {validation_output.length.delta_seconds:.2f} "
                        "seconds longer than the input)\n"
                    )
                else:
                    msg += (
                        f" (the output is {-validation_output.length.delta_seconds:.2f}"
                        " seconds shorter than the input)\n"
                    )
            if validation_output.latency.manual is None:
                if validation_output.latency.calibration.warnings.matches_lookahead:
                    msg += (
                        "\t\t* The calibrated latency is the maximum allowed. This is "
                        "probably because the latency calibration was triggered by noise.\n"
                    )
                if validation_output.latency.calibration.warnings.disagreement_too_high:
                    msg += "\t\t* The calculated latencies are too different from each other.\n"
            if not validation_output.checks.passed:
                msg += "\t\t* A data check failed (TODO in more detail).\n"
            if not validation_output.pytorch.passed:
                msg += "\t\t* PyTorch data set errors:\n"
                for split in _Split:
                    split_validation = getattr(validation_output.pytorch, split.value)
                    if not split_validation.passed:
                        msg += f"   * {split.value:10s}: {split_validation.msg}\n"
            return msg

        # Validate input
        input_validation = _core.validate_input(input_path)
        if not input_validation.passed:
            self._wait_while_func(
                (lambda resume, *args, **kwargs: _OkModal(resume, *args, **kwargs)),
                f"Input file {input_path} is not recognized as a standardized input "
                "file.\nTraining cannot proceed.",
            )
            return False

        user_latency = self.advanced_options.latency
        file_validation_outputs = {
            output_path: _core.validate_data(
                input_path,
                output_path,
                user_latency,
            )
            for output_path in output_paths
        }
        if any(not fv.passed for fv in file_validation_outputs.values()):
            msg = "The following output files failed checks:\n" + "".join(
                [
                    make_message_for_file(output_path, fv)
                    for output_path, fv in file_validation_outputs.items()
                    if not fv.passed
                ]
            )
            if all(fv.passed_critical for fv in file_validation_outputs.values()):
                msg += "\nIgnore and proceed?"

                # Hacky to listen to the modal:
                modal_listener = {"proceed": False, "still_open": True}

                def on_yes():
                    modal_listener["proceed"] = True

                def on_no():
                    modal_listener["proceed"] = False

                def on_close():
                    if modal_listener["proceed"]:
                        self._train2(ignore_checks=True)

                self._wait_while_func(
                    (
                        lambda resume, on_yes, on_no, *args, **kwargs: _YesNoModal(
                            on_yes, on_no, resume, *args, **kwargs
                        )
                    ),
                    on_yes=on_yes,
                    on_no=on_no,
                    msg=msg,
                    on_close=on_close,
                    label_kwargs={"justify": "left"},
                )
                return False  # we still failed checks so say so.
            else:
                msg += "\nCritical errors found, cannot ignore."
                self._wait_while_func(
                    lambda resume, msg, **kwargs: _OkModal(resume, msg, **kwargs),
                    msg=msg,
                    label_kwargs={"justify": "left"},
                )
                return False

        return True

    def _wait_while_func(self, func, *args, **kwargs):
        """
        Disable this GUI while something happens.
        That function _needs_ to call the provided self._resume when it's ready to
        release me!
        """
        self._disable()
        func(self._resume, *args, **kwargs)


# some typing functions
def _non_negative_int(val):
    val = int(val)
    if val < 0:
        val = 0
    return val


class _TypeOrNull(object):
    def __init__(self, T, null_str=""):
        """
        :param T: tpe to cast to on .forward()
        """
        self._T = T
        self._null_str = null_str

    @property
    def null_str(self) -> str:
        """
        What str is displayed when for "None"
        """
        return self._null_str

    def forward(self, val: str):
        val = val.rstrip()
        return None if val == self._null_str else self._T(val)

    def inverse(self, val) -> str:
        return self._null_str if val is None else str(val)


_int_or_null = _TypeOrNull(int)
_float_or_null = _TypeOrNull(float)


def _rstripped_str(val):
    return str(val).rstrip()


class _SettingWidget(_abc.ABC):
    """
    A widget for the user to interact with to set something
    """

    @_abc.abstractmethod
    def get(self):
        pass


class LabeledOptionMenu(_SettingWidget):
    """
    Label (left) and radio buttons (right)
    """

    def __init__(
        self,
        frame: _tk.Frame,
        label: str,
        choices: _Enum,
        default: _Optional[_Enum] = None,
    ):
        """
        :param command: Called to propagate option selection. Is provided with the
            value corresponding to the radio button selected.
        """
        self._frame = frame
        self._choices = choices
        height = _BUTTON_HEIGHT
        bg = None
        self._label = _tk.Label(
            frame,
            width=_ADVANCED_OPTIONS_LEFT_WIDTH,
            height=height,
            bg=bg,
            anchor="w",
            text=label,
        )
        self._label.pack(side=_tk.LEFT)

        frame_menu = _tk.Frame(frame)
        frame_menu.pack(side=_tk.RIGHT)

        self._selected_value = None
        default = (list(choices)[0] if default is None else default).value
        self._menu = _tk.OptionMenu(
            frame_menu,
            _tk.StringVar(master=frame, value=default, name=label),
            # default,
            *[choice.value for choice in choices],  #  if choice.value!=default],
            command=self._set,
        )
        self._menu.config(width=_ADVANCED_OPTIONS_RIGHT_WIDTH)
        self._menu.pack(side=_tk.RIGHT)
        # Initialize
        self._set(default)

    def get(self) -> _Enum:
        return self._selected_value

    def _set(self, val: str):
        """
        Set the value selected
        """
        self._selected_value = self._choices(val)


class _Hovertip(Hovertip):
    """
    Adjustments:

    * Always black text (macOS)
    """

    def showcontents(self):
        # Override
        label = _tk.Label(
            self.tipwindow,
            text=self.text,
            justify=_tk.LEFT,
            background="#ffffe0",
            relief=_tk.SOLID,
            borderwidth=1,
            fg="black",
        )
        label.pack()


class LabeledText(_SettingWidget):
    """
    Label (left) and text input (right)
    """

    def __init__(
        self,
        frame: _tk.Frame,
        label: str,
        default=None,
        type=None,
        left_width=_ADVANCED_OPTIONS_LEFT_WIDTH,
        right_width=_ADVANCED_OPTIONS_RIGHT_WIDTH,
    ):
        """
        :param command: Called to propagate option selection. Is provided with the
            value corresponding to the radio button selected.
        :param type: If provided, casts value to given type
        :param left_width: How much space to use on the left side (text)
        :param right_width: How much space for the Text field
        """
        self._frame = frame
        label_height = 2
        text_height = 1
        self._label = _tk.Label(
            frame,
            width=left_width,
            height=label_height,
            bg=None,
            anchor="e",
            text=label,
        )
        self._label.pack(side=_tk.LEFT)

        self._text = _tk.Text(
            frame,
            width=right_width,
            height=text_height,
            bg=None,
        )
        self._text.pack(side=_tk.RIGHT)

        self._type = (lambda x: x) if type is None else type

        if default is not None:
            self._text.insert("1.0", str(default))

        # You can assign a tooltip for the label if you'd like.
        self.label_tooltip: _Optional[_Hovertip] = None

    @property
    def label(self) -> _tk.Label:
        return self._label

    def get(self):
        """
        Attempt to get and return the value.
        May throw a tk.TclError indicating something went wrong getting the value.
        """
        # "1.0" means Line 1, character zero (wat)
        return self._type(self._text.get("1.0", _tk.END))


class AdvancedOptionsGUI(object):
    """
    A window to hold advanced options (Architecture and number of epochs)
    """

    def __init__(self, resume_main, parent: GUI):
        self._parent = parent
        self._root = _TopLevelWithOk(self.apply, resume_main)
        self._root.title("Advanced Options")

        self.pack()

        # "Ok": apply and destroy
        self._frame_ok = _tk.Frame(self._root)
        self._frame_ok.pack()
        self._button_ok = _tk.Button(
            self._frame_ok,
            text="Ok",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_ok=True),
        )
        self._button_ok.pack()

    def apply(self):
        """
        Set values to parent and destroy this object
        """

        def safe_apply(name):
            try:
                setattr(
                    self._parent.advanced_options, name, getattr(self, "_" + name).get()
                )
            except ValueError:
                pass

        # TODO could clean up more / see `.pack_options()`
        for name in ("architecture", "num_epochs", "latency", "threshold_esr"):
            safe_apply(name)

    def pack(self):
        # TODO things that are `_SettingWidget`s are named carefully, need to make this
        # easier to work with.

        # Architecture: radio buttons
        self._frame_architecture = _tk.Frame(self._root)
        self._frame_architecture.pack()
        self._architecture = LabeledOptionMenu(
            self._frame_architecture,
            "Architecture",
            _core.Architecture,
            default=self._parent.advanced_options.architecture,
        )

        # Number of epochs: text box
        self._frame_epochs = _tk.Frame(self._root)
        self._frame_epochs.pack()

        self._num_epochs = LabeledText(
            self._frame_epochs,
            "Epochs",
            default=str(self._parent.advanced_options.num_epochs),
            type=_non_negative_int,
        )

        # Delay: text box
        self._frame_latency = _tk.Frame(self._root)
        self._frame_latency.pack()

        self._latency = LabeledText(
            self._frame_latency,
            "Reamp latency",
            default=_int_or_null.inverse(self._parent.advanced_options.latency),
            type=_int_or_null.forward,
        )

        # Threshold ESR
        self._frame_threshold_esr = _tk.Frame(self._root)
        self._frame_threshold_esr.pack()
        self._threshold_esr = LabeledText(
            self._frame_threshold_esr,
            "Threshold ESR",
            default=_float_or_null.inverse(self._parent.advanced_options.threshold_esr),
            type=_float_or_null.forward,
        )


class UserMetadataGUI(object):
    # Things that are auto-filled:
    # Model date
    # gain
    def __init__(self, resume_main, parent: GUI):
        self._parent = parent
        self._root = _TopLevelWithOk(self.apply, resume_main)
        self._root.title("Metadata")

        # Pack all the widgets
        self.pack()

        # "Ok": apply and destroy
        self._frame_ok = _tk.Frame(self._root)
        self._frame_ok.pack()
        self._button_ok = _tk.Button(
            self._frame_ok,
            text="Ok",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_ok=True),
        )
        self._button_ok.pack()

    def apply(self):
        """
        Set values to parent and destroy this object
        """

        def safe_apply(name):
            try:
                setattr(
                    self._parent.user_metadata, name, getattr(self, "_" + name).get()
                )
            except ValueError:
                pass

        # TODO could clean up more / see `.pack()`
        for name in (
            "name",
            "modeled_by",
            "gear_make",
            "gear_model",
            "gear_type",
            "tone_type",
            "input_level_dbu",
            "output_level_dbu",
        ):
            safe_apply(name)
        self._parent.user_metadata_flag = True

    def pack(self):
        # TODO things that are `_SettingWidget`s are named carefully, need to make this
        # easier to work with.

        LabeledText_ = _partial(
            LabeledText,
            left_width=_METADATA_LEFT_WIDTH,
            right_width=_METADATA_RIGHT_WIDTH,
        )
        parent = self._parent

        # Name
        self._frame_name = _tk.Frame(self._root)
        self._frame_name.pack()
        self._name = LabeledText_(
            self._frame_name,
            "NAM name",
            default=parent.user_metadata.name,
            type=_rstripped_str,
        )
        # Modeled by
        self._frame_modeled_by = _tk.Frame(self._root)
        self._frame_modeled_by.pack()
        self._modeled_by = LabeledText_(
            self._frame_modeled_by,
            "Modeled by",
            default=parent.user_metadata.modeled_by,
            type=_rstripped_str,
        )
        # Gear make
        self._frame_gear_make = _tk.Frame(self._root)
        self._frame_gear_make.pack()
        self._gear_make = LabeledText_(
            self._frame_gear_make,
            "Gear make",
            default=parent.user_metadata.gear_make,
            type=_rstripped_str,
        )
        # Gear model
        self._frame_gear_model = _tk.Frame(self._root)
        self._frame_gear_model.pack()
        self._gear_model = LabeledText_(
            self._frame_gear_model,
            "Gear model",
            default=parent.user_metadata.gear_model,
            type=_rstripped_str,
        )
        # Calibration: input & output dBu
        self._frame_input_dbu = _tk.Frame(self._root)
        self._frame_input_dbu.pack()
        self._input_level_dbu = LabeledText_(
            self._frame_input_dbu,
            "Reamp send level (dBu)",
            default=_float_or_null.inverse(parent.user_metadata.input_level_dbu),
            type=_float_or_null.forward,
        )
        self._input_level_dbu.label_tooltip = _Hovertip(
            anchor_widget=self._input_level_dbu.label,
            text=(
                "(Ok to leave blank)\n\n"
                "Play a sine wave with frequency 1kHz and peak amplitude 0dBFS. Use\n"
                "a multimeter to measure the RMS voltage of the signal at the jack\n"
                "that connects to your gear, and convert to dBu.\n"
                "Record the value here."
            ),
        )
        self._frame_output_dbu = _tk.Frame(self._root)
        self._frame_output_dbu.pack()
        self._output_level_dbu = LabeledText_(
            self._frame_output_dbu,
            "Reamp return level (dBu)",
            default=_float_or_null.inverse(parent.user_metadata.output_level_dbu),
            type=_float_or_null.forward,
        )
        self._output_level_dbu.label_tooltip = _Hovertip(
            anchor_widget=self._output_level_dbu.label,
            text=(
                "(Ok to leave blank)\n\n"
                "Play a sine wave with frequency 1kHz into your interface where\n"
                "you're recording your gear. Keeping the interface's input gain\n"
                "trimmed as you will use it when recording, adjust the sine wave\n"
                "until the input peaks at exactly 0dBFS in your DAW. Measure the RMS\n"
                "voltage and convert to dBu.\n"
                "Record the value here."
            ),
        )
        # Gear type
        self._frame_gear_type = _tk.Frame(self._root)
        self._frame_gear_type.pack()
        self._gear_type = LabeledOptionMenu(
            self._frame_gear_type,
            "Gear type",
            _GearType,
            default=parent.user_metadata.gear_type,
        )
        # Tone type
        self._frame_tone_type = _tk.Frame(self._root)
        self._frame_tone_type.pack()
        self._tone_type = LabeledOptionMenu(
            self._frame_tone_type,
            "Tone type",
            _ToneType,
            default=parent.user_metadata.tone_type,
        )


def _install_error():
    window = _tk.Tk()
    window.title("ERROR")
    label = _tk.Label(
        window,
        width=45,
        height=2,
        text="The NAM training software has not been installed correctly.",
    )
    label.pack()
    button = _tk.Button(window, width=10, height=2, text="Quit", command=window.destroy)
    button.pack()
    window.mainloop()


def run():
    if _install_is_valid:
        _gui = GUI()
        _gui.mainloop()
        print("Shut down NAM trainer")
    else:
        _install_error()


if __name__ == "__main__":
    run()
