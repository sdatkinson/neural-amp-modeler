# File: gui.py
# Created Date: Saturday February 25th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
GUI for training

Usage:
>>> from nam.train.gui import run
>>> run()
"""

import re
import requests
import tkinter as tk
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict, NamedTuple, Optional, Sequence

try:  # 3rd-party and 1st-party imports
    import torch

    from nam import __version__
    from nam.data import Split
    from nam.train import core
    from nam.train.gui._resources import settings
    from nam.models.metadata import GearType, UserMetadata, ToneType

    # Ok private access here--this is technically allowed access
    from nam.train import metadata
    from nam.train._names import INPUT_BASENAMES, LATEST_VERSION
    from nam.train._version import Version, get_current_version

    _install_is_valid = True
    _HAVE_ACCELERATOR = torch.cuda.is_available() or torch.backends.mps.is_available()
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
_METADATA_RIGHT_WIDTH = 60


def _is_mac() -> bool:
    return sys.platform == "darwin"


_SYSTEM_TEXT_COLOR = "systemTextColor" if _is_mac() else "black"


@dataclass
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

    architecture: core.Architecture
    num_epochs: int
    latency: Optional[int]
    ignore_checks: bool
    threshold_esr: Optional[float]


class _PathType(Enum):
    FILE = "file"
    DIRECTORY = "directory"
    MULTIFILE = "multifile"


class _PathButton(object):
    """
    Button and the path
    """

    def __init__(
        self,
        frame: tk.Frame,
        button_text: str,
        info_str: str,
        path_type: _PathType,
        path_key: settings.PathKey,
        hooks: Optional[Sequence[Callable[[], None]]] = None,
        color_when_not_set: str = "#EF0000",  # Darker red
        color_when_set: str = _SYSTEM_TEXT_COLOR,
        default: Optional[Path] = None,
    ):
        """
        :param hooks: Callables run at the end of setting the value.
        """
        self._button_text = button_text
        self._info_str = info_str
        self._path: Optional[Path] = default
        self._path_type = path_type
        self._path_key = path_key
        self._frame = frame
        self._widgets = {}
        self._widgets["button"] = tk.Button(
            self._frame,
            text=button_text,
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._set_val,
        )
        self._widgets["button"].pack(side=tk.LEFT)
        self._widgets["label"] = tk.Label(
            self._frame,
            width=_TEXT_WIDTH,
            height=_BUTTON_HEIGHT,
            bg=None,
            anchor="w",
        )
        self._widgets["label"].pack(side=tk.LEFT)
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
    def val(self) -> Optional[Path]:
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
        last_path = settings.get_last_path(self._path_key)
        if last_path is None:
            initial_dir = None
        elif not last_path.is_dir():
            initial_dir = last_path.parent
        else:
            initial_dir = last_path
        result = {
            _PathType.FILE: filedialog.askopenfilename,
            _PathType.DIRECTORY: filedialog.askdirectory,
            _PathType.MULTIFILE: filedialog.askopenfilenames,
        }[self._path_type](initialdir=str(initial_dir))
        if result != "":
            self._path = result
            settings.set_last_path(
                self._path_key,
                Path(result[0] if self._path_type == _PathType.MULTIFILE else result),
            )
        self._set_text()

        if self._hooks is not None:
            for h in self._hooks:
                h()


class _InputPathButton(_PathButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Download the training file!
        self._widgets["button_download_input"] = tk.Button(
            self._frame,
            text="Download input file",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._download_input_file,
        )
        self._widgets["button_download_input"].pack(side=tk.RIGHT)

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
        for input_basename in INPUT_BASENAMES:
            name = input_basename.name
            url = file_urls.get(name)
            if url:
                if name != LATEST_VERSION.name:
                    print(
                        f"WARNING: File {name} is out of date. "
                        "This needs to be updated!"
                    )
                webbrowser.open(url)
                return


class _ClearablePathButton(_PathButton):
    """
    Can clear a path
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, color_when_not_set="black", **kwargs)
        # Download the training file!
        self._widgets["button_clear"] = tk.Button(
            self._frame,
            text="Clear",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._clear_path,
        )
        self._widgets["button_clear"].pack(side=tk.RIGHT)

    def _clear_path(self):
        self._path = None
        self._set_text()


class _CheckboxKeys(Enum):
    """
    Keys for checkboxes
    """

    SILENT_TRAINING = "silent_training"
    SAVE_PLOT = "save_plot"


class _TopLevelWithOk(tk.Toplevel):
    """
    Toplevel with an Ok button (provide yourself!)
    """

    def __init__(
        self, on_ok: Callable[[None], None], resume_main: Callable[[None], None]
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


class _TopLevelWithYesNo(tk.Toplevel):
    """
    Toplevel holding functions for yes/no buttons to close
    """

    def __init__(
        self,
        on_yes: Callable[[None], None],
        on_no: Callable[[None], None],
        on_close: Optional[Callable[[None], None]],
        resume_main: Callable[[None], None],
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

    def __init__(self, resume_main, msg: str):
        self._root = _TopLevelWithOk((lambda: None), resume_main)
        self._text = tk.Label(self._root, text=msg)
        self._text.pack()
        self._ok = tk.Button(
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
        on_yes: Callable[[None], None],
        on_no: Callable[[None], None],
        resume_main,
        msg: str,
        on_close: Optional[Callable[[None], None]] = None,
        label_kwargs: Optional[dict] = None,
    ):
        label_kwargs = {} if label_kwargs is None else label_kwargs
        self._root = _TopLevelWithYesNo(on_yes, on_no, on_close, resume_main)
        self._text = tk.Label(self._root, text=msg, **label_kwargs)
        self._text.pack()
        self._buttons_frame = tk.Frame(self._root)
        self._buttons_frame.pack()
        self._yes = tk.Button(
            self._buttons_frame,
            text="Yes",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_yes=True),
        )
        self._yes.pack(side=tk.LEFT)
        self._no = tk.Button(
            self._buttons_frame,
            text="No",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_no=True),
        )
        self._no.pack(side=tk.RIGHT)


class _GUIWidgets(Enum):
    INPUT_PATH = "input_path"
    OUTPUT_PATH = "output_path"
    TRAINING_DESTINATION = "training_destination"
    METADATA = "metadata"
    ADVANCED_OPTIONS = "advanced_options"
    TRAIN = "train"
    UPDATE = "update"


@dataclass
class Checkbox(object):
    variable: tk.BooleanVar
    check_button: tk.Checkbutton


class GUI(object):
    def __init__(self):
        self._root = tk.Tk()
        self._root.title(f"NAM Trainer - v{__version__}")
        self._widgets = {}

        # Buttons for paths:
        self._frame_input = tk.Frame(self._root)
        self._frame_input.pack(anchor="w")
        self._widgets[_GUIWidgets.INPUT_PATH] = _InputPathButton(
            self._frame_input,
            "Input Audio",
            f"Select input (DI) file (e.g. {LATEST_VERSION.name})",
            _PathType.FILE,
            settings.PathKey.INPUT_FILE,
            hooks=[self._check_button_states],
        )

        self._frame_output_path = tk.Frame(self._root)
        self._frame_output_path.pack(anchor="w")
        self._widgets[_GUIWidgets.OUTPUT_PATH] = _PathButton(
            self._frame_output_path,
            "Output Audio",
            "Select output (reamped) file - (Choose MULTIPLE FILES to enable BATCH TRAINING)",
            _PathType.MULTIFILE,
            settings.PathKey.OUTPUT_FILE,
            hooks=[self._check_button_states],
        )

        self._frame_train_destination = tk.Frame(self._root)
        self._frame_train_destination.pack(anchor="w")
        self._widgets[_GUIWidgets.TRAINING_DESTINATION] = _PathButton(
            self._frame_train_destination,
            "Train Destination",
            "Select training output directory",
            _PathType.DIRECTORY,
            settings.PathKey.TRAINING_DESTINATION,
            hooks=[self._check_button_states],
        )

        # Metadata
        self.user_metadata = UserMetadata()
        self._frame_metadata = tk.Frame(self._root)
        self._frame_metadata.pack(anchor="w")
        self._widgets["metadata"] = tk.Button(
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
        self._frame_advanced_options = tk.Frame(self._root)
        self._frame_train = tk.Frame(self._root)
        self._frame_update = tk.Frame(self._root)
        # Pack must be in reverse order
        self._frame_update.pack(side=tk.BOTTOM, anchor="e")
        self._frame_train.pack(side=tk.BOTTOM, anchor="e")
        self._frame_advanced_options.pack(side=tk.BOTTOM, anchor="e")

        # Advanced options for training
        default_architecture = core.Architecture.STANDARD
        self.advanced_options = AdvancedOptions(
            default_architecture,
            _DEFAULT_NUM_EPOCHS,
            _DEFAULT_DELAY,
            _DEFAULT_IGNORE_CHECKS,
            _DEFAULT_THRESHOLD_ESR,
        )
        # Window to edit them:

        self._widgets[_GUIWidgets.ADVANCED_OPTIONS] = tk.Button(
            self._frame_advanced_options,
            text="Advanced options...",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._open_advanced_options,
        )
        self._widgets[_GUIWidgets.ADVANCED_OPTIONS].pack()

        # Train button

        self._widgets[_GUIWidgets.TRAIN] = tk.Button(
            self._frame_train,
            text="Train",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=self._train,
        )
        self._widgets[_GUIWidgets.TRAIN].pack()

        self._pack_update_button_if_update_is_available()

        self._check_button_states()

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
            self._widgets[_GUIWidgets.TRAIN]["state"] = tk.DISABLED
            return
        self._widgets[_GUIWidgets.TRAIN]["state"] = tk.NORMAL

    def _get_additional_options_frame(self):
        # Checkboxes
        # TODO get these definitions into __init__()
        self._frame_checkboxes = tk.Frame(self._root)
        self._frame_checkboxes.pack(side=tk.LEFT)
        row = 1

        def make_checkbox(
            key: _CheckboxKeys, text: str, default_value: bool
        ) -> Checkbox:
            variable = tk.BooleanVar()
            variable.set(default_value)
            check_button = tk.Checkbutton(
                self._frame_checkboxes, text=text, variable=variable
            )
            self._checkboxes[key] = Checkbox(variable, check_button)
            self._widgets[key] = check_button  # For tracking in set-all-widgets ops

        self._checkboxes: Dict[_CheckboxKeys, Checkbox] = dict()
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
        self._set_all_widget_states_to(tk.DISABLED)

    def _open_advanced_options(self):
        """
        Open window for advanced options
        """

        self._wait_while_func(lambda resume: AdvancedOptionsGUI(resume, self))

    def _open_metadata(self):
        """
        Open window for metadata
        """

        self._wait_while_func(lambda resume: _UserMetadataGUI(resume, self))

    def _pack_update_button(self, version_from: Version, version_to: Version):
        """
        Pack a button that a user can click to update
        """

        def update_nam():
            result = subprocess.run(
                [
                    f"{sys.executable}",
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

        self._widgets[_GUIWidgets.UPDATE] = tk.Button(
            self._frame_update,
            text=f"Update ({str(version_from)} -> {str(version_to)})",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=update_nam,
        )
        self._widgets[_GUIWidgets.UPDATE].pack()

    def _pack_update_button_if_update_is_available(self):
        class UpdateInfo(NamedTuple):
            available: bool
            current_version: Version
            new_version: Optional[Version]

        def get_info() -> UpdateInfo:
            # TODO error handling
            url = f"https://api.github.com/repos/sdatkinson/neural-amp-modeler/releases"
            current_version = get_current_version()
            try:
                response = requests.get(url)
            except requests.exceptions.ConnectionError:
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
                            this_version = Version.from_string(tag[1:])
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
        self._set_all_widget_states_to(tk.NORMAL)
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

        # Advanced-er options
        # If you're poking around looking for these, then maybe it's time to learn to
        # use the command-line scripts ;)
        lr = 0.004
        lr_decay = _DEFAULT_LR_DECAY
        batch_size = _DEFAULT_BATCH_SIZE
        seed = 0
        # Run it
        for file in file_list:
            print(f"Now training {file}")
            basename = re.sub(r"\.wav$", "", file.split("/")[-1])
            user_metadata = (
                self.user_metadata if self.user_metadata_flag else UserMetadata()
            )

            train_output = core.train(
                input_path,
                file,
                self._widgets[_GUIWidgets.TRAINING_DESTINATION].val,
                epochs=num_epochs,
                latency=user_latency,
                architecture=architecture,
                batch_size=batch_size,
                lr=lr,
                lr_decay=lr_decay,
                seed=seed,
                silent=self._checkboxes[_CheckboxKeys.SILENT_TRAINING].variable.get(),
                save_plot=self._checkboxes[_CheckboxKeys.SAVE_PLOT].variable.get(),
                modelname=basename,
                ignore_checks=ignore_checks,
                local=True,
                fit_mrstft=self.get_mrstft_fit(),
                threshold_esr=threshold_esr,
                user_metadata=user_metadata,
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
                    metadata.TRAINING_KEY: train_output.metadata.model_dump()
                },
            )
            print("Done!")

        # Metadata was only valid for 1 run, so make sure it's not used again unless
        # the user re-visits the window and clicks "ok"
        self.user_metadata_flag = False

    def _validate_all_data(
        self, input_path: Path, output_paths: Sequence[Path]
    ) -> bool:
        """
        Validate all the data.
        If something doesn't pass, then alert the user and ask them whether they
        want to continue.

        :return: whether we passed (NOTE: Training in spite of failure is
            triggered by a modal that is produced on failure.)
        """

        def make_message_for_file(
            output_path: str, validation_output: core.DataValidationOutput
        ) -> str:
            """
            File and explain what's wrong with it.
            """
            # TODO put this closer to what it looks at, i.e. core.DataValidationOutput
            msg = (
                f"\t{Path(output_path).name}:\n"  # They all have the same directory so
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
                for split in Split:
                    split_validation = getattr(validation_output.pytorch, split.value)
                    if not split_validation.passed:
                        msg += f"   * {split.value:10s}: {split_validation.msg}\n"
            return msg

        # Validate input
        input_validation = core.validate_input(input_path)
        if not input_validation.passed:
            self._wait_while_func(
                (lambda resume, *args, **kwargs: _OkModal(resume, *args, **kwargs)),
                f"Input file {input_path} is not recognized as a standardized input "
                "file.\nTraining cannot proceed.",
            )
            return False

        user_latency = self.advanced_options.latency
        file_validation_outputs = {
            output_path: core.validate_data(
                input_path,
                output_path,
                user_latency,
            )
            for output_path in output_paths
        }
        if any(not fv.passed for fv in file_validation_outputs.values()):
            msg = (
                "The following output files failed checks:\n"
                + "".join(
                    [
                        make_message_for_file(output_path, fv)
                        for output_path, fv in file_validation_outputs.items()
                        if not fv.passed
                    ]
                )
                + "\nIgnore and proceed?"
            )

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


def _type_or_null(T, val):
    val = val.rstrip()
    if val == "null":
        return val
    return T(val)


_int_or_null = partial(_type_or_null, int)
_float_or_null = partial(_type_or_null, float)


def _type_or_null_inv(val):
    return "null" if val is None else str(val)


def _rstripped_str(val):
    return str(val).rstrip()


class _LabeledOptionMenu(object):
    """
    Label (left) and radio buttons (right)
    """

    def __init__(
        self, frame: tk.Frame, label: str, choices: Enum, default: Optional[Enum] = None
    ):
        """
        :param command: Called to propagate option selection. Is provided with the
            value corresponding to the radio button selected.
        """
        self._frame = frame
        self._choices = choices
        height = _BUTTON_HEIGHT
        bg = None
        self._label = tk.Label(
            frame,
            width=_ADVANCED_OPTIONS_LEFT_WIDTH,
            height=height,
            bg=bg,
            anchor="w",
            text=label,
        )
        self._label.pack(side=tk.LEFT)

        frame_menu = tk.Frame(frame)
        frame_menu.pack(side=tk.RIGHT)

        self._selected_value = None
        default = (list(choices)[0] if default is None else default).value
        self._menu = tk.OptionMenu(
            frame_menu,
            tk.StringVar(master=frame, value=default, name=label),
            # default,
            *[choice.value for choice in choices],  #  if choice.value!=default],
            command=self._set,
        )
        self._menu.config(width=_ADVANCED_OPTIONS_RIGHT_WIDTH)
        self._menu.pack(side=tk.RIGHT)
        # Initialize
        self._set(default)

    def get(self) -> Enum:
        return self._selected_value

    def _set(self, val: str):
        """
        Set the value selected
        """
        self._selected_value = self._choices(val)


class _LabeledText(object):
    """
    Label (left) and text input (right)
    """

    def __init__(
        self,
        frame: tk.Frame,
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
        """
        self._frame = frame
        label_height = 2
        text_height = 1
        self._label = tk.Label(
            frame,
            width=left_width,
            height=label_height,
            bg=None,
            anchor="w",
            text=label,
        )
        self._label.pack(side=tk.LEFT)

        self._text = tk.Text(
            frame,
            width=right_width,
            height=text_height,
            bg=None,
        )
        self._text.pack(side=tk.RIGHT)

        self._type = type

        if default is not None:
            self._text.insert("1.0", str(default))

    def get(self):
        try:
            val = self._text.get("1.0", tk.END)  # Line 1, character zero (wat)
            if self._type is not None:
                val = self._type(val)
            return val
        except tk.TclError:
            return None


class AdvancedOptionsGUI(object):
    """
    A window to hold advanced options (Architecture and number of epochs)
    """

    def __init__(self, resume_main, parent: GUI):
        self._parent = parent
        self._root = _TopLevelWithOk(self.apply, resume_main)
        self._root.title("Advanced Options")

        self.pack_options()

        # "Ok": apply and destroy
        self._frame_ok = tk.Frame(self._root)
        self._frame_ok.pack()
        self._button_ok = tk.Button(
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
        self._parent.advanced_options.architecture = self._architecture.get()
        epochs = self._epochs.get()
        if epochs is not None:
            self._parent.advanced_options.num_epochs = epochs
        latency = self._latency.get()
        # Value None is returned as "null" to disambiguate from non-set.
        if latency is not None:
            self._parent.advanced_options.latency = (
                None if latency == "null" else latency
            )
        threshold_esr = self._threshold_esr.get()
        if threshold_esr is not None:
            self._parent.advanced_options.threshold_esr = (
                None if threshold_esr == "null" else threshold_esr
            )

    def pack_options(self):
        # Architecture: radio buttons
        self._frame_architecture = tk.Frame(self._root)
        self._frame_architecture.pack()
        self._architecture = _LabeledOptionMenu(
            self._frame_architecture,
            "Architecture",
            core.Architecture,
            default=self._parent.advanced_options.architecture,
        )

        # Number of epochs: text box
        self._frame_epochs = tk.Frame(self._root)
        self._frame_epochs.pack()

        self._epochs = _LabeledText(
            self._frame_epochs,
            "Epochs",
            default=str(self._parent.advanced_options.num_epochs),
            type=_non_negative_int,
        )

        # Delay: text box
        self._frame_latency = tk.Frame(self._root)
        self._frame_latency.pack()

        self._latency = _LabeledText(
            self._frame_latency,
            "Reamp latency",
            default=_type_or_null_inv(self._parent.advanced_options.latency),
            type=_int_or_null,
        )

        # Threshold ESR
        self._frame_threshold_esr = tk.Frame(self._root)
        self._frame_threshold_esr.pack()
        self._threshold_esr = _LabeledText(
            self._frame_threshold_esr,
            "Threshold ESR",
            default=_type_or_null_inv(self._parent.advanced_options.threshold_esr),
            type=_float_or_null,
        )


class _UserMetadataGUI(object):
    # Things that are auto-filled:
    # Model date
    # gain
    def __init__(self, resume_main, parent: GUI):
        self._parent = parent
        self._root = _TopLevelWithOk(self._apply, resume_main)
        self._root.title("Metadata")

        LabeledText = partial(_LabeledText, right_width=_METADATA_RIGHT_WIDTH)

        # Name
        self._frame_name = tk.Frame(self._root)
        self._frame_name.pack()
        self._name = LabeledText(
            self._frame_name,
            "NAM name",
            default=parent.user_metadata.name,
            type=_rstripped_str,
        )
        # Modeled by
        self._frame_modeled_by = tk.Frame(self._root)
        self._frame_modeled_by.pack()
        self._modeled_by = LabeledText(
            self._frame_modeled_by,
            "Modeled by",
            default=parent.user_metadata.modeled_by,
            type=_rstripped_str,
        )
        # Gear make
        self._frame_gear_make = tk.Frame(self._root)
        self._frame_gear_make.pack()
        self._gear_make = LabeledText(
            self._frame_gear_make,
            "Gear make",
            default=parent.user_metadata.gear_make,
            type=_rstripped_str,
        )
        # Gear model
        self._frame_gear_model = tk.Frame(self._root)
        self._frame_gear_model.pack()
        self._gear_model = LabeledText(
            self._frame_gear_model,
            "Gear model",
            default=parent.user_metadata.gear_model,
            type=_rstripped_str,
        )
        # Gear type
        self._frame_gear_type = tk.Frame(self._root)
        self._frame_gear_type.pack()
        self._gear_type = _LabeledOptionMenu(
            self._frame_gear_type,
            "Gear type",
            GearType,
            default=parent.user_metadata.gear_type,
        )
        # Tone type
        self._frame_tone_type = tk.Frame(self._root)
        self._frame_tone_type.pack()
        self._tone_type = _LabeledOptionMenu(
            self._frame_tone_type,
            "Tone type",
            ToneType,
            default=parent.user_metadata.tone_type,
        )

        # "Ok": apply and destroy
        self._frame_ok = tk.Frame(self._root)
        self._frame_ok.pack()
        self._button_ok = tk.Button(
            self._frame_ok,
            text="Ok",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            command=lambda: self._root.destroy(pressed_ok=True),
        )
        self._button_ok.pack()

    def _apply(self):
        """
        Set values to parent and destroy this object
        """
        self._parent.user_metadata.name = self._name.get()
        self._parent.user_metadata.modeled_by = self._modeled_by.get()
        self._parent.user_metadata.gear_make = self._gear_make.get()
        self._parent.user_metadata.gear_model = self._gear_model.get()
        self._parent.user_metadata.gear_type = self._gear_type.get()
        self._parent.user_metadata.tone_type = self._tone_type.get()
        self._parent.user_metadata_flag = True


def _install_error():
    window = tk.Tk()
    window.title("ERROR")
    label = tk.Label(
        window,
        width=45,
        height=2,
        text="The NAM training software has not been installed correctly.",
    )
    label.pack()
    button = tk.Button(window, width=10, height=2, text="Quit", command=window.destroy)
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
