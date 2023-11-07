# File: gui.py
# Created Date: Saturday February 25th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
GUI for training

Usage:
>>> from nam.train.gui import run
>>> run()
"""


# Hack to recover graceful shutdowns in Windows.
# This has to happen ASAP
# See:
# https://github.com/sdatkinson/neural-amp-modeler/issues/105
# https://stackoverflow.com/a/44822794
def _ensure_graceful_shutdowns():
    import os

    if os.name == "nt":  # OS is Windows
        os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


_ensure_graceful_shutdowns()

import re
import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict, Optional, Sequence

try:  # 3rd-party and 1st-party imports
    import torch

    from nam import __version__
    from nam.train import core
    from nam.models.metadata import GearType, UserMetadata, ToneType

    # Ok private access here--this is technically allowed access
    from nam.train._names import LATEST_VERSION

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

_ADVANCED_OPTIONS_LEFT_WIDTH = 12
_ADVANCED_OPTIONS_RIGHT_WIDTH = 12
_METADATA_RIGHT_WIDTH = 60


@dataclass
class _AdvancedOptions(object):
    architecture: core.Architecture
    num_epochs: int
    delay: Optional[int]
    ignore_checks: bool


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
        hooks: Optional[Sequence[Callable[[], None]]] = None,
    ):
        self._button_text = button_text
        self._info_str = info_str
        self._path: Optional[Path] = None
        self._path_type = path_type
        self._button = tk.Button(
            frame,
            text=button_text,
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            fg="black",
            command=self._set_val,
        )
        self._button.pack(side=tk.LEFT)
        self._label = tk.Label(
            frame,
            width=_TEXT_WIDTH,
            height=_BUTTON_HEIGHT,
            fg="black",
            bg=None,
            anchor="w",
        )
        self._label.pack(side=tk.RIGHT)
        self._hooks = hooks
        self._set_text()

    @property
    def val(self) -> Optional[Path]:
        return self._path

    def _set_text(self):
        if self._path is None:
            self._label["fg"] = "red"
            self._label["text"] = self._info_str
        else:
            val = self.val
            val = val[0] if isinstance(val, tuple) and len(val) == 1 else val
            self._label["fg"] = "black"
            self._label["text"] = f"{self._button_text.capitalize()} set to {val}"

    def _set_val(self):
        res = {
            _PathType.FILE: filedialog.askopenfilename,
            _PathType.DIRECTORY: filedialog.askdirectory,
            _PathType.MULTIFILE: filedialog.askopenfilenames,
        }[self._path_type]()
        if res != "":
            self._path = res
        self._set_text()

        if self._hooks is not None:
            for h in self._hooks:
                h()


class _CheckboxKeys(Enum):
    """
    Keys for checkboxes
    """

    FIT_CAB = "fit_cab"
    SILENT_TRAINING = "silent_training"
    SAVE_PLOT = "save_plot"
    IGNORE_DATA_CHECKS = "ignore_data_checks"


class _GUI(object):
    def __init__(self):
        self._root = tk.Tk()
        self._root.title(f"NAM Trainer - v{__version__}")

        # Buttons for paths:
        self._frame_input_path = tk.Frame(self._root)
        self._frame_input_path.pack()
        self._path_button_input = _PathButton(
            self._frame_input_path,
            "Input Audio",
            f"Select input DI file (e.g. {LATEST_VERSION.name})",
            _PathType.FILE,
            hooks=[self._check_button_states],
        )

        self._frame_output_path = tk.Frame(self._root)
        self._frame_output_path.pack()
        self._path_button_output = _PathButton(
            self._frame_output_path,
            "Output Audio",
            "Select output (reamped) audio - choose multiple files to enable batch training",
            _PathType.MULTIFILE,
            hooks=[self._check_button_states],
        )

        self._frame_train_destination = tk.Frame(self._root)
        self._frame_train_destination.pack()
        self._path_button_train_destination = _PathButton(
            self._frame_train_destination,
            "Train Destination",
            "Select training output directory",
            _PathType.DIRECTORY,
            hooks=[self._check_button_states],
        )

        # Metadata
        self.user_metadata = UserMetadata()
        self._frame_metadata = tk.Frame(self._root)
        self._frame_metadata.pack()
        self._button_metadata = tk.Button(
            self._frame_metadata,
            text="Metadata...",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            fg="black",
            command=self._open_metadata,
        )
        self._button_metadata.pack()
        self.user_metadata_flag = False

        # This should probably be to the right somewhere
        self._get_additional_options_frame()

        # Advanced options for training
        default_architecture = core.Architecture.STANDARD
        self.advanced_options = _AdvancedOptions(
            default_architecture,
            _DEFAULT_NUM_EPOCHS,
            _DEFAULT_DELAY,
            _DEFAULT_IGNORE_CHECKS,
        )
        # Window to edit them:
        self._frame_advanced_options = tk.Frame(self._root)
        self._frame_advanced_options.pack()
        self._button_advanced_options = tk.Button(
            self._frame_advanced_options,
            text="Advanced options...",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            fg="black",
            command=self._open_advanced_options,
        )
        self._button_advanced_options.pack()

        # Train button
        self._frame_train = tk.Frame(self._root)
        self._frame_train.pack()
        self._button_train = tk.Button(
            self._frame_train,
            text="Train",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            fg="black",
            command=self._train,
        )
        self._button_train.pack()

        self._check_button_states()

    def _check_button_states(self):
        """
        Determine if any buttons should be disabled
        """
        # Train button is disabled unless all paths are set
        if any(
            pb.val is None
            for pb in (
                self._path_button_input,
                self._path_button_output,
                self._path_button_train_destination,
            )
        ):
            self._button_train["state"] = tk.DISABLED
            return
        self._button_train["state"] = tk.NORMAL

    def _get_additional_options_frame(self):
        # Checkboxes
        # TODO get these definitions into __init__()
        self._frame_checkboxes = tk.Frame(self._root)
        self._frame_checkboxes.pack(side=tk.LEFT)
        row = 1

        @dataclass
        class Checkbox(object):
            variable: tk.BooleanVar
            check_button: tk.Checkbutton

        def make_checkbox(
            key: _CheckboxKeys, text: str, default_value: bool
        ) -> Checkbox:
            variable = tk.BooleanVar()
            variable.set(default_value)
            check_button = tk.Checkbutton(
                self._frame_checkboxes, text=text, variable=variable
            )
            self._checkboxes[key] = Checkbox(variable, check_button)

        self._checkboxes: Dict[_CheckboxKeys, Checkbox] = dict()
        make_checkbox(_CheckboxKeys.FIT_CAB, "Cab modeling", False)
        make_checkbox(
            _CheckboxKeys.SILENT_TRAINING,
            "Silent run (suggested for batch training)",
            False,
        )
        make_checkbox(_CheckboxKeys.SAVE_PLOT, "Save ESR plot automatically", True)
        make_checkbox(
            _CheckboxKeys.IGNORE_DATA_CHECKS,
            "Ignore data quality checks (DO AT YOUR OWN RISK!)",
            False,
        )

        # Grid them:
        row = 1
        for v in self._checkboxes.values():
            v.check_button.grid(row=row, column=1, sticky="W")
            row += 1

    def mainloop(self):
        self._root.mainloop()

    def _open_advanced_options(self):
        """
        Open advanced options
        """
        ao = _AdvancedOptionsGUI(self)
        # I should probably disable the main GUI...
        ao.mainloop()
        # ...and then re-enable it once it gets closed.

    def _open_metadata(self):
        """
        Open dialog for metadata
        """
        mdata = _UserMetadataGUI(self)
        # I should probably disable the main GUI...
        mdata.mainloop()

    def _train(self):
        # Advanced options:
        num_epochs = self.advanced_options.num_epochs
        architecture = self.advanced_options.architecture
        delay = self.advanced_options.delay
        file_list = self._path_button_output.val

        # Advanced-er options
        # If you're poking around looking for these, then maybe it's time to learn to
        # use the command-line scripts ;)
        lr = 0.004
        lr_decay = _DEFAULT_LR_DECAY
        batch_size = _DEFAULT_BATCH_SIZE
        seed = 0

        # Run it
        for file in file_list:
            print("Now training {}".format(file))
            basename = re.sub(r"\.wav$", "", file.split("/")[-1])

            trained_model = core.train(
                self._path_button_input.val,
                file,
                self._path_button_train_destination.val,
                epochs=num_epochs,
                delay=delay,
                architecture=architecture,
                batch_size=batch_size,
                lr=lr,
                lr_decay=lr_decay,
                seed=seed,
                silent=self._checkboxes[_CheckboxKeys.SILENT_TRAINING].variable.get(),
                save_plot=self._checkboxes[_CheckboxKeys.SAVE_PLOT].variable.get(),
                modelname=basename,
                ignore_checks=self._checkboxes[
                    _CheckboxKeys.IGNORE_DATA_CHECKS
                ].variable.get(),
                local=True,
                fit_cab=self._checkboxes[_CheckboxKeys.FIT_CAB].variable.get(),
            )
            if trained_model is None:
                print("Model training failed! Skip exporting...")
                continue
            print("Model training complete!")
            print("Exporting...")
            outdir = self._path_button_train_destination.val
            print(f"Exporting trained model to {outdir}...")
            trained_model.net.export(
                outdir,
                basename=basename,
                user_metadata=self.user_metadata
                if self.user_metadata_flag
                else UserMetadata(),
            )
            print("Done!")

        # Metadata was only valid for 1 run, so make sure it's not used again unless
        # the user re-visits the window and clicks "ok"
        self.user_metadata_flag = False


# some typing functions
def _non_negative_int(val):
    val = int(val)
    if val < 0:
        val = 0
    return val


def _int_or_null(val):
    val = val.rstrip()
    if val == "null":
        return val
    return int(val)


def _int_or_null_inv(val):
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
        fg = "black"
        self._label = tk.Label(
            frame,
            width=_ADVANCED_OPTIONS_LEFT_WIDTH,
            height=height,
            fg=fg,
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
            fg="black",
            bg=None,
            anchor="w",
            text=label,
        )
        self._label.pack(side=tk.LEFT)

        self._text = tk.Text(
            frame,
            width=right_width,
            height=text_height,
            fg="black",
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


class _AdvancedOptionsGUI(object):
    """
    A window to hold advanced options (Architecture and number of epochs)
    """

    def __init__(self, parent: _GUI):
        self._parent = parent
        self._root = tk.Tk()
        self._root.title("Advanced Options")

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
        self._frame_delay = tk.Frame(self._root)
        self._frame_delay.pack()

        self._delay = _LabeledText(
            self._frame_delay,
            "Delay",
            default=_int_or_null_inv(self._parent.advanced_options.delay),
            type=_int_or_null,
        )

        # "Ok": apply and destory
        self._frame_ok = tk.Frame(self._root)
        self._frame_ok.pack()
        self._button_ok = tk.Button(
            self._frame_ok,
            text="Ok",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            fg="black",
            command=self._apply_and_destroy,
        )
        self._button_ok.pack()

    def mainloop(self):
        self._root.mainloop()

    def _apply_and_destroy(self):
        """
        Set values to parent and destroy this object
        """
        self._parent.advanced_options.architecture = self._architecture.get()
        epochs = self._epochs.get()
        if epochs is not None:
            self._parent.advanced_options.num_epochs = epochs
        delay = self._delay.get()
        # Value None is returned as "null" to disambiguate from non-set.
        if delay is not None:
            self._parent.advanced_options.delay = None if delay == "null" else delay
        self._root.destroy()


class _UserMetadataGUI(object):
    # Things that are auto-filled:
    # Model date
    # gain
    def __init__(self, parent: _GUI):
        self._parent = parent
        self._root = tk.Tk()
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

        # "Ok": apply and destory
        self._frame_ok = tk.Frame(self._root)
        self._frame_ok.pack()
        self._button_ok = tk.Button(
            self._frame_ok,
            text="Ok",
            width=_BUTTON_WIDTH,
            height=_BUTTON_HEIGHT,
            fg="black",
            command=self._apply_and_destroy,
        )
        self._button_ok.pack()

    def mainloop(self):
        self._root.mainloop()

    def _apply_and_destroy(self):
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

        self._root.destroy()


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
        _gui = _GUI()
        _gui.mainloop()
    else:
        _install_error()


if __name__ == "__main__":
    run()
