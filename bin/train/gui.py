# File: gui.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
GUI for training
"""

import tkinter as tk
from enum import Enum
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional, Sequence

from nam.models.base import Model
from nam.train import gui as _gui

BUTTON_WIDTH = 20
BUTTON_HEIGHT = 2
TEXT_WIDTH=70


class _PathType(Enum):
    FILE = "file"
    DIRECTORY = "directory"


class _PathButton(object):
    """
    Button and the path
    """

    def __init__(
        self,
        frame: tk.Frame,
        button_text,
        info_str: str,
        path_type: _PathType,
        hooks: Optional[Sequence[Callable[[], None]]] = None,
    ):
        self._info_str = info_str
        self._path: Optional[Path] = None
        self._path_type = path_type
        self._button = tk.Button(
            frame,
            text=button_text,
            width=BUTTON_WIDTH,
            height=BUTTON_HEIGHT,
            fg="black",
            command=self._set_val,
        )
        self._button.pack(side=tk.LEFT)
        self._label = tk.Label(frame, width=TEXT_WIDTH, height=BUTTON_HEIGHT, fg="black", bg=None, anchor="w")
        self._label.pack(side=tk.RIGHT)
        self._hooks = hooks
        self._set_text()

    @property
    def val(self) -> Optional[Path]:
        return self._path

    def _set_text(self):
        if self._path is None:
            self._label["fg"] = "red"
            self._label["text"] = f"{self._info_str} is not set!"
        else:
            self._label["fg"] = "black"
            self._label["text"] = f"{self._info_str} set to {self.val}"

    def _set_val(self):
        res = {
            _PathType.FILE: filedialog.askopenfilename,
            _PathType.DIRECTORY: filedialog.askdirectory,
        }[self._path_type]()
        if res != "":
            self._path = res
        self._set_text()

        if self._hooks is not None:
            for h in self._hooks:
                h()


class GUI(object):
    def __init__(self):
        self._root = tk.Tk()

        # Buttons for paths:
        self._frame_input_path = tk.Frame(self._root)
        self._frame_input_path.pack()
        self._path_button_input = _PathButton(
            self._frame_input_path,
            "Input Audio",
            "Input audio",
            _PathType.FILE,
            hooks=[self._check_button_states],
        )

        self._frame_output_path = tk.Frame(self._root)
        self._frame_output_path.pack()
        self._path_button_output = _PathButton(
            self._frame_output_path,
            "Output Audio",
            "Output audio",
            _PathType.FILE,
            hooks=[self._check_button_states],
        )

        self._frame_train_destination = tk.Frame(self._root)
        self._frame_train_destination.pack()
        self._path_button_train_destination = _PathButton(
            self._frame_train_destination,
            "Train Destination",
            "Train destination",
            _PathType.DIRECTORY,
            hooks=[self._check_button_states],
        )

        # Train button
        self._frame_train = tk.Frame(self._root)
        self._frame_train.pack()
        self._button_train = tk.Button(
            self._frame_train,
            text="Train",
            width=BUTTON_WIDTH,
            height=BUTTON_HEIGHT,
            fg="black",
            command=self._train,
        )
        self._button_train.pack()

        self._check_button_states()

    def mainloop(self):
        self._root.mainloop()

    def _train(self):
        epochs = 100
        delay = None
        architecture = _gui.Architecture.STANDARD
        lr = 0.004
        lr_decay = 0.007
        seed = 0
        trained_model = _gui.train(
            self._path_button_input.val,
            self._path_button_output.val,
            self._path_button_train_destination.val,
            epochs=epochs,
            delay=delay,
            architecture=architecture,
            lr=lr,
            lr_decay=lr_decay,
            seed=seed,
        )
        print("Model training complete!")
        print("Exporting...")
        outdir = self._path_button_train_destination.val
        print(f"Exporting trained model to {outdir}...")
        trained_model.net.export(outdir)
        print("Done!")

    def _check_button_states(self):
        """
        Determine if any buttons should be disabled
        """
        # Train button is diabled unless all paths are set
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


def main():
    gui = GUI()
    gui.mainloop()


if __name__ == "__main__":
    main()
