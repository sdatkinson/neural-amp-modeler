# File: gui.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
GUI for training
"""

import tkinter as tk
from tkinter import filedialog
from typing import Optional

from nam.models.base import Model
from nam.train import gui as _gui


class Paths(object):
    def __init__(self):
        self.input_path = None
        self.output_path = None
        self.train_destination = None

    def set_input(self):
        res = filedialog.askopenfilename()
        if res != "":
            self.input_path = res
            print(f"Input set to {self.input_path}")

    def set_output(self):
        res = filedialog.askopenfilename()
        if res != "":
            self.output_path = res
            print(f"Output set to {self.output_path}")

    def set_train_destination(self):
        res = filedialog.askdirectory()
        if res != "":
            self.train_destination = res
            print(f"Training destination set to {self.train_destination}")


class Buttons(object):
    def __init__(self, gui):
        self._buttons = {}
        button_width = 20
        button_height = 2
        self._register_button(
            "input",
            tk.Button(
                gui.window,
                text="Input Audio",
                width=button_width,
                height=button_height,
                fg="black",
                command=gui.paths.set_input,
            ),
        )
        self._register_button(
            "output",
            tk.Button(
                gui.window,
                text="Output Audio",
                width=button_width,
                height=button_height,
                fg="black",
                command=gui.paths.set_output,
            ),
        )
        self._register_button(
            "train_dest",
            tk.Button(
                gui.window,
                text="Train destination",
                width=button_width,
                height=button_height,
                fg="black",
                command=gui.paths.set_train_destination,
            ),
        )
        self._register_button(
            "train",
            tk.Button(
                gui.window,
                text="Train",
                width=button_width,
                height=button_height,
                fg="black",
                command=gui.train,
            ),
        )
        self._register_button(
            "export",
            tk.Button(
                gui.window,
                text="Export",
                width=button_width,
                height=button_height,
                fg="black",
                command=gui.export,
            ),
        )

    def pack(self):
        for button in self._buttons.values():
            button.pack()

    def _register_button(self, name: str, button: tk.Button):
        if name in self._buttons:
            raise RuntimeError(f"Button name {name} is already registered!")
        self._buttons[name] = button


class GUI(object):
    def __init__(self):
        self._window = tk.Tk()
        self._paths = Paths()
        self._buttons = Buttons(self)
        self._buttons.pack()
        self._trained_model: Optional[Model] = None

    @property
    def paths(self) -> Paths:
        return self._paths

    @property
    def window(self) -> tk.Tk:
        return self._window

    def export(self):
        if self._trained_model is None:
            print("No model trained; can't export.")
            return
        outdir = filedialog.askdirectory()
        if outdir != "":
            print(f"Exporting trained model to {outdir}...")
            self._trained_model.net.export(outdir)
            print("Done!")

    def mainloop(self):
        self.window.mainloop()

    def train(self):
        epochs = 100
        delay = None
        stage_1_channels = 16
        stage_2_channels = 8
        head_scale: float = 0.02
        lr = 0.004
        lr_decay = 0.007
        seed = 0
        self._trained_model = _gui.train(
            self.paths.input_path,
            self.paths.output_path,
            self.paths.train_destination,
            epochs=epochs,
            delay=delay,
            stage_1_channels=stage_1_channels,
            stage_2_channels=stage_2_channels,
            head_scale=head_scale,
            lr=lr,
            lr_decay=lr_decay,
            seed=seed,
        )
        print("Model training complete!")


def main():
    gui = GUI()
    gui.mainloop()


if __name__ == "__main__":
    main()
