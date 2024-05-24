# File: test_gui.py
# Created Date: Friday May 24th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import tkinter as tk

import pytest

from nam.train import gui


class TestPathButton(object):
    def test_system_text_color(self):
        """
        Issue 428
        """
        top_level = tk.Toplevel()
        label = tk.Label(master=top_level, text="My text", fg=gui._SYSTEM_TEXT_COLOR)
        label.pack()


if __name__ == "__main__":
    pytest.main()
