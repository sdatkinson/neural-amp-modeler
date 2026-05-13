# File: test_gui.py
# Created Date: Friday May 24th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import inspect
import tkinter as tk

import pytest

from nam.train import gui

# class TestPathButton(object):
#     def test_system_text_color(self):
#         """
#         Issue 428
#         """
#         top_level = tk.Toplevel()
#         label = tk.Label(master=top_level, text="My text", fg=gui._SYSTEM_TEXT_COLOR)
#         label.pack()


def test_get_current_version():
    """
    Make sure this at least runs!
    See #516
    """
    v = gui._get_current_version()


def test_gui_does_not_depend_on_core_architecture():
    source = inspect.getsource(gui)
    assert "_core.Architecture" not in source


if __name__ == "__main__":
    pytest.main()
