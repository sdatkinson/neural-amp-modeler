# File: test_gui.py
# Created Date: Sunday July 7th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import os
from pathlib import Path

import pytest


def test_extensions():
    """
    Test that we can use a simple extension.
    """
    # DRY: Make sure this matches the code!
    extensions_path = Path(
        os.path.join(os.environ["HOME"], ".neural-amp-modeler", "extensions")
    )

    def get_name():
        i = 0
        while True:
            basename = f"test_extension_{i}.py"
            path = Path(extensions_path, basename)
            if not path.exists():
                return path
            else:
                i += 1

    path = get_name()

    try:
        # Make the extension
        # It's going to set an attribute inside nam.core. We'll know the extension worked if
        # that attr is set.
        attr_name = "my_test_attr"
        attr_val = "THIS IS A TEST ATTRIBUTE I SHOULDN'T BE HERE"
        with open(path, "w") as f:
            f.writelines(
                [
                    'print("RUNNING TEST!")\n',
                    "from nam.train import core\n",
                    f'name = "{attr_name}"\n',
                    "assert not hasattr(core, name)\n"
                    f'setattr(core, name, "{attr_val}")\n',
                ]
            )

        # Now trigger the extension by importing from the GUI module:
        # NOTE: We may have to trigger re-loading the module.
        from nam.train import core

        assert hasattr(core, attr_name)
        assert getattr(core, attr_name) == attr_val
    finally:
        if path.exists():
            path.unlink()
        # You might want to comment that .unlink() and uncomment this if this test isn't
        # passing and you're struggling:
        # pass
