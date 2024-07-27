# File: test_cli.py
# Created Date: Saturday July 27th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import importlib
import os
from pathlib import Path

import pytest


def test_extensions():
    """
    Test that we can use a simple extension.
    """
    # DRY: Make sure this matches the code!
    home_path = os.environ["HOMEPATH"] if os.name == "nt" else os.environ["HOME"]
    extensions_path = os.path.join(home_path, ".neural-amp-modeler", "extensions")

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
    path.parent.mkdir(parents=True, exist_ok=True)

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

        # Now trigger the extension by importing the CLI module:
        from nam import cli

        # If some other test already imported this, then we need to trigger a re-load or
        # else the extension won't get picked up!
        importlib.reload(cli)

        # Now let's have a look:
        from nam.train import core

        assert hasattr(core, attr_name)
        assert getattr(core, attr_name) == attr_val
    finally:
        if path.exists():
            path.unlink()
        # You might want to comment that .unlink() and uncomment this if this test isn't
        # passing and you're struggling:
        # pass


if __name__ == "__main__":
    pytest.main()
