# File: cli.py
# Created Date: Saturday July 27th 2024
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Command line interface entry points (GUI trainer, full trainer)
"""


# This must happen first
def _ensure_graceful_shutdowns():
    """
    Hack to recover graceful shutdowns in Windows.
    This has to happen ASAP
    See:
    https://github.com/sdatkinson/neural-amp-modeler/issues/105
    https://stackoverflow.com/a/44822794
    """
    import os

    if os.name == "nt":  # OS is Windows
        os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


_ensure_graceful_shutdowns()


# This must happen ASAP but not before the graceful shutdown hack
def _apply_extensions():
    """
    Find and apply extensions to NAM
    """

    def removesuffix(s: str, suffix: str) -> str:
        # Remove once 3.8 is dropped
        if len(suffix) == 0:
            return s
        return s[: -len(suffix)] if s.endswith(suffix) else s

    import importlib
    import os
    import sys

    # DRY: Make sure this matches the test!
    home_path = os.environ["HOMEPATH"] if os.name == "nt" else os.environ["HOME"]
    extensions_path = os.path.join(home_path, ".neural-amp-modeler", "extensions")
    if not os.path.exists(extensions_path):
        return
    if not os.path.isdir(extensions_path):
        print(
            f"WARNING: non-directory object found at expected extensions path {extensions_path}; skip"
        )
    print("Applying extensions...")
    if extensions_path not in sys.path:
        sys.path.append(extensions_path)
        extensions_path_not_in_sys_path = True
    else:
        extensions_path_not_in_sys_path = False
    for name in os.listdir(extensions_path):
        if name in {"__pycache__", ".DS_Store"}:
            continue
        try:
            importlib.import_module(removesuffix(name, ".py"))  # Runs it
            print(f"  {name} [SUCCESS]")
        except Exception as e:
            print(f"  {name} [FAILED]")
            print(e)
    if extensions_path_not_in_sys_path:
        for i, p in enumerate(sys.path):
            if p == extensions_path:
                sys.path = sys.path[:i] + sys.path[i + 1 :]
                break
        else:
            raise RuntimeError("Failed to remove extensions path from sys.path?")
    print("Done!")


_apply_extensions()

import json as _json
from argparse import ArgumentParser as _ArgumentParser
from pathlib import Path as _Path

from nam.train.full import main as _nam_full
from nam.train.gui import run as nam_gui  # noqa F401 Used as an entry point
from nam.util import timestamp as _timestamp


def nam_hello_world():
    """
    This is a minimal CLI entry point that's meant to be used to ensure that NAM
    was installed successfully
    """
    from nam import __version__

    msg = f"""
    Neural Amp Modeler

    by Steven Atkinson

    Version {__version__}
    """
    print(msg)


def nam_full():
    parser = _ArgumentParser()
    parser.add_argument("data_config_path", type=str)
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("learning_config_path", type=str)
    parser.add_argument("outdir")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")

    args = parser.parse_args()

    def ensure_outdir(outdir: str) -> _Path:
        outdir = _Path(outdir, _timestamp())
        outdir.mkdir(parents=True, exist_ok=False)
        return outdir

    outdir = ensure_outdir(args.outdir)
    # Read
    with open(args.data_config_path, "r") as fp:
        data_config = _json.load(fp)
    with open(args.model_config_path, "r") as fp:
        model_config = _json.load(fp)
    with open(args.learning_config_path, "r") as fp:
        learning_config = _json.load(fp)
    _nam_full(data_config, model_config, learning_config, outdir, args.no_show)
