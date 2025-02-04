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
    Find and apply extensions to NAM with security measures
    """
    import importlib
    import os
    import sys
    import re

    # Define allowed module pattern
    ALLOWED_MODULE_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')
    
    # Validate module name
    def is_safe_module_name(name: str) -> bool:
        """
        Validate module name:
        - Must match allowed pattern
        - Must not be a system module
        - Must not contain path traversal attempts
        """
        if not ALLOWED_MODULE_PATTERN.match(name):
            return False
        if name in sys.modules:
            return False
        if '..' in name or '/' in name or '\\' in name:
            return False
        return True

    # Get extensions path securely
    home_path = os.environ.get("HOMEPATH" if os.name == "nt" else "HOME", "")
    if not home_path:
        print("WARNING: Home path not found, skipping extensions")
        return
        
    extensions_path = os.path.join(home_path, ".neural-amp-modeler", "extensions")
    if not os.path.exists(extensions_path):
        return
    if not os.path.isdir(extensions_path):
        print(f"WARNING: non-directory object found at expected extensions path {extensions_path}; skip")
        return

    print("Applying extensions...")
    extensions_path_not_in_sys_path = False
    if extensions_path not in sys.path:
        sys.path.append(extensions_path)
        extensions_path_not_in_sys_path = True

    for name in os.listdir(extensions_path):
        if name in {"__pycache__", ".DS_Store"} or name.endswith(".py"):
            continue
            
        if not is_safe_module_name(name):
            print(f"  {name} [SKIPPED] Invalid module name")
            continue
            
        try:
            importlib.import_module(name)
            print(f"  {name} [SUCCESS]")
        except Exception as e:
            print(f"  {name} [FAILED]")
            print(e)

    # Clean up sys.path
    if extensions_path_not_in_sys_path:
        sys.path.remove(extensions_path)
    
    print("Done!")


_apply_extensions()

import json as _json
from argparse import ArgumentParser as _ArgumentParser
from pathlib import Path as _Path
import os

from nam.train.full import main as _nam_full
from nam.train.gui import run as _nam_gui  # noqa F401 Used as an entry point
from nam.util import timestamp as _timestamp


def validate_path(path_str: str) -> _Path:
    """verified path"""
    try:
        # transform to absolute path
        abs_path = os.path.abspath(path_str)
        # ensure path is within allowed directory
        if not abs_path.startswith(os.getcwd()):
            raise ValueError("path must be within allowed directory")
        return _Path(abs_path)
    except Exception as e:
        raise ValueError(f"invalid path: {str(e)}")


def nam_full():
    parser = _ArgumentParser()
    parser.add_argument("data_config_path", type=str)
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("learning_config_path", type=str)
    parser.add_argument("outdir")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")

    args = parser.parse_args()

    # Validate all paths
    outdir = validate_path(args.outdir)
    data_config_path = validate_path(args.data_config_path)
    model_config_path = validate_path(args.model_config_path)
    learning_config_path = validate_path(args.learning_config_path)

    # Read configs with validated paths
    with open(data_config_path, "r") as fp:
        data_config = _json.load(fp)
    with open(model_config_path, "r") as fp:
        model_config = _json.load(fp)
    with open(learning_config_path, "r") as fp:
        learning_config = _json.load(fp)
    
    _nam_full(data_config, model_config, learning_config, outdir, args.no_show)
