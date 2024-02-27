# File: __init__.py
# File Created: Tuesday, 2nd February 2021 9:42:50 pm
# Author: Steven Atkinson (steven@atkinson.mn)


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

from ._version import __version__  # Must be before models or else circular

from . import _core  # noqa F401
from . import data  # noqa F401
from . import models  # noqa F401
from . import util  # noqa F401
from . import train  # noqa F401
