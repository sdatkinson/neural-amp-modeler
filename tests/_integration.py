"""
Integration helpers (e.g. running external tools like NeuralAmpModelerCore loadmodel).
"""

import subprocess as _subprocess
import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
# _NEURAL_AMP_MODELER_CORE_DIR = _REPO_ROOT.parent / "NeuralAmpModelerCore"
_NEURAL_AMP_MODELER_CORE_DIR = _REPO_ROOT / "temp" / "NeuralAmpModelerCore"


def loadmodel_exe_path():
    """Path to the loadmodel executable if it exists, else None."""
    if not _NEURAL_AMP_MODELER_CORE_DIR.exists():
        return None
    build = _NEURAL_AMP_MODELER_CORE_DIR / "build"
    if not build.exists():
        return None
    if _sys.platform == "win32":
        exe = build / "tools" / "loadmodel.exe"
    else:
        exe = build / "tools" / "loadmodel"
    return exe if exe.exists() else None


def run_loadmodel(
    model_path: _Path, *, timeout: float = 10.0
) -> _subprocess.CompletedProcess:
    """
    Run NeuralAmpModelerCore's loadmodel tool on a .nam model path.

    :param model_path: Path to a .nam file (or directory containing one).
    :param timeout: Seconds before the subprocess is killed.
    :return: CompletedProcess from subprocess.run.
    :raises: FileNotFoundError if loadmodel executable is not found.
    """
    exe = loadmodel_exe_path()
    if exe is None:
        raise FileNotFoundError(
            "NeuralAmpModelerCore loadmodel not found: either "
            f"{_NEURAL_AMP_MODELER_CORE_DIR!s} is missing or build/tools/loadmodel is not built."
        )
    return _subprocess.run(
        [str(exe), str(model_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
