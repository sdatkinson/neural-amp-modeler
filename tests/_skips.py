"""
Tools for skipping tests
"""

import pytest as _pytest
import torch as _torch

from tests._integration import loadmodel_exe_path as _loadmodel_exe_path

_has_loadmodel = _loadmodel_exe_path() is not None

requires_neural_amp_modeler_core_loadmodel = _pytest.mark.skipif(
    not _has_loadmodel,
    reason="NeuralAmpModelerCore not present or loadmodel tool not built",
)
requires_cuda = _pytest.mark.skipif(
    not _torch.cuda.is_available(), reason="CUDA-specific test"
)
requires_mps = _pytest.mark.skipif(
    not _torch.backends.mps.is_available(), reason="MPS-specific test"
)
