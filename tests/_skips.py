"""
Tools for skipping tests
"""

import pytest as _pytest
import torch as _torch

requires_cuda = _pytest.mark.skipif(
    not _torch.cuda.is_available(), reason="CUDA-specific test"
)
requires_mps = _pytest.mark.skipif(
    not _torch.backends.mps.is_available(), reason="MPS-specific test"
)
