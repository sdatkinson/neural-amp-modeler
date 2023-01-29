# File: test_exportable.py
# Created Date: Sunday January 29th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Test export behavior of models
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from nam.models import _exportable

def test_export():
    """
    Does it work?
    """
    class Model(nn.Module, _exportable.Exportable):
        def __init__(self):
            super().__init__()
            self._scale = nn.Parameter(torch.tensor(0.0))
            self._bias = nn.Parameter(torch.tensor(0.0))

        def forward(self, x: torch.Tensor):
            return self._scale * x + self._bias

        def export_cpp_header(self, filename: Path):
            pass

        def _export_config(self):
            return {}

        def _export_input_output(self) -> Tuple[np.ndarray, np.ndarray]:
            x = 0.01 * np.random.randn(3,)
            y = self(torch.Tensor(x)).detach().cpu().numpy()
            return x, y

        def _export_weights(self) -> np.ndarray:
            return torch.stack([self._scale, self._bias]).detach().cpu().numpy()

    model = Model()
    with TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        model_basename = "model.nam"
        model_path = Path(tmpdir, model_basename)
        assert model_path.exists()
        with open(model_path, "r") as fp:
            model_dict = json.load(fp)
        required_keys = {"version", "architecture", "config", "weights"}
        for key in required_keys:
            assert key in model_dict
        weights_list = model_dict["weights"]
        assert isinstance(weights_list, list)
        assert len(weights_list) == 2
        assert all(isinstance(w, float) for w in weights_list)
            