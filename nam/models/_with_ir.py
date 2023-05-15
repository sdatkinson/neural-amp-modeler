# File: _with_ir.py
# Created Date: Sunday May 14th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Model + IR
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import REQUIRED_RATE
from ._base import BaseNet

__all__ = ["WithIR"]


class WithIR(BaseNet):
    def __init__(self, model: BaseNet, ir: torch.Tensor, trainable_ir: bool = False):
        super().__init__()
        self._model = model
        if ir.ndim != 1:
            raise ValueError(f"Expect 1D IR; got ndim={ir.ndim} instead.")

        self._ir_weights = nn.Parameter(
            self._prepare_ir(ir), requires_grad=trainable_ir
        )

    @property
    def ir_length(self) -> int:
        return self._ir_weights.shape[2]

    @property
    def receptive_field(self) -> int:
        model_rf = self._model.receptive_field - 1
        ir_rf = self.ir_length - 1
        return 1 + model_rf + ir_rf

    @property
    def pad_start_default(self) -> bool:
        return self._model.pad_start_default

    def export_cpp_header(self, filename: Path):
        print("WARNING: IR skipped!")
        self._model.export_cpp_header(filename)

    def set_ir(self, ir: torch.Tensor):
        self._ir_weights.data = self._prepare_ir(ir)

    @classmethod
    def _prepare_ir(cls, ir: torch.Tensor) -> torch.Tensor:
        """
        Reverse the array and add dimensions.

        Equivalent to `ir[None, None, ::-1]`.
        """
        device = ir.device
        with torch.no_grad():
            return torch.Tensor(ir.cpu().numpy()[::-1].copy())[None, None, :].to(device)

    def _forward(self, *args, **kwargs):
        kwargs = deepcopy(kwargs)
        kwargs["pad_start"] = False  # Taken care of by self.forward
        model_out = self._model(*args, **kwargs)
        ndim = model_out.ndim
        process_in, process_out = {
            1: (
                (lambda model_out: model_out[None, None, :]),
                (lambda ir_out: ir_out[0, 0, :]),
            ),
            2: (
                (lambda model_out: model_out[:, None, :]),
                (lambda ir_out: ir_out[:, 0, :]),
            ),
            3: ((lambda model_out: model_out), (lambda ir_out: ir_out)),
        }[ndim]
        model_out = process_in(model_out)
        ir_out = F.conv1d(model_out, self._ir_weights)
        ir_out = process_out(ir_out)
        return ir_out

    def _export_config(self):
        c = self._model._export_config()
        with torch.no_grad():
            # TODO: The real-time code should be told not to rescale the weights it's
            # given in order to achieve the intended scaling.
            c["post_impulse_response"] = {
                "sample_rate": REQUIRED_RATE,
                "samples": self._ir_weights.data[0, 0, :].cpu().numpy()[::-1].tolist(),
            }
        return c

    def _export_weights(self) -> np.ndarray:
        return self._model._export_weights()
