# File: linear.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Linear model
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .._version import __version__
from ._base import BaseNet


class Linear(BaseNet):
    def __init__(self, receptive_field: int, *args, bias: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._net = nn.Conv1d(1, 1, receptive_field, bias=bias)

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.weight.shape[2]

    def export(self, outdir: Path):
        training = self.training
        self.eval()
        with open(Path(outdir, "config.json"), "w") as fp:
            json.dump(
                {
                    "version": __version__,
                    "architecture": self.__class__.__name__,
                    "config": {
                        "receptive_field": self.receptive_field,
                        "bias": self._bias,
                    },
                },
                fp,
                indent=4,
            )

        params = [self._net.weight.flatten()]
        if self._bias:
            params.append(self._net.bias.flatten())
        params = torch.cat(params).detach().cpu().numpy()
        # Hope I don't regret using np.save...
        np.save(Path(outdir, "weights.npy"), params)

        # And an input/output to verify correct computation:
        x, y = self._export_input_output()
        np.save(Path(outdir, "input.npy"), x.detach().cpu().numpy())
        np.save(Path(outdir, "output.npy"), y.detach().cpu().numpy())

        # And resume training state
        self.train(training)

    def export_cpp_header(self):
        raise NotImplementedError()

    @property
    def _bias(self) -> bool:
        return self._net.bias is not None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x[:, None])[:, 0]

    def _export_config(self):
        raise NotImplementedError()

    def _export_weights(self) -> np.ndarray:
        raise NotImplementedError()
