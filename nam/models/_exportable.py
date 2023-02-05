# File: _exportable.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from .._version import __version__
from ..data import np_to_wav

logger = logging.getLogger(__name__)


class Exportable(abc.ABC):
    """
    Interface for my custon export format for use in the plugin.
    """

    def export(self, outdir: Path, include_snapshot: bool = False):
        """
        Interface for exporting.
        You should create at least a `config.json` containing the two fields:
        * "version" (str)
        * "architecture" (str)
        * "config": (dict w/ other necessary data like tensor shapes etc)

        :param outdir: Assumed to exist. Can be edited inside at will.
        :param include_snapshots: If True, outputs `input.npy` and `output.npy`
            Containing an example input/output pair that the model creates. This
            Can be used to debug e.g. the implementation of the model in the
            plugin.
        """
        training = self.training
        self.eval()
        with open(Path(outdir, "model.nam"), "w") as fp:
            json.dump(
                {
                    "version": __version__,
                    "architecture": self.__class__.__name__,
                    "config": self._export_config(),
                    "weights": self._export_weights().tolist(),
                },
                fp,
                indent=4,
            )
        if include_snapshot:
            x, y = self._export_input_output()
            x_path = Path(outdir, "test_inputs.npy")
            y_path = Path(outdir, "test_outputs.npy")
            logger.debug(f"Saving snapshot input to {x_path}")
            np.save(x_path, x)
            logger.debug(f"Saving snapshot output to {y_path}")
            np.save(y_path, y)

        # And resume training state
        self.train(training)

    @abc.abstractmethod
    def export_cpp_header(self, filename: Path):
        """
        Export a .h file to compile into the plugin with the weights written right out
        as text
        """
        pass

    @abc.abstractmethod
    def _export_config(self):
        """
        Creates the JSON of the model's archtecture hyperparameters (number of layers,
        number of units, etc)

        :return: a JSON serializable object
        """
        pass

    @abc.abstractmethod
    def _export_input_output(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create an input and corresponding output signal to verify its behavior.

        They should be the same length, but the start of the output might have transient
        effects. Up to you to interpret.
        """
        pass

    @abc.abstractmethod
    def _export_weights(self) -> np.ndarray:
        """
        Flatten the weights out to a 1D array
        """
        pass
