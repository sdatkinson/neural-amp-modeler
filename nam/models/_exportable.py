# File: _exportable.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .._version import __version__
from ..data import np_to_wav


class Exportable(abc.ABC):
    """
    Interface for my custon export format for use in the plugin.
    """

    def export(self, outdir: Path):
        """
        Interface for exporting.
        You should create at least a `config.json` containing the two fields:
        * "version" (str)
        * "architecture" (str)
        * "config": (dict w/ other necessary data like tensor shapes etc)

        :param outdir: Assumed to exist. Can be edited inside at will.
        """
        training = self.training
        self.eval()
        with open(Path(outdir, "config.json"), "w") as fp:
            json.dump(
                {
                    "version": __version__,
                    "architecture": self.__class__.__name__,
                    "config": self._export_config(),
                },
                fp,
                indent=4,
            )
        np.save(Path(outdir, "weights.npy"), self._export_weights())
        x, y = self._export_input_output()
        np.save(Path(outdir, "inputs.npy"), x)
        np.save(Path(outdir, "outputs.npy"), y)
        np_to_wav(x, Path(outdir, "input.wav"))
        np_to_wav(y, Path(outdir, "output.wav"))

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
        """
        pass

    @abc.abstractmethod
    def _export_weights(self) -> np.ndarray:
        """
        Flatten the weights out to a 1D array
        """
        pass
