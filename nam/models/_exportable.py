# File: _exportable.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..data import np_to_wav
from .metadata import Date, UserMetadata

logger = logging.getLogger(__name__)

# Model version is independent from package version as of package version 0.5.2 so that
# the API of the package can iterate at a different pace from that of the model files.
_MODEL_VERSION = "0.5.2"


def _cast_enums(d: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Casts enum-type keys to their values
    """
    out = {}
    for key, val in d.items():
        if isinstance(val, Enum):
            val = val.value
        out[key] = val
    return out


class Exportable(abc.ABC):
    """
    Interface for my custon export format for use in the plugin.
    """

    def export(
        self,
        outdir: Path,
        include_snapshot: bool = False,
        basename: str = "model",
        user_metadata: Optional[UserMetadata] = None,
    ):
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
        model_dict = self._get_export_dict()
        model_dict["metadata"].update(
            {} if user_metadata is None else _cast_enums(user_metadata.dict())
        )

        training = self.training
        self.eval()
        with open(Path(outdir, f"{basename}.nam"), "w") as fp:
            json.dump(model_dict, fp)
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

    def export_onnx(self, filename: Path):
        """
        Export model in format for ONNX Runtime
        """
        raise NotImplementedError(
            "Exporting to ONNX is not supported for models of type "
            f"{self.__class__.__name__}"
        )

    def import_weights(self, weights: Sequence[float]):
        """
        Inverse of `._export_weights()
        """
        raise NotImplementedError(
            f"Importing weights for models of type {self.__class__.__name__} isn't "
            "implemented yet."
        )

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

    def _get_export_dict(self):
        return {
            "version": _MODEL_VERSION,
            "metadata": self._get_non_user_metadata(),
            "architecture": self.__class__.__name__,
            "config": self._export_config(),
            "weights": self._export_weights().tolist(),
        }

    def _get_non_user_metadata(self) -> Dict[str, Union[str, int, float]]:
        """
        Get any metadata that's non-user-provided (date, loudness, gain)
        """
        t = datetime.now()
        return {
            "date": Date(
                year=t.year,
                month=t.month,
                day=t.day,
                hour=t.hour,
                minute=t.minute,
                second=t.second,
            ).dict()
        }
