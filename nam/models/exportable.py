# File: _exportable.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc as _abc
import json as _json
import logging as _logging
from datetime import datetime as _datetime
from enum import Enum as _Enum
from pathlib import Path as _Path
from typing import (
    Any as _Any,
    Dict as _Dict,
    Optional as _Optional,
    Sequence as _Sequence,
    Tuple as _Tuple,
    Union as _Union,
)

import numpy as _np

from .metadata import Date as _Date, UserMetadata as _UserMetadata

logger = _logging.getLogger(__name__)

# Model version is independent from package version as of package version 0.5.2 so that
# the API of the package can iterate at a different pace from that of the model files.
_MODEL_VERSION = "0.5.4"


def _cast_enums(d: _Dict[_Any, _Any]) -> _Dict[_Any, _Any]:
    """
    Casts enum-type keys to their values
    """
    out = {}
    for key, val in d.items():
        if isinstance(val, _Enum):
            val = val.value
        if isinstance(val, dict):
            val = _cast_enums(val)
        out[key] = val
    return out


class Exportable(_abc.ABC):
    """
    Interface for my custon export format for use in the plugin.
    """

    FILE_EXTENSION = ".nam"

    def export(
        self,
        outdir: _Path,
        include_snapshot: bool = False,
        basename: str = "model",
        user_metadata: _Optional[_UserMetadata] = None,
        other_metadata: _Optional[dict] = None,
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
            {} if user_metadata is None else _cast_enums(user_metadata.model_dump())
        )
        if other_metadata is not None:
            overwritten_keys = []
            for key in other_metadata:
                if key in model_dict["metadata"]:
                    overwritten_keys.append(key)
            if overwritten_keys:
                logger.warning(
                    "other_metadata provided keys that will overwrite existing keys!\n "
                    + "\n ".join(overwritten_keys)
                )
            model_dict["metadata"].update(_cast_enums(other_metadata))

        training = self.training
        self.eval()
        with open(_Path(outdir, f"{basename}{self.FILE_EXTENSION}"), "w") as fp:
            _json.dump(model_dict, fp)
        if include_snapshot:
            x, y = self._export_input_output()
            x_path = _Path(outdir, "test_inputs.npy")
            y_path = _Path(outdir, "test_outputs.npy")
            logger.debug(f"Saving snapshot input to {x_path}")
            _np.save(x_path, x)
            logger.debug(f"Saving snapshot output to {y_path}")
            _np.save(y_path, y)

        # And resume training state
        self.train(training)

    def export_onnx(self, filename: _Path):
        """
        Export model in format for ONNX Runtime
        """
        raise NotImplementedError(
            "Exporting to ONNX is not supported for models of type "
            f"{self.__class__.__name__}"
        )

    def import_weights(self, weights: _Sequence[float]):
        """
        Inverse of `._export_weights()
        """
        raise NotImplementedError(
            f"Importing weights for models of type {self.__class__.__name__} isn't "
            "implemented yet."
        )

    @_abc.abstractmethod
    def _export_config(self):
        """
        Creates the JSON of the model's archtecture hyperparameters (number of layers,
        number of units, etc)

        :return: a JSON serializable object
        """
        pass

    @_abc.abstractmethod
    def _export_input_output(self) -> _Tuple[_np.ndarray, _np.ndarray]:
        """
        Create an input and corresponding output signal to verify its behavior.

        They should be the same length, but the start of the output might have transient
        effects. Up to you to interpret.
        """
        pass

    @_abc.abstractmethod
    def _export_weights(self) -> _np.ndarray:
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

    def _get_non_user_metadata(self) -> _Dict[str, _Union[str, int, float]]:
        """
        Get any metadata that's non-user-provided (date, loudness, gain)
        """
        t = _datetime.now()
        return {
            "date": _Date(
                year=t.year,
                month=t.month,
                day=t.day,
                hour=t.hour,
                minute=t.minute,
                second=t.second,
            ).model_dump()
        }
