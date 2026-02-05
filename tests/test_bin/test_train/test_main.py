# File: test_main.py
# Created Date: Sunday April 30th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import json
from copy import deepcopy
from enum import Enum
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Dict, Tuple, Union

import numpy as np
import pytest
import torch

from nam.data import np_to_wav

# Configs are loaded from nam_full_configs and patched for minimal fast runs.
_NAM_FULL_CONFIGS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "nam_full_configs"
)


def _load_json(config_dir: str, name: str) -> dict:
    path = _NAM_FULL_CONFIGS_DIR / config_dir / f"{name}.json"
    with open(path, "r") as f:
        data = json.load(f)
    data.pop("_notes", None)
    data.pop("_comments", None)
    return data


class _Device(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MPS = "mps"


class Test(object):
    @classmethod
    def setup_class(cls):
        cls._num_samples = 128
        cls._num_samples_validation = 15
        cls._ny = 2
        cls._batch_size = 2

    def test_cpu(self):
        self._t_main(_Device.CPU)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
    def test_gpu(self):
        self._t_main(_Device.GPU)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS test")
    def test_mps(self):
        self._t_main(_Device.MPS)

    @classmethod
    def _data_config_path(cls, root_path: Path) -> Path:
        return Path(cls._input_path(root_path), "data_config.json")

    def _get_configs(
        self, root_path: Path, device: _Device
    ) -> Tuple[Dict, Dict, Dict]:  # TODO pydantic models
        # Data: from single_pair; patch paths and trim for minimal 128 samples
        data_config = _load_json("data", "single_pair")
        data_config = deepcopy(data_config)
        common = data_config["common"]
        common["x_path"] = str(self._x_path(root_path))
        common["y_path"] = str(self._y_path(root_path))
        common["require_input_pre_silence"] = None
        data_config["train"] = {
            "start": None,
            "stop": -self._num_samples_validation,
            "ny": self._ny,
        }
        data_config["validation"] = {
            "start": -self._num_samples_validation,
            "stop": None,
            "ny": None,
        }

        # Model: from demonet (kept tiny so this test runs fast)
        model_config = _load_json("models", "demonet")

        # Learning: from demo; patch device and short run
        learning_config = _load_json("learning", "demo")
        learning_config = deepcopy(learning_config)
        extra_trainer_kwargs: Dict[str, Union[int, str]] = {
            _Device.CPU: {"accelerator": "cpu"},
            _Device.GPU: {"accelerator": "gpu", "devices": 1},
            _Device.MPS: {"accelerator": "mps", "devices": 1},
        }.get(device, {})
        learning_config["trainer"]["max_epochs"] = 2
        learning_config["trainer"].update(extra_trainer_kwargs)
        learning_config["train_dataloader"]["batch_size"] = 3

        return data_config, model_config, learning_config

    def _get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: (N,), (N,)
        """
        x = np.random.rand(self._num_samples) - 0.5
        y = 1.1 * x
        return x, y

    @classmethod
    def _input_path(cls, root_path: Path, ensure: bool = False) -> Path:
        p = Path(root_path, "inputs")
        if ensure:
            p.mkdir()
        return p

    @classmethod
    def _learning_config_path(cls, root_path: Path) -> Path:
        return Path(cls._input_path(root_path), "learning_config.json")

    @classmethod
    def _model_config_path(cls, root_path: Path) -> Path:
        return Path(cls._input_path(root_path), "model_config.json")

    @classmethod
    def _output_path(cls, root_path: Path, ensure: bool = False) -> Path:
        p = Path(root_path, "outputs")
        if ensure:
            p.mkdir()
        return p

    def _setup_files(self, root_path: Path, device: _Device):
        x, y = self._get_data()
        np_to_wav(x, self._x_path(root_path))
        np_to_wav(y, self._y_path(root_path))
        data_config, model_config, learning_config = self._get_configs(
            root_path, device
        )
        with open(self._data_config_path(root_path), "w") as fp:
            json.dump(data_config, fp)
        with open(self._model_config_path(root_path), "w") as fp:
            json.dump(model_config, fp)
        with open(self._learning_config_path(root_path), "w") as fp:
            json.dump(learning_config, fp)

    def _t_main(self, device: _Device):
        """
        End-to-end test of bin/train/main.py
        """
        with TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            self._input_path(tempdir, ensure=True)
            self._setup_files(tempdir, device)
            check_call(
                [
                    "nam-full",  # HACK not DRY w/ setup.py
                    str(self._data_config_path(tempdir)),
                    str(self._model_config_path(tempdir)),
                    str(self._learning_config_path(tempdir)),
                    str(self._output_path(tempdir, ensure=True)),
                    "--no-show",
                ]
            )

    @classmethod
    def _x_path(cls, root_path: Path) -> Path:
        return Path(cls._input_path(root_path), "input.wav")

    @classmethod
    def _y_path(cls, root_path: Path) -> Path:
        return Path(cls._input_path(root_path), "output.wav")


if __name__ == "__main__":
    pytest.main()
