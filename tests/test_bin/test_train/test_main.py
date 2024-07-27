# File: test_main.py
# Created Date: Sunday April 30th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import json
from enum import Enum
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Dict, Tuple, Union

import numpy as np
import pytest
import torch

from nam.data import np_to_wav


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
        data_config = {
            "train": {
                "start": None,
                "stop": -self._num_samples_validation,
                "ny": self._ny,
            },
            "validation": {
                "start": -self._num_samples_validation,
                "stop": None,
                "ny": None,
            },
            "common": {
                "x_path": str(self._x_path(root_path)),
                "y_path": str(self._y_path(root_path)),
                "delay": 0,
                "require_input_pre_silence": None,
            },
        }
        stage_channels = (3, 2)
        model_config = {
            "net": {
                "name": "WaveNet",
                "config": {
                    "layers_configs": [
                        {
                            "condition_size": 1,
                            "input_size": 1,
                            "channels": stage_channels[0],
                            "head_size": stage_channels[1],
                            "kernel_size": 3,
                            "dilations": [1],
                            "activation": "Tanh",
                            "gated": False,
                            "head_bias": False,
                        },
                        {
                            "condition_size": 1,
                            "input_size": stage_channels[0],
                            "channels": stage_channels[1],
                            "head_size": 1,
                            "kernel_size": 3,
                            "dilations": [2],
                            "activation": "Tanh",
                            "gated": False,
                            "head_bias": False,
                        },
                    ],
                    "head_scale": 0.02,
                },
            },
            "optimizer": {"lr": 0.004},
            "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
        }

        def extra_trainer_kwargs(device) -> Dict[str, Union[int, str]]:
            return {
                _Device.GPU: {"accelerator": "gpu", "devices": 1},
                _Device.MPS: {"accelerator": "mps", "devices": 1},
            }.get(device, {})

        learning_config = {
            "train_dataloader": {
                "batch_size": 3,
                "shuffle": True,
                "pin_memory": True,
                "drop_last": True,
                "num_workers": 0,
            },
            "val_dataloader": {},
            "trainer": {"max_epochs": 2, **extra_trainer_kwargs(device)},
            "trainer_fit_kwargs": {},
        }

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
