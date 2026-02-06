# File: test_wavenet.py
# Created Date: Friday May 5th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import pytest as _pytest
import torch as _torch

from nam.models._activations import (
    PairBlend as _PairBlend,
    PairMultiply as _PairMultiply,
)
from nam.models.wavenet import WaveNet as _WaveNet
from nam.train.core import (
    Architecture as _Architecture,
    get_wavenet_config as _get_wavenet_config,
)
from nam.train.lightning_module import LightningModule as _LightningModule
from tests._integration import run_loadmodel as _run_loadmodel
from tests._skips import (
    requires_neural_amp_modeler_core_loadmodel as _requires_neural_amp_modeler_core_loadmodel,
)

from .base import Base as _Base

# Activations supported by both Python _activations and NeuralAmpModelerCore loadmodel.
# (Fasttanh is C++-only; omit it since we build the model in Python.)
_LOADMODEL_ACTIVATIONS = [
    "Tanh",
    "Hardtanh",
    # "Fasttanh",  # C++ approximation of Tanh; not used in the trainer.
    "ReLU",
    "LeakyReLU",
    "PReLU",
    "Sigmoid",
    "SiLU",
    "Hardswish",
    "LeakyHardtanh",
    "Softsign",
    {
        "name": "PairBlend",
        "primary": "Tanh",
        "secondary": "Sigmoid",
    },
    {
        "name": "PairMultiply",
        "primary": "Tanh",
        "secondary": "Sigmoid",
    },
]

_NAM_FULL_CONFIGS_DIR = _Path(__file__).resolve().parents[3] / "nam_full_configs"


def _load_demonet_config() -> dict:
    path = _NAM_FULL_CONFIGS_DIR / "models" / "demonet.json"
    data = _json.loads(path.read_text())
    data.pop("_notes", None)
    data.pop("_comments", None)
    return data


class TestWaveNet(_Base):
    @classmethod
    def setup_class(cls):
        C = _WaveNet
        args = ()
        kwargs = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 1,
                    "kernel_size": 1,
                    "dilations": [1],
                }
            ]
        }
        super().setup_class(C, args, kwargs)

    def test_import_weights(self):
        config = _get_wavenet_config(_Architecture.FEATHER)
        model_1 = _WaveNet.init_from_config(config)
        model_2 = _WaveNet.init_from_config(config)

        batch_size = 2
        x = _torch.randn(batch_size, model_1.receptive_field + 23)

        y1 = model_1(x)
        y2_before = model_2(x)

        model_2.import_weights(model_1._export_weights())
        y2_after = model_2(x)

        assert not _torch.allclose(y2_before, y1)
        assert _torch.allclose(y2_after, y1)

    def test_init_from_config_activation_list_of_str(self):
        """WaveNet.init_from_config accepts activation as a list of str per layer."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": ["Tanh", "ReLU"],
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)  # Pre-pads
        assert y.shape == x.shape

    @_pytest.mark.parametrize("pairing_name", ["PairMultiply", "PairBlend"])
    def test_init_from_config_activation_dict_pairing(self, pairing_name: str):
        """
        WaveNet.init_from_config accepts activation as a dict for
        PairMultiply/PairBlend.
        """
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": {
                        "name": pairing_name,
                        "primary": "Tanh",
                        "secondary": "Sigmoid",
                    },
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 4)
        y = model(x)  # Pre-pads
        assert y.shape == x.shape
        pairing_cls = _PairMultiply if pairing_name == "PairMultiply" else _PairBlend
        assert isinstance(
            model._net._layer_arrays[0]._layers[0]._activation, pairing_cls
        )

    def test_init_from_config_activation_mixed_per_layer(self):
        """WaveNet.init_from_config accepts different activations (basic and pair) per layer."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": [
                        "ReLU",
                        {
                            "name": "PairBlend",
                            "primary": "Tanh",
                            "secondary": "Sigmoid",
                        },
                    ],
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 4)
        y = model(x)  # Pre-pads
        assert y.shape == x.shape
        layers = model._net._layer_arrays[0]._layers
        assert isinstance(layers[0]._activation, _torch.nn.ReLU)
        assert isinstance(layers[1]._activation, _PairBlend)

    @_requires_neural_amp_modeler_core_loadmodel
    @_pytest.mark.parametrize("activation", _LOADMODEL_ACTIVATIONS)
    def test_export_nam_loadmodel_can_load(self, activation: str):
        """
        LightningModule.init_from_config(demonet with activation replaced) -> .export()
        -> loadmodel can load the resulting .nam.
        """
        config = _load_demonet_config()
        for layer in config["net"]["config"]["layers_configs"]:
            layer["activation"] = activation
        module = _LightningModule.init_from_config(config)
        module.net.sample_rate = 48000
        with _TemporaryDirectory() as tmpdir:
            outdir = _Path(tmpdir)
            module.net.export(outdir, basename="model")
            nam_path = outdir / "model.nam"
            assert nam_path.exists()
            result = _run_loadmodel(nam_path)
            assert result.returncode == 0, (
                f"loadmodel failed for activation={activation!r}: "
                f"stderr={result.stderr!r} stdout={result.stdout!r}"
            )

    @_requires_neural_amp_modeler_core_loadmodel
    def test_export_nam_loadmodel_can_load_different_activation_per_layer(self):
        """
        Same as test_export_nam_loadmodel_can_load but with a different activation
        for each layer in the layer array (loadmodel still loads the .nam).
        """
        config = _load_demonet_config()
        layers_configs = config["net"]["config"]["layers_configs"]
        # Use a distinct loadmodel-supported activation for each layer in the array.
        per_layer_activations = ["Tanh", "ReLU"]
        for i, layer in enumerate(layers_configs):
            layer["activation"] = per_layer_activations[i]
        module = _LightningModule.init_from_config(config)
        module.net.sample_rate = 48000
        with _TemporaryDirectory() as tmpdir:
            outdir = _Path(tmpdir)
            module.net.export(outdir, basename="model")
            nam_path = outdir / "model.nam"
            assert nam_path.exists()
            result = _run_loadmodel(nam_path)
            assert result.returncode == 0, (
                "loadmodel failed for per-layer activations: "
                f"stderr={result.stderr!r} stdout={result.stdout!r}"
            )


if __name__ == "__main__":
    _pytest.main()
