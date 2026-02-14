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
from nam.models.wavenet import _FiLM
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

_FILM_SLOTS = (
    "conv_pre_film",
    "conv_post_film",
    "input_mixin_pre_film",
    "input_mixin_post_film",
    "activation_pre_film",
    "activation_post_film",
    "layer1x1_post_film",
    "head1x1_post_film",
)


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

    def test_init_from_config_bottleneck_int(self):
        """WaveNet.init_from_config accepts bottleneck as int (same for all layers)."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "bottleneck": 2,
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        assert y.shape == x.shape

    def test_init_from_config_bottleneck_defaults_to_channels(self):
        """When bottleneck is omitted, it defaults to channels (no compression)."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        assert y.shape == x.shape
        exported = model._export_config()
        assert exported["layers"][0]["bottleneck"] == 4

    def test_init_from_config_layer1x1_active(self):
        """WaveNet.init_from_config uses layer1x1 by default (active=True)."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "bottleneck": 2,
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        layer = model._net._layer_arrays[0]._layers[0]
        assert layer._layer1x1 is not None
        assert layer._layer1x1.in_channels == 2
        assert layer._layer1x1.out_channels == 4
        assert layer._layer1x1.groups == 1

    def test_init_from_config_layer1x1_groups(self):
        """WaveNet.init_from_config accepts layer_1x1_config.groups."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "bottleneck": 4,
                    "layer_1x1_config": {"active": True, "groups": 2},
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        layer = model._net._layer_arrays[0]._layers[0]
        assert layer._layer1x1 is not None
        assert layer._layer1x1.groups == 2

    def test_init_from_config_groups_input_and_groups_input_mixin(self):
        """WaveNet.init_from_config accepts groups_input and groups_input_mixin."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 2,  # divisible by groups_input_mixin for build
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "bottleneck": 4,
                    "groups_input": 2,
                    "groups_input_mixin": 2,
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        layer = model._net._layer_arrays[0]._layers[0]
        assert layer.conv.groups == 2
        assert layer._input_mixer.groups == 2

    def test_export_config_includes_groups_input(self):
        """Exported layer config includes groups_input and groups_input_mixin."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 2,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "groups_input": 2,
                    "groups_input_mixin": 2,
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        exported = model._export_config()
        assert exported["layers"][0]["groups_input"] == 2
        assert exported["layers"][0]["groups_input_mixin"] == 2
        # Default (1) when omitted
        config_default = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model_default = _WaveNet.init_from_config(config_default)
        exported_default = model_default._export_config()
        assert exported_default["layers"][0]["groups_input"] == 1
        assert exported_default["layers"][0]["groups_input_mixin"] == 1

    def test_init_from_config_layer1x1_inactive(self):
        """WaveNet.init_from_config accepts layer_1x1_config with active=False."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "bottleneck": 4,  # Needs to match channels if layer1x1 is inactive
                    "layer_1x1_config": {"active": False},
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        assert y.shape == x.shape
        layer = model._net._layer_arrays[0]._layers[0]
        assert layer._layer1x1 is None

    def test_import_weights_layer1x1_inactive(self):
        """Weight import/export roundtrip works with layer1x1 inactive."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "bottleneck": 2,  # Must match channels if layer1x1 is inactive
                    "layer_1x1_config": {"active": False},
                }
            ],
            "head_scale": 1.0,
        }
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

    def test_export_config_includes_layer1x1(self):
        """Exported layer config includes layer1x1 (default active=True)."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        exported = model._export_config()
        assert "layer1x1" in exported["layers"][0]
        assert exported["layers"][0]["layer1x1"]["active"] is True
        assert exported["layers"][0]["layer1x1"]["groups"] == 1

    def test_init_from_config_head1x1_active(self):
        """WaveNet.init_from_config accepts head_1x1_config with active=True."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "bottleneck": 2,
                    "head_1x1_config": {
                        "active": True,
                        "out_channels": 2,
                        "groups": 1,
                    },
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        assert y.shape == x.shape
        layer = model._net._layer_arrays[0]._layers[0]
        assert layer._head1x1 is not None
        assert layer._head1x1.out_channels == 2

    def test_init_from_config_head1x1_out_channels_independent_of_bottleneck(self):
        """head1x1 out_channels can differ from bottleneck; head_rechannel uses it."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "bottleneck": 2,
                    "head_1x1_config": {
                        "active": True,
                        "out_channels": 4,  # Differs from bottleneck (2)
                        "groups": 1,
                    },
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        assert y.shape == x.shape
        layer_array = model._net._layer_arrays[0]
        assert layer_array._head_rechannel.in_channels == 4

    def test_import_weights_head1x1(self):
        """Weight import/export roundtrip works with head1x1 active."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "bottleneck": 2,
                    "head_1x1_config": {
                        "active": True,
                        "out_channels": 2,
                        "groups": 1,
                    },
                }
            ],
            "head_scale": 1.0,
        }
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

    def test_init_from_config_bottleneck_with_pairing_activation(self):
        """Bottleneck works with PairMultiply activation."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": {
                        "name": "PairMultiply",
                        "primary": "Tanh",
                        "secondary": "Sigmoid",
                    },
                    "bottleneck": 2,
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        assert y.shape == x.shape

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
    def test_export_nam_loadmodel_can_load_with_bottleneck(self):
        """
        LightningModule with bottleneck -> .export() -> loadmodel can load the .nam.
        """
        config = _load_demonet_config()
        for layer in config["net"]["config"]["layers_configs"]:
            layer["bottleneck"] = 2
        module = _LightningModule.init_from_config(config)
        module.net.sample_rate = 48000
        with _TemporaryDirectory() as tmpdir:
            outdir = _Path(tmpdir)
            module.net.export(outdir, basename="model")
            nam_path = outdir / "model.nam"
            assert nam_path.exists()
            result = _run_loadmodel(nam_path)
            assert result.returncode == 0, (
                "loadmodel failed for bottleneck: "
                f"stderr={result.stderr!r} stdout={result.stdout!r}"
            )

    @_requires_neural_amp_modeler_core_loadmodel
    def test_export_nam_loadmodel_can_load_with_groups_input(self):
        """
        LightningModule with groups_input=2 -> .export() -> loadmodel can load the .nam.
        """
        config = _load_demonet_config()
        layers_configs = config["net"]["config"]["layers_configs"]
        # Second layer has channels=2, so groups_input=2 is valid (depthwise).
        layers_configs[1]["groups_input"] = 2
        module = _LightningModule.init_from_config(config)
        module.net.sample_rate = 48000
        with _TemporaryDirectory() as tmpdir:
            outdir = _Path(tmpdir)
            module.net.export(outdir, basename="model")
            nam_path = outdir / "model.nam"
            assert nam_path.exists()
            result = _run_loadmodel(nam_path)
            assert result.returncode == 0, (
                "loadmodel failed for groups_input=2: "
                f"stderr={result.stderr!r} stdout={result.stdout!r}"
            )

    @_requires_neural_amp_modeler_core_loadmodel
    def test_export_nam_loadmodel_can_load_with_head1x1(self):
        """
        LightningModule with head1x1 active -> .export() -> loadmodel can load the .nam.
        """
        config = _load_demonet_config()
        layers_configs = config["net"]["config"]["layers_configs"]
        # head_input is summed across layer arrays, so all must use the same out_channels
        head1x1_out_channels = layers_configs[0]["head_size"]
        for layer in layers_configs:
            layer["head_1x1_config"] = {
                "active": True,
                "out_channels": head1x1_out_channels,
                "groups": 1,
            }
        module = _LightningModule.init_from_config(config)
        module.net.sample_rate = 48000
        with _TemporaryDirectory() as tmpdir:
            outdir = _Path(tmpdir)
            module.net.export(outdir, basename="model")
            nam_path = outdir / "model.nam"
            assert nam_path.exists()
            result = _run_loadmodel(nam_path)
            assert result.returncode == 0, (
                "loadmodel failed for head1x1: "
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

    @_requires_neural_amp_modeler_core_loadmodel
    @_pytest.mark.parametrize("film_slot", _FILM_SLOTS)
    def test_export_nam_loadmodel_can_load_with_film(self, film_slot: str):
        """
        LightningModule with one FiLM slot active -> .export() -> loadmodel can load the .nam.
        """
        config = _load_demonet_config()
        layers_configs = config["net"]["config"]["layers_configs"]
        head_size = layers_configs[0]["head_size"]
        for layer in layers_configs:
            layer["film_params"] = {
                film_slot: {"active": True, "shift": True, "groups": 1},
            }
            if film_slot == "head1x1_post_film":
                layer["head_1x1_config"] = {
                    "active": True,
                    "out_channels": head_size,
                    "groups": 1,
                }
        module = _LightningModule.init_from_config(config)
        module.net.sample_rate = 48000
        with _TemporaryDirectory() as tmpdir:
            outdir = _Path(tmpdir)
            module.net.export(outdir, basename="model")
            nam_path = outdir / "model.nam"
            assert nam_path.exists()
            result = _run_loadmodel(nam_path)
            assert result.returncode == 0, (
                f"loadmodel failed for FiLM slot {film_slot!r}: "
                f"stderr={result.stderr!r} stdout={result.stdout!r}"
            )

    # --- WaveNet + FiLM tests ---

    @_pytest.mark.parametrize("film_slot", _FILM_SLOTS)
    def test_init_from_config_conv_pre_film(self, film_slot: str):
        """WaveNet.init_from_config accepts each FiLM slot and forward runs."""
        layer_config = {
            "input_size": 1,
            "condition_size": 1,
            "head_size": 1,
            "channels": 2,
            "kernel_size": 2,
            "dilations": [1],
            "activation": "Tanh",
            "film_params": {
                film_slot: {"active": True, "shift": True, "groups": 1},
            },
        }
        if film_slot == "head1x1_post_film":
            layer_config["head_1x1_config"] = {
                "active": True,
                "out_channels": 1,
                "groups": 1,
            }
        config = {
            "layers_configs": [layer_config],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field >= 1
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        assert y.shape == x.shape
        layer = model._net._layer_arrays[0]._layers[0]
        attr = f"_{film_slot}"
        assert getattr(layer, attr) is not None

    @_pytest.mark.parametrize("film_slot", _FILM_SLOTS)
    def test_export_config_includes_film(self, film_slot: str):
        """Exported config includes FiLM keys when film_params present."""
        layer_config = {
            "input_size": 1,
            "condition_size": 1,
            "head_size": 1,
            "channels": 2,
            "kernel_size": 2,
            "dilations": [1],
            "activation": "Tanh",
            "film_params": {
                film_slot: {"active": True, "shift": True, "groups": 1},
            },
        }
        if film_slot == "head1x1_post_film":
            layer_config["head_1x1_config"] = {
                "active": True,
                "out_channels": 1,
                "groups": 1,
            }
        config = {
            "layers_configs": [layer_config],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        exported = model._export_config()
        assert film_slot in exported["layers"][0]
        assert exported["layers"][0][film_slot]["active"] is True
        assert exported["layers"][0][film_slot]["shift"] is True
        assert exported["layers"][0][film_slot]["groups"] == 1

    @_pytest.mark.parametrize("film_slot", _FILM_SLOTS)
    def test_import_weights_with_film(self, film_slot: str):
        """Weight import/export roundtrip works with FiLM active."""
        layer_config = {
            "input_size": 1,
            "condition_size": 1,
            "head_size": 1,
            "channels": 2,
            "kernel_size": 2,
            "dilations": [1],
            "activation": "Tanh",
            "film_params": {
                film_slot: {"active": True, "shift": True, "groups": 1},
            },
        }
        if film_slot == "head1x1_post_film":
            layer_config["head_1x1_config"] = {
                "active": True,
                "out_channels": 1,
                "groups": 1,
            }
        config = {
            "layers_configs": [layer_config],
            "head_scale": 1.0,
        }
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

    def test_film_layer1x1_post_film_requires_layer1x1(self):
        """layer1x1_post_film cannot be active when layer1x1 is inactive."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "bottleneck": 2,
                    "layer_1x1_config": {"active": False},
                    "film_params": {
                        "layer1x1_post_film": {"active": True},
                    },
                }
            ],
            "head_scale": 1.0,
        }
        with _pytest.raises(ValueError, match="layer1x1_post_film cannot be active"):
            _WaveNet.init_from_config(config)

    def test_film_head1x1_post_film_requires_head1x1(self):
        """head1x1_post_film cannot be active when head1x1 is inactive."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "film_params": {
                        "head1x1_post_film": {"active": True},
                    },
                }
            ],
            "head_scale": 1.0,
        }
        with _pytest.raises(ValueError, match="head1x1_post_film cannot be active"):
            _WaveNet.init_from_config(config)

    # --- condition_dsp tests ---

    def test_condition_dsp_none_forward(self):
        """Without condition_dsp, condition is the raw input and forward runs."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model._net._condition_dsp is None
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        # With pad_start_default True, output length equals input length
        assert y.shape == x.shape

    def test_condition_dsp_wavenet_forward(self):
        """With condition_dsp (nested WaveNet), condition is processed and forward runs."""
        # Condition DSP: 1 in -> 2 out. Main: condition_size 2, one layer array.
        config = {
            "condition_dsp": {
                "name": "WaveNet",
                "config": {
                    "layers_configs": [
                        {
                            "input_size": 1,
                            "condition_size": 1,
                            "head_size": 2,
                            "channels": 2,
                            "kernel_size": 2,
                            "dilations": [1],
                            "activation": "Tanh",
                        }
                    ],
                    "head_scale": 1.0,
                },
            },
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 2,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model._net._condition_dsp is not None
        x = _torch.randn(1, model.receptive_field + 8)
        y = model(x)
        # With pad_start_default True, output length equals input length
        assert y.shape == x.shape

    def test_condition_dsp_export_config_includes_condition_dsp(self):
        """
        Exported config includes condition_dsp (full export dict) when set.
        """
        config = {
            "condition_dsp": {
                "name": "WaveNet",
                "config": {
                    "layers_configs": [
                        {
                            "input_size": 1,
                            "condition_size": 1,
                            "head_size": 1,
                            "channels": 2,
                            "kernel_size": 2,
                            "dilations": [1],
                            "activation": "Tanh",
                        }
                    ],
                    "head_scale": 1.0,
                },
            },
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        exported = model._export_config()
        assert "condition_dsp" in exported
        cd = exported["condition_dsp"]
        assert cd["architecture"] == "WaveNet"
        assert "config" in cd
        assert "weights" in cd
        assert "layers" in cd["config"]

    def test_condition_dsp_export_weights_main_only(self):
        """Main model export_weights does not include condition_dsp weights."""
        config_no_cd = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        config_with_cd = {
            **config_no_cd,
            "condition_dsp": {
                "name": "WaveNet",
                "config": {
                    "layers_configs": [
                        {
                            "input_size": 1,
                            "condition_size": 1,
                            "head_size": 1,
                            "channels": 2,
                            "kernel_size": 2,
                            "dilations": [1],
                            "activation": "Tanh",
                        }
                    ],
                    "head_scale": 1.0,
                },
            },
        }
        model_no_cd = _WaveNet.init_from_config(config_no_cd)
        model_with_cd = _WaveNet.init_from_config(config_with_cd)
        main_weights_no_cd = model_no_cd._export_weights()
        main_weights_with_cd = model_with_cd._export_weights()
        # Same main layout: one layer array + head_scale. Same length.
        assert len(main_weights_no_cd) == len(main_weights_with_cd)
        # condition_dsp has its own weights in the export dict, not in main
        exported = model_with_cd._export_config()
        assert "condition_dsp" in exported
        assert "weights" in exported["condition_dsp"]
        assert len(exported["condition_dsp"]["weights"]) > 0

    # --- Bug fix regression tests (see commit "Fix bugs in WaveNet") ---

    def test_condition_dsp_with_conv_pre_film_forward(self):
        """
        Bug fix: When condition_dsp shortens c, x and c have different lengths in
        the first layer. conv_pre_film must slice both to min length before FiLM.
        Without the fix: shape mismatch in FiLM (x and c different L).
        """
        config = {
            "condition_dsp": {
                "name": "WaveNet",
                "config": {
                    "layers_configs": [
                        {
                            "input_size": 1,
                            "condition_size": 1,
                            "head_size": 2,
                            "channels": 2,
                            "kernel_size": 2,
                            "dilations": [1],
                            "activation": "Tanh",
                        }
                    ],
                    "head_scale": 1.0,
                },
            },
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 2,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "film_params": {"conv_pre_film": {"active": True, "shift": True, "groups": 1}},
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        x = _torch.randn(1, model.receptive_field + 16)
        y = model(x)
        assert y.shape == x.shape

    def test_condition_dsp_zconv_mix_out_length_mismatch_forward(self):
        """
        Bug fix: When condition_dsp shortens c, zconv (from conv on x) can be longer
        than mix_out (from input_mixer on c). Must use min length and slice both.
        Without the fix: zconv + mix_out fails (different shapes).
        """
        # Condition DSP: k=2,d=1 -> rf=2, c has length L-1.
        # Main: k=2,d=1, so first layer zconv has length L-1 from x (length L).
        # Wait - x gets rechanneled, so it has length L. Conv reduces to L-1.
        # c has length L-1 (condition_dsp output). So zconv L-1, mix_out from
        # mixer(c) has length L-1. They match. Need condition_dsp with larger rf
        # so c is much shorter. Cond: k=3, d=2 -> rf=1+2*2=5, c has L-4.
        # Main: k=2,d=1, zconv has L-1. So zconv (L-1) > mix_out (L-4). Good.
        config = {
            "condition_dsp": {
                "name": "WaveNet",
                "config": {
                    "layers_configs": [
                        {
                            "input_size": 1,
                            "condition_size": 1,
                            "head_size": 2,
                            "channels": 2,
                            "kernel_size": 3,
                            "dilations": [2],  # rf = 1 + (3-1)*2 = 5, c has L-4
                            "activation": "Tanh",
                        }
                    ],
                    "head_scale": 1.0,
                },
            },
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 2,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],  # zconv has L-1
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        x = _torch.randn(1, model.receptive_field + 20)
        y = model(x)
        assert y.shape == x.shape

    def test_condition_dsp_out_length_uses_min_of_x_and_c(self):
        """
        Bug fix: out_length must be min(x.shape[2], c.shape[2]) - (rf-1) so we
        don't request more samples than c provides. Without the fix: index error
        or wrong output shape when c is shorter than x.
        """
        config = {
            "condition_dsp": {
                "name": "WaveNet",
                "config": {
                    "layers_configs": [
                        {
                            "input_size": 1,
                            "condition_size": 1,
                            "head_size": 2,
                            "channels": 2,
                            "kernel_size": 2,
                            "dilations": [1, 2],
                            "activation": "Tanh",
                        }
                    ],
                    "head_scale": 1.0,
                },
            },
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 2,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        x = _torch.randn(1, model.receptive_field + 16)
        y = model(x)
        assert y.shape == x.shape

    def test_receptive_field_includes_condition_dsp(self):
        """
        Bug fix: receptive_field must include condition_dsp's receptive field when
        condition_dsp is present. Without the fix: under-reported rf.
        """
        cond_config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 2,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                }
            ],
            "head_scale": 1.0,
        }
        main_layer_config = {
            "input_size": 1,
            "condition_size": 2,
            "head_size": 1,
            "channels": 2,
            "kernel_size": 2,
            "dilations": [1],
            "activation": "Tanh",
        }
        cond_rf = 1 + (2 - 1) * 1  # 2
        main_rf = 1 + (2 - 1) * 1  # 2
        expected_rf = 1 + (main_rf - 1) + (cond_rf - 1)  # 1 + 1 + 1 = 3
        config = {
            "condition_dsp": {"name": "WaveNet", "config": cond_config},
            "layers_configs": [main_layer_config],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        assert model.receptive_field == expected_rf, (
            f"receptive_field should be {expected_rf} (main + condition_dsp), "
            f"got {model.receptive_field}"
        )

    @_requires_neural_amp_modeler_core_loadmodel
    def test_export_nam_loadmodel_can_load_with_condition_dsp(self):
        """
        WaveNet with condition_dsp: export to .nam -> loadmodel can load the file.
        """
        # Condition DSP outputs 2 channels; main first layer has condition_size 2.
        config = {
            "net": {
                "name": "WaveNet",
                "config": {
                    "condition_dsp": {
                        "name": "WaveNet",
                        "config": {
                            "layers_configs": [
                                {
                                    "input_size": 1,
                                    "condition_size": 1,
                                    "head_size": 2,
                                    "channels": 2,
                                    "kernel_size": 2,
                                    "dilations": [1],
                                    "activation": "Tanh",
                                }
                            ],
                            "head_scale": 1.0,
                        },
                    },
                    "layers_configs": [
                        {
                            "input_size": 1,
                            "condition_size": 2,
                            "head_size": 1,
                            "channels": 2,
                            "kernel_size": 2,
                            "dilations": [1],
                            "activation": "Tanh",
                        }
                    ],
                    "head_scale": 1.0,
                },
            },
            "optimizer": {"lr": 0.004},
            "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.993}},
        }
        module = _LightningModule.init_from_config(config)
        module.net.sample_rate = 48000
        with _TemporaryDirectory() as tmpdir:
            outdir = _Path(tmpdir)
            module.net.export(outdir, basename="model")
            nam_path = outdir / "model.nam"
            assert nam_path.exists()
            result = _run_loadmodel(nam_path)
            assert result.returncode == 0, (
                "loadmodel failed for condition_dsp: "
                f"stderr={result.stderr!r} stdout={result.stdout!r}"
            )


class TestFiLM:
    """Tests for the _FiLM class itself."""

    def test_forward_shape(self):
        """_FiLM forward returns correct shape (scale-only and scale+shift)."""
        B, C, Dc, L = 2, 4, 3, 10
        x = _torch.randn(B, C, L)
        c = _torch.randn(B, Dc, L)
        for shift in (True, False):
            film = _FiLM(condition_size=Dc, input_dim=C, shift=shift, groups=1)
            out = film(x, c)
            assert out.shape == (B, C, L)

    def test_shift_false_scale_only(self):
        """_FiLM with shift=False applies only scale (no additive shift)."""
        B, C, Dc, L = 2, 4, 3, 10
        x = _torch.randn(B, C, L)
        c = _torch.randn(B, Dc, L)
        film = _FiLM(condition_size=Dc, input_dim=C, shift=False, groups=1)
        out = film(x, c)
        # With zero input, output should be zero (no shift, scale multiplies against
        # zero input)
        out_zero = film(_torch.zeros(B, C, L), c)
        assert out_zero.abs().max() < 1e-5

    def test_export_import_weights_roundtrip(self):
        """_FiLM export_weights/import_weights roundtrip preserves behavior."""
        B, C, Dc, L = 2, 4, 3, 10
        film1 = _FiLM(condition_size=Dc, input_dim=C, shift=True, groups=1)
        film2 = _FiLM(condition_size=Dc, input_dim=C, shift=True, groups=1)
        x = _torch.randn(B, C, L)
        c = _torch.randn(B, Dc, L)
        y1 = film1(x, c)
        film2.import_weights(film1.export_weights(), 0)
        y2 = film2(x, c)
        assert _torch.allclose(y1, y2)


if __name__ == "__main__":
    _pytest.main()
