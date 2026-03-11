# File: test_wavenet.py
# Created Date: Friday May 5th 2023
# Author: Steven Atkinson (steven@atkinson.mn)

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.models._activations import PairBlend as _PairBlend
from nam.models._activations import PairMultiply as _PairMultiply
from nam.models.wavenet import WaveNet as _WaveNet
from nam.models.wavenet._film import FiLM as _FiLM
from nam.models.wavenet._slimmable import Slimmable as _Slimmable
from nam.models.wavenet._slimmable_conv import (
    SlimmableConv1dBase as _SlimmableConv1dBase,
)
from nam.models.wavenet._slimmable_conv import class_set as _slimmable_class_set
from nam.train.core import Architecture as _Architecture
from nam.train.core import get_wavenet_config as _get_wavenet_config

from .base import Base as _Base

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


class TestWaveNet(_Base):
    """Tests for WaveNet. Uses init_from_config for construction."""

    _DEFAULT_CONFIG = {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "head_size": 1,
                "channels": 1,
                "kernel_size": 1,
                "dilations": [1],
                "activation": "Tanh",
            }
        ],
        "head_scale": 1.0,
    }

    @classmethod
    def setup_class(cls):
        C = _WaveNet
        args = ()
        kwargs = {"config": cls._DEFAULT_CONFIG}
        super().setup_class(C, args, kwargs)

    def _construct(self, C=None, args=None, kwargs=None):
        """Build WaveNet via init_from_config; kwargs may contain 'config'."""
        C = self._C if C is None else C
        kwargs = self._kwargs if kwargs is None else kwargs
        config = kwargs.get("config", self._DEFAULT_CONFIG)
        return C.init_from_config(config)

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

    @_pytest.mark.parametrize("pairing_name", ["PairMultiply", "PairBlend"])
    def test_layer_bottleneck_property_returns_internal_channels_for_pairing(
        self, pairing_name: str
    ):
        """Layer.bottleneck returns internal channel count (not conv.out_channels) for pairing activations."""
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
                        "name": pairing_name,
                        "primary": "Tanh",
                        "secondary": "Sigmoid",
                    },
                    "bottleneck": 2,
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        layer = model._net._layer_arrays[0]._layers[0]
        # For pairing, conv outputs 2*bottleneck=4; bottleneck property should return 2.
        assert layer.conv.out_channels == 4
        assert layer.bottleneck == 2
        exported = model._net._layer_arrays[0].export_config()
        assert exported["bottleneck"] == 2

    def test_layer_bottleneck_property_returns_conv_out_channels_for_simple_activation(
        self,
    ):
        """Layer.bottleneck returns conv.out_channels for non-pairing activations."""
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
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        layer = model._net._layer_arrays[0]._layers[0]
        assert layer.conv.out_channels == 2
        assert layer.bottleneck == 2

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
                    "film_params": {
                        "conv_pre_film": {"active": True, "shift": True, "groups": 1}
                    },
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


class TestSlimmableWaveNet:
    """Tests for slimmable (channel-slicing) WaveNet training."""

    def test_slimmable_config_builds_and_forward(self):
        """WaveNet with slimmable config builds and forward runs in train and eval."""
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
                    "head_bias": True,
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)

        model.train()
        y_train = model(x)
        assert y_train.shape == x.shape

        model.eval()
        y_eval = model(x)
        assert y_eval.shape == x.shape

    def test_slimmable_export_full_size(self):
        """Slimmable WaveNet exports full-size weights."""
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
                    "head_bias": True,
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                }
            ],
            "head_scale": 1.0,
        }
        model = _WaveNet.init_from_config(config)
        weights = model._export_weights()
        assert isinstance(weights, (list, _torch.Tensor, _np.ndarray))
        if hasattr(weights, "shape"):
            assert weights.ndim == 1
        else:
            assert len(weights) > 0

    def test_slimmable_with_head1x1_raises(self):
        """Slimmable + head 1x1 raises NotImplementedError."""
        config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 2,
                    "channels": 4,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "head_bias": True,
                    "head_1x1_config": {"active": True, "out_channels": 2},
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                }
            ],
            "head_scale": 1.0,
        }
        with _pytest.raises(NotImplementedError, match="head 1x1"):
            _WaveNet.init_from_config(config)

    def test_slimmable_with_film_raises(self):
        """Slimmable + FiLM raises NotImplementedError."""
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
                    "head_bias": True,
                    "film_params": {"conv_pre_film": {"active": True}},
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                }
            ],
            "head_scale": 1.0,
        }
        with _pytest.raises(NotImplementedError, match="FiLM"):
            _WaveNet.init_from_config(config)

    def test_slimmable_with_condition_dsp_raises(self):
        """Slimmable + condition_dsp raises NotImplementedError."""
        cond_config = {
            "layers_configs": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "head_bias": True,
                }
            ],
            "head_scale": 1.0,
        }
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
                    "head_bias": True,
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                }
            ],
            "head_scale": 1.0,
            "condition_dsp": {"name": "WaveNet", "config": cond_config},
        }
        with _pytest.raises(NotImplementedError, match="condition_dsp"):
            _WaveNet.init_from_config(config)

    def test_slimmable_with_groups_raises(self):
        """Slimmable + groups > 1 raises NotImplementedError."""
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
                    "head_bias": True,
                    "groups_input": 2,
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                }
            ],
            "head_scale": 1.0,
        }
        with _pytest.raises(NotImplementedError, match="groups"):
            _WaveNet.init_from_config(config)

    def test_slimmable_with_multiple_layer_arrays_raises(self):
        """Slimmable + more than one layer array raises NotImplementedError."""
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
                    "head_bias": True,
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                },
                {
                    "input_size": 4,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 2,
                    "kernel_size": 2,
                    "dilations": [1],
                    "activation": "Tanh",
                    "head_bias": True,
                    "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
                },
            ],
            "head_scale": 1.0,
        }
        with _pytest.raises(NotImplementedError, match="more than one layer array"):
            _WaveNet.init_from_config(config)

    def test_slimmable_unsupported_format_raises(self):
        """Slimmable with unsupported format raises NotImplementedError."""
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
                    "head_bias": True,
                    "slimmable": {},  # Need to define how to slim!
                }
            ],
            "head_scale": 1.0,
        }
        with _pytest.raises(NotImplementedError, match="slice_channels_uniform"):
            _WaveNet.init_from_config(config)

        # Some other unsupported method
        config["layers_configs"][0]["slimmable"] = {
            "method": "other_method",
            "kwargs": {},
        }
        with _pytest.raises(NotImplementedError, match="slice_channels_uniform"):
            _WaveNet.init_from_config(config)

    # --- Extended slimmable capability tests ---

    def _slimmable_config(self, **layer_overrides):
        """Base slimmable config with optional layer overrides."""
        base = {
            "input_size": 1,
            "condition_size": 1,
            "head_size": 1,
            "channels": 4,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "Tanh",
            "head_bias": True,
            "slimmable": {"method": "slice_channels_uniform", "kwargs": {}},
        }
        base.update(layer_overrides)
        return {"layers_configs": [base], "head_scale": 1.0}

    def test_slimmable_is_slimmable_returns_true(self):
        """Slimmable model reports is_slimmable() as True."""
        config = self._slimmable_config()
        model = _WaveNet.init_from_config(config)
        assert model._net.is_slimmable() is True

    def test_slimmable_set_slimming_forward_at_different_ratios(self):
        """Forward runs at various set_slimming values (0.25, 0.5, 1.0)."""
        config = self._slimmable_config()
        model = _WaveNet.init_from_config(config)
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)
        model.eval()
        for ratio in (0.25, 0.5, 1.0):
            model._net.set_slimming(ratio)
            y = model(x)
            assert y.shape == x.shape
        model._net.set_slimming(1.0)

    def test_slimmable_context_adjust_to_random_restores(self):
        """context_adjust_to_random restores full width after exit."""
        config = self._slimmable_config()
        model = _WaveNet.init_from_config(config)
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)
        model.eval()
        y_full = model(x)
        with model._net.context_adjust_to_random():
            _ = model(x)
        # After context exit, should be back to full
        y_after = model(x)
        assert _torch.allclose(y_full, y_after)

    def test_slimmable_weight_import_export_roundtrip(self):
        """Weight export/import roundtrip preserves full-width behavior."""
        config = self._slimmable_config()
        model_1 = _WaveNet.init_from_config(config)
        model_2 = _WaveNet.init_from_config(config)
        rf = model_1.receptive_field
        x = _torch.randn(2, rf + 23)
        model_1.eval()
        model_2.eval()
        y1 = model_1(x)
        y2_before = model_2(x)
        model_2._net.import_weights(
            _torch.from_numpy(model_1._export_weights().astype(_np.float32)), 0
        )
        y2_after = model_2(x)
        assert not _torch.allclose(y2_before, y1)
        assert _torch.allclose(y2_after, y1)

    def test_slimmable_with_layer1x1_builds_and_forward(self):
        """Slimmable with layer1x1 (bottleneck == channels) builds and runs."""
        config = self._slimmable_config(
            channels=4,
            bottleneck=4,
            layer_1x1_config={"active": True, "groups": 1},
        )
        model = _WaveNet.init_from_config(config)
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)
        model.eval()
        for ratio in (0.5, 1.0):
            model._net.set_slimming(ratio)
            y = model(x)
            assert y.shape == x.shape

    def test_slimmable_with_pairing_activation_builds_and_forward(self):
        """Slimmable with PairMultiply activation builds and runs."""
        config = self._slimmable_config(
            channels=4,
            activation={
                "name": "PairMultiply",
                "primary": "Tanh",
                "secondary": "Sigmoid",
            },
        )
        model = _WaveNet.init_from_config(config)
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)
        model.eval()
        for ratio in (0.5, 1.0):
            model._net.set_slimming(ratio)
            y = model(x)
            assert y.shape == x.shape

    def test_slimmable_export_config_includes_slimmable(self):
        """Exported config includes slimmable when model is slimmable."""
        config = self._slimmable_config()
        model = _WaveNet.init_from_config(config)
        exported = model._export_config()
        assert "slimmable" in exported["layers"][0]
        assert exported["layers"][0]["slimmable"] == {
            "method": "slice_channels_uniform",
            "kwargs": {},
        }

    def test_slimmable_layer1x1_groups_raises(self):
        """Slimmable + layer1x1 groups != 1 raises NotImplementedError."""
        config = self._slimmable_config()
        config["layers_configs"][0]["layer_1x1_config"] = {
            "active": True,
            "groups": 2,
        }
        with _pytest.raises(NotImplementedError, match="layer 1x1 groups"):
            _WaveNet.init_from_config(config)

    # --- allowed_channels tests ---

    def _allowed_channels_config(
        self,
        channels: int = 12,
        allowed: tuple[int, ...] = (3, 12),
        layer1x1_active: bool = True,
        activation: str | dict = "LeakyReLU",
    ) -> dict:
        """Config with allowed_channels. Single layer array, no head1x1/FiLM."""
        layer = {
            "input_size": 1,
            "condition_size": 1,
            "head_size": 1,
            "channels": channels,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": activation,
            "head_bias": True,
            "slimmable": {
                "method": "slice_channels_uniform",
                "kwargs": {"allowed_channels": list(allowed)},
            },
        }
        if layer1x1_active:
            layer["layer_1x1_config"] = {"active": True, "groups": 1}
        return {"layers_configs": [layer], "head_scale": 1.0}

    def _ratio_to_channels(self, ratio: float, allowed: tuple[int, ...]) -> int:
        """Same logic as slimmable_conv._ratio_to_channels."""
        n = len(allowed)
        i = min(int(_np.floor(ratio * n)), n - 1)
        return allowed[max(0, i)]

    def _collect_slimmable_convs(self, model: _WaveNet):
        """Collect all SlimmableConv1dBase modules (excluding Head1x1)."""
        convs = []
        for m in model._net.modules():
            if isinstance(m, _SlimmableConv1dBase):
                # Skip Head1x1 (not implemented for slimmable)
                if "Head1x1" in type(m).__name__:
                    continue
                convs.append(m)
        return convs

    def test_allowed_channels_adjusted_weight_bias_shapes(self):
        """_get_adjusted_weight_and_bias returns expected shapes at each slimming ratio."""
        config = self._allowed_channels_config(channels=12, allowed=(3, 12))
        model = _WaveNet.init_from_config(config)
        model.eval()
        net = model._net
        convs = self._collect_slimmable_convs(model)

        for ratio in (0.0, 1.0):
            net.set_slimming(ratio)
            ch = self._ratio_to_channels(ratio, (3, 12))

            for conv in convs:
                w, b = conv._get_adjusted_weight_and_bias()
                # Conv1d weight: (out_ch, in_ch, kernel)
                assert w.ndim == 3
                out_ch, in_ch, k = w.shape
                assert out_ch >= 1 and in_ch >= 1 and k >= 1
                # No dimension exceeds full size
                assert out_ch <= conv.out_channels
                assert in_ch <= conv.in_channels
                assert k == conv.weight.shape[2]
                if b is not None:
                    assert b.ndim == 1
                    assert b.shape[0] == out_ch
                # Bias should match output channels
                assert b is None or b.numel() == out_ch

    def test_allowed_channels_forward_passes(self):
        """Forward runs at all slimming ratios for allowed_channels config."""
        config = self._allowed_channels_config(channels=12, allowed=(3, 12))
        model = _WaveNet.init_from_config(config)
        model.eval()
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)

        for ratio in (0.0, 0.5, 1.0):
            model._net.set_slimming(ratio)
            y = model(x)
            assert y.shape == x.shape
        model._net.set_slimming(1.0)

    def test_allowed_channels_with_pairing_activation_forward(self):
        """allowed_channels + PairMultiply forward at different ratios."""
        config = self._allowed_channels_config(
            channels=12,
            allowed=(3, 12),
            activation={
                "name": "PairMultiply",
                "primary": "Tanh",
                "secondary": "Sigmoid",
            },
        )
        model = _WaveNet.init_from_config(config)
        model.eval()
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)

        for ratio in (0.0, 0.5, 1.0):
            model._net.set_slimming(ratio)
            y = model(x)
            assert y.shape == x.shape
        model._net.set_slimming(1.0)

    # --- init_strategy smallest_and_zeros tests ---

    def _init_strategy_config(
        self,
        channels: int = 12,
        allowed: tuple[int, ...] = (3, 12),
        init_strategy: str = "smallest_and_zeros",
    ) -> dict:
        """Config with allowed_channels and init_strategy."""
        layer = {
            "input_size": 1,
            "condition_size": 1,
            "head_size": 1,
            "channels": channels,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "LeakyReLU",
            "head_bias": True,
            "slimmable": {
                "method": "slice_channels_uniform",
                "kwargs": {
                    "allowed_channels": list(allowed),
                    "init_strategy": init_strategy,
                },
            },
        }
        layer["layer_1x1_config"] = {"active": True, "groups": 1}
        return {"layers_configs": [layer], "head_scale": 1.0}

    def test_init_strategy_smallest_and_zeros_weight_bias_layout(self):
        """With init_strategy=smallest_and_zeros, smallest slice has standard init, rest is zero."""
        config = self._init_strategy_config(channels=12, allowed=(3, 12))
        model = _WaveNet.init_from_config(config)
        convs = self._collect_slimmable_convs(model)

        for conv in convs:
            min_in = conv._allowed_in_channels[0]
            min_out = conv._allowed_out_channels[0]
            w = conv.weight
            # Smallest slice should have non-zero init (standard Kaiming/etc)
            small_slice = w[:min_out, :min_in, :]
            assert small_slice.numel() > 0
            assert small_slice.abs().sum().item() > 0.0
            # Extra output channels should be zero
            if min_out < w.shape[0]:
                assert w[min_out:, :, :].abs().sum().item() == 0.0
            # Extra input channels should be zero
            if min_in < w.shape[1]:
                assert w[:, min_in:, :].abs().sum().item() == 0.0
            # Bias: smallest slice non-zero (if bias exists), rest zero
            if conv.bias is not None:
                assert conv.bias[:min_out].abs().sum().item() > 0.0
                if min_out < conv.bias.shape[0]:
                    assert conv.bias[min_out:].abs().sum().item() == 0.0

    def test_init_strategy_smallest_and_zeros_forward_passes(self):
        """Model with init_strategy=smallest_and_zeros builds and forward runs at all ratios."""
        config = self._init_strategy_config(channels=12, allowed=(3, 12))
        model = _WaveNet.init_from_config(config)
        model.eval()
        rf = model.receptive_field
        x = _torch.randn(2, rf + 16)

        for ratio in (0.0, 0.5, 1.0):
            model._net.set_slimming(ratio)
            y = model(x)
            assert y.shape == x.shape
        model._net.set_slimming(1.0)

    def test_init_strategy_smallest_and_zeros_direct_layer(self):
        """Direct SlimmableConv1dBase with init_strategy has correct weight/bias layout."""
        layer = _slimmable_class_set.LayerConv(
            in_channels=12,
            out_channels=24,
            kernel_size=3,
            allowed_in_channels=(3, 12),
            allowed_out_channels=(6, 24),
            output_paired=True,
            init_strategy="smallest_and_zeros",
        )
        min_in, min_out = 3, 6
        w = layer.weight
        assert w[:min_out, :min_in, :].abs().sum().item() > 0.0
        assert w[min_out:, :, :].abs().sum().item() == 0.0
        assert w[:, min_in:, :].abs().sum().item() == 0.0
        if layer.bias is not None:
            assert layer.bias[:min_out].abs().sum().item() > 0.0
            assert layer.bias[min_out:].abs().sum().item() == 0.0

    def test_init_strategy_none_uses_default_init(self):
        """With init_strategy absent, model uses default PyTorch init (all params non-zero)."""
        config = self._allowed_channels_config(channels=12, allowed=(3, 12))
        # No init_strategy in kwargs
        model = _WaveNet.init_from_config(config)
        convs = self._collect_slimmable_convs(model)
        for conv in convs:
            assert conv.weight.abs().sum().item() > 0.0
            if conv.bias is not None:
                assert conv.bias.abs().sum().item() > 0.0

    # --- Boosting tests ---

    def test_boosting_same_weight_bias_values_as_non_boosting(self):
        """
        Weight and bias tensors from _get_adjusted_weight_and_bias are identical
        regardless of whether _boosting is True or False.
        """
        allowed = (3, 12)
        slimming_ratio = 1.0  # Uses 12 channels, previous would be 3
        for layer_cls, layer_kwargs in [
            (
                _slimmable_class_set.LayerConv,
                {
                    "in_channels": 12,
                    "out_channels": 24,
                    "kernel_size": 3,
                    "allowed_in_channels": allowed,
                    "allowed_out_channels": (6, 24),
                    "output_paired": True,
                },
            ),
            (
                _slimmable_class_set.InputMixer,
                {
                    "in_channels": 1,
                    "out_channels": 24,
                    "kernel_size": 1,
                    "allowed_in_channels": (1,),
                    "allowed_out_channels": (6, 24),
                    "output_paired": True,
                    "bias": False,
                },
            ),
            (
                _slimmable_class_set.Layer1x1,
                {
                    "in_channels": 12,
                    "out_channels": 12,
                    "kernel_size": 1,
                    "allowed_in_channels": allowed,
                    "allowed_out_channels": allowed,
                },
            ),
            (
                _slimmable_class_set.HeadRechannel,
                {
                    "in_channels": 12,
                    "out_channels": 1,
                    "kernel_size": 1,
                    "allowed_in_channels": allowed,
                    "allowed_out_channels": (1,),
                    "is_last": True,
                },
            ),
        ]:
            layer_no_boost = layer_cls(boosting=False, **layer_kwargs)
            layer_boost = layer_cls(boosting=True, **layer_kwargs)
            with _torch.no_grad():
                layer_boost.weight.copy_(layer_no_boost.weight)
                if layer_no_boost.bias is not None:
                    layer_boost.bias.copy_(layer_no_boost.bias)
            layer_no_boost.set_slimming(slimming_ratio)
            layer_boost.set_slimming(slimming_ratio)
            w_no, b_no = layer_no_boost._get_adjusted_weight_and_bias()
            w_yes, b_yes = layer_boost._get_adjusted_weight_and_bias()
            assert _torch.allclose(
                w_no, w_yes
            ), f"{layer_cls.__name__}: weight mismatch"
            if b_no is not None and b_yes is not None:
                assert _torch.allclose(
                    b_no, b_yes
                ), f"{layer_cls.__name__}: bias mismatch"

    def test_boosting_grads_to_previous_entries_are_zero(self):
        """
        When backpropagating with _boosting=True, gradients to the 'previous'
        (next-smallest channel count) entries in weight and bias are zero.
        """
        allowed = (3, 12)
        slimming_ratio = 1.0  # Current=12, previous=3
        layer = _slimmable_class_set.LayerConv(
            in_channels=12,
            out_channels=24,
            kernel_size=3,
            allowed_in_channels=allowed,
            allowed_out_channels=(6, 24),
            output_paired=True,
            boosting=True,
        )
        assert isinstance(layer, _Slimmable)
        layer.set_slimming(slimming_ratio)
        # Previous slice: out_prev=6, in_prev=3
        out_prev, in_prev = 6, 3
        x = _torch.randn(2, 12, 20, requires_grad=True)
        w, b = layer._get_adjusted_weight_and_bias()
        out = _torch.nn.functional.conv1d(x, w, b, stride=1, padding=layer.padding)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        grad_prev = layer.weight.grad[:out_prev, :in_prev, :]
        assert _torch.allclose(
            grad_prev, _torch.zeros_like(grad_prev)
        ), "Gradients to previous weight entries should be zero"
        if layer.bias is not None and layer.bias.requires_grad:
            assert layer.bias.grad is not None
            grad_bias_prev = layer.bias.grad[:out_prev]
            assert _torch.allclose(
                grad_bias_prev, _torch.zeros_like(grad_bias_prev)
            ), "Gradients to previous bias entries should be zero"


if __name__ == "__main__":
    _pytest.main()
