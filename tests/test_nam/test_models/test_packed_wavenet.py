import json as _json

import pytest as _pytest
import torch as _torch

from nam import data as _data
from nam.models.wavenet import PackedWaveNet as _PackedWaveNet
from nam.models.wavenet import WaveNet as _WaveNet
from nam.models.wavenet._packed_conv import PackedConv1dBase as _PackedConv1dBase

_DEFAULT_HEAD_SCALE = 0.25


def _wavenet_config(channels: int, *, dilations=None, activation="Tanh"):
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": channels,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "kernel_size": [2, 3],
                "dilations": [1, 2] if dilations is None else dilations,
                "activation": activation,
            }
        ],
        "head": None,
        "head_scale": _DEFAULT_HEAD_SCALE,
    }


def _packed_config():
    return {
        "submodels": [
            {"name": "small", "config": _wavenet_config(3)},
            {"name": "large", "config": _wavenet_config(8)},
        ],
        "export": {"container_max_values": "uniform"},
    }


def _two_array_wavenet_config(channels_0: int, channels_1: int):
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": channels_0,
                "head": {
                    "out_channels": channels_1,
                    "kernel_size": 1,
                    "bias": False,
                },
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            },
            {
                "input_size": channels_0,
                "condition_size": 1,
                "channels": channels_1,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            },
        ],
        "head": None,
        "head_scale": _DEFAULT_HEAD_SCALE,
    }


def test_packed_wavenet_forward_shape():
    model = _PackedWaveNet.init_from_config(_packed_config())
    x = _torch.randn(4, model.receptive_field + 11)
    y = model(x)
    assert y.shape == (4, 2, x.shape[-1])


def test_packed_wavenet_mps_fallback_stitches_time_axis():
    """Packed fallback must stitch chunks along time, not submodels."""
    model = _PackedWaveNet.init_from_config(_packed_config())
    model.eval()
    batch_size = 1
    input_length = 65_536 + model.receptive_field + 1
    x = _torch.linspace(-1.0, 1.0, input_length).repeat(batch_size, 1)

    with _torch.no_grad():
        y_reference = model(x, pad_start=False)
        model._mps_65536_fallback = True
        y_fallback = model(x, pad_start=False)

    expected_length = input_length - model.receptive_field + 1
    assert y_fallback.shape == y_reference.shape
    assert y_fallback.shape == (batch_size, model.num_submodels, expected_length)
    assert _torch.allclose(y_fallback, y_reference, atol=1.0e-6)


def test_packed_wavenet_supports_compatible_multiple_layer_arrays():
    model = _PackedWaveNet.init_from_config(
        {
            "submodels": [
                {"name": "small", "config": _two_array_wavenet_config(3, 2)},
                {"name": "large", "config": _two_array_wavenet_config(8, 4)},
            ]
        }
    )
    x = _torch.randn(4, model.receptive_field + 11)
    assert model(x).shape == (4, 2, x.shape[-1])


def test_packed_wavenet_rejects_mismatched_head_path_between_layer_arrays():
    config = {
        "submodels": [
            {"name": "small", "config": _two_array_wavenet_config(3, 2)},
            {"name": "large", "config": _two_array_wavenet_config(8, 4)},
        ]
    }
    config["submodels"][1]["config"]["layers_configs"][0]["head"]["out_channels"] = 3
    with _pytest.raises(ValueError, match="head channels"):
        _PackedWaveNet.init_from_config(config)


@_pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda c: c["submodels"][1]["config"]["layers_configs"].append(
                c["submodels"][1]["config"]["layers_configs"][0].copy()
            ),
            "same number of layer arrays",
        ),
        (
            lambda c: c["submodels"][1]["config"]["layers_configs"][0].update(
                {"dilations": [1, 4]}
            ),
            "dilations",
        ),
        (
            lambda c: c["submodels"][1]["config"]["layers_configs"][0].update(
                {"activation": "ReLU"}
            ),
            "activations",
        ),
        (
            lambda c: c["submodels"][1]["config"].update({"condition_dsp": {}}),
            "condition_dsp",
        ),
        (
            lambda c: c["submodels"][1]["config"]["layers_configs"][0].update(
                {"groups_input": 2}
            ),
            "grouped",
        ),
        (
            lambda c: c["submodels"][1]["config"]["layers_configs"][0].update(
                {
                    "activation": {
                        "name": "PairMultiply",
                        "primary": "Tanh",
                        "secondary": "Sigmoid",
                    }
                }
            ),
            "paired/gated",
        ),
    ),
)
def test_packed_wavenet_validation_rejects_incompatible_configs(mutate, match):
    config = _packed_config()
    mutate(config)
    with _pytest.raises((ValueError, NotImplementedError), match=match):
        _PackedWaveNet.init_from_config(config)


def test_packed_conv_masks_invalid_weights_and_gradients():
    model = _PackedWaveNet.init_from_config(_packed_config())
    conv = next(m for m in model.modules() if isinstance(m, _PackedConv1dBase))
    with _torch.no_grad():
        conv.weight.fill_(1.0)
    model.apply_mask()
    conv._assert_masked()
    x = _torch.randn(2, model.receptive_field + 8)
    loss = model(x).sum()
    loss.backward()
    invalid_grad = conv.weight.grad * (1.0 - conv._weight_mask)
    assert _torch.allclose(invalid_grad, _torch.zeros_like(invalid_grad))


def test_packed_forward_matches_imported_ordinary_submodels():
    ordinary = [
        _WaveNet.init_from_config(_wavenet_config(3)),
        _WaveNet.init_from_config(_wavenet_config(8)),
    ]
    packed = _PackedWaveNet.init_from_config(_packed_config())
    for i, submodel in enumerate(ordinary):
        packed.import_submodel(i, submodel)

    x = _torch.randn(3, packed.receptive_field + 13)
    y_packed = packed(x)
    for i, submodel in enumerate(ordinary):
        assert _torch.allclose(y_packed[:, i, :], submodel(x), atol=1.0e-6)


def test_packed_extraction_matches_packed_output():
    packed = _PackedWaveNet.init_from_config(_packed_config())
    x = _torch.randn(3, packed.receptive_field + 13)
    y_packed = packed(x)
    for i in range(packed.num_submodels):
        extracted = packed.extract_submodel(i)
        assert _torch.allclose(extracted(x), y_packed[:, i, :], atol=1.0e-6)


def test_packed_export_writes_slimmable_container(tmp_path):
    model = _PackedWaveNet.init_from_config({**_packed_config(), "sample_rate": 48_000})
    container = model.export_container(tmp_path)
    with open(tmp_path / "model.nam", "r") as fp:
        from_disk = _json.load(fp)
    assert from_disk == container
    _assert_container_contains_two_wavenets(container)
    highest_quality = max(
        container["config"]["submodels"], key=lambda entry: entry["max_value"]
    )["model"]
    assert container["metadata"]["loudness"] == highest_quality["metadata"]["loudness"]
    assert container["metadata"]["gain"] == highest_quality["metadata"]["gain"]


def test_packed_export_refreshes_loudness_after_head_scale_compensation(tmp_path):
    """
    When an export hook scales `head_scale` (e.g. the dataset normalization
    handshake), the exported `metadata.loudness` must describe the *compensated*
    model that will be loaded at inference, not the pre-compensation snapshot.

    Output of WaveNet (no top-level head) and SlimmableContainer is linear in
    `head_scale`, so loudness moves by `20 * log10(scale)` exactly.
    """
    import math as _math

    model = _PackedWaveNet.init_from_config({**_packed_config(), "sample_rate": 48_000})
    pre_container = model.export_container(tmp_path)
    pre_submodel_loudnesses = [
        entry["model"]["metadata"]["loudness"]
        for entry in pre_container["config"]["submodels"]
    ]
    pre_highest_quality_loudness = max(
        pre_container["config"]["submodels"],
        key=lambda entry: entry["max_value"],
    )["model"]["metadata"]["loudness"]

    scale = 2.0
    model.export_model_dict_post_hooks.append(_data.Dataset._ScaleOutputHook(scale=scale))
    post_container = model.export_container(tmp_path)

    offset_db = 20.0 * _math.log10(scale)
    assert post_container["metadata"]["loudness"] == _pytest.approx(
        pre_highest_quality_loudness + offset_db, abs=1e-3
    )
    for entry, pre_loudness in zip(
        post_container["config"]["submodels"], pre_submodel_loudnesses
    ):
        assert entry["model"]["metadata"]["loudness"] == _pytest.approx(
            pre_loudness + offset_db, abs=1e-3
        )
        # head_scale was actually compensated on disk
        assert entry["model"]["config"]["head_scale"] == _pytest.approx(
            _DEFAULT_HEAD_SCALE * scale
        )


def test_packed_export_applies_model_dict_post_hooks(tmp_path):
    model = _PackedWaveNet.init_from_config({**_packed_config(), "sample_rate": 48_000})
    model.export_model_dict_post_hooks.append(_data.Dataset._ScaleOutputHook(scale=2.0))

    container = model.export_container(tmp_path)
    with open(tmp_path / "model.nam", "r") as fp:
        from_disk = _json.load(fp)
    submodels = [item["model"] for item in container["config"]["submodels"]]

    assert from_disk == container
    assert [model["config"]["head_scale"] for model in submodels] == [
        _pytest.approx(0.5),
        _pytest.approx(0.5),
    ]
    assert [model["weights"][-1] for model in submodels] == [
        _pytest.approx(0.5),
        _pytest.approx(0.5),
    ]


def test_packed_export_loads_lightning_checkpoint(tmp_path):
    from nam.train.lightning_module import (
        PackedLightningModule as _PackedLightningModule,
    )

    packed = _PackedWaveNet.init_from_config(
        {**_packed_config(), "sample_rate": 48_000}
    )
    checkpoint_path = tmp_path / "lightning.ckpt"
    _torch.save(
        {
            "state_dict": _PackedLightningModule(packed).state_dict(),
            "sample_rate": 48_000,
        },
        checkpoint_path,
    )

    exporter = _PackedWaveNet.init_from_config(
        {**_packed_config(), "sample_rate": 48_000}
    )
    container = exporter.export_container(
        tmp_path,
        checkpoint_paths_by_submodel=[checkpoint_path, checkpoint_path],
    )

    _assert_container_contains_two_wavenets(container)


def test_packed_export_loads_raw_packed_state_dict_checkpoint(tmp_path):
    packed = _PackedWaveNet.init_from_config(
        {**_packed_config(), "sample_rate": 48_000}
    )
    checkpoint_path = tmp_path / "raw.ckpt"
    _torch.save(
        {"state_dict": packed.state_dict(), "sample_rate": 48_000},
        checkpoint_path,
    )

    exporter = _PackedWaveNet.init_from_config(
        {**_packed_config(), "sample_rate": 48_000}
    )
    container = exporter.export_container(
        tmp_path,
        checkpoint_paths_by_submodel=[checkpoint_path, checkpoint_path],
    )

    _assert_container_contains_two_wavenets(container)


def test_packed_export_rejects_corrupt_checkpoint(tmp_path):
    packed = _PackedWaveNet.init_from_config(
        {**_packed_config(), "sample_rate": 48_000}
    )
    state_dict = dict(packed.state_dict())
    key, value = state_dict.popitem()
    state_dict[f"corrupt.{key}"] = value
    checkpoint_path = tmp_path / "corrupt.ckpt"
    _torch.save(
        {"state_dict": state_dict, "sample_rate": 48_000},
        checkpoint_path,
    )

    exporter = _PackedWaveNet.init_from_config(
        {**_packed_config(), "sample_rate": 48_000}
    )
    with _pytest.raises(RuntimeError) as exc_info:
        exporter.export_container(
            tmp_path,
            checkpoint_paths_by_submodel=[checkpoint_path, checkpoint_path],
        )
    message = str(exc_info.value)
    assert str(checkpoint_path) in message
    assert "missing keys" in message
    assert "unexpected keys" in message


def _assert_container_contains_two_wavenets(container):
    assert container["architecture"] == "SlimmableContainer"
    assert container["weights"] == []
    assert len(container["config"]["submodels"]) == 2
    assert container["config"]["submodels"][-1]["max_value"] == 1.0
    assert [
        item["model"]["architecture"] for item in container["config"]["submodels"]
    ] == ["WaveNet", "WaveNet"]


def test_packed_export_uses_checkpoint_selected_for_each_submodel(tmp_path):
    def set_parameter_values(model, value):
        with _torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(value)
        model.apply_mask()

    def save_lightning_checkpoint(model, path):
        from nam.train.lightning_module import (
            PackedLightningModule as _PackedLightningModule,
        )

        _torch.save(
            {
                "state_dict": _PackedLightningModule(model).state_dict(),
                "sample_rate": model.sample_rate,
            },
            path,
        )

    config = {**_packed_config(), "sample_rate": 48_000}
    base = _PackedWaveNet.init_from_config(config)
    source_a = _PackedWaveNet.init_from_config(config)
    source_b = _PackedWaveNet.init_from_config(config)
    set_parameter_values(base, -3.0)
    set_parameter_values(source_a, 1.0)
    set_parameter_values(source_b, 2.0)

    checkpoint_a = tmp_path / "source_a.ckpt"
    checkpoint_b = tmp_path / "source_b.ckpt"
    save_lightning_checkpoint(source_a, checkpoint_a)
    save_lightning_checkpoint(source_b, checkpoint_b)

    container = base.export_container(
        tmp_path,
        checkpoint_paths_by_submodel=[str(checkpoint_a), str(checkpoint_b)],
    )

    exported_submodels = [item["model"] for item in container["config"]["submodels"]]
    expected_submodel_0 = source_a.extract_submodel(0)._get_export_dict()
    expected_submodel_1 = source_b.extract_submodel(1)._get_export_dict()

    assert container["architecture"] == "SlimmableContainer"
    assert [model["architecture"] for model in exported_submodels] == [
        "WaveNet",
        "WaveNet",
    ]
    assert exported_submodels[0]["weights"] == _pytest.approx(
        expected_submodel_0["weights"]
    )
    assert exported_submodels[1]["weights"] == _pytest.approx(
        expected_submodel_1["weights"]
    )
