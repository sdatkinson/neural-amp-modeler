import json as _json
from collections.abc import Mapping as _Mapping
from functools import lru_cache as _lru_cache
from typing import Optional as _Optional
from typing import cast as _cast

import pytest as _pytest
import torch as _torch
import torch.nn as _nn

from nam.models.parametric import Hypernetwork as _Hypernetwork
from nam.models.wavenet._wavenet import WaveNet as _WaveNet

# The model wrapper (Task 5) will pass this so dilated convs stay shared. The generic
# Hypernetwork is net-agnostic and targets ALL params by default.
_CHEAP_SUBSET = {"exclude_suffixes": ["_conv.weight"]}


@_lru_cache(maxsize=1)
def _channels_8_named_shapes() -> dict[str, _torch.Size]:
    with open("nam/train/_resources/config_model_packed.json") as fp:
        cfg = _json.load(fp)
    subconfig = next(
        submodel
        for submodel in cfg["net"]["config"]["submodels"]
        if submodel["name"] == "channels_8"
    )["config"]
    template = _WaveNet.init_from_config(subconfig)
    return {name: parameter.shape for name, parameter in template.named_parameters()}


def _resolve_output_widths(
    named_shapes: _Mapping[str, _torch.Size],
    target_names: tuple[str, ...],
    *,
    mode: str,
    rank: _Optional[int] = None,
) -> dict[str, int]:
    widths = {}
    for name in target_names:
        shape = named_shapes[name]
        out_features = int(shape[0]) if len(shape) > 0 else 0
        rest_features = int(shape[1:].numel()) if len(shape) > 1 else 1
        if mode == "low_rank" and out_features > 1 and rest_features > 1:
            assert rank is not None
            widths[name] = min(rank, out_features, rest_features) * (
                out_features + rest_features
            )
        else:
            widths[name] = shape.numel()
    return widths


def test_default_selector_targets_all_parameters():
    # Net-agnostic default: no selector -> every parameter is a target (incl. dilated convs).
    named_shapes = _channels_8_named_shapes()

    hypernet = _Hypernetwork.from_config(input_dim=4, named_shapes=named_shapes)

    assert hypernet.target_names == tuple(named_shapes)
    assert len(hypernet.target_names) == 118
    assert sum(named_shapes[name].numel() for name in hypernet.target_names) == 12145


def test_explicit_selector_excludes_conv_weights():
    named_shapes = _channels_8_named_shapes()

    hypernet = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config={"selector": _CHEAP_SUBSET}
    )

    expected_target_names = tuple(
        name for name in named_shapes if not name.endswith("_conv.weight")
    )
    excluded_names = tuple(
        name for name in named_shapes if name.endswith("_conv.weight")
    )
    assert hypernet.target_names == expected_target_names
    assert len(hypernet.target_names) == 95
    assert sum(named_shapes[name].numel() for name in hypernet.target_names) == 2161
    assert len(excluded_names) == 23
    assert sum(named_shapes[name].numel() for name in excluded_names) == 9984


def test_default_mode_is_low_rank_not_full():
    # "Start small": the default must NOT be full-rank generation.
    named_shapes = _channels_8_named_shapes()

    hypernet = _Hypernetwork.from_config(input_dim=4, named_shapes=named_shapes)

    assert hypernet.config["mode"] == "low_rank"
    assert hypernet.config["rank"] == 2
    assert hypernet.config["hidden_sizes"] == []  # linear trunk by default


def test_generate_preserves_target_shapes_for_vector_and_batch_inputs():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config={"selector": _CHEAP_SUBSET}
    )

    vector_params = _torch.randn(4)
    batch_params = _torch.randn(3, 4)

    vector_deltas = hypernet.generate(vector_params)
    batch_deltas = hypernet.generate(batch_params)

    assert set(vector_deltas) == set(hypernet.target_names)
    assert set(batch_deltas) == set(hypernet.target_names)
    for name in hypernet.target_names:
        assert vector_deltas[name].shape == named_shapes[name]
        assert batch_deltas[name].shape == (3, *named_shapes[name])


def test_zero_init_produces_exactly_zero_deltas():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config={"selector": _CHEAP_SUBSET}
    )

    vector_deltas = hypernet.generate(_torch.randn(4))
    batch_deltas = hypernet.generate(_torch.randn(2, 4))

    for deltas in (vector_deltas, batch_deltas):
        for tensor in deltas.values():
            assert _torch.count_nonzero(tensor) == 0


def test_low_rank_seed_is_independent_of_global_rng():
    # The default (low_rank, empty trunk) hypernet has no trunk params; its only state is the
    # zeroed readout weight plus the locally-seeded anchor buffer. Because that seed uses a local
    # generator, construction is fully deterministic regardless of the ambient global RNG --
    # building under two different global seeds yields identical parameters.
    named_shapes = _channels_8_named_shapes()
    config = {"mode": "low_rank", "rank": 2, "selector": _CHEAP_SUBSET}

    _torch.manual_seed(1)
    first = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config=config
    )
    _torch.manual_seed(99999)
    second = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config=config
    )

    first_state = first.state_dict()
    second_state = second.state_dict()
    assert set(first_state) == set(second_state)
    assert first_state  # not empty: there is a readout to compare
    for key, value in first_state.items():
        assert _torch.equal(value, second_state[key]), key
    # And the locally-seeded anchor buffer is genuinely nonzero (the symmetry-breaking happened).
    assert _torch.count_nonzero(first.state_dict()["_low_rank_anchor"]) > 0


def test_empty_trunk_construction_does_not_advance_global_rng():
    hypernet_kwargs: dict[str, object] = dict(
        input_dim=3,
        named_shapes={"weight": _torch.Size((4, 4))},
        target_names=("weight",),
        mode="low_rank",
        rank=2,
    )

    _torch.manual_seed(1234)
    expected = _torch.rand(5)

    _torch.manual_seed(1234)
    _Hypernetwork(
        input_dim=_cast(int, hypernet_kwargs["input_dim"]),
        named_shapes=_cast(dict[str, _torch.Size], hypernet_kwargs["named_shapes"]),
        target_names=_cast(tuple[str, ...], hypernet_kwargs["target_names"]),
        mode=_cast(str, hypernet_kwargs["mode"]),
        rank=_cast(int, hypernet_kwargs["rank"]),
    )
    observed = _torch.rand(5)

    assert _torch.equal(observed, expected)


def test_low_rank_mode_shapes_and_rank_bound():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4,
        named_shapes=named_shapes,
        config={
            "mode": "low_rank",
            "rank": 2,
            "hidden_sizes": (5,),
            "selector": _CHEAP_SUBSET,
        },
    )

    widths = _resolve_output_widths(
        named_shapes, hypernet.target_names, mode="low_rank", rank=2
    )
    layer_name = "_layer_arrays.0._layers.0._layer1x1.weight"
    bias_name = "_layer_arrays.0._layers.0._conv.bias"
    bias_offset = sum(
        widths[name]
        for name in hypernet.target_names[: hypernet.target_names.index(bias_name)]
    )
    layer_offset = sum(
        widths[name]
        for name in hypernet.target_names[: hypernet.target_names.index(layer_name)]
    )

    with _torch.no_grad():
        hypernet._final.bias.zero_()
        hypernet._final.bias[bias_offset : bias_offset + widths[bias_name]] = (
            _torch.arange(1, widths[bias_name] + 1, dtype=hypernet._final.bias.dtype)
        )
        hypernet._final.bias[layer_offset : layer_offset + widths[layer_name]] = (
            _torch.tensor(
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    2.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    1.0,
                    0.0,
                    2.0,
                ],
                dtype=hypernet._final.bias.dtype,
            )
        )

    deltas = hypernet.generate(_torch.zeros(4))

    assert deltas[layer_name].shape == named_shapes[layer_name]
    assert deltas[bias_name].shape == named_shapes[bias_name]
    assert _torch.equal(
        deltas[bias_name],
        _torch.arange(1, widths[bias_name] + 1, dtype=deltas[bias_name].dtype),
    )
    matrix = deltas[layer_name].reshape(named_shapes[layer_name][0], -1)
    assert _torch.linalg.matrix_rank(matrix) <= 2


def test_low_rank_zero_init_still_backprops_into_both_factor_slices():
    hypernet = _Hypernetwork(
        input_dim=3,
        named_shapes={"weight": _torch.Size((4, 4))},
        target_names=("weight",),
        hidden_sizes=(5,),
        mode="low_rank",
        rank=2,
    )

    deltas = hypernet.generate(_torch.randn(2, 3))
    assert _torch.count_nonzero(deltas["weight"]) == 0

    loss = (deltas["weight"] - 1.0).square().mean()
    loss.backward()

    assert hypernet._final.bias.grad is not None
    u_width = hypernet._target_layouts[0].u_width
    assert _torch.count_nonzero(hypernet._final.bias.grad[:u_width]) > 0
    assert _torch.count_nonzero(hypernet._final.bias.grad[u_width:]) > 0


def test_true_init_gradient_warmup():
    # Documents the one-step warmup: at true zero-init the readout gets a gradient but the
    # trunk does not (final weight is zero -> dL/dh == 0). Use a loss linear in the deltas so
    # dL/ddelta is nonzero even though every delta starts at exactly zero.
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4,
        named_shapes=named_shapes,
        config={
            "hidden_sizes": [8],
            "mode": "low_rank",
            "rank": 2,
            "selector": _CHEAP_SUBSET,
        },
    )

    deltas = hypernet.generate(_torch.randn(2, 4))
    loss = _torch.stack([delta.sum() for delta in deltas.values()]).sum()
    loss.backward()

    assert hypernet._final.weight.grad is not None
    assert _torch.count_nonzero(hypernet._final.weight.grad) > 0

    trunk = _cast(_nn.Sequential, hypernet._trunk)
    trunk_linear = next(m for m in trunk if isinstance(m, _nn.Linear))
    assert (
        trunk_linear.weight.grad is None
        or _torch.count_nonzero(trunk_linear.weight.grad) == 0
    )


@_pytest.mark.parametrize(
    "config, match",
    [
        ({"unknown": True}, "unexpected keys"),
        ({"selector": {"unknown": True}}, "selector config has unexpected keys"),
        ({"selector": {"include_names": ["not_real"]}}, "unknown parameter names"),
        ({"mode": "low_rank", "rank": 0}, "rank must be a positive integer"),
        ({"mode": "bogus"}, "Unsupported hypernetwork mode"),
        ({"activation": "NotAnActivation"}, "Invalid activation"),
    ],
)
def test_invalid_config_raises_clear_errors(config, match):
    named_shapes = _channels_8_named_shapes()

    with _pytest.raises(ValueError, match=match):
        _Hypernetwork.from_config(input_dim=4, named_shapes=named_shapes, config=config)


def test_param_count_positive_and_gradients_flow():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4,
        named_shapes=named_shapes,
        config={
            "hidden_sizes": (8,),
            "mode": "low_rank",
            "rank": 2,
            "selector": _CHEAP_SUBSET,
        },
    )

    assert hypernet.param_count() > 0

    with _torch.no_grad():
        hypernet._final.weight.normal_(mean=0.0, std=0.01)

    deltas = hypernet.generate(_torch.randn(2, 4))
    loss = _torch.stack([delta.square().mean() for delta in deltas.values()]).sum()
    loss.backward()

    nonzero_grad_names = {
        name
        for name, parameter in hypernet.named_parameters()
        if parameter.grad is not None and _torch.count_nonzero(parameter.grad) > 0
    }
    assert "_final.weight" in nonzero_grad_names
    assert any(name.startswith("_trunk.") for name in nonzero_grad_names)


@_pytest.mark.parametrize(
    ("mode", "rank"),
    (
        ("low_rank", 2),
        ("full", None),
    ),
)
def test_serialized_state_matches_state_dict_size(mode: str, rank: int | None):
    hypernet = _Hypernetwork(
        input_dim=3,
        named_shapes={"weight": _torch.Size((4, 4))},
        target_names=("weight",),
        mode=mode,
        rank=rank,
    )

    assert hypernet.state_count() == sum(
        tensor.numel() for tensor in hypernet.state_dict().values()
    )
    assert hypernet.export_state().numel() == hypernet.state_count()


def test_import_state_rejects_non_flat_tensor():
    hypernet = _Hypernetwork(
        input_dim=3,
        named_shapes={"weight": _torch.Size((4, 4))},
        target_names=("weight",),
        mode="low_rank",
        rank=2,
    )

    with _pytest.raises(ValueError, match="flat 1-D tensor"):
        hypernet.import_state(hypernet.export_state().reshape(-1, 1))


def test_config_round_trips_with_explicit_selector():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4,
        named_shapes=named_shapes,
        config={
            "mode": "low_rank",
            "rank": 2,
            "hidden_sizes": [8],
            "activation": "Tanh",
            "selector": _CHEAP_SUBSET,
        },
    )

    assert hypernet.config == {
        "hidden_sizes": [8],
        "activation": "Tanh",
        "mode": "low_rank",
        "rank": 2,
        "selector": {"exclude_suffixes": ["_conv.weight"]},
    }

    round_tripped = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config=hypernet.config
    )

    assert round_tripped.config == hypernet.config
    assert round_tripped.target_names == hypernet.target_names


def test_default_config_round_trips_without_selector_key():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(input_dim=4, named_shapes=named_shapes)

    assert hypernet.config == {
        "hidden_sizes": [],
        "activation": "ReLU",
        "mode": "low_rank",
        "rank": 2,
    }

    round_tripped = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config=hypernet.config
    )
    assert round_tripped.config == hypernet.config
    assert round_tripped.target_names == hypernet.target_names


def test_direct_subset_config_round_trips_via_synthesized_selector():
    named_shapes = {
        "a": _torch.Size((2, 2)),
        "b": _torch.Size((2, 2)),
        "c": _torch.Size((2, 2)),
    }
    hypernet = _Hypernetwork(
        input_dim=3,
        named_shapes=named_shapes,
        target_names=("a", "c"),
    )

    assert hypernet.config["selector"] == {"include_names": ["a", "c"]}

    round_tripped = _Hypernetwork.from_config(
        input_dim=3, named_shapes=named_shapes, config=hypernet.config
    )
    assert round_tripped.target_names == hypernet.target_names


def test_full_mode_config_omits_rank():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4,
        named_shapes=named_shapes,
        config={"mode": "full", "rank": 5, "selector": _CHEAP_SUBSET},
    )

    # rank is meaningless in full mode and must not be persisted.
    assert "rank" not in hypernet.config
    assert hypernet.config["mode"] == "full"


def test_config_activation_is_copied():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4,
        named_shapes=named_shapes,
        config={"activation": {"name": "ELU"}, "hidden_sizes": [4]},
    )

    returned = hypernet.config
    returned["activation"]["name"] = "MUTATED"

    assert hypernet.config["activation"] == {"name": "ELU"}


def test_generate_rejects_bad_input_shape():
    named_shapes = _channels_8_named_shapes()
    hypernet = _Hypernetwork.from_config(
        input_dim=4, named_shapes=named_shapes, config={"selector": _CHEAP_SUBSET}
    )

    with _pytest.raises(ValueError, match="trailing dimension 4"):
        hypernet.generate(_torch.zeros(2, 5))

    with _pytest.raises(ValueError, match="shape \\(E,\\) or \\(B, E\\)"):
        hypernet.generate(_torch.zeros(1, 2, 4))


def _grouped_hypernet() -> _Hypernetwork:
    hypernet = _Hypernetwork.from_config(
        input_dim=4,
        named_shapes=_channels_8_named_shapes(),
        config={
            "hidden_sizes": [8],
            "mode": "low_rank",
            "rank": 2,
            "selector": _CHEAP_SUBSET,
        },
    )
    with _torch.no_grad():
        hypernet._final.weight.normal_(0.0, 0.05)
        hypernet._final.bias.normal_(0.0, 0.05)
        _cast(_torch.Tensor, hypernet._low_rank_anchor).normal_(0.0, 0.05)
    return hypernet


def test_decode_groups_partition_the_targets_in_order():
    hypernet = _grouped_hypernet()

    grouped_names = [name for group in hypernet._decode_groups for name in group.names]
    assert sorted(grouped_names) == sorted(hypernet.target_names)
    assert len(grouped_names) == len(set(grouped_names))
    # Grouping is what makes the decode cheap; if every target landed in its own group the
    # batched ops would have degenerated back to per-target ones.
    assert len(hypernet._decode_groups) < len(hypernet.target_names)


def test_generate_groups_stack_matches_generate():
    hypernet = _grouped_hypernet()

    for params in (_torch.randn(4), _torch.randn(3, 4)):
        deltas = hypernet.generate(params)
        for group, stacked in hypernet.generate_groups(params):
            assert stacked.shape[group.stack_dim] == group.size
            for name, member in zip(group.names, stacked.unbind(group.stack_dim)):
                assert _torch.equal(member, deltas[name])


def test_decode_group_index_buffers_stay_out_of_the_serialized_state():
    hypernet = _grouped_hypernet()

    state_keys = set(hypernet.state_dict())
    assert "_low_rank_anchor" in state_keys
    assert not any(key.startswith("_group") for key in state_keys)
    assert hypernet.state_count() == hypernet.export_state().numel()


def test_mutating_the_anchor_still_changes_generated_deltas():
    # The grouped decode gathers the anchor per call rather than caching it in group form.
    hypernet = _grouped_hypernet()
    params = _torch.randn(4)
    before = hypernet.generate(params)

    with _torch.no_grad():
        _cast(_torch.Tensor, hypernet._low_rank_anchor).add_(0.125)
    after = hypernet.generate(params)

    low_rank = [
        layout.name for layout in hypernet._target_layouts if layout.is_low_rank
    ]
    assert low_rank
    assert all(not _torch.equal(before[name], after[name]) for name in low_rank)
