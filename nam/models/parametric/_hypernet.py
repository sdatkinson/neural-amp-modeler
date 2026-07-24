"""
Net-agnostic hypernetwork utilities for parametric NAM models.

A ``Hypernetwork`` maps an *encoded* parameter vector to additive **deltas** for a
configurable subset of some template network's weight tensors. It knows nothing about
WaveNet/LSTM internals -- it operates purely on the ``{name: shape}`` mapping passed in --
so the same class is reused across architectures. The net-specific knowledge (e.g. "share
the dilated convs") lives in the *caller*: the model wrapper (``HyperWaveNet``, Task 5) is
responsible for passing a ``selector`` that excludes the tensors it wants to keep shared.
Because of that, the generic default here is intentionally "target every parameter"; it is
NOT WaveNet-aware. Callers that want the cheap-subset behaviour must say so explicitly via
``selector={"exclude_suffixes": ["_conv.weight"]}``.

Defaults favour *starting small* (per project direction: scale capacity up only if
validation shows it's needed). The default mode is ``low_rank`` with a small rank and no
hidden trunk, so the generated deltas are low-rank and the param->delta map is simple. The
two scale-up levers are ``hidden_sizes`` (add param-space nonlinearity) and ``rank`` (raise
the per-tensor delta rank); ``mode="full"`` removes the rank constraint entirely.

Init contract: the readout is zero-initialized so every delta == 0 at construction (the
model equals the bare template for all params). For ``low_rank`` targets we also register a
tiny factor-anchor seed from a *local* generator: alternating rank components get a
nonzero U anchor or a nonzero V anchor, never both, so the anchored product is still
exactly zero while the first backward pass reaches both factor slices. That seed is
deterministic and independent of the ambient global ``torch`` RNG state. (The trunk's
``Linear`` layers still use standard PyTorch init, which draws from the global RNG like any
``nn.Module`` -- only the extra low-rank anchoring is made global-RNG-neutral.)
"""

import math as _math
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Optional as _Optional
from typing import cast as _cast

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

from .._activations import get_activation as _get_activation

_FULL = "full"
_LOW_RANK = "low_rank"
_VALID_MODES = (_FULL, _LOW_RANK)
_ALLOWED_CONFIG_KEYS = frozenset(
    {"activation", "hidden_sizes", "mode", "rank", "selector"}
)
_ALLOWED_SELECTOR_KEYS = frozenset(
    {"exclude_names", "exclude_suffixes", "include_names", "include_suffixes"}
)
# "Start small" defaults: low-rank deltas, smallest real rank, linear (no-hidden) trunk.
_DEFAULT_MODE = _LOW_RANK
_DEFAULT_RANK = 2
_DEFAULT_HIDDEN_SIZES: tuple[int, ...] = ()
_DEFAULT_ACTIVATION = "ReLU"
# Std of the local symmetry-breaking seed on the low-rank factor anchors.
_LOW_RANK_SEED_STD = 1.0e-2
# Fixed seed for the local generator so init is deterministic without touching global RNG.
_DEFAULT_LOW_RANK_SEED = 0


@_dataclass(frozen=True)
class _TargetLayout:
    name: str
    shape: _torch.Size
    output_width: int
    mode: str
    rank: _Optional[int] = None
    out_features: _Optional[int] = None
    rest_features: _Optional[int] = None

    @property
    def is_low_rank(self) -> bool:
        return self.mode == _LOW_RANK

    @property
    def u_width(self) -> int:
        if self.rank is None or self.out_features is None:
            raise RuntimeError(
                f"Low-rank target layout for {self.name!r} is incomplete"
            )
        return self.out_features * self.rank

    @property
    def v_width(self) -> int:
        if self.rank is None or self.rest_features is None:
            raise RuntimeError(
                f"Low-rank target layout for {self.name!r} is incomplete"
            )
        return self.rank * self.rest_features


@_dataclass(frozen=True)
class _DecodeGroup:
    """
    A set of targets that share an identical decode layout, decoded together.

    Every target used to be decoded on its own: slice it out of the readout output, reshape,
    add its anchor, multiply its two low-rank factors. On a small net that is a few hundred
    tiny CUDA kernels per step, each doing a handful of elements, and the launch cost swamps
    the arithmetic. Targets with the same shape can share all of that work instead -- one
    gather, one add, one batched matmul for the whole group -- which is what this describes.

    ``index``/``v_index`` name int64 buffers holding the readout columns this group's members
    occupy (``v_*`` is the low-rank V block; the plain fields are the U block, or the whole
    slot for a full-mode group). They are ``None`` when the members already sit in one
    contiguous run, in which case ``start``/``width`` name that run and a free view replaces
    the gather.
    """

    names: tuple[str, ...]
    shape: _torch.Size
    is_low_rank: bool
    index: _Optional[str]
    start: int
    width: int
    rank: int = 0
    out_features: int = 0
    rest_features: int = 0
    v_index: _Optional[str] = None
    v_start: int = 0
    v_width: int = 0

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def stack_dim(self) -> int:
        """Which dim of a decoded group tensor indexes its members."""
        return -(len(self.shape) + 1)


def _is_contiguous_run(starts: _Sequence[int], width: int) -> bool:
    return all(start == starts[0] + i * width for i, start in enumerate(starts))


class _ZeroLinear(_nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = _nn.Parameter(_torch.zeros((out_features, in_features)))
        self.bias = _nn.Parameter(_torch.zeros((out_features,)))

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return _F.linear(x, self.weight, self.bias)


def _validate_positive_int(value: _Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _validate_hidden_sizes(hidden_sizes: _Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_sizes, (str, bytes)):
        raise ValueError("hidden_sizes must be a sequence of positive integers")
    return tuple(
        _validate_positive_int(hidden_size, name="hidden_sizes entries")
        for hidden_size in hidden_sizes
    )


def _coerce_string_sequence(value: _Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a string")
    return tuple(str(item) for item in value)


def _validate_named_shapes(
    named_shapes: _Mapping[str, _torch.Size],
) -> tuple[tuple[str, _torch.Size], ...]:
    items = []
    for name, shape in named_shapes.items():
        if not name:
            raise ValueError("named_shapes keys must be non-empty strings")
        items.append((str(name), _torch.Size(shape)))
    if len(items) == 0:
        raise ValueError("named_shapes must contain at least one parameter tensor")
    return tuple(items)


def _resolve_target_names(
    named_shape_items: tuple[tuple[str, _torch.Size], ...],
    selector: _Optional[dict[str, _Any]],
) -> tuple[str, ...]:
    ordered_names = tuple(name for name, _ in named_shape_items)
    if selector is None:
        # Net-agnostic default: target every parameter. The model wrapper (Task 5) is
        # responsible for passing a selector to keep tensors (e.g. dilated convs) shared.
        return ordered_names
    selector_config = _normalize_selector_config(selector)
    known_names = set(ordered_names)
    include_names = selector_config["include_names"]
    exclude_names = selector_config["exclude_names"]
    unknown_names = sorted((set(include_names) | set(exclude_names)) - known_names)
    if unknown_names:
        raise ValueError(
            f"Hypernetwork selector references unknown parameter names: {unknown_names}"
        )

    include_suffixes = selector_config["include_suffixes"]
    exclude_suffixes = selector_config["exclude_suffixes"]

    def _matches_suffixes(name: str, suffixes: tuple[str, ...]) -> bool:
        return any(name.endswith(str(suffix)) for suffix in suffixes)

    use_includes = len(include_names) > 0 or len(include_suffixes) > 0
    resolved = []
    for name in ordered_names:
        selected = (
            name in include_names or _matches_suffixes(name, include_suffixes)
            if use_includes
            else True
        )
        if not selected:
            continue
        if name in exclude_names or _matches_suffixes(name, exclude_suffixes):
            continue
        resolved.append(name)
    if len(resolved) == 0:
        raise ValueError("Hypernetwork must target at least one parameter tensor")
    return tuple(resolved)


def _normalize_selector_config(
    selector: dict[str, _Any],
) -> dict[str, tuple[str, ...]]:
    selector_config = dict(selector)
    unexpected = sorted(set(selector_config) - _ALLOWED_SELECTOR_KEYS)
    if unexpected:
        raise ValueError(
            f"Hypernetwork selector config has unexpected keys: {unexpected}"
        )
    return {
        "include_names": _coerce_string_sequence(
            selector_config.get("include_names"),
            name="Hypernetwork selector include_names",
        ),
        "exclude_names": _coerce_string_sequence(
            selector_config.get("exclude_names"),
            name="Hypernetwork selector exclude_names",
        ),
        "include_suffixes": _coerce_string_sequence(
            selector_config.get("include_suffixes"),
            name="Hypernetwork selector include_suffixes",
        ),
        "exclude_suffixes": _coerce_string_sequence(
            selector_config.get("exclude_suffixes"),
            name="Hypernetwork selector exclude_suffixes",
        ),
    }


def _serialize_selector_config(
    selector_config: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    return {
        key: list(values) for key, values in selector_config.items() if len(values) > 0
    }


def _build_target_layout(
    *,
    name: str,
    shape: _torch.Size,
    mode: str,
    rank: _Optional[int],
) -> _TargetLayout:
    if mode == _FULL:
        return _TargetLayout(
            name=name,
            shape=shape,
            output_width=shape.numel(),
            mode=_FULL,
        )

    if rank is None:
        raise ValueError("Low-rank hypernetwork mode requires a positive integer rank")
    rank = _validate_positive_int(rank, name="rank")

    out_features = int(shape[0]) if len(shape) > 0 else 0
    rest_features = int(_math.prod(shape[1:])) if len(shape) > 1 else 1
    if out_features <= 1 or rest_features <= 1:
        return _TargetLayout(
            name=name,
            shape=shape,
            output_width=shape.numel(),
            mode=_FULL,
        )

    capped_rank = min(rank, out_features, rest_features)
    return _TargetLayout(
        name=name,
        shape=shape,
        output_width=capped_rank * (out_features + rest_features),
        mode=_LOW_RANK,
        rank=capped_rank,
        out_features=out_features,
        rest_features=rest_features,
    )


class Hypernetwork(_nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        named_shapes: _Mapping[str, _torch.Size],
        target_names: _Sequence[str],
        hidden_sizes: _Sequence[int] = _DEFAULT_HIDDEN_SIZES,
        activation: _Any = _DEFAULT_ACTIVATION,
        mode: str = _DEFAULT_MODE,
        rank: _Optional[int] = _DEFAULT_RANK,
        selector: _Optional[dict[str, _Any]] = None,
        low_rank_seed_std: float = _LOW_RANK_SEED_STD,
        seed: _Optional[int] = None,
    ):
        super().__init__()
        self._input_dim = _validate_positive_int(input_dim, name="input_dim")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unsupported hypernetwork mode {mode!r}; expected one of {list(_VALID_MODES)!r}"
            )
        # Validate the activation eagerly so a bad name fails at construction even when the
        # trunk is empty (hidden_sizes == ()) and would otherwise never instantiate it.
        if not isinstance(activation, (str, dict)):
            raise ValueError(
                "activation must be a string name or an activation config dict; "
                f"got {type(activation).__name__}"
            )
        try:
            _get_activation(activation)
        except Exception as exc:
            raise ValueError(f"Invalid activation {activation!r}: {exc}") from exc

        named_shape_items = _validate_named_shapes(named_shapes)
        target_names_tuple = tuple(str(name) for name in target_names)
        if len(target_names_tuple) == 0:
            raise ValueError("target_names must contain at least one parameter tensor")
        if len(set(target_names_tuple)) != len(target_names_tuple):
            raise ValueError("target_names must be unique")

        ordered_names = tuple(name for name, _ in named_shape_items)
        self._ordered_names = ordered_names
        unknown_target_names = sorted(set(target_names_tuple) - set(ordered_names))
        if unknown_target_names:
            raise ValueError(
                f"target_names contain unknown parameter names: {unknown_target_names}"
            )
        ordered_named_shapes = dict(named_shape_items)
        ordered_targets = tuple(
            name for name in ordered_names if name in target_names_tuple
        )
        if ordered_targets != target_names_tuple:
            raise ValueError(
                "target_names must follow the same deterministic order as named_shapes"
            )

        hidden_sizes_tuple = _validate_hidden_sizes(hidden_sizes)
        selector_config = (
            None if selector is None else _normalize_selector_config(selector)
        )
        if selector is not None:
            selected_targets = _resolve_target_names(named_shape_items, selector)
            if selected_targets != target_names_tuple:
                raise ValueError(
                    "selector must resolve to the same deterministic target_names order"
                )
        target_layouts = tuple(
            _build_target_layout(
                name=name,
                shape=ordered_named_shapes[name],
                mode=mode,
                rank=rank,
            )
            for name in ordered_targets
        )

        trunk_layers: list[_nn.Module] = []
        in_features = self._input_dim
        for hidden_size in hidden_sizes_tuple:
            trunk_layers.append(_nn.Linear(in_features, hidden_size))
            trunk_layers.append(_get_activation(activation))
            in_features = hidden_size
        self._trunk = (
            _nn.Sequential(*trunk_layers) if len(trunk_layers) > 0 else _nn.Identity()
        )

        self._target_layouts = target_layouts
        self.target_names = tuple(layout.name for layout in target_layouts)
        self._mode = mode
        self._rank = rank
        self._hidden_sizes = hidden_sizes_tuple
        self._activation = activation
        self._selector_config = selector_config
        self._low_rank_seed_std = float(low_rank_seed_std)
        self._seed = seed
        final_output_dim = sum(layout.output_width for layout in target_layouts)
        self._final = _ZeroLinear(in_features, final_output_dim)
        # Zero-init the readout so every delta == 0 at construction (model == base net for
        # all params). Consequence: at the very first optimizer step the trunk receives no
        # gradient (dL/dh = Wᵀ·dL/dflat == 0 while W == 0); the readout itself DOES get a
        # gradient from the upstream weight loss, becomes nonzero, and the trunk trains from
        # then on. This one-step warmup is expected -- do not "fix" it by dropping zero-init.
        self.register_buffer("_low_rank_anchor", self._build_low_rank_anchor())
        # Purely a decode-order view of `_target_layouts`: same flat readout layout, same
        # anchors, same math -- just evaluated a shape-group at a time. Built last because it
        # registers index buffers and reads the layouts above.
        self._decode_groups = self._build_decode_groups()

    @classmethod
    def from_config(
        cls,
        *,
        input_dim: int,
        named_shapes: _Mapping[str, _torch.Size],
        config: _Optional[dict[str, _Any]] = None,
    ) -> "Hypernetwork":
        config_dict = {} if config is None else dict(config)
        unexpected = sorted(set(config_dict) - _ALLOWED_CONFIG_KEYS)
        if unexpected:
            raise ValueError(f"Hypernetwork config has unexpected keys: {unexpected}")

        named_shape_items = _validate_named_shapes(named_shapes)
        selector = config_dict.get("selector")
        target_names = _resolve_target_names(named_shape_items, selector)
        return cls(
            input_dim=input_dim,
            named_shapes=dict(named_shape_items),
            target_names=target_names,
            hidden_sizes=config_dict.get("hidden_sizes", _DEFAULT_HIDDEN_SIZES),
            activation=config_dict.get("activation", _DEFAULT_ACTIVATION),
            mode=config_dict.get("mode", _DEFAULT_MODE),
            rank=config_dict.get("rank", _DEFAULT_RANK),
            selector=selector,
        )

    @property
    def config(self) -> dict[str, _Any]:
        # Enough to reconstruct the same targeting/architecture given input_dim + named_shapes.
        # The deterministic low-rank anchors use fixed defaults, so the init-only knobs remain
        # intentionally omitted here.
        config: dict[str, _Any] = {
            "hidden_sizes": list(self._hidden_sizes),
            # deepcopy so a caller mutating the returned dict can't alias internal state.
            "activation": _deepcopy(self._activation),
            "mode": self._mode,
        }
        # rank only matters in low_rank mode; emitting it for full mode would persist a
        # meaningless field and confuse readers.
        if self._mode == _LOW_RANK:
            config["rank"] = self._rank
        selector_config = self._selector_config
        if selector_config is None and self.target_names != self._ordered_names:
            selector_config = {
                "include_names": self.target_names,
                "exclude_names": (),
                "include_suffixes": (),
                "exclude_suffixes": (),
            }
        if selector_config is not None:
            config["selector"] = _serialize_selector_config(selector_config)
        return config

    def _build_decode_groups(self) -> tuple[_DecodeGroup, ...]:
        offsets = []
        offset = 0
        for layout in self._target_layouts:
            offsets.append(offset)
            offset += layout.output_width

        # dict preserves insertion order, so group order (and therefore buffer naming) is a
        # deterministic function of the target order.
        members: dict[_Any, list[int]] = {}
        for i, layout in enumerate(self._target_layouts):
            key = (
                layout.mode,
                tuple(layout.shape),
                layout.rank,
                layout.out_features,
                layout.rest_features,
            )
            members.setdefault(key, []).append(i)

        groups = []
        for group_index, indices in enumerate(members.values()):
            first = self._target_layouts[indices[0]]
            names = tuple(self._target_layouts[i].name for i in indices)

            def _slot(
                starts: list[int], width: int, suffix: str
            ) -> tuple[_Optional[str], int, int]:
                if _is_contiguous_run(starts, width):
                    return None, starts[0], width * len(starts)
                buffer_name = f"_group{group_index}{suffix}_index"
                self.register_buffer(
                    buffer_name,
                    _torch.cat(
                        [
                            _torch.arange(start, start + width, dtype=_torch.long)
                            for start in starts
                        ]
                    ),
                    # Derived from the layouts, so it must not ride along in checkpoints or
                    # in the exported state (which stays parameters + anchor, unchanged).
                    persistent=False,
                )
                return buffer_name, starts[0], width * len(starts)

            if not first.is_low_rank:
                index, start, width = _slot(
                    [offsets[i] for i in indices], first.shape.numel(), ""
                )
                groups.append(
                    _DecodeGroup(
                        names=names,
                        shape=first.shape,
                        is_low_rank=False,
                        index=index,
                        start=start,
                        width=width,
                    )
                )
                continue

            u_width, v_width = first.u_width, first.v_width
            index, start, width = _slot([offsets[i] for i in indices], u_width, "u")
            v_index, v_start, v_total = _slot(
                [offsets[i] + u_width for i in indices], v_width, "v"
            )
            groups.append(
                _DecodeGroup(
                    names=names,
                    shape=first.shape,
                    is_low_rank=True,
                    index=index,
                    start=start,
                    width=width,
                    rank=_cast(int, first.rank),
                    out_features=_cast(int, first.out_features),
                    rest_features=_cast(int, first.rest_features),
                    v_index=v_index,
                    v_start=v_start,
                    v_width=v_total,
                )
            )
        return tuple(groups)

    def _take(
        self,
        source: _torch.Tensor,
        index: _Optional[str],
        start: int,
        width: int,
    ) -> _torch.Tensor:
        if index is None:
            return source.narrow(-1, start, width)
        return source.index_select(-1, _cast(_torch.Tensor, getattr(self, index)))

    def _readout(self, p: _torch.Tensor) -> _torch.Tensor:
        reference = self._final.weight
        p = _torch.as_tensor(p, device=reference.device, dtype=reference.dtype)
        if p.ndim not in (1, 2):
            raise ValueError(
                f"Expected encoded params to have shape (E,) or (B, E); got {tuple(p.shape)}"
            )
        if p.shape[-1] != self._input_dim:
            raise ValueError(
                f"Expected encoded params trailing dimension {self._input_dim}; got {p.shape[-1]}"
            )
        return self._final(self._trunk(p))

    def generate_groups(
        self, p: _torch.Tensor
    ) -> list[tuple[_DecodeGroup, _torch.Tensor]]:
        """
        Deltas decoded a shape-group at a time.

        Each entry is a group paired with its members' deltas stacked along
        ``group.stack_dim``, i.e. shape ``(*leading, group.size, *group.shape)``. Callers that
        can consume the stacked form (see ``HyperWaveNet``) keep the whole group on one add;
        ``generate`` unbinds it back to one tensor per target.
        """
        flat = self._readout(p)
        leading = tuple(flat.shape[:-1])
        anchor = _cast(_torch.Tensor, self._low_rank_anchor)

        generated = []
        for group in self._decode_groups:
            if not group.is_low_rank:
                chunk = self._take(flat, group.index, group.start, group.width)
                generated.append(
                    (group, chunk.reshape(*leading, group.size, *group.shape))
                )
                continue

            u_shape = (group.size, group.out_features, group.rank)
            v_shape = (group.size, group.rank, group.rest_features)
            # The anchor is gathered per call rather than cached in group form so that
            # mutating `_low_rank_anchor` (import_state, tests) still takes effect.
            u = self._take(flat, group.index, group.start, group.width).reshape(
                *leading, *u_shape
            ) + self._take(anchor, group.index, group.start, group.width).reshape(
                *u_shape
            )
            v = self._take(flat, group.v_index, group.v_start, group.v_width).reshape(
                *leading, *v_shape
            ) + self._take(anchor, group.v_index, group.v_start, group.v_width).reshape(
                *v_shape
            )
            generated.append(
                (group, (u @ v).reshape(*leading, group.size, *group.shape))
            )
        return generated

    def generate(self, p: _torch.Tensor) -> dict[str, _torch.Tensor]:
        deltas = {}
        for group, stacked in self.generate_groups(p):
            for name, delta in zip(group.names, stacked.unbind(group.stack_dim)):
                deltas[name] = delta
        # Re-key in target order; grouping reorders the decode, not the public contract.
        return {layout.name: deltas[layout.name] for layout in self._target_layouts}

    def export_target_metadata(self) -> list[dict[str, _Any]]:
        metadata = []
        for layout in self._target_layouts:
            target: dict[str, _Any] = {
                "name": layout.name,
                "mode": layout.mode,
                "numel": layout.shape.numel(),
            }
            if layout.is_low_rank:
                if (
                    layout.rank is None
                    or layout.out_features is None
                    or layout.rest_features is None
                ):
                    raise RuntimeError(
                        f"Low-rank target layout for {layout.name!r} is incomplete"
                    )
                target["rank"] = layout.rank
                target["out_features"] = layout.out_features
                target["rest_features"] = layout.rest_features
            metadata.append(target)
        return metadata

    def param_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def state_count(self) -> int:
        return sum(tensor.numel() for tensor in self._serialized_tensors())

    def export_state(self) -> _torch.Tensor:
        return _torch.cat(
            [tensor.detach().reshape(-1).cpu() for tensor in self._serialized_tensors()]
        )

    def import_state(self, weights: _torch.Tensor, i: int = 0) -> int:
        if weights.ndim != 1:
            raise ValueError(
                f"Hypernetwork state must be a flat 1-D tensor; got shape {tuple(weights.shape)}"
            )
        expected = self.state_count()
        remaining = len(weights) - i
        if remaining != expected:
            raise ValueError(
                f"Expected {expected} serialized hypernetwork values, but found {remaining}"
            )

        with _torch.no_grad():
            for tensor in self._serialized_tensors():
                n = tensor.numel()
                tensor.copy_(
                    weights[i : i + n]
                    .to(device=tensor.device, dtype=tensor.dtype)
                    .reshape(tensor.shape)
                )
                i += n
        return i

    def _serialized_tensors(self) -> tuple[_torch.Tensor, ...]:
        return tuple(parameter for _, parameter in self.named_parameters()) + (
            _cast(_torch.Tensor, self._low_rank_anchor),
        )

    def _build_low_rank_anchor(self) -> _torch.Tensor:
        anchor = self._final.bias.detach().clone()
        if self._low_rank_seed_std <= 0.0:
            return anchor

        # Local generator: the symmetry-breaking seed is deterministic and does not consume
        # the global torch RNG. The trunk's standard Linear init still draws from the global
        # RNG -- that part is unavoidable.
        generator = _torch.Generator(device=self._final.bias.device)
        generator.manual_seed(
            _DEFAULT_LOW_RANK_SEED if self._seed is None else self._seed
        )
        offset = 0
        with _torch.no_grad():
            for layout in self._target_layouts:
                next_offset = offset + layout.output_width
                if layout.is_low_rank:
                    if (
                        layout.rank is None
                        or layout.out_features is None
                        or layout.rest_features is None
                    ):
                        raise RuntimeError(
                            f"Low-rank target layout for {layout.name!r} is incomplete"
                        )
                    # Alternate anchored rank components between U and V. Each component has
                    # exactly one anchored side, so the seeded U@V product is still exactly 0
                    # while the first backward pass reaches both factor slices.
                    u_anchor = anchor[offset : offset + layout.u_width].reshape(
                        layout.out_features, layout.rank
                    )
                    v_anchor = anchor[offset + layout.u_width : next_offset].reshape(
                        layout.rank, layout.rest_features
                    )
                    for rank_index in range(layout.rank):
                        if rank_index % 2 == 0:
                            u_anchor[:, rank_index].normal_(
                                0.0, self._low_rank_seed_std, generator=generator
                            )
                        else:
                            v_anchor[rank_index].normal_(
                                0.0, self._low_rank_seed_std, generator=generator
                            )
                offset = next_offset
        return anchor
