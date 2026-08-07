"""
Parametric datasets that pair fixed control settings with stock NAM audio windows.
"""

from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Optional as _Optional

import torch as _torch

from ..._core import InitializableFromConfig as _InitializableFromConfig
from ...data import AbstractDataset as _AbstractDataset
from ...data import Dataset as _Dataset
from ._spec import ParamSpec as _ParamSpec


def _is_non_string_iterable(value: _Any) -> bool:
    return isinstance(value, _Iterable) and not isinstance(value, (str, bytes))


def _coerce_param_specs(raw_param_specs: _Any) -> tuple[_ParamSpec, ...]:
    if not _is_non_string_iterable(raw_param_specs):
        raise ValueError("param_specs must be a non-string iterable of ParamSpec definitions")
    param_specs = []
    for i, raw_param_spec in enumerate(raw_param_specs):
        if isinstance(raw_param_spec, _ParamSpec):
            spec = raw_param_spec
        elif isinstance(raw_param_spec, _Mapping):
            spec = _ParamSpec.from_dict(dict(raw_param_spec))
        else:
            raise ValueError(
                f"param_specs[{i}] must be a ParamSpec or mapping, got {type(raw_param_spec).__name__}"
            )
        param_specs.append(spec)
    if len(param_specs) == 0:
        raise ValueError("param_specs must contain at least one parameter definition")
    param_names = tuple(spec.name for spec in param_specs)
    if len(set(param_names)) != len(param_names):
        raise ValueError("param_specs must define unique parameter names")
    return tuple(param_specs)


def _resolve_switch_value(
    name: str, raw_value: _Any, enum_names: tuple[str, ...]
) -> float:
    if isinstance(raw_value, str):
        try:
            return float(enum_names.index(raw_value))
        except ValueError as exc:
            raise ValueError(
                f"params[{name!r}]={raw_value!r} is not a valid enum name; expected one of "
                f"{list(enum_names)}"
            ) from exc

    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"params[{name!r}] must be an enum name or integer index; got {raw_value!r}"
        ) from exc
    if not numeric_value.is_integer():
        raise ValueError(
            f"params[{name!r}] must be an integer switch index; got {raw_value!r}"
        )
    index = int(numeric_value)
    if index < 0 or index >= len(enum_names):
        raise ValueError(
            f"params[{name!r}] index {index} is out of range for enum names {list(enum_names)}"
        )
    return float(index)


def _resolve_continuous_value(name: str, raw_value: _Any) -> float:
    # Continuous values are passed through as the raw user-facing number. We deliberately
    # do NOT clamp/validate against the spec's [min, max] here: out-of-range captures are
    # allowed and the model linearly rescales them at normalization time (no clamping).
    if isinstance(raw_value, str):
        raise ValueError(
            f"params[{name!r}] must be numeric for a continuous parameter; got {raw_value!r}"
        )
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"params[{name!r}] must be numeric for a continuous parameter; got {raw_value!r}"
        ) from exc
    if not _torch.isfinite(_torch.tensor(value)):
        raise ValueError(f"params[{name!r}] must be finite; got {raw_value!r}")
    return value


def resolve_named_params(
    raw_params: _Any,
    param_specs: tuple[_ParamSpec, ...],
) -> _torch.Tensor:
    """
    Resolve a named ``{name: value}`` mapping into a positional ``(P,)`` float tensor in the
    declared ``param_specs`` order. Switches accept an enum name or integer index; continuous
    values pass through unclamped. Shared by the dataset and the export bake path so both use
    one resolver.
    """
    if not isinstance(raw_params, _Mapping):
        raise ValueError("params must be a mapping from parameter name to raw value")

    param_names = tuple(spec.name for spec in param_specs)
    expected_names = set(param_names)
    provided_names = {str(name) for name in raw_params.keys()}
    missing_names = [name for name in param_names if name not in provided_names]
    if missing_names:
        raise ValueError(
            "params is missing declared parameter name(s): " + ", ".join(missing_names)
        )
    unknown_names = sorted(provided_names - expected_names)
    if unknown_names:
        raise ValueError(
            "params contains unknown parameter name(s): " + ", ".join(unknown_names)
        )

    resolved_values = []
    for spec in param_specs:
        raw_value = raw_params[spec.name]
        if spec.type == "switch":
            if spec.enum_names is None:
                raise RuntimeError(
                    f"Switch ParamSpec {spec.name!r} is missing enum_names after validation"
                )
            resolved_values.append(
                _resolve_switch_value(spec.name, raw_value, spec.enum_names)
            )
        else:
            resolved_values.append(_resolve_continuous_value(spec.name, raw_value))
    return _torch.tensor(resolved_values, dtype=_torch.float32)


class ParametricDataset(_AbstractDataset, _InitializableFromConfig):
    """
    Compose a stock :class:`~nam.data.Dataset` with a fixed parameter vector.

    Each item is ``(x, params, y)`` — audio input first, to match the model's
    ``forward(x, params)`` signature so the stock training loop's ``self(*args)`` works
    unchanged. ``params`` is a 1D float32 tensor in the model's declared parameter order.
    Its entries carry mixed semantics, resolved against the ParamSpecs at build time and
    decoded again by the model:

    - continuous params  -> the raw user-facing value (e.g. ``8.0``)
    - switch params      -> the integer class index (e.g. ``"crunch"`` -> ``1.0``)

    Normalization (continuous) and one-hot expansion (switch) are intentionally left to
    the model wrapper, so the dataset stays a dumb carrier of values.
    """

    def __init__(self, dataset: _Dataset, params: _torch.Tensor):
        if params.ndim != 1:
            raise ValueError(
                f"ParametricDataset params must be a 1D tensor; got shape {tuple(params.shape)}"
            )
        if not _torch.isfinite(params).all():
            raise ValueError("ParametricDataset params must all be finite")
        self._dataset = dataset
        self._params = params.detach().clone().to(dtype=_torch.float32)

    def __getitem__(self, idx: int) -> _Any:
        x, y = self._dataset[idx]
        return x, self._params, y

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> _Dataset:
        return self._dataset

    @property
    def params(self) -> _torch.Tensor:
        return self._params

    @property
    def nx(self) -> int:
        return self._dataset.nx

    @property
    def ny(self) -> int:
        return self._dataset.ny

    @property
    def sample_rate(self) -> _Optional[float]:
        return self._dataset.sample_rate

    @classmethod
    def parse_config(cls, config: dict[str, _Any]) -> dict[str, _Any]:
        config = _deepcopy(config)
        raw_params = config.pop("params", None)
        if raw_params is None:
            raise ValueError("ParametricDataset config is missing required field: params")
        param_specs = cls._parse_param_specs(config)
        return {
            "dataset": _Dataset.init_from_config(config),
            "params": cls._resolve_params(raw_params, param_specs),
        }

    @classmethod
    def _resolve_params(
        cls,
        raw_params: _Any,
        param_specs: tuple[_ParamSpec, ...],
    ) -> _torch.Tensor:
        return resolve_named_params(raw_params, param_specs)

    @classmethod
    def _parse_param_specs(cls, config: dict[str, _Any]) -> tuple[_ParamSpec, ...]:
        raw_param_specs = config.pop("param_specs", None)
        if raw_param_specs is None:
            raise ValueError(
                "ParametricDataset config is missing required field: param_specs"
            )
        return _coerce_param_specs(raw_param_specs)

def init_dataset(config: dict[str, _Any]) -> _AbstractDataset:
    # One capture -> one ParametricDataset. Stock ``nam.data.init_dataset`` fans a list of
    # captures out across this initializer and wraps them in a ConcatDataset itself, so a
    # list of train (or validation) settings needs no special handling here.
    return ParametricDataset.init_from_config(config)


def data_config_from_model(
    data_config: dict[str, _Any], model_config: dict[str, _Any]
) -> dict[str, _Any]:
    """
    Return a copy of ``data_config`` with ``common.param_specs`` injected from
    ``model_config["net"]["config"]["params"]``.
    """

    data_config = _deepcopy(data_config)
    try:
        model_param_specs = model_config["net"]["config"]["params"]
    except KeyError as exc:
        raise ValueError(
            "Model config must define net.config.params for parametric dataset loading"
        ) from exc
    param_specs = [spec.to_dict() for spec in _coerce_param_specs(model_param_specs)]
    common = data_config.setdefault("common", {})
    if not isinstance(common, dict):
        raise ValueError("Data config common section must be a mapping")
    if "param_specs" in common:
        raise ValueError("Data config common section already defines param_specs")
    common["param_specs"] = param_specs
    return data_config
