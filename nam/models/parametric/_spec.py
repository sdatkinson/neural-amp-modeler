"""
Parameter metadata for parametric NAM models.
"""

import math as _math
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Optional as _Optional


_CONTINUOUS = "continuous"
_SWITCH = "switch"
# Signed min-max scaling maps the user-facing range [min, max] to [-1, 1].
_MIN_MAX_SIGNED = "min_max_signed"
# Unit min-max scaling maps the user-facing range [min, max] to [0, 1].
_MIN_MAX_UNIT = "min_max_unit"
_VALID_TYPES = (_CONTINUOUS, _SWITCH)


def _is_finite(value: _Any) -> bool:
    try:
        return _math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_int_like(value: _Any) -> bool:
    return _is_finite(value) and float(value).is_integer()


def _coerce_enum_names(name: str, raw_enum_names: _Any) -> tuple[str, ...]:
    if isinstance(raw_enum_names, (str, bytes)):
        raise ValueError(
            f"Switch ParamSpec {name!r} enum_names must be a sequence of names, not a string"
        )
    try:
        enum_names = tuple(str(value) for value in raw_enum_names)
    except TypeError as exc:
        raise ValueError(
            f"Switch ParamSpec {name!r} enum_names must be an iterable of names"
        ) from exc
    return enum_names


@_dataclass(frozen=True)
class ParamSpec:
    name: str
    min: float
    max: float
    default: float
    type: str = _CONTINUOUS
    enum_names: _Optional[tuple[str, ...]] = None
    # Capture-grid step (realizability metadata only; it does not affect normalization
    # or training). ``None`` defers to the consumer's default step.
    step: _Optional[float] = None
    # Capture-planning metadata: when true, quantization never emits an exact zero for
    # this continuous param, snapping to the next reachable grid value instead. Guards
    # against gain/drive knobs that mute the rig at zero and destabilize training. Like
    # ``step``, it has no effect on normalization or training.
    avoid_zero: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError("ParamSpec name must be non-empty")
        if self.type not in _VALID_TYPES:
            raise ValueError(
                f"Unsupported ParamSpec type {self.type!r}; expected one of {_VALID_TYPES}"
            )
        if not all(_is_finite(v) for v in (self.min, self.max, self.default)):
            raise ValueError("ParamSpec min/max/default must all be finite")
        if self.min >= self.max:
            raise ValueError(
                f"ParamSpec {self.name!r} must satisfy min < max; got {self.min} >= {self.max}"
            )
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"ParamSpec {self.name!r} default must satisfy min <= default <= max; "
                f"got {self.min} <= {self.default} <= {self.max}"
            )

        if self.type == _CONTINUOUS:
            if self.enum_names is not None:
                raise ValueError(
                    f"Continuous ParamSpec {self.name!r} cannot define enum_names"
                )
            if self.step is not None:
                if not _is_finite(self.step) or float(self.step) <= 0.0:
                    raise ValueError(
                        f"ParamSpec {self.name!r} step must be a positive finite number; "
                        f"got {self.step}"
                    )
                if float(self.step) > float(self.max) - float(self.min):
                    raise ValueError(
                        f"ParamSpec {self.name!r} step must not exceed the range width; "
                        f"got {self.step} > {float(self.max) - float(self.min)}"
                    )
                object.__setattr__(self, "step", float(self.step))
            object.__setattr__(self, "avoid_zero", bool(self.avoid_zero))
            object.__setattr__(self, "min", float(self.min))
            object.__setattr__(self, "max", float(self.max))
            object.__setattr__(self, "default", float(self.default))
            return

        if self.step is not None:
            raise ValueError(f"Switch ParamSpec {self.name!r} cannot define step")
        if self.avoid_zero:
            raise ValueError(f"Switch ParamSpec {self.name!r} cannot set avoid_zero")
        if self.enum_names is None:
            raise ValueError(f"Switch ParamSpec {self.name!r} requires enum_names")
        enum_names = _coerce_enum_names(self.name, self.enum_names)
        if len(enum_names) < 2:
            raise ValueError(
                f"Switch ParamSpec {self.name!r} must define at least two enum_names"
            )
        if len(set(enum_names)) != len(enum_names):
            raise ValueError(
                f"Switch ParamSpec {self.name!r} enum_names must be unique"
            )
        if not all(name for name in enum_names):
            raise ValueError(
                f"Switch ParamSpec {self.name!r} enum_names must all be non-empty"
            )
        if not all(_is_int_like(v) for v in (self.min, self.max, self.default)):
            raise ValueError(
                f"Switch ParamSpec {self.name!r} min/max/default must be integer indices"
            )

        min_index = int(self.min)
        max_index = int(self.max)
        default_index = int(self.default)
        expected_max = len(enum_names) - 1
        if min_index != 0 or max_index != expected_max:
            raise ValueError(
                f"Switch ParamSpec {self.name!r} must use min/max index bounds [0, {expected_max}]"
            )
        if not (min_index <= default_index <= max_index):
            raise ValueError(
                f"Switch ParamSpec {self.name!r} default index {default_index} "
                f"must be within [{min_index}, {max_index}]"
            )

        object.__setattr__(self, "min", min_index)
        object.__setattr__(self, "max", max_index)
        object.__setattr__(self, "default", default_index)
        object.__setattr__(self, "enum_names", enum_names)

    @property
    def normalization(self) -> str:
        # Normalization mode is an internal training detail, not a user-configurable field.
        return _MIN_MAX_UNIT if self.type == _SWITCH else _MIN_MAX_SIGNED

    @property
    def normalized_min(self) -> float:
        return -1.0 if self.normalization == _MIN_MAX_SIGNED else 0.0

    @property
    def normalized_max(self) -> float:
        return 1.0

    @property
    def num_inputs(self) -> int:
        if self.type == _CONTINUOUS:
            return 1
        if self.enum_names is None:
            raise RuntimeError(
                f"Switch ParamSpec {self.name!r} is missing enum_names after validation"
            )
        return len(self.enum_names)

    def to_dict(self) -> dict[str, _Any]:
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "default": self.default,
            "type": self.type,
            "enum_names": None if self.enum_names is None else list(self.enum_names),
            "step": self.step,
            "avoid_zero": self.avoid_zero,
        }

    @classmethod
    def from_dict(cls, d: dict[str, _Any]) -> "ParamSpec":
        config = dict(d)
        required_keys = ("name", "min", "max", "default")
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            missing_keys_str = ", ".join(missing_keys)
            raise ValueError(
                f"ParamSpec config is missing required field(s): {missing_keys_str}"
            )
        raw_enum_names = config.get("enum_names")
        enum_names = (
            None
            if raw_enum_names is None
            else _coerce_enum_names(str(config["name"]), raw_enum_names)
        )
        spec = cls(
            name=config["name"],
            min=config["min"],
            max=config["max"],
            default=config["default"],
            type=config.get("type", _CONTINUOUS),
            enum_names=enum_names,
            step=config.get("step"),
            avoid_zero=config.get("avoid_zero", False),
        )
        return spec
