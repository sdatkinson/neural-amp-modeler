"""
User-facing continuous knob specifications for the capture app.

A :class:`KnobSpec` is what the app asks the user for: a knob's name, its printed
range, and the increment they can realistically dial it to. It converts to the
training-side :class:`nam.models.parametric.ParamSpec`, which is what the planner,
dataset, and generated model configs consume.

This module is deliberately torch-free so the GUI can import it at startup.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Optional as _Optional
from typing import Sequence as _Sequence

from ..models.parametric._spec import ParamSpec as _ParamSpec


DEFAULT_KNOB_STEP = 0.5


def _snap(value: float, minimum: float, maximum: float, step: float) -> float:
    snapped = minimum + round((value - minimum) / step) * step
    return min(max(snapped, minimum), maximum)


@_dataclass(frozen=True)
class KnobSpec:
    """
    A continuous knob to be captured.

    :param name: User-facing knob name (e.g. "Gain").
    :param min: Value at the knob's minimum position, in the units printed on the amp.
    :param max: Value at the knob's maximum position.
    :param step: Capture grid increment: planned settings are snapped to
        ``min + k * step`` so every planned value is one a human can actually dial.
    :param default: Resting value baked into the generated model config; the knob's
        midpoint (snapped to the grid) when omitted.
    :param avoid_zero: When true, no planned capture sets this knob to zero; a draw that
        would snap to zero uses the next reachable grid value instead. Use for gain/drive
        knobs that mute the rig at zero and can destabilize training.
    :param is_gain: Marks this knob as the amp's gain/drive control. At most one knob may
        be marked. It changes how the initial corner captures are generated (the gain knob
        gets its own min/max sweep so the EQ params' corners are learned at both gain
        extremes); it is capture-only metadata and does not affect the training-side
        :class:`ParamSpec`.
    """

    name: str
    min: float
    max: float
    step: float = DEFAULT_KNOB_STEP
    default: _Optional[float] = None
    avoid_zero: bool = False
    is_gain: bool = False

    def __post_init__(self):
        if not str(self.name).strip():
            raise ValueError("Knob name must be non-empty")
        object.__setattr__(self, "name", str(self.name).strip())
        for field in ("min", "max", "step"):
            value = getattr(self, field)
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Knob {self.name!r} {field} must be a number; got {value!r}"
                ) from exc
            if not _math.isfinite(value):
                raise ValueError(f"Knob {self.name!r} {field} must be finite")
            object.__setattr__(self, field, value)
        if self.min >= self.max:
            raise ValueError(
                f"Knob {self.name!r} must satisfy min < max; got {self.min} >= {self.max}"
            )
        if self.step <= 0.0:
            raise ValueError(f"Knob {self.name!r} step must be positive; got {self.step}")
        if self.step > self.max - self.min:
            raise ValueError(
                f"Knob {self.name!r} step must not exceed the range width; "
                f"got {self.step} > {self.max - self.min}"
            )
        if self.default is None:
            object.__setattr__(
                self,
                "default",
                _snap(0.5 * (self.min + self.max), self.min, self.max, self.step),
            )
        else:
            default = float(self.default)
            if not _math.isfinite(default):
                raise ValueError(f"Knob {self.name!r} default must be finite")
            if not (self.min <= default <= self.max):
                raise ValueError(
                    f"Knob {self.name!r} default must be within [{self.min}, {self.max}]; "
                    f"got {default}"
                )
            object.__setattr__(self, "default", default)
        object.__setattr__(self, "avoid_zero", bool(self.avoid_zero))
        object.__setattr__(self, "is_gain", bool(self.is_gain))

    def to_param_spec(self) -> _ParamSpec:
        return _ParamSpec(
            name=self.name,
            min=self.min,
            max=self.max,
            default=self.default,
            type="continuous",
            step=self.step,
            avoid_zero=self.avoid_zero,
        )

    def to_dict(self) -> dict[str, _Any]:
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "default": self.default,
            "avoid_zero": self.avoid_zero,
            "is_gain": self.is_gain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, _Any]) -> "KnobSpec":
        config = dict(d)
        missing = [key for key in ("name", "min", "max") if key not in config]
        if missing:
            raise ValueError(
                f"Knob config is missing required field(s): {', '.join(missing)}"
            )
        return cls(
            name=config["name"],
            min=config["min"],
            max=config["max"],
            step=config.get("step", DEFAULT_KNOB_STEP),
            default=config.get("default"),
            avoid_zero=config.get("avoid_zero", False),
            is_gain=config.get("is_gain", False),
        )


def validate_knobs(knobs: _Sequence[KnobSpec]) -> tuple[KnobSpec, ...]:
    """
    Validate a knob set as a whole and return it as a tuple.

    Names must be unique case-insensitively: capture filenames encode params via
    case-insensitive unique-prefix abbreviations, so "gain" and "Gain" would collide.
    """
    knobs = tuple(knobs)
    if len(knobs) == 0:
        raise ValueError("At least one knob is required")
    seen: dict[str, str] = {}
    for knob in knobs:
        key = knob.name.lower()
        if key in seen:
            raise ValueError(
                f"Duplicate knob name: {knob.name!r} collides with {seen[key]!r}"
            )
        seen[key] = knob.name
    gain_names = [knob.name for knob in knobs if knob.is_gain]
    if len(gain_names) > 1:
        raise ValueError(
            "At most one knob may be marked Gain/Drive; "
            f"got {', '.join(repr(name) for name in gain_names)}"
        )
    return knobs


def gain_knob_index(knobs: _Sequence[KnobSpec]) -> _Optional[int]:
    """
    Position of the knob marked Gain/Drive, or ``None`` if none is. Assumes the knob set
    has passed :func:`validate_knobs` (at most one gain knob).
    """
    for index, knob in enumerate(knobs):
        if knob.is_gain:
            return index
    return None
