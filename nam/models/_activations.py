# File: _activations.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc as _abc
from copy import deepcopy as _deepcopy
from typing import (
    Any as _Any,
    Dict as _Dict,
    Optional as _Optional,
    Tuple as _Tuple,
    Union as _Union,
)

import torch as _torch
import torch.nn as _nn
from pydantic import BaseModel as _BaseModel


class _GeneralizedActivation(_nn.Module, _abc.ABC):
    """Base for NAM activations that support C++ export via export_config()."""

    @_abc.abstractmethod
    def export_config(self) -> _Union[_Dict[str, _Any], _Tuple]:
        """
        Return C++-compatible export format.
        Simple: {"type": name, ...}. Pair: (primary, gating_mode, secondary).
        """
        pass


class _Softsign(_GeneralizedActivation):
    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return x / (1.0 + _torch.abs(x))

    def export_config(self) -> _Dict[str, str]:
        return {"type": "Softsign"}


class _LeakyHardtanh(_GeneralizedActivation):
    """
    Leaky hard tanh: linear slopes outside [min_val, max_val], identity inside.
    Matches NeuralAmpModelerCore leaky_hardtanh (activations.h).
    """

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        min_slope: float = 0.01,
        max_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self._min_val = min_val
        self._max_val = max_val
        self._min_slope = min_slope
        self._max_slope = max_slope

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        # Below min_val: (x - min_val) * min_slope + min_val
        # Above max_val: (x - max_val) * max_slope + max_val
        # Else: x
        out = _torch.where(
            x < self._min_val,
            (x - self._min_val) * self._min_slope + self._min_val,
            x,
        )
        out = _torch.where(
            x > self._max_val,
            (x - self._max_val) * self._max_slope + self._max_val,
            out,
        )
        return out

    def export_config(self) -> _Dict[str, _Any]:
        return {
            "type": "LeakyHardtanh",
            "min_val": self._min_val,
            "max_val": self._max_val,
            "min_slope": self._min_slope,
            "max_slope": self._max_slope,
        }


class _Softsigmoid(_GeneralizedActivation):
    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return 0.5 * (1.0 + x / (1.0 + _torch.abs(x)))

    def export_config(self) -> _Dict[str, str]:
        return {"type": "Softsigmoid"}


class PairingActivation(_GeneralizedActivation, _abc.ABC):
    """
    Activations that pair inputs.
    E.g. gating activations where half is the activation and the other half
    scales the output.
    """

    def __init__(self, primary, secondary):
        super().__init__()
        self._primary = primary
        self._secondary = secondary

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @_abc.abstractmethod
    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        """
        :param x: (N,2D,...)
        :return: (N,D,...)
        """
        pass


class PairMultiply(PairingActivation):
    """
    aka "gating" activation

    The output of a primary activation is multiplied by a secondary ("gating") activation.
    """

    def __init__(self, primary, secondary):
        super().__init__(primary=primary, secondary=secondary)

    def export_config(
        self,
    ) -> _Tuple[_Dict[str, _Any], str, _Dict[str, _Any]]:
        prim = _export_pytorch_activation_config(self._primary)
        sec = _export_pytorch_activation_config(self._secondary)
        return {
            "type": "PairMultiply",
            "primary": prim,
            "secondary": sec,
        }

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        dim = x.shape[1] // 2
        x1, x2 = x.split(dim, dim=1)
        return self._primary(x1) * self._secondary(x2)


class PairBlend(PairingActivation):
    """
    Blending activation

    The output of a primary activation is blended with the original input according to
    the output of a secondary activation.

    The seconddary activation is "supposed" to output a value between 0 and 1
    (like a sigmoid might), but why not do other cool things? ;)
    """

    def __init__(self, primary, secondary):
        super().__init__(primary=primary, secondary=secondary)

    def export_config(
        self,
    ) -> _Tuple[_Dict[str, _Any], str, _Dict[str, _Any]]:
        prim = _export_pytorch_activation_config(self._primary)
        sec = _export_pytorch_activation_config(self._secondary)
        return {
            "type": "PairBlend",
            "primary": prim,
            "secondary": sec,
        }

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        dim = x.shape[1] // 2
        x1, x2 = x.split(dim, dim=1)
        blend = self._secondary(x2)
        return blend * self._primary(x1) + (1 - blend) * x1


class _BasicActivationConfig(_BaseModel):
    name: str
    kwargs: _Optional[_Dict[str, _Any]] = None

    def create(self) -> _nn.Module:
        name = self.name
        kwargs = dict() if self.kwargs is None else self.kwargs
        # Special NAM activations (check first so we don't call getattr(_nn, name) for these)
        special = {
            "LeakyHardtanh": _LeakyHardtanh,
            "Softsigmoid": _Softsigmoid,
            "Softsign": _Softsign,
        }
        if name in special:
            return special[name](**kwargs)
        return getattr(_nn, name)(**kwargs)


class _PairingActivationConfig(_BaseModel):
    name: str
    primary: str
    secondary: str
    kwargs: _Optional[_Dict[str, _Any]] = None

    def create(self) -> PairingActivation:
        primary = get_activation(self.primary)
        secondary = get_activation(self.secondary)
        kwargs = dict() if self.kwargs is None else self.kwargs
        return {
            "PairMultiply": PairMultiply,
            "PairBlend": PairBlend,
        }[
            self.name
        ](primary=primary, secondary=secondary, **kwargs)


ActivationConfig = _Union[_BasicActivationConfig, _PairingActivationConfig]


def parse_activation_config(config: _Dict[str, _Any]) -> ActivationConfig:
    # Bah wish I could do this smoother
    name = config.pop("name")
    if all(k in config for k in ["primary", "secondary"]):
        primary = config.pop("primary")
        secondary = config.pop("secondary")
        return _PairingActivationConfig(
            name=name, primary=primary, secondary=secondary, kwargs=config
        )
    else:
        return _BasicActivationConfig(name=name, kwargs=config)


def get_activation(name: _Union[str, _Dict[str, _Any]], **kwargs) -> _nn.Module:
    if isinstance(name, dict):
        return parse_activation_config(_deepcopy(name)).create()
    return parse_activation_config(config={"name": name, **kwargs}).create()


def _export_pytorch_activation_config(module: _nn.Module) -> _Dict[str, _Any]:
    """
    Convert a simple (non-pair) activation to C++ export format.
    Uses export_config() when available, else fallbacks for PyTorch activations.
    Always returns {"type": name, ...}.
    """
    assert not isinstance(module, _GeneralizedActivation)

    # Fallbacks for PyTorch activations
    def _class_name(m: _nn.Module) -> str:
        return m.__class__.__name__.lstrip("_")

    # Special cases:
    if isinstance(module, _nn.LeakyReLU):
        return {"type": "LeakyReLU", "negative_slope": module.negative_slope}
    if isinstance(module, _nn.PReLU):
        w = module.weight
        if w.numel() == 1:
            return {"type": "PReLU", "negative_slope": float(w.item())}
        return {"type": "PReLU", "negative_slopes": w.detach().cpu().tolist()}
    if isinstance(module, _nn.Hardtanh):
        return {
            "type": "Hardtanh",
            "min_val": module.min_val,
            "max_val": module.max_val,
        }
    # Basic case: no parameters
    return {"type": _class_name(module)}


def export_activation_config(
    activation: _nn.Module,
) -> _Tuple[_Dict[str, _Any], str, _Optional[_Dict[str, _Any]]]:
    """
    Convert an activation module to the C++-compatible export format.

    Returns (primary_activation, gating_mode, secondary_activation) where:
    - primary_activation: {"type": name, ...} for ActivationConfig::from_json
    - gating_mode: "gated" | "blended" | "none"
    - secondary_activation: {"type": name, ...} for gated/blended, None for none

    For simple activations, gating_mode is "none" and secondary_activation is None.
    """
    if isinstance(activation, _GeneralizedActivation):
        return activation.export_config()
    else:
        return _export_pytorch_activation_config(activation)
