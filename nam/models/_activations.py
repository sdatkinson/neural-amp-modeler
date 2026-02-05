# File: _activations.py
# Created Date: Friday July 29th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc as _abc
from typing import Any as _Any, Dict as _Dict, Optional as _Optional, Union as _Union

import torch as _torch
import torch.nn as _nn
from pydantic import BaseModel as _BaseModel


class _Softsign(_nn.Module):
    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return x / (1.0 + _torch.abs(x))


class _Softsigmoid(_nn.Module):
    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return 0.5 * (1.0 + x / (1.0 + _torch.abs(x)))


class PairingActivation(_nn.Module, _abc.ABC):
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


def get_activation(name: str, **kwargs) -> _nn.Module:
    return parse_activation_config(config={"name": name, **kwargs}).create()
