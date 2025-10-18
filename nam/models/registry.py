"""
Registry of factories for models
"""

from importlib import import_module as _import_module
import logging as _logging
from typing import Any as _Any
from typing import Callable as _Callable
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Sequence as _Sequence

from .base import BaseNet as _BaseNet
from .conv_net import ConvNet as _ConvNet
from .linear import Linear as _Linear
from .recurrent import LSTM as _LSTM
from .wavenet import WaveNet as _WaveNet
from .sequential import Sequential as _Sequential

_logger = _logging.getLogger(__name__)

_model_net_init_registry = {
    "ConvNet": _ConvNet.init_from_config,
    "Linear": _Linear.init_from_config,
    "LSTM": _LSTM.init_from_config,
    "Sequential": _Sequential.init_from_config,
    "WaveNet": _WaveNet.init_from_config,
}


def register(
    name: str, constructor: _Callable[[_Any], _BaseNet], overwrite: bool = False
):
    if name in _model_net_init_registry:
        if not overwrite:
            raise KeyError(
                f"A constructor for net name '{name}' is already registered!"
            )
        else:
            _logger.warning(f"Overwriting constructor for net name '{name}'")
    _model_net_init_registry[name] = constructor


def init(
    name: str,
    args: _Optional[_Sequence[_Any]] = None,
    kwargs: _Optional[_Dict[str, _Any]] = None,
) -> _BaseNet:
    if name in _model_net_init_registry:
        return _model_net_init_registry[name](*args, **kwargs)
    else:
        _logger.info(
            f"name {name} not in model registry; attempting import-based initialization"
        )

        # TODO pull this out.
        module_name, factory_name = name.rsplit(".", 1)
        try:
            module = _import_module(module_name)
        except ImportError:
            raise KeyError(f"No importable module found for name '{name}'")
        try:
            factory = getattr(module, factory_name)
        except AttributeError:
            raise KeyError(
                f"No factory found for name '{name}' within module '{module_name}'"
            )
        return factory(*args, **kwargs)
