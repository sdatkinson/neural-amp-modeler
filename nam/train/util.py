from typing import Any as _Any
from typing import Mapping as _Mapping
from typing import Type as _Type

from nam.train import lightning_module as _lightning_module


def resolve_lightning_module_class(
    model_config: _Mapping[str, _Any],
) -> _Type[_lightning_module.LightningModule]:
    return (
        _lightning_module.PackedLightningModule
        if model_config["net"]["name"] == "PackedWaveNet"
        else _lightning_module.LightningModule
    )
