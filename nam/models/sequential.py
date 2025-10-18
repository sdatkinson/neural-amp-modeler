"""
Sequential compositional models
"""

from typing import Sequence as _Sequence

import torch as _torch

from .base import BaseNet as _BaseNet


class Sequential(_BaseNet):
    def __init__(self, *args, models: _Sequence[_BaseNet], **kwargs):
        pad_start_default, sample_rate = self._validate_models(models)
        super().__init__(*args, sample_rate=sample_rate, **kwargs)
        self._models = models
        self._pad_start_default = pad_start_default

    @property
    def pad_start_default(self) -> bool:
        return self._pad_start_default

    @property
    def receptive_field(self) -> int:
        return 1 + sum([model.receptive_field - 1 for model in self._models])

    @classmethod
    def parse_config(cls, config):
        # Delayed import to avoid circular imports
        from .registry import init as _init_model

        config = super().parse_config(config)
        config["models"] = [
            _init_model(name=model_config["name"], kwargs=model_config["config"])
            for model_config in config.pop("models")
        ]
        return config

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        for model in self.models:
            x = model(x)
        return x

    @classmethod
    def _validate_models(cls, models: _Sequence[_BaseNet]):
        if not models:
            raise ValueError("Sequential models must be non-empty")
        if not all(isinstance(model, _BaseNet) for model in models):
            raise ValueError("Sequential models must be instances of BaseNet")
        if not all(model.receptive_field > 0 for model in models):
            raise ValueError("Sequential models must have a positive receptive field")

        # Assert that some properties are consistent:
        def assert_consistent(property_name: str):
            value = getattr(models[0], property_name)
            if not all(getattr(model, property_name) == value for model in models):
                raise ValueError(
                    f"Sequential models must have a consistent {property_name}"
                )
            return value

        pad_start_default = assert_consistent("pad_start_default")
        sample_rate = assert_consistent("sample_rate")
        return pad_start_default, sample_rate
