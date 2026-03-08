"""
Slimmable mixin abstract class
"""

import random as _random
from contextlib import contextmanager as _contextmanager

import torch.nn as _nn

SLIMMABLE_METHOD = "slice_channels_uniform"  # TODO other methods in the future


class Slimmable(_nn.Module):
    """
    Mixin for slimmable modules.
    """

    def __init__(self):
        super().__init__()
        self._slimming_value: float = 1.0
        self.set_slimming(1.0)

    def set_slimming(self, value: float):
        self._set_slimming(value=value, recurse=True)

    @_contextmanager
    def context_adjust_to_random(self):
        value = _random.uniform(0.0, 1.0)
        self.set_slimming(value=value)
        try:
            yield
        finally:
            self.set_slimming(value=1.0)

    def _set_slimming(self, value: float, recurse: bool):
        """
        Set how much the module is slimmed to.
        """
        self._slimming_value = value
        if recurse:
            for module in self.modules():
                if module is self:
                    continue
                if isinstance(module, Slimmable):
                    module._set_slimming(value=value, recurse=False)
