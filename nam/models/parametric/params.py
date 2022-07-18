# File: params.py
# Created Date: Sunday July 17th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Handling parametric inputs
"""

import abc
import inspect
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

from ..._core import InitializableFromConfig


# class ParamType(Enum):
#     CONTINUOUS = "continuous"
#     BOOLEAN = "boolean"


@dataclass
class Param(InitializableFromConfig):
    default_value: Any

    @classmethod
    def init_from_config(cls, config):
        C, kwargs = cls.parse_config(config)
        return C(**kwargs)

    @classmethod
    def parse_config(cls, config):
        for C in [
            _C
            for _C in globals().values()
            if inspect.isclass(_C) and _C is not Param and issubclass(_C, Param)
        ]:
            if C.typestr() == config["type"]:
                config.pop("type")
                break
        else:
            raise ValueError(f"Unrecognized aprameter type {config['type']}")
        return C, config

    @abc.abstractclassmethod
    def typestr(cls) -> str:
        pass

    def to_json(self):
        return {
            "type": self.typestr(),
            **{f.name: getattr(self, f.name) for f in fields(self)},
        }


@dataclass
class BooleanParam(Param):
    @classmethod
    def typestr(cls) -> str:
        return "boolean"


@dataclass
class ContinuousParam(Param):
    minval: float
    maxval: float

    @classmethod
    def typestr(self) -> str:
        return "continuous"
