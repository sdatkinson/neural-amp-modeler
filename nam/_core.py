# File: core.py
# Created Date: Saturday February 5th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from copy import deepcopy


class InitializableFromConfig(object):
    @classmethod
    def init_from_config(cls, config):
        return cls(**cls.parse_config(config))

    @classmethod
    def parse_config(cls, config):
        return deepcopy(config)
