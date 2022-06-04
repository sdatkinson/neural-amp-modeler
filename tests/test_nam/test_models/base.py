# File: base.py
# Created Date: Saturday June 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc


class Base(abc.ABC):
    @classmethod
    def setup_class(cls, args=None, kwargs=None):
        cls._args = () if args is None else args
        cls._kwargs = {} if kwargs is None else kwargs

    @abc.abstractmethod
    def test_init(self):
        pass

    @abc.abstractmethod
    def test_export(self):
        pass
