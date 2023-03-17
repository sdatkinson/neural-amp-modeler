# File: base.py
# Created Date: Saturday June 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
from pathlib import Path
from tempfile import TemporaryDirectory


class Base(abc.ABC):
    @classmethod
    def setup_class(cls, C, args=None, kwargs=None):
        cls._C = C
        cls._args = () if args is None else args
        cls._kwargs = {} if kwargs is None else kwargs

    def test_init(self, args=None, kwargs=None):
        obj = self._construct(args=args, kwargs=kwargs)
        assert isinstance(obj, self._C)

    def test_export(self, args=None, kwargs=None):
        model = self._construct(args=args, kwargs=kwargs)
        with TemporaryDirectory() as tmpdir:
            model.export(Path(tmpdir))

    def _construct(self, C=None, args=None, kwargs=None):
        C = self._C if C is None else C
        args = args if args is not None else self._args
        kwargs = kwargs if kwargs is not None else self._kwargs
        return C(*args, **kwargs)
