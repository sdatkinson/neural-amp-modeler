# File: base.py
# Created Date: Saturday June 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc as _abc
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import pytest as _pytest
import torch as _torch


class Base(_abc.ABC):
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
        with _TemporaryDirectory() as tmpdir:
            model.export(_Path(tmpdir))

    @_pytest.mark.parametrize(
        "device",
        (
            _pytest.param(
                "cuda",
                marks=_pytest.mark.skipif(
                    not _torch.cuda.is_available(), reason="CUDA-specific test"
                ),
            ),
            _pytest.param(
                "mps",
                marks=_pytest.mark.skipif(
                    not _torch.backends.mps.is_available(), reason="MPS-specific test"
                ),
            ),
        ),
    )
    def test_process_input_longer_than_65536_on(self, device: str):
        """
        Processing inputs longer than 65,536 samples using various accelerator
        backends can cause problems.

        See:
        * https://github.com/sdatkinson/neural-amp-modeler/issues/505
        * https://github.com/sdatkinson/neural-amp-modeler/issues/512

        (Funny that both have the same length limit--65,536...)

        Assert that precautions are taken.
        """
        x = _torch.zeros((65_536 + 1,)).to(device)
        model = self._construct().to(device)
        model(x)

    def _construct(self, C=None, args=None, kwargs=None):
        C = self._C if C is None else C
        args = args if args is not None else self._args
        kwargs = kwargs if kwargs is not None else self._kwargs
        return C(*args, **kwargs)
