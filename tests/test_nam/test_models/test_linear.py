# File: test_linear.py
# Created Date: Saturday November 23rd 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest as _pytest

from nam.models import linear as _linear

from ._convolutional import Convolutional as _Convolutional


class TestLinear(_Convolutional):
    @classmethod
    def setup_class(cls):
        C = _linear.Linear
        args = ()
        kwargs = {"receptive_field": 2, "sample_rate": 44100}
        super().setup_class(C, args, kwargs)
