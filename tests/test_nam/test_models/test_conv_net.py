# File: test_conv_net.py
# Created Date: Friday May 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest as _pytest

from nam.models import conv_net

from .base import Base as _Base


class TestConvNet(_Base):
    @classmethod
    def setup_class(cls):
        channels = 3
        dilations = [1, 2, 4]
        return super().setup_class(
            conv_net.ConvNet,
            (channels, dilations),
            {"batchnorm": False, "activation": "Tanh"},
        )

    @_pytest.mark.parametrize(
        ("batchnorm,activation"), ((False, "ReLU"), (True, "Tanh"))
    )
    def test_init(self, batchnorm, activation):
        super().test_init(kwargs={"batchnorm": batchnorm, "activation": activation})

    @_pytest.mark.parametrize(
        ("batchnorm,activation"), ((False, "ReLU"), (True, "Tanh"))
    )
    def test_export(self, batchnorm, activation):
        super().test_export(kwargs={"batchnorm": batchnorm, "activation": activation})


if __name__ == "__main__":
    _pytest.main()
