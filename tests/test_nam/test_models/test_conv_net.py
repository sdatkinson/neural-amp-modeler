# File: test_conv_net.py
# Created Date: Friday May 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from nam.models import conv_net

from .base import Base


class TestConvNet(Base):
    @classmethod
    def setup_class(cls):
        channels = 3
        dilations = [1, 2, 4]
        return super().setup_class(
            conv_net.ConvNet,
            (channels, dilations),
            {"batchnorm": False, "activation": "Tanh"},
        )

    @pytest.mark.parametrize(
        ("batchnorm,activation"), ((False, "ReLU"), (True, "Tanh"))
    )
    def test_init(self, batchnorm, activation):
        super().test_init(kwargs={"batchnorm": batchnorm, "activation": activation})

    @pytest.mark.parametrize(
        ("batchnorm,activation"), ((False, "ReLU"), (True, "Tanh"))
    )
    def test_export(self, batchnorm, activation):
        super().test_export(kwargs={"batchnorm": batchnorm, "activation": activation})


if __name__ == "__main__":
    pytest.main()
