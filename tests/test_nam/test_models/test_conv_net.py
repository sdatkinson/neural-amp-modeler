# File: test_conv_net.py
# Created Date: Friday May 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest

from nam.models import conv_net


class TestConvNet(object):
    @pytest.mark.parametrize(
        ("batchnorm,activation"), ((False, "ReLU"), (True, "Tanh"))
    )
    def test_init(self, batchnorm, activation):
        channels = 3
        dilations = [1, 2, 4]
        conv_net.ConvNet(
            channels, dilations, batchnorm=batchnorm, activation=activation
        )
