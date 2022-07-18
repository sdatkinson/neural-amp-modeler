# File: test_hyper_net.py
# Created Date: Saturday June 4th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from nam.models.parametric import hyper_net

from ..base import Base


class TestHyperConvNet(Base):
    @classmethod
    def setup_class(cls):
        return super().setup_class(hyper_net.HyperConvNet, (), {})

    @pytest.mark.parametrize(
        ("batchnorm,activation"), ((False, "ReLU"), (True, "Tanh"))
    )
    def test_init(self, batchnorm, activation):
        # TODO refactor
        channels = 3
        dilations = [1, 2, 4]
        assert isinstance(
            self._construct(
                self._config(
                    batchnorm=batchnorm,
                    activation=activation,
                    dilations=dilations,
                    channels=channels,
                )
            ),
            hyper_net.HyperConvNet,
        )

    @pytest.mark.parametrize(
        ("batchnorm,activation"), ((False, "ReLU"), (True, "Tanh"))
    )
    def test_export(self, batchnorm, activation):
        # TODO refactor
        channels = 3
        dilations = [1, 2, 4]
        model = self._construct(
            self._config(
                batchnorm=batchnorm,
                activation=activation,
                dilations=dilations,
                channels=channels,
            )
        )
        with TemporaryDirectory() as tmpdir:
            model.export(Path(tmpdir))

    def test_export_cpp_header(self):
        # TODO refactor
        with TemporaryDirectory() as tmpdir:
            self._construct().export_cpp_header(Path(tmpdir, "model.h"))

    def _config(self, batchnorm=True, activation="Tanh", dilations=None, channels=7):
        dilations = [1, 2, 4] if dilations is None else dilations
        return {
            "net": {
                "channels": channels,
                "dilations": dilations,
                "batchnorm": batchnorm,
                "activation": activation,
            },
            "hyper_net": {
                "num_inputs": 3,
                "num_layers": 2,
                "num_units": 11,
                "batchnorm": True,
            },
        }

    def _construct(self, config=None):
        # Override for simplicity...
        config = self._config() if config is None else config
        return self._C.init_from_config(config)


if __name__ == "__main__":
    pytest.main()
