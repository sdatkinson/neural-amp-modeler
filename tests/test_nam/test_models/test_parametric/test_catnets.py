# File: test_catnets.py
# Created Date: Sunday July 17th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ..base import Base

from nam.models.parametric import catnets, params


_mock_params = {
    "gain": params.ContinuousParam(0.5, 0.0, 1.0),
    "tone": params.ContinuousParam(0.5, 0.0, 1.0),
    "level": params.ContinuousParam(0.5, 0.0, 1.0),
}


class _ParametricBase(Base):
    pass


class TestCatLSTM(_ParametricBase):
    @classmethod
    def setup_class(cls):
        # Using init_from_config
        return super().setup_class(
            catnets.CatLSTM,
            args=(),
            kwargs={
                "num_layers": 1,
                "hidden_size": 2,
                "train_truncate": 11,
                "train_burn_in": 7,
                "input_size": 1 + len(_mock_params),
            },
        )

    def test_export(self, args=None, kwargs=None):
        # Override to provide params info
        model = self._construct(args=args, kwargs=kwargs)
        with TemporaryDirectory() as tmpdir:
            model.export(Path(tmpdir), _mock_params)


if __name__ == "__main__":
    pytest.main()
