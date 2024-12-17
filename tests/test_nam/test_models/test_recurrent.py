# File: test_recurrent.py
# Created Date: Sunday July 17th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest as _pytest
import torch as _torch

from nam.models import recurrent as _recurrent

from .base import Base as _Base

_metadata_loudness_x_mocked = 0.1 * _torch.randn((11,))  # Shorter for speed


class TestLSTM(_Base):
    @classmethod
    def setup_class(cls):
        class LSTMWithMocks(_recurrent.LSTM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._get_initial_state_burn_in = 7

            @classmethod
            def _metadata_loudness_x(cls) -> _torch.Tensor:
                return _metadata_loudness_x_mocked

        num_layers = 2
        hidden_size = 3
        super().setup_class(
            LSTMWithMocks,
            args=(hidden_size,),
            kwargs={"train_burn_in": 3, "train_truncate": 5, "num_layers": num_layers},
        )
        cls._num_layers = num_layers
        cls._hidden_size = hidden_size

    @_pytest.mark.parametrize(
        "device",
        (
            "cpu",
            _pytest.param(
                "cuda",
                marks=_pytest.mark.skipif(
                    not _torch.cuda.is_available(), reason="GPU test"
                ),
            ),
        ),
    )
    def test_get_initial_state_on(self, device: str):
        model = self._construct().to(device)
        h, c = model._get_initial_state()
        assert isinstance(h, _torch.Tensor)
        assert isinstance(c, _torch.Tensor)
