# File: test_recurrent.py
# Created Date: Sunday July 17th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from nam.models import recurrent

from .base import Base

_metadata_loudness_x_mocked = 0.1 * torch.randn((11,))  # Shorter for speed


class TestLSTM(Base):
    @classmethod
    def setup_class(cls):
        class LSTMWithMocks(recurrent.LSTM):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._get_initial_state_burn_in = 7

            @classmethod
            def _metadata_loudness_x(cls) -> torch.Tensor:
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

    def test_get_initial_state_cpu(self):
        return self._t_initial_state("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
    def test_get_initial_state_gpu(self):
        self._t_initial_state("cuda")

    def _t_initial_state(self, device):
        model = self._construct().to(device)
        h, c = model._get_initial_state()
        assert isinstance(h, torch.Tensor)
        assert isinstance(c, torch.Tensor)
