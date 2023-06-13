# File: test_recurrent.py
# Created Date: Sunday July 17th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import onnxruntime
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

    def test_export_onnx(self):
        model = self._construct()
        with TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir, "model.onnx")
            model.export_onnx(filename)
            onnx_model = onnx.load(filename)
            session = onnxruntime.InferenceSession(str(filename))
        onnx.checker.check_model(onnx_model)
        wrapped_model = recurrent._ONNXWrapped(model)
        x = torch.Tensor([0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2])
        hin = torch.zeros((self._num_layers, self._hidden_size))
        cin = torch.zeros((self._num_layers, self._hidden_size))

        with torch.no_grad():
            y_expected, hout_expected, cout_expected = [
                z.detach().cpu().numpy() for z in wrapped_model(x, hin, cin)
            ]

        input_names = [z.name for z in session.get_inputs()]
        onnx_inputs = {
            i: z.detach().cpu().numpy() for i, z in zip(input_names, (x, hin, cin))
        }
        y_actual, hout_actual, cout_actual = session.run([], onnx_inputs)

        def approx(val):
            return pytest.approx(val, rel=1.0e-6, abs=1.0e-6)

        assert y_expected == approx(y_actual)
        assert hout_expected == approx(hout_actual)
        assert cout_expected == approx(cout_actual)
