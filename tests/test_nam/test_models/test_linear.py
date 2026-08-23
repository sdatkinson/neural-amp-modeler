# File: test_linear.py
# Created Date: Saturday November 23rd 2024
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest as _pytest
import torch as _torch

from nam.models import linear as _linear

from .base import Base as _Base


class TestLinear(_Base):
    @classmethod
    def setup_class(cls):
        C = _linear.Linear
        args = ()
        kwargs = {"receptive_field": 2, "sample_rate": 44100}
        super().setup_class(C, args, kwargs)

    def test_export_weights_are_chronological(self):
        model = _linear.Linear(receptive_field=3)
        model._net.weight.data.copy_(_torch.tensor([[[0.5, -0.25, 0.125]]]))

        exported_weights = model._export_weights()
        impulse_response = model(_torch.tensor([1.0, 0.0, 0.0]))

        _torch.testing.assert_close(
            _torch.from_numpy(exported_weights), impulse_response
        )
        _torch.testing.assert_close(
            _torch.from_numpy(exported_weights),
            _torch.tensor([0.125, -0.25, 0.5]),
        )

    def test_import_weights_are_chronological(self):
        weights = _torch.tensor([0.5, -0.25, 0.125])
        model = _linear.Linear(receptive_field=3)

        model.import_weights(weights)

        _torch.testing.assert_close(
            model._net.weight.data, _torch.tensor([[[0.125, -0.25, 0.5]]])
        )
        _torch.testing.assert_close(
            model(_torch.tensor([1.0, 0.0, 0.0])), weights
        )

    def test_import_export_weights_round_trip_with_bias(self):
        model = _linear.Linear(receptive_field=3, bias=True)
        model._net.weight.data.copy_(_torch.tensor([[[0.5, -0.25, 0.125]]]))
        model._net.bias.data.copy_(_torch.tensor([0.75]))
        exported_weights = model._export_weights()
        model2 = _linear.Linear(receptive_field=3, bias=True)

        model2.import_weights(exported_weights)

        _torch.testing.assert_close(
            _torch.from_numpy(exported_weights),
            _torch.tensor([0.125, -0.25, 0.5, 0.75]),
        )
        _torch.testing.assert_close(model2._net.weight, model._net.weight)
        _torch.testing.assert_close(model2._net.bias, model._net.bias)
        _torch.testing.assert_close(
            _torch.from_numpy(model2._export_weights()),
            _torch.from_numpy(exported_weights),
        )
