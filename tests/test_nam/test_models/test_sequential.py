# File: test_sequential.py
# Created Date: [Current Date]
# Author: [Author]

import pytest as _pytest
import torch as _torch

from nam.models import linear as _linear
from nam.models import conv_net as _conv_net
from nam.models import sequential as _sequential

from .base import Base as _Base


class TestSequential(_Base):
    @classmethod
    def setup_class(cls):
        # Create a simple sequential model with two linear models
        sample_rate = 44100
        linear1 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        linear2 = _linear.Linear(receptive_field=3, sample_rate=sample_rate)
        models = [linear1, linear2]

        C = _sequential.Sequential
        args = ()
        kwargs = {"models": models}
        super().setup_class(C, args, kwargs)

    def test_receptive_field_calculation(self):
        """Test that receptive field is calculated correctly for sequential models."""
        sample_rate = 44_100
        linear1 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        linear2 = _linear.Linear(receptive_field=3, sample_rate=sample_rate)
        models = [linear1, linear2]

        seq_model = _sequential.Sequential(models=models)
        expected_rf = 1 + (2 - 1) + (3 - 1)  # 1 + 1 + 2 = 4
        assert seq_model.receptive_field == expected_rf

    def test_pad_start_default_property(self):
        """Test that pad_start_default is consistent across models."""
        sample_rate = 44_100
        linear1 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        linear2 = _linear.Linear(receptive_field=3, sample_rate=sample_rate)
        models = [linear1, linear2]

        seq_model = _sequential.Sequential(models=models)
        assert (
            seq_model.pad_start_default == True
        )  # Linear models have pad_start_default=True

    @_pytest.mark.parametrize("pad_start", [True, False])
    def test_forward_pass(self, pad_start: bool):
        """Test that forward pass works correctly through sequential models."""
        # Create models with known behavior
        sample_rate = 44_100
        linear1 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        linear2 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        models = [linear1, linear2]

        seq_model = _sequential.Sequential(models=models)

        # Test input
        x = _torch.randn(1000)  # Long enough for receptive field

        expected_samples_lost = 0 if pad_start else seq_model.receptive_field - 1
        # Forward pass
        y = seq_model(x, pad_start=pad_start)

        # Check output shape and properties
        assert isinstance(y, _torch.Tensor)
        assert x.ndim == 1
        assert y.ndim == 1
        actual_samples_lost = len(x) - len(y)
        assert actual_samples_lost == expected_samples_lost

    def test_validate_models_empty(self):
        """Test that validation fails for empty model list."""
        with _pytest.raises(ValueError, match="Sequential models must be non-empty"):
            _sequential.Sequential(models=[])

    def test_validate_models_non_basenet(self):
        """Test that validation fails for non-BaseNet models."""

        class NotBaseNet:
            def __init__(self):
                pass

        with _pytest.raises(
            ValueError, match="Sequential models must be instances of BaseNet"
        ):
            _sequential.Sequential(models=[NotBaseNet()])

    def test_validate_models_inconsistent_pad_start(self):
        """Test that validation fails for inconsistent pad_start_default."""
        # Create models with different pad_start_default
        sample_rate = 44_100
        linear1 = _linear.Linear(
            receptive_field=2, sample_rate=sample_rate
        )  # pad_start_default=True

        # Mock a model with different pad_start_default
        class MockModel(_linear.Linear):
            @property
            def pad_start_default(self):
                return False

        mock_model = MockModel(receptive_field=2, sample_rate=sample_rate)

        with _pytest.raises(
            ValueError,
            match="Sequential models must have a consistent pad_start_default",
        ):
            _sequential.Sequential(models=[linear1, mock_model])

    def test_validate_models_inconsistent_sample_rate(self):
        """Test that validation fails for inconsistent sample rates."""
        sample_rate = 44_100
        linear1 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        linear2 = _linear.Linear(receptive_field=2, sample_rate=48000)

        with _pytest.raises(
            ValueError, match="Sequential models must have a consistent sample_rate"
        ):
            _sequential.Sequential(models=[linear1, linear2])

    def test_validate_models_zero_receptive_field(self):
        """Test that validation fails for models with zero receptive field."""
        # Mock a model with zero receptive field
        sample_rate = 44_100

        class ZeroRFModel(_linear.Linear):
            @property
            def receptive_field(self):
                return 0

        zero_rf_model = ZeroRFModel(receptive_field=2, sample_rate=sample_rate)

        with _pytest.raises(
            ValueError, match="Sequential models must have a positive receptive field"
        ):
            _sequential.Sequential(models=[zero_rf_model])

    def test_single_model_sequential(self):
        """Test sequential model with a single sub-model."""
        sample_rate = 44_100
        linear = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        models = [linear]

        seq_model = _sequential.Sequential(models=models)

        assert seq_model.receptive_field == 2  # Should be same as the single model
        assert seq_model.pad_start_default == True

        # Test forward pass
        x = _torch.randn(1000)
        y = seq_model(x, pad_start=False)
        assert y.shape[0] == x.shape[0] - seq_model.receptive_field + 1

    def test_mixed_model_types(self):
        """Test sequential model with different types of models."""
        sample_rate = 44_100
        linear = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        conv = _conv_net.ConvNet(channels=4, dilations=[1, 2], sample_rate=sample_rate)
        models = [linear, conv]

        seq_model = _sequential.Sequential(models=models)

        # Test that it can be created and has reasonable properties
        assert seq_model.receptive_field > 0
        assert (
            seq_model.pad_start_default == True
        )  # Both should have pad_start_default=True

        # Test forward pass
        x = _torch.randn(2000)  # Need longer input for conv model
        y = seq_model(x, pad_start=False)
        assert isinstance(y, _torch.Tensor)
        assert y.shape[0] == x.shape[0] - seq_model.receptive_field + 1

    def test_parse_config(self):
        """Test parse_config method with model registry."""
        sample_rate = 44_100
        config = {
            "sample_rate": sample_rate,
            "models": [
                {"name": "Linear", "config": {"receptive_field": 2}},
                {"name": "Linear", "config": {"receptive_field": 3}},
            ],
        }

        parsed_config = _sequential.Sequential.parse_config(config)

        assert "sample_rate" in parsed_config
        assert parsed_config["sample_rate"] == sample_rate
        assert "models" in parsed_config
        assert len(parsed_config["models"]) == 2
        assert all(
            isinstance(model, _linear.Linear) for model in parsed_config["models"]
        )

    def test_receptive_field(self):
        """Test some receptive field arithmetic"""
        # Create models with larger receptive fields
        sample_rate = 44_100
        linear1 = _linear.Linear(receptive_field=10, sample_rate=sample_rate)
        linear2 = _linear.Linear(receptive_field=15, sample_rate=sample_rate)
        linear3 = _linear.Linear(receptive_field=20, sample_rate=sample_rate)
        models = [linear1, linear2, linear3]
        seq_model = _sequential.Sequential(models=models)
        expected_rf = 1 + (10 - 1) + (15 - 1) + (20 - 1)  # 1 + 9 + 14 + 19 = 43
        assert seq_model.receptive_field == expected_rf

        # Test with input that's just long enough
        x = _torch.randn(expected_rf)
        y = seq_model(x, pad_start=False)
        assert y.shape[0] == 1  # Should output just 1 sample

    def test_batch_processing(self):
        """Test that sequential model works with batched inputs."""
        sample_rate = 44_100
        linear1 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        linear2 = _linear.Linear(receptive_field=2, sample_rate=sample_rate)
        models = [linear1, linear2]

        seq_model = _sequential.Sequential(models=models)

        # Test batched input
        x = _torch.randn(3, 1000)  # Batch of 3, length 1000
        y = seq_model(x, pad_start=False)

        assert y.shape[0] == 3  # Batch dimension preserved
        assert y.shape[1] == x.shape[1] - seq_model.receptive_field + 1


if __name__ == "__main__":
    _pytest.main()
