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
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=3, sample_rate=44100)
        models = [linear1, linear2]
        
        C = _sequential.Sequential
        args = ()
        kwargs = {"models": models, "sample_rate": 44100}
        super().setup_class(C, args, kwargs)

    def test_receptive_field_calculation(self):
        """Test that receptive field is calculated correctly for sequential models."""
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=3, sample_rate=44100)
        models = [linear1, linear2]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        expected_rf = 1 + (2 - 1) + (3 - 1)  # 1 + 1 + 2 = 4
        assert seq_model.receptive_field == expected_rf

    def test_pad_start_default_property(self):
        """Test that pad_start_default is consistent across models."""
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=3, sample_rate=44100)
        models = [linear1, linear2]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        assert seq_model.pad_start_default == True  # Linear models have pad_start_default=True

    def test_forward_pass(self):
        """Test that forward pass works correctly through sequential models."""
        # Create models with known behavior
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=2, sample_rate=44100)
        models = [linear1, linear2]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        
        # Test input
        x = _torch.randn(1000)  # Long enough for receptive field
        
        # Forward pass
        y = seq_model(x)
        
        # Check output shape and properties
        assert isinstance(y, _torch.Tensor)
        assert y.shape[0] == x.shape[0] - seq_model.receptive_field + 1

    def test_validate_models_empty(self):
        """Test that validation fails for empty model list."""
        with _pytest.raises(ValueError, match="Sequential models must be non-empty"):
            _sequential.Sequential(models=[], sample_rate=44100)

    def test_validate_models_non_basenet(self):
        """Test that validation fails for non-BaseNet models."""
        class NotBaseNet:
            def __init__(self):
                pass
        
        with _pytest.raises(ValueError, match="Sequential models must be instances of BaseNet"):
            _sequential.Sequential(models=[NotBaseNet()], sample_rate=44100)

    def test_validate_models_inconsistent_pad_start(self):
        """Test that validation fails for inconsistent pad_start_default."""
        # Create models with different pad_start_default
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)  # pad_start_default=True
        
        # Mock a model with different pad_start_default
        class MockModel(_linear.Linear):
            @property
            def pad_start_default(self):
                return False
        
        mock_model = MockModel(receptive_field=2, sample_rate=44100)
        
        with _pytest.raises(ValueError, match="Sequential models must have a consistent pad_start_default"):
            _sequential.Sequential(models=[linear1, mock_model], sample_rate=44100)

    def test_validate_models_inconsistent_sample_rate(self):
        """Test that validation fails for inconsistent sample rates."""
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=2, sample_rate=48000)
        
        with _pytest.raises(ValueError, match="Sequential models must have a consistent sample_rate"):
            _sequential.Sequential(models=[linear1, linear2], sample_rate=44100)

    def test_validate_models_zero_receptive_field(self):
        """Test that validation fails for models with zero receptive field."""
        # Mock a model with zero receptive field
        class ZeroRFModel(_linear.Linear):
            @property
            def receptive_field(self):
                return 0
        
        zero_rf_model = ZeroRFModel(receptive_field=2, sample_rate=44100)
        
        with _pytest.raises(ValueError, match="Sequential models must have a positive receptive field"):
            _sequential.Sequential(models=[zero_rf_model], sample_rate=44100)

    def test_single_model_sequential(self):
        """Test sequential model with a single sub-model."""
        linear = _linear.Linear(receptive_field=2, sample_rate=44100)
        models = [linear]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        
        assert seq_model.receptive_field == 2  # Should be same as the single model
        assert seq_model.pad_start_default == True
        
        # Test forward pass
        x = _torch.randn(1000)
        y = seq_model(x)
        assert y.shape[0] == x.shape[0] - seq_model.receptive_field + 1

    def test_mixed_model_types(self):
        """Test sequential model with different types of models."""
        linear = _linear.Linear(receptive_field=2, sample_rate=44100)
        conv = _conv_net.ConvNet(channels=4, dilations=[1, 2], sample_rate=44100)
        models = [linear, conv]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        
        # Test that it can be created and has reasonable properties
        assert seq_model.receptive_field > 0
        assert seq_model.pad_start_default == True  # Both should have pad_start_default=True
        
        # Test forward pass
        x = _torch.randn(2000)  # Need longer input for conv model
        y = seq_model(x)
        assert isinstance(y, _torch.Tensor)
        assert y.shape[0] == x.shape[0] - seq_model.receptive_field + 1

    def test_parse_config(self):
        """Test parse_config method with model registry."""
        config = {
            "sample_rate": 44100,
            "models": [
                {"name": "Linear", "config": {"receptive_field": 2}},
                {"name": "Linear", "config": {"receptive_field": 3}}
            ]
        }
        
        parsed_config = _sequential.Sequential.parse_config(config)
        
        assert "sample_rate" in parsed_config
        assert parsed_config["sample_rate"] == 44100
        assert "models" in parsed_config
        assert len(parsed_config["models"]) == 2
        assert all(isinstance(model, _linear.Linear) for model in parsed_config["models"])

    def test_models_property(self):
        """Test that the models property is accessible."""
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=3, sample_rate=44100)
        models = [linear1, linear2]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        
        # Check that we can access the models (though it's a private attribute)
        assert hasattr(seq_model, '_models')
        assert len(seq_model._models) == 2
        assert seq_model._models[0] is linear1
        assert seq_model._models[1] is linear2

    def test_large_receptive_field(self):
        """Test sequential model with large receptive field."""
        # Create models with larger receptive fields
        linear1 = _linear.Linear(receptive_field=10, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=15, sample_rate=44100)
        linear3 = _linear.Linear(receptive_field=20, sample_rate=44100)
        models = [linear1, linear2, linear3]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        
        expected_rf = 1 + (10 - 1) + (15 - 1) + (20 - 1)  # 1 + 9 + 14 + 19 = 43
        assert seq_model.receptive_field == expected_rf
        
        # Test with input that's just long enough
        x = _torch.randn(expected_rf)
        y = seq_model(x)
        assert y.shape[0] == 1  # Should output just 1 sample

    def test_batch_processing(self):
        """Test that sequential model works with batched inputs."""
        linear1 = _linear.Linear(receptive_field=2, sample_rate=44100)
        linear2 = _linear.Linear(receptive_field=2, sample_rate=44100)
        models = [linear1, linear2]
        
        seq_model = _sequential.Sequential(models=models, sample_rate=44100)
        
        # Test batched input
        x = _torch.randn(3, 1000)  # Batch of 3, length 1000
        y = seq_model(x)
        
        assert y.shape[0] == 3  # Batch dimension preserved
        assert y.shape[1] == x.shape[1] - seq_model.receptive_field + 1


if __name__ == "__main__":
    _pytest.main()
