# File: test_registry.py
# Created Date: [Current Date]
# Author: [Author]

import copy as _copy
import contextlib as _contextlib
import pytest as _pytest

from nam.models import registry as _registry
from nam.models import linear as _linear
from nam.models import conv_net as _conv_net
from nam.models import sequential as _sequential
from nam.models import recurrent as _recurrent
from nam.models import wavenet as _wavenet


@_contextlib.contextmanager
def _registry_backup():
    """
    Context manager to safely backup and restore the registry state.

    This context manager creates a deep copy of the registry before running
    the wrapped test, and ensures that the original registry is restored
    even if the test fails or raises an exception.

    Usage:
        with _registry_backup():
            # Test code that modifies the registry
            _registry.register("TestModel", some_constructor)
            # ... test logic ...
    """
    # Get the original registry state
    original_registry = _copy.deepcopy(_registry._model_net_init_registry)

    try:
        yield
    except Exception as e:
        # Re-raise any exceptions that occurred during the test
        raise e
    finally:
        # Always restore the original registry state
        _registry._model_net_init_registry.clear()
        _registry._model_net_init_registry.update(original_registry)


class TestRegistry:
    """Test the model registry functionality."""

    @_pytest.mark.parametrize(
        "model_name,config,expected_type",
        [
            ("Linear", {"receptive_field": 2, "sample_rate": 44100}, _linear.Linear),
            (
                "ConvNet",
                {"channels": 4, "dilations": [1, 2], "sample_rate": 44100},
                _conv_net.ConvNet,
            ),
            ("LSTM", {"hidden_size": 10, "sample_rate": 44100}, _recurrent.LSTM),
            (
                "WaveNet",
                {
                    "layers_configs": [
                        {
                            "condition_size": 1,
                            "input_size": 1,
                            "channels": 4,
                            "head_size": 2,
                            "kernel_size": 3,
                            "dilations": [1, 2],
                            "activation": "Tanh",
                            "gated": False,
                            "head_bias": False,
                        }
                    ],
                    "head_scale": 0.02,
                    "sample_rate": 44100,
                },
                _wavenet.WaveNet,
            ),
            (
                "Sequential",
                {
                    "models": [
                        {
                            "name": "Linear",
                            "config": {"receptive_field": 2, "sample_rate": 44100},
                        },
                        {
                            "name": "Linear",
                            "config": {"receptive_field": 3, "sample_rate": 44100},
                        },
                    ]
                },
                _sequential.Sequential,
            ),
        ],
    )
    def test_default_registry_contents(self, model_name, config, expected_type):
        """Test that all expected models are registered by default."""
        model = _registry.init(model_name, kwargs={"config": config})
        assert isinstance(model, expected_type)

    def test_register_successful(self):
        """Test successful registration of a new model."""
        with _registry_backup():
            # Create a mock constructor
            def mock_constructor(*args, **kwargs):
                return _linear.Linear(receptive_field=2, sample_rate=44100)

            # Register the mock constructor
            _registry.register("MockModel", mock_constructor)

            # Test that we can initialize it
            model = _registry.init("MockModel")
            assert isinstance(model, _linear.Linear)

    def test_register_duplicate_without_overwrite(self):
        """Test that registering a duplicate name without overwrite raises KeyError."""
        with _registry_backup():
            with _pytest.raises(
                KeyError,
                match="A constructor for net name 'Linear' is already registered!",
            ):
                _registry.register(
                    "Linear", _linear.Linear.init_from_config, overwrite=False
                )

    def test_register_duplicate_with_overwrite(self):
        """Test that registering a duplicate name with overwrite=True succeeds."""
        with _registry_backup():
            # Create a mock constructor
            def mock_constructor(*args, **kwargs):
                return _linear.Linear(receptive_field=2, sample_rate=44100)

            # This should not raise an error
            _registry.register("Linear", mock_constructor, overwrite=True)

            # Test that the new constructor is used
            model = _registry.init("Linear")
            assert isinstance(model, _linear.Linear)

    def test_init_with_args_and_kwargs(self):
        """Test init function with both args and kwargs."""
        sample_rate = 44100

        # Test with kwargs only
        model1 = _registry.init(
            "Linear",
            kwargs={"config": {"receptive_field": 2, "sample_rate": sample_rate}},
        )
        assert isinstance(model1, _linear.Linear)

        # Test with args only (empty kwargs)
        model2 = _registry.init(
            "Linear", args=({"receptive_field": 2, "sample_rate": sample_rate},)
        )
        assert isinstance(model2, _linear.Linear)

        # Test with both args and kwargs
        model3 = _registry.init(
            "Linear",
            args=(),
            kwargs={"config": {"receptive_field": 2, "sample_rate": sample_rate}},
        )
        assert isinstance(model3, _linear.Linear)

    def test_init_with_none_args_kwargs(self):
        """Test init function with None args and kwargs."""
        sample_rate = 44100

        # Test with None args and kwargs
        model = _registry.init(
            "Linear",
            args=None,
            kwargs={"config": {"receptive_field": 2, "sample_rate": sample_rate}},
        )
        assert isinstance(model, _linear.Linear)

    def test_init_import_based_fallback(self):
        """Test import-based initialization fallback."""
        sample_rate = 44100

        # Test with a fully qualified module name
        model = _registry.init(
            "nam.models.linear.Linear",
            kwargs={"receptive_field": 2, "sample_rate": sample_rate},
        )
        assert isinstance(model, _linear.Linear)

    def test_init_import_based_fallback_missing_module(self):
        """Test import-based initialization with missing module."""
        with _pytest.raises(
            KeyError,
            match="No importable module found for name 'nonexistent.module.Factory'",
        ):
            _registry.init("nonexistent.module.Factory")

    def test_init_import_based_fallback_missing_factory(self):
        """Test import-based initialization with missing factory."""
        with _pytest.raises(
            KeyError,
            match="No factory found for name 'nam.models.linear.NonExistentFactory' within module 'nam.models.linear'",
        ):
            _registry.init("nam.models.linear.NonExistentFactory")

    def test_init_unknown_model(self):
        """Test init function with unknown model name."""
        with _pytest.raises(ValueError, match="not enough values to unpack"):
            _registry.init("UnknownModel")

    def test_init_with_invalid_args(self):
        """Test init function with invalid arguments."""
        # Test with invalid kwargs that should cause the model constructor to fail
        with _pytest.raises(Exception):  # The specific exception depends on the model
            _registry.init("Linear", kwargs={"invalid_param": "invalid_value"})

    def test_multiple_registrations(self):
        """Test registering multiple models."""
        with _registry_backup():
            # Create mock constructors
            def mock_constructor1(*args, **kwargs):
                return _linear.Linear(receptive_field=2, sample_rate=44100)

            def mock_constructor2(*args, **kwargs):
                return _conv_net.ConvNet(
                    channels=4, dilations=[1, 2], sample_rate=44100
                )

            # Register both
            _registry.register("TestModel1", mock_constructor1)
            _registry.register("TestModel2", mock_constructor2)

            # Test both
            model1 = _registry.init("TestModel1")
            model2 = _registry.init("TestModel2")

            assert isinstance(model1, _linear.Linear)
            assert isinstance(model2, _conv_net.ConvNet)

    def test_registry_with_complex_models(self):
        """Test registry with more complex model configurations."""
        sample_rate = 44100

        # Test Sequential model with multiple sub-models
        seq_config = {
            "models": [
                {
                    "name": "Linear",
                    "config": {"receptive_field": 2, "sample_rate": sample_rate},
                },
                {
                    "name": "Linear",
                    "config": {"receptive_field": 3, "sample_rate": sample_rate},
                },
            ]
        }
        seq_model = _registry.init("Sequential", kwargs={"config": seq_config})
        assert isinstance(seq_model, _sequential.Sequential)
        assert seq_model.receptive_field == 4  # 1 + (2-1) + (3-1) = 4

        # Test ConvNet with specific configuration
        conv_model = _registry.init(
            "ConvNet",
            kwargs={
                "config": {
                    "channels": 8,
                    "dilations": [1, 2, 4, 8],
                    "sample_rate": sample_rate,
                }
            },
        )
        assert isinstance(conv_model, _conv_net.ConvNet)

    def test_registry_error_handling(self):
        """Test various error conditions in the registry."""
        # Test with empty string name
        with _pytest.raises(ValueError, match="not enough values to unpack"):
            _registry.init("")

        # Test with None name
        with _pytest.raises(AttributeError):
            _registry.init(None)

    def test_registry_backup_handles_exceptions(self):
        """
        Meta-test that the registry backup context manager properly handles exceptions.
        """
        # Store the original registry state
        original_registry = _copy.deepcopy(_registry._model_net_init_registry)

        class TestException(Exception):
            pass

        try:
            with _registry_backup():
                # Register a test model
                def test_constructor(*args, **kwargs):
                    return _linear.Linear(receptive_field=1, sample_rate=44100)

                _registry.register("TestExceptionModel", test_constructor)

                # Verify it's registered
                assert "TestExceptionModel" in _registry._model_net_init_registry

                # Intentionally raise an exception
                raise TestException("Test exception")
        except TestException as e:
            # Verify the exception was re-raised
            assert str(e) == "Test exception"

        # Verify the registry was restored to its original state
        assert _registry._model_net_init_registry == original_registry
        assert "TestExceptionModel" not in _registry._model_net_init_registry


if __name__ == "__main__":
    _pytest.main()
