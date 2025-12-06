"""Unit tests for custom exception hierarchy."""

import pytest

from fluxflow.exceptions import (
    CheckpointError,
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
    ConvergenceError,
    DataError,
    DataLoaderError,
    DatasetError,
    EMAError,
    FluxFlowError,
    ForwardPassError,
    GenerationError,
    GradientError,
    ImageDecodingError,
    InvalidCaptionError,
    InvalidImageError,
    IOError,
    LoadError,
    ModelArchitectureError,
    ModelConfigError,
    ModelError,
    OptimizerError,
    PromptError,
    SamplingError,
    SaveError,
    SchedulerError,
    TrainingError,
    handle_exception,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_base_exception_is_exception(self):
        """Test that FluxFlowError inherits from Exception."""
        assert issubclass(FluxFlowError, Exception)

    def test_data_exceptions_inherit_from_base(self):
        """Test that data exceptions inherit from FluxFlowError."""
        assert issubclass(DataError, FluxFlowError)
        assert issubclass(DatasetError, DataError)
        assert issubclass(DataLoaderError, DataError)
        assert issubclass(InvalidImageError, DataError)
        assert issubclass(InvalidCaptionError, DataError)

    def test_model_exceptions_inherit_from_base(self):
        """Test that model exceptions inherit from FluxFlowError."""
        assert issubclass(ModelError, FluxFlowError)
        assert issubclass(CheckpointError, ModelError)
        assert issubclass(ModelConfigError, ModelError)
        assert issubclass(ModelArchitectureError, ModelError)
        assert issubclass(ForwardPassError, ModelError)

    def test_training_exceptions_inherit_from_base(self):
        """Test that training exceptions inherit from FluxFlowError."""
        assert issubclass(TrainingError, FluxFlowError)
        assert issubclass(OptimizerError, TrainingError)
        assert issubclass(SchedulerError, TrainingError)
        assert issubclass(ConvergenceError, TrainingError)
        assert issubclass(GradientError, TrainingError)
        assert issubclass(EMAError, TrainingError)

    def test_generation_exceptions_inherit_from_base(self):
        """Test that generation exceptions inherit from FluxFlowError."""
        assert issubclass(GenerationError, FluxFlowError)
        assert issubclass(PromptError, GenerationError)
        assert issubclass(SamplingError, GenerationError)
        assert issubclass(ImageDecodingError, GenerationError)

    def test_config_exceptions_inherit_from_base(self):
        """Test that config exceptions inherit from FluxFlowError."""
        assert issubclass(ConfigError, FluxFlowError)
        assert issubclass(ConfigFileError, ConfigError)
        assert issubclass(ConfigValidationError, ConfigError)

    def test_io_exceptions_inherit_from_base(self):
        """Test that I/O exceptions inherit from FluxFlowError."""
        assert issubclass(IOError, FluxFlowError)
        assert issubclass(SaveError, IOError)
        assert issubclass(LoadError, IOError)


class TestExceptionRaising:
    """Tests for raising and catching exceptions."""

    def test_raise_base_exception(self):
        """Test raising the base FluxFlowError."""
        with pytest.raises(FluxFlowError) as exc_info:
            raise FluxFlowError("Base error")
        assert str(exc_info.value) == "Base error"

    def test_raise_dataset_error(self):
        """Test raising DatasetError."""
        with pytest.raises(DatasetError) as exc_info:
            raise DatasetError("Dataset not found")
        assert str(exc_info.value) == "Dataset not found"

    def test_raise_checkpoint_error(self):
        """Test raising CheckpointError."""
        with pytest.raises(CheckpointError) as exc_info:
            raise CheckpointError("Checkpoint corrupted")
        assert str(exc_info.value) == "Checkpoint corrupted"

    def test_raise_optimizer_error(self):
        """Test raising OptimizerError."""
        with pytest.raises(OptimizerError) as exc_info:
            raise OptimizerError("Invalid learning rate")
        assert str(exc_info.value) == "Invalid learning rate"

    def test_raise_prompt_error(self):
        """Test raising PromptError."""
        with pytest.raises(PromptError) as exc_info:
            raise PromptError("Prompt too long")
        assert str(exc_info.value) == "Prompt too long"

    def test_raise_config_error(self):
        """Test raising ConfigFileError."""
        with pytest.raises(ConfigFileError) as exc_info:
            raise ConfigFileError("Missing config field")
        assert str(exc_info.value) == "Missing config field"


class TestExceptionCatching:
    """Tests for catching exceptions at different hierarchy levels."""

    def test_catch_specific_exception(self):
        """Test catching a specific exception type."""
        with pytest.raises(DatasetError):
            raise DatasetError("Specific error")

    def test_catch_parent_exception(self):
        """Test catching an exception using its parent class."""
        with pytest.raises(DataError):
            raise DatasetError("Dataset error")

    def test_catch_base_exception(self):
        """Test catching any FluxFlow exception using base class."""
        with pytest.raises(FluxFlowError):
            raise CheckpointError("Checkpoint error")

    def test_multiple_exception_types(self):
        """Test that different exception types can be distinguished."""
        exceptions = [
            DatasetError("data"),
            CheckpointError("checkpoint"),
            OptimizerError("optimizer"),
            PromptError("prompt"),
        ]

        for exc in exceptions:
            assert isinstance(exc, FluxFlowError)
            with pytest.raises(type(exc)):
                raise exc


class TestExceptionHandler:
    """Tests for the handle_exception utility function."""

    def test_handle_exception_without_logger(self):
        """Test handle_exception without a logger."""
        exc = FluxFlowError("Test error")
        with pytest.raises(FluxFlowError):
            handle_exception(exc, logger=None, reraise=True)

    def test_handle_exception_no_reraise(self):
        """Test handle_exception without re-raising."""
        exc = FluxFlowError("Test error")
        # Should not raise
        handle_exception(exc, logger=None, reraise=False)

    def test_handle_exception_with_mock_logger(self):
        """Test handle_exception with a mock logger."""
        from unittest.mock import MagicMock

        logger = MagicMock()
        exc = DatasetError("Dataset error")

        with pytest.raises(DatasetError):
            handle_exception(exc, logger=logger, reraise=True)

        # Verify logger was called
        logger.error.assert_called_once()
        call_args = logger.error.call_args[0][0]
        assert "DatasetError" in call_args
        assert "Dataset error" in call_args

    def test_handle_non_fluxflow_exception(self):
        """Test handle_exception with a non-FluxFlow exception."""
        from unittest.mock import MagicMock

        logger = MagicMock()
        exc = ValueError("Standard Python error")

        with pytest.raises(ValueError):
            handle_exception(exc, logger=logger, reraise=True)

        # Verify logger was called with "Unexpected error"
        logger.error.assert_called_once()
        call_args = logger.error.call_args[0][0]
        assert "Unexpected error" in call_args


class TestExceptionMessages:
    """Tests for exception message handling."""

    def test_exception_with_simple_message(self):
        """Test exception with a simple string message."""
        exc = DatasetError("Simple message")
        assert str(exc) == "Simple message"

    def test_exception_with_formatted_message(self):
        """Test exception with formatted message."""
        path = "/path/to/data"
        exc = DatasetError(f"Dataset not found at: {path}")
        assert path in str(exc)

    def test_exception_with_multiple_args(self):
        """Test exception with multiple arguments."""
        exc = CheckpointError("Error", "Additional info")
        # Python Exception handles multiple args as tuple
        assert "Error" in str(exc)

    def test_exception_repr(self):
        """Test exception repr contains class name."""
        exc = ModelConfigError("Invalid config")
        repr_str = repr(exc)
        assert "ModelConfigError" in repr_str


class TestExceptionUsageExamples:
    """Tests demonstrating proper exception usage patterns."""

    def test_data_pipeline_error_handling(self):
        """Example: Error handling in data pipeline."""

        def load_dataset(path):
            """Simulate dataset loading."""
            if not path:
                raise DatasetError("Dataset path cannot be empty")
            if path == "/invalid":
                raise InvalidImageError("Image format not supported")
            return "dataset"

        # Test successful case
        assert load_dataset("/valid/path") == "dataset"

        # Test dataset error
        with pytest.raises(DatasetError):
            load_dataset("")

        # Test image error
        with pytest.raises(InvalidImageError):
            load_dataset("/invalid")

        # Catch all data errors
        try:
            load_dataset("/invalid")
        except DataError as e:
            assert isinstance(e, InvalidImageError)

    def test_training_error_handling(self):
        """Example: Error handling in training."""

        def check_loss(loss_value):
            """Simulate loss checking."""
            if loss_value != loss_value:  # NaN check
                raise ConvergenceError("NaN detected in loss")
            if loss_value == float("inf"):
                raise ConvergenceError("Inf detected in loss")
            return True

        # Test successful case
        assert check_loss(0.5) is True

        # Test NaN detection
        with pytest.raises(ConvergenceError) as exc_info:
            check_loss(float("nan"))
        assert "NaN" in str(exc_info.value)

        # Test Inf detection
        with pytest.raises(ConvergenceError) as exc_info:
            check_loss(float("inf"))
        assert "Inf" in str(exc_info.value)

    def test_checkpoint_error_handling(self):
        """Example: Error handling in checkpoint loading."""

        def load_checkpoint(path):
            """Simulate checkpoint loading."""
            if not path.endswith(".safetensors"):
                raise CheckpointError("Only .safetensors format supported")
            if "corrupted" in path:
                raise CheckpointError("Checkpoint file is corrupted")
            return {"model": "weights"}

        # Test successful case
        assert "model" in load_checkpoint("model.safetensors")

        # Test format error
        with pytest.raises(CheckpointError) as exc_info:
            load_checkpoint("model.pth")
        assert "safetensors" in str(exc_info.value)

        # Test corruption error
        with pytest.raises(CheckpointError):
            load_checkpoint("corrupted.safetensors")
