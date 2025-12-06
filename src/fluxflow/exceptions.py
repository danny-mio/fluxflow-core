"""Custom exception hierarchy for FluxFlow.

This module defines a comprehensive exception hierarchy for better error handling
and debugging throughout the FluxFlow codebase.
"""


class FluxFlowError(Exception):
    """Base exception for all FluxFlow-specific errors.

    All custom exceptions in FluxFlow inherit from this base class,
    making it easy to catch all FluxFlow-related errors.

    Example:
        >>> try:
        ...     # FluxFlow code
        ...     pass
        ... except FluxFlowError as e:
        ...     logger.error(f"FluxFlow error: {e}")
    """


# ==================== Data Exceptions ====================


class DataError(FluxFlowError):
    """Base exception for data-related errors."""


class DatasetError(DataError):
    """Error related to dataset initialization or access.

    Raised when there are issues with dataset setup, file access,
    or dataset corruption.

    Example:
        >>> raise DatasetError("Dataset directory not found: /path/to/data")
    """


class DataLoaderError(DataError):
    """Error related to DataLoader configuration or operation.

    Raised when there are issues with DataLoader setup, batching,
    or sampling.

    Example:
        >>> raise DataLoaderError("Invalid batch size: must be positive integer")
    """


class InvalidImageError(DataError):
    """Error for invalid or corrupted image data.

    Raised when an image cannot be loaded, decoded, or has invalid dimensions.

    Example:
        >>> raise InvalidImageError(f"Image dimensions too small: {width}x{height}")
    """


class InvalidCaptionError(DataError):
    """Error for invalid or missing caption data.

    Raised when caption files are malformed or captions are missing.

    Example:
        >>> raise InvalidCaptionError("Caption file missing required columns")
    """


# ==================== Model Exceptions ====================


class ModelError(FluxFlowError):
    """Base exception for model-related errors."""


class CheckpointError(ModelError):
    """Error related to checkpoint loading or saving.

    Raised when checkpoints are corrupted, missing required keys,
    or cannot be loaded/saved.

    Example:
        >>> raise CheckpointError("Checkpoint missing required key: 'diffuser.compressor'")
    """


class ModelConfigError(ModelError):
    """Error for invalid model configuration.

    Raised when model parameters are invalid or incompatible.

    Example:
        >>> raise ModelConfigError("VAE dimension must be divisible by 8")
    """


class ModelArchitectureError(ModelError):
    """Error for model architecture mismatches.

    Raised when trying to load weights into a model with different architecture.

    Example:
        >>> raise ModelArchitectureError("Layer count mismatch: expected 12, got 6")
    """


class ForwardPassError(ModelError):
    """Error during model forward pass.

    Raised when shape mismatches or other issues occur during forward pass.

    Example:
        >>> raise ForwardPassError(f"Shape mismatch: expected {expected}, got {actual}")
    """


# ==================== Training Exceptions ====================


class TrainingError(FluxFlowError):
    """Base exception for training-related errors."""


class OptimizerError(TrainingError):
    """Error related to optimizer setup or configuration.

    Raised when optimizer parameters are invalid or optimizer fails.

    Example:
        >>> raise OptimizerError("Invalid learning rate: must be positive")
    """


class SchedulerError(TrainingError):
    """Error related to learning rate scheduler setup or operation.

    Raised when scheduler configuration is invalid.

    Example:
        >>> raise SchedulerError("Invalid scheduler type: 'InvalidScheduler'")
    """


class ConvergenceError(TrainingError):
    """Error indicating training convergence issues.

    Raised when NaN or Inf values are detected in loss or gradients.

    Example:
        >>> raise ConvergenceError(f"NaN detected in loss at step {step}")
    """


class GradientError(TrainingError):
    """Error related to gradient computation or clipping.

    Raised when gradients are invalid or gradient operations fail.

    Example:
        >>> raise GradientError("Gradient norm is NaN")
    """


class EMAError(TrainingError):
    """Error related to Exponential Moving Average operations.

    Raised when EMA initialization or update fails.

    Example:
        >>> raise EMAError("EMA model mismatch with source model")
    """


# ==================== Generation Exceptions ====================


class GenerationError(FluxFlowError):
    """Base exception for generation pipeline errors."""


class PromptError(GenerationError):
    """Error for invalid or problematic prompts.

    Raised when prompts are too long, empty, or contain invalid tokens.

    Example:
        >>> raise PromptError("Prompt exceeds maximum length of 77 tokens")
    """


class SamplingError(GenerationError):
    """Error during sampling or denoising process.

    Raised when sampling fails or produces invalid results.

    Example:
        >>> raise SamplingError(f"Sampling failed at timestep {t}")
    """


class ImageDecodingError(GenerationError):
    """Error during latent to image decoding.

    Raised when VAE decoder produces invalid output.

    Example:
        >>> raise ImageDecodingError("Decoded image has invalid pixel values")
    """


# ==================== Configuration Exceptions ====================


class ConfigError(FluxFlowError):
    """Base exception for configuration-related errors."""


class ConfigFileError(ConfigError):
    """Error related to configuration file parsing.

    Raised when config files are malformed or missing required fields.

    Example:
        >>> raise ConfigFileError("Missing required field: 'model.vae_dim'")
    """


class ConfigValidationError(ConfigError):
    """Error during configuration validation.

    Raised when config values fail validation checks.

    Example:
        >>> raise ConfigValidationError("Batch size must be at least 1")
    """


# ==================== I/O Exceptions ====================


class IOError(FluxFlowError):
    """Base exception for input/output errors."""


class SaveError(IOError):
    """Error during file save operations.

    Raised when files cannot be written to disk.

    Example:
        >>> raise SaveError(f"Failed to save checkpoint: {path}")
    """


class LoadError(IOError):
    """Error during file load operations.

    Raised when files cannot be read from disk.

    Example:
        >>> raise LoadError(f"Failed to load model: {path}")
    """


# ==================== Utility Functions ====================


def handle_exception(exc: Exception, logger=None, reraise: bool = True):
    """Handle exceptions with optional logging.

    Args:
        exc: The exception to handle
        logger: Optional logger instance for logging
        reraise: Whether to re-raise the exception after handling

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     handle_exception(e, logger=logger, reraise=True)
    """
    if logger is not None:
        if isinstance(exc, FluxFlowError):
            logger.error(f"FluxFlow error: {exc.__class__.__name__}: {exc}")
        else:
            logger.error(f"Unexpected error: {exc.__class__.__name__}: {exc}")

    if reraise:
        raise exc
