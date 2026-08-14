"""FluxFlow: Text-to-image generation with VAE and flow-based diffusion transformers."""

__version__ = "0.10.0"

# Import config classes
from .config import (
    DataConfig,
    FluxFlowConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    TrainingConfig,
    create_default_config,
    load_config,
)

# Import exceptions (always available)
from .exceptions import (
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

# Shared text-token length default (training <-> generation, see text_length.py)
from .text_length import DEFAULT_MAX_TEXT_LENGTH

__all__ = [
    # Version
    "__version__",
    # Text length
    "DEFAULT_MAX_TEXT_LENGTH",
    # Exceptions
    "FluxFlowError",
    "DataError",
    "DatasetError",
    "DataLoaderError",
    "InvalidImageError",
    "InvalidCaptionError",
    "ModelError",
    "CheckpointError",
    "ModelConfigError",
    "ModelArchitectureError",
    "ForwardPassError",
    "TrainingError",
    "OptimizerError",
    "SchedulerError",
    "ConvergenceError",
    "GradientError",
    "EMAError",
    "GenerationError",
    "PromptError",
    "SamplingError",
    "ImageDecodingError",
    "ConfigError",
    "ConfigFileError",
    "ConfigValidationError",
    "IOError",
    "SaveError",
    "LoadError",
    "handle_exception",
    # Configuration
    "FluxFlowConfig",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "OptimizationConfig",
    "OutputConfig",
    "load_config",
    "create_default_config",
]
