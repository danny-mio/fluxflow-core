# CLAUDE.md - FluxFlow Core

This document provides comprehensive guidance for AI assistants working with the FluxFlow Core codebase.

## Project Overview

FluxFlow Core is a text-to-image generation library using VAE (Variational Autoencoder) and flow-based diffusion transformers. The library provides:

- **VAE Models**: Image compression/decompression (FluxCompressor/FluxExpander)
- **Flow Models**: Flow-based diffusion transformers (FluxFlowProcessor)
- **Text Encoders**: DistilBERT-based text encoding
- **Conditioning**: FiLM, SPADE, Gated Context Injection modules
- **Pipeline**: Complete inference pipeline orchestration

## Quick Commands

```bash
# Installation
pip install -e ".[dev]"           # Development mode (quotes required!)

# Testing
make test                         # Run all tests
pytest tests/unit/test_foo.py -v                    # Single test file
pytest tests/unit/test_foo.py::test_bar -v          # Single test function
pytest tests/unit/test_foo.py::TestClass::test_bar  # Single method in class

# Code Quality
make lint                         # Run flake8, black --check, isort --check
make format                       # Format with black + isort
mypy src/                         # Type checking (run directly)
pre-commit run --all-files        # Run all pre-commit hooks

# Build
make build                        # Build distribution package
make clean                        # Clean build artifacts
```

**Available Makefile Targets** (run `make help` to verify):
- `install` - Install package
- `install-dev` - Install with dev dependencies
- `test` - Run tests
- `lint` - Run linting checks
- `format` - Format code
- `clean` - Clean build artifacts
- `build` - Build distribution package

Note: `make type-check`, `make check`, `make test-fast`, and `make install-hooks` do NOT exist in Makefile. Use `mypy src/` directly for type checking.

## Architecture

```
Text -> BertTextEncoder -> embeddings
                             |
Image -> FluxCompressor -> latent -> FluxFlowProcessor -> denoised -> FluxExpander -> output
            (VAE enc)         ^            (flow)                       (VAE dec)
                              |
                    PatchDiscriminator (training only)
```

**Model Sizes (default configuration: vae_dim=128, feat_dim=128):**
- Total: ~183M parameters (~732 MB)
- FluxCompressor: ~12.6M params
- FluxFlowProcessor: ~5.4M params
- FluxExpander: ~94.0M params
- BertTextEncoder: ~71.0M params
- PatchDiscriminator: ~45.1M params (training only)

## Directory Structure

```
fluxflow-core/
├── src/fluxflow/
│   ├── __init__.py           # Package init, version
│   ├── config.py             # Pydantic-based configuration
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── types.py              # Type definitions
│   ├── models/
│   │   ├── activations.py    # Bezier activations, transforms
│   │   ├── conditioning.py   # FiLM, SPADE, context modules
│   │   ├── diffusion_pipeline.py # FluxFlowPipeline (high-level API)
│   │   ├── discriminators.py # PatchDiscriminator for GAN
│   │   ├── encoders.py       # BERT text encoder, image encoder
│   │   ├── flow.py           # FluxFlowProcessor, transformer blocks
│   │   ├── pipeline.py       # FluxPipeline (low-level API)
│   │   └── vae.py            # FluxCompressor, FluxExpander
│   └── utils/
│       ├── io.py             # Checkpoint save/load utilities
│       ├── logger.py         # Logging setup
│       └── visualization.py  # Sample generation, visualization
├── tests/
│   ├── conftest.py           # Shared pytest fixtures
│   └── unit/                 # Unit tests
├── docs/
│   └── ARCHITECTURE.md       # Detailed architecture docs
├── pyproject.toml            # Project configuration
├── Makefile                  # Build automation
└── .pre-commit-config.yaml   # Pre-commit hooks
```

## Code Conventions

### Python Version and Dependencies

- **Python >= 3.10** required (supports 3.10, 3.11, 3.12)
- Use type hints on all public APIs
- Key dependencies: torch, transformers, diffusers, einops, pydantic

### Formatting and Linting

- **Black**: line-length=100, target Python 3.10+
- **isort**: profile=black, line_length=100
- **flake8**: max-line-length=100, max-complexity=15
- **mypy**: strict type checking enabled

### Import Order

```python
# stdlib
import math
from pathlib import Path

# third-party
import torch
import torch.nn as nn
from einops import rearrange

# local
from .activations import BezierActivation
from ..exceptions import FluxFlowError
```

### Docstrings (Google-style)

```python
def function(arg1: str, arg2: int = 10) -> dict:
    """
    Short description of the function.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong

    Example:
        >>> function("test")
        {'result': 'test'}
    """
```

### Naming Conventions

- **Functions/variables**: snake_case
- **Classes**: PascalCase
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: _leading_underscore

### Error Handling

Always use custom exceptions from `fluxflow.exceptions`:

```python
from fluxflow.exceptions import FluxFlowError, ConfigurationError

if invalid_config:
    raise ConfigurationError("Invalid configuration: ...")
```

### Logging

Use the project's logging utilities:

```python
from fluxflow.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Processing started")
```

## Testing Guidelines

### Test Structure

```python
import pytest

class TestYourClass:
    """Tests for YourClass."""

    def test_basic_functionality(self):
        """Test basic functionality works."""
        result = your_function()
        assert result == expected

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            your_function(invalid_input)
```

### Test Markers

- `@pytest.mark.slow` - Tests > 1s execution time
- `@pytest.mark.gpu` - Tests requiring GPU/CUDA

### Available Fixtures (conftest.py)

- **Device fixtures**: `device`, `cpu_device`
- **Tensor fixtures**: `random_tensor_2d`, `random_tensor_3d`, `random_tensor_4d`, `random_image_tensor`
- **Mock fixtures**: `mock_tokenizer`, `mock_image_dataset`, `mock_dimension_cache`
- **Config fixtures**: `mock_vae_config`, `mock_flow_config`, `mock_discriminator_config`
- **Helper assertions**: `assert_shape`, `assert_close`, `assert_finite`

### Test Coverage

Tests are run with pytest-cov. Aim for high coverage on new code.

## Key Patterns and Conventions

### Bezier Activations

Custom activation functions using Bezier curves with learnable control points:
- Input channels must be divisible by 5 (t, p0, p1, p2, p3)
- Used throughout for smooth, adaptive nonlinearities

### v-Prediction

Flow model predicts velocity: `v = alpha_t * noise - sigma_t * signal`
- More stable than epsilon-prediction during training

### Latent Packet Format

- Shape: `[B, T+1, D]` where T = H*W/256 tokens + 1 HW dimension token
- First K tokens used for context pooling

### Memory Optimization

- Gradient checkpointing enabled in VAE and flow blocks
- 16x spatial compression (4 downscales of 2x each)

### Configuration

Uses Pydantic models with YAML support:

```python
from fluxflow.config import FluxFlowConfig

config = FluxFlowConfig.from_yaml("config.yaml")
# or
config = FluxFlowConfig(latent_dim=256, num_layers=12)
```

### Using BertTextEncoder

The text encoder uses DistilBERT from HuggingFace transformers. Here's how to properly initialize and use it:

**Basic initialization:**

```python
from fluxflow.models.encoders import BertTextEncoder
from transformers import DistilBertTokenizer

# With pretrained weights (recommended)
encoder = BertTextEncoder(embed_dim=256, pretrain_model='distilbert-base-uncased')

# Without pretrained weights (random initialization)
encoder = BertTextEncoder(embed_dim=256)
```

**Pre-downloading the model cache:**

The encoder downloads DistilBERT to `./_cache` directory on first use. To pre-download:

```python
from transformers import DistilBertModel, DistilBertTokenizer

# Download model and tokenizer to cache
model_name = 'distilbert-base-uncased'
DistilBertModel.from_pretrained(model_name, cache_dir='./_cache')
DistilBertTokenizer.from_pretrained(model_name, cache_dir='./_cache')
```

Or via command line:

```bash
python -c "from transformers import DistilBertModel, DistilBertTokenizer; \
    DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir='./_cache'); \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./_cache')"
```

**Complete usage example:**

```python
import torch
from transformers import DistilBertTokenizer
from fluxflow.models.encoders import BertTextEncoder

# Initialize tokenizer and encoder
tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-uncased',
    cache_dir='./_cache'
)
encoder = BertTextEncoder(
    embed_dim=256,
    pretrain_model='distilbert-base-uncased'
)

# Tokenize text
text = "a beautiful sunset over mountains"
tokens = tokenizer(
    text,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=512
)

# Get embeddings
with torch.no_grad():
    embeddings = encoder(
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask']
    )
# embeddings shape: [1, 256]
```

**Important notes:**
- When using `pretrain_model`, DistilBERT weights are **frozen** (not trainable)
- Cache directory is `./_cache` relative to working directory
- Requires `input_ids` tensor, not raw text strings
- Output shape is `[batch_size, embed_dim]`

**Saving and loading checkpoints:**

```python
# Save encoder (output layers only, DistilBERT reloaded from pretrained)
encoder.save_checkpoint('checkpoints/text_encoder.safetensors')

# Save with full DistilBERT weights included
encoder.save_checkpoint('checkpoints/', save_language_model=True)

# Load from checkpoint
encoder = BertTextEncoder.from_checkpoint(
    'checkpoints/text_encoder.safetensors',
    embed_dim=256,
    pretrain_model='distilbert-base-uncased'
)

# Load to specific device
encoder = BertTextEncoder.from_checkpoint(
    'checkpoints/',
    embed_dim=256,
    device='cuda'
)
```

## Development Workflow

### Before Making Changes

1. Create a feature branch
2. Install dev dependencies: `make install-dev`
3. Run existing tests: `make test`

### Making Changes

1. Write code following conventions above
2. Add/update tests for new functionality
3. Keep functions < 50 lines, complexity < 15
4. Use type hints on public APIs

### Before Committing

1. Format code: `make format`
2. Run linting: `make lint`
3. Run tests: `make test`
4. Pre-commit hooks: `pre-commit run --all-files`

### Commit Messages

Use clear, descriptive commit messages:
- `Add feature X for Y`
- `Fix bug in Z when A`
- `Refactor X for better performance`

## Common Tasks

### Adding a New Model Component

1. Create file in `src/fluxflow/models/`
2. Follow existing patterns (see `activations.py`, `conditioning.py`)
3. Export in `models/__init__.py`
4. Add tests in `tests/unit/`
5. Update documentation if needed

### Adding New Tests

1. Create test file: `tests/unit/test_component.py`
2. Use fixtures from `conftest.py`
3. Follow `Test*` class and `test_*` function naming
4. Use appropriate markers (`@pytest.mark.slow`, `@pytest.mark.gpu`)

### Modifying Configuration

1. Update Pydantic models in `config.py`
2. Update default values as needed
3. Add tests for new configuration options

### Adding Dependencies

1. Add to `dependencies` in `pyproject.toml` (runtime)
2. Or add to `dev` optional dependencies (development only)
3. Update `requirements.txt` or `requirements-dev.txt` as needed

## CI/CD

GitHub Actions runs on push/PR to main and develop branches:

1. Linting (flake8)
2. Format checking (black)
3. Type checking (mypy)
4. Tests with coverage (pytest-cov)
5. Coverage upload (codecov)

Publishing to PyPI triggers on version tags (`v*`).

## Related Repositories

- **fluxflow-training**: Training tools and scripts
- **fluxflow-ui**: Web interface
- **fluxflow-comfyui**: ComfyUI integration

## Pipeline Validation Features

### Checkpoint Version Validation

Both `FluxPipeline` and `FluxFlowPipeline` validate checkpoint versions on load:
- Supported versions are checked against `SUPPORTED_CHECKPOINT_VERSIONS`
- Legacy checkpoints (no version) are assumed compatible
- Newer major versions raise an error with upgrade instructions
- Unknown versions log a warning but attempt to load

### Latent Format Validation

`FluxFlowPipeline` validates latent tensors when passed directly:
- Shape must be `[B, T+1, D]` (batch, sequence + HW token, dimension)
- Validates batch size and VAE dimension match
- Warns for unusually small or large implied image sizes

### Configurable Scheduler

`FluxFlowPipeline.from_pretrained()` supports custom schedulers:

```python
from fluxflow.models import FluxFlowPipeline
from diffusers import EulerDiscreteScheduler

# Pass custom scheduler instance
scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
pipeline = FluxFlowPipeline.from_pretrained(
    "checkpoint.safetensors",
    scheduler=scheduler
)

# Or configure default DPMSolverMultistepScheduler
pipeline = FluxFlowPipeline.from_pretrained(
    "checkpoint.safetensors",
    scheduler_config={"use_karras_sigmas": True, "solver_order": 3}
)
```

## Known Issues

**setup.sh and Makefile potential issues:**
- Both may use unquoted `.[dev]` which can fail in zsh and some other shells
- Use `pip install -e ".[dev]"` with quotes instead

## Additional Documentation

- `README.md` - Project overview and quick start with API comparison
- `CONTRIBUTING.md` - Contribution guidelines
- `docs/ARCHITECTURE.md` - Deep dive into architecture and design decisions
- `AGENTS.md` - Quick reference for AI assistants
