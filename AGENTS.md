# FluxFlow Core - Agent Instructions

## Project Overview

**Repository**: `fluxflow-core`  
**Purpose**: Core model architectures, inference pipeline, and foundational components for FluxFlow text-to-image generation  
**Package Name**: `fluxflow` (on PyPI)  
**Dependencies**: PyTorch, safetensors, diffusers  
**Role**: Foundation library - other repos depend on this

## Cross-Project References (CRITICAL)

- **NEVER** reference files in other projects using relative filesystem paths (e.g., `../fluxflow-training/`)
- **ALWAYS** use GitHub URLs when referencing other projects' documentation/code
- **NEVER** include local filesystem paths (e.g., `/Users/`, `/Volumes/`) in any committed documentation
- Each project is a standalone repository; cross-references must use public URLs only
- Example: `See [fluxflow-training PIPELINE_ARCHITECTURE.md](https://github.com/danny-mio/fluxflow-training/blob/main/docs/PIPELINE_ARCHITECTURE.md)`

## Git Workflow

- Default branch: `main` (not `develop` - GitHub default)
- Create feature branches from `main`, PR to `main`
- Branch naming: `feature/add-*`, `fix/*`, `docs/*`, `refactor/*`
- Commit messages: Imperative mood, concise, explain WHY not WHAT

## Package Structure

```
fluxflow-core/
├── src/fluxflow/
│   ├── __init__.py           # Package exports, version
│   ├── config.py             # Pydantic configuration models
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── types.py              # Type definitions and protocols
│   ├── models/               # Model architectures
│   │   ├── __init__.py       # Model exports
│   │   ├── activations.py    # BezierActivation, TrainableBezier
│   │   ├── conditioning.py   # FiLM, SPADE, context modules
│   │   ├── discriminators.py # PatchDiscriminator (GAN training)
│   │   ├── encoders.py       # BertTextEncoder, ImageEncoder
│   │   ├── flow.py           # FluxFlowProcessor (diffusion)
│   │   ├── pipeline.py       # FluxPipeline (base, nn.Module)
│   │   ├── diffusion_pipeline.py  # FluxFlowPipeline (DiffusionPipeline)
│   │   └── vae.py            # FluxCompressor, FluxExpander
│   └── utils/                # Utilities
│       ├── io.py             # Checkpoint save/load
│       ├── logger.py         # Logging setup
│       └── visualization.py  # Sample generation
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   └── unit/                 # Unit tests
├── docs/
│   ├── ARCHITECTURE.md       # Architecture documentation
│   ├── BEZIER_ACTIVATIONS.md # Technical deep dive
│   └── INTUITION.md          # High-level explanation
├── pyproject.toml            # Package configuration
├── Makefile                  # Development automation
└── .pre-commit-config.yaml   # Code quality hooks
```

## Build/Test Commands

```bash
# Install with dev dependencies
make install-dev              # Or: pip install -e ".[dev]"

# Run all tests
make test                     # Or: pytest tests/

# Run specific test
pytest tests/unit/test_activations.py::test_bezier_forward -v

# Code quality
make lint                     # flake8 + black check + isort check
make format                   # black + isort (auto-fix)
mypy src/                     # Type checking

# Pre-commit hooks (run before committing)
pre-commit run --all-files
```

## Code Style (Python ≥3.10)

- **Formatting**: Black (line-length=100), isort (profile=black)
- **Linting**: flake8 (max-complexity=15, max-line-length=100)
- **Type Hints**: Required on public APIs, optional on internal helpers
- **Docstrings**: Google style for public classes/functions
- **Naming**: 
  - snake_case: functions, variables, modules
  - PascalCase: classes
  - UPPER_SNAKE: constants
- **Imports**: stdlib → third-party → local (blank line between groups)
- **Function Length**: < 50 lines ideal (exceptions allowed for init methods)

## Testing Requirements

### Before Committing

- ✅ All tests pass: `make test`
- ✅ Linting clean: `make lint`
- ✅ Type checking: `mypy src/`
- ✅ Pre-commit hooks: `pre-commit run --all-files`

### Test Structure

```python
"""Tests for BezierActivation."""
import pytest
import torch
from fluxflow.models.activations import BezierActivation


class TestBezierActivation:
    """Test suite for BezierActivation."""

    def test_forward_shape(self):
        """Test output shape matches expected dimensions."""
        act = BezierActivation()
        x = torch.randn(2, 5, 32, 32)  # [B, t+p0+p1+p2+p3, H, W]
        output = act(x)
        assert output.shape == (2, 1, 32, 32)  # [B, 1, H, W]

    def test_gradient_flow(self):
        """Test gradients propagate correctly."""
        act = BezierActivation()
        x = torch.randn(2, 5, 8, 8, requires_grad=True)
        output = act(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

## Documentation Standards

### Markdown Files

- Use GitHub-flavored Markdown
- Reference other repos via GitHub URLs (NOT relative paths)
- Code blocks: Always specify language (```python, ```bash)
- Tables: Use consistent alignment
- Images: Use GitHub raw URLs for cross-platform rendering
- No emojis unless explicitly requested
- No marketing language ("amazing", "revolutionary") - be precise

### Code Comments

- Inline comments: Explain WHY, not WHAT
- Complex algorithms: Add references to papers/sources
- Magic numbers: Extract to named constants with comments

### API Documentation

- All public classes/functions must have docstrings
- Include Args, Returns, Raises sections
- Add Examples for non-obvious usage

Example:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply Bezier activation to input tensor.

    Args:
        x: Input tensor of shape [B, D*5, ...] where D*5 represents
           concatenated [t, p0, p1, p2, p3] parameters.

    Returns:
        torch.Tensor: Bezier-transformed output of shape [B, D, ...].

    Raises:
        ValueError: If input shape is not divisible by 5.

    Example:
        >>> act = BezierActivation()
        >>> x = torch.randn(2, 5, 32, 32)
        >>> out = act(x)
        >>> out.shape
        torch.Size([2, 1, 32, 32])
    """
```

## Publishing Workflow (PyPI)

**CRITICAL**: `fluxflow-core` publishes as `fluxflow` on PyPI. Other packages depend on it.

**Publishing Order**:
1. `fluxflow` (this repo) - NO dependencies on other FluxFlow packages
2. `fluxflow-training` - depends on `fluxflow>=0.3.0`
3. `fluxflow-ui` - depends on `fluxflow-training>=0.3.0`
4. `fluxflow-comfyui` - depends on `fluxflow>=0.3.0` (parallel with ui)

**Release Process**:
```bash
# 1. Ensure all tests pass and code is merged to main
git checkout main
git pull origin main

# 2. Update version in pyproject.toml
# Verify CHANGELOG.md is updated

# 3. Create and push git tag (triggers GitHub Actions)
git tag -a v0.3.1 -m "Release v0.3.1: Summary"
git push origin v0.3.1

# 4. GitHub Actions auto-publishes if pipeline succeeds
# Monitor: https://github.com/danny-mio/fluxflow-core/actions

# 5. Verify on PyPI (~5 min after CI completes)
# https://pypi.org/project/fluxflow/
```

**Never publish dependent packages before this one is live on PyPI.**

## Common Tasks

### Adding a New Model Component

1. Create module in `src/fluxflow/models/`
2. Add type hints and docstrings
3. Add unit tests in `tests/unit/`
4. Export from `src/fluxflow/models/__init__.py`
5. Update `docs/ARCHITECTURE.md` if significant
6. Run `make lint && make test && mypy src/`

### Fixing a Bug

1. Create bugfix branch: `git checkout -b fix/issue-description`
2. Write a test that reproduces the bug
3. Fix the bug
4. Verify test passes: `pytest tests/unit/test_your_fix.py -v`
5. Run full test suite: `make test`
6. Create PR to `main`

### Updating Documentation

1. Check for duplicate content across docs (avoid redundancy)
2. Use cross-references instead of copy-pasting
3. Verify all GitHub URLs resolve correctly
4. Update CHANGELOG.md if user-facing change
5. Run `pre-commit run --all-files` (checks Markdown linting)

### Adding a Dependency

1. Add to `dependencies` in `pyproject.toml`
2. Pin minimum version if specific features required
3. Document why dependency is needed in PR description
4. Check for conflicts with `fluxflow-training`, `fluxflow-ui`
5. Update setup-time dependencies in CI (`.github/workflows/`)

## Security Best Practices

- **Never commit**:
  - API keys, tokens, credentials
  - Local filesystem paths (`/Users/`, `/Volumes/`)
  - Checkpoints (use Git LFS for model files)
  
- **Use environment variables** for:
  - HuggingFace tokens
  - Weights & Biases API keys
  - Download URLs

- **Validate user inputs**:
  - Sanitize file paths (prevent directory traversal)
  - Validate tensor shapes before operations
  - Check for NaN/Inf in model outputs

## Model Architecture Guidelines

### Bezier Activation Placement

**Use BezierActivation when**:
- High expressiveness needed (VAE encoder/decoder, flow transformer)
- Training from scratch (can learn optimal curves)
- Memory not critical

**Avoid BezierActivation when**:
- Memory critical (e.g., discriminator with 2× forward passes)
- Simple transformations (e.g., SPADE normalization - use ReLU)
- Binary classification tasks

### Configuration Patterns

```python
# VAE Encoder (Image → Latent)
BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu")

# VAE Decoder (Latent → Image, final layer)
BezierActivation(t_pre_activation="silu", p_preactivation="tanh")

# Transformer MLP (latent space)
BezierActivation()  # No pre-activation, max flexibility
```

## Training Status Awareness

**Current Status**: Models are in Phase 1 validation training (December 2025).

**Documentation Rules**:
- Mark all unvalidated claims as "Training in progress" or "Target"
- Use "Expected" or "Theoretical" for unproven metrics
- Update `MODEL_ZOO.md` as empirical results become available
- Reference `TRAINING_VALIDATION_PLAN.md` for timeline

**Language to use**:
- ✅ "Target FID ≤ 15 (empirical validation pending)"
- ✅ "Expected 38% speedup based on parameter counting"
- ❌ "FluxFlow achieves 38% speedup" (not yet proven)
- ❌ "Bezier activations are better" (subjective, unvalidated)

## Issue Labels

When creating issues or PRs:

- `bug`: Something isn't working correctly
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `performance`: Performance optimization
- `testing`: Test coverage or infrastructure
- `refactor`: Code cleanup without behavior change
- `dependencies`: Dependency updates
- `security`: Security-related changes

## Getting Help

- **Technical Questions**: [GitHub Discussions](https://github.com/danny-mio/fluxflow-core/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/danny-mio/fluxflow-core/issues)
- **Security Issues**: See [SECURITY.md](SECURITY.md) - do NOT open public issues

## Related Repositories

- **fluxflow-training**: [https://github.com/danny-mio/fluxflow-training](https://github.com/danny-mio/fluxflow-training)
- **fluxflow-ui**: [https://github.com/danny-mio/fluxflow-ui](https://github.com/danny-mio/fluxflow-ui)
- **fluxflow-comfyui**: [https://github.com/danny-mio/fluxflow-comfyui](https://github.com/danny-mio/fluxflow-comfyui)

Always use GitHub URLs when referencing these repositories in documentation.

---

**Last Updated**: December 14, 2025
