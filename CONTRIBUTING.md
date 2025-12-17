# Contributing to FluxFlow

Thank you for your interest in contributing to FluxFlow! This guide will help you get started with development.

## Quick Start for Developers

### 1. Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/danny-mio/fluxflow-core.git
cd fluxflow-core

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 2. Development Workflow

#### Before Making Changes

```bash
# Create a new branch
git checkout -b feature/your-feature-name
```

#### During Development

```bash
# Format your code
make format

# Run linting
make lint

# Run type checking
mypy src/

# Run tests
make test
```

#### Before Committing

Pre-commit hooks will automatically run when you commit. To run them manually:

```bash
# Run all pre-commit checks
pre-commit run --all-files
```

### 3. Code Quality Standards

FluxFlow uses several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run them manually:

```bash
make format     # Format code
make lint       # Check linting
mypy src/       # Check types
make test       # Run tests
```

## Code Style Guidelines

### Python Code

- Follow PEP 8 (enforced by flake8)
- Use Black formatting (100 character line limit)
- Add type hints to public APIs
- Write docstrings for modules, classes, and functions
- Keep functions focused and small (< 50 lines ideally)

### Example

```python
"""Module docstring explaining the module's purpose."""

from typing import Optional

from fluxflow.utils.logger import setup_logger

logger = setup_logger(__name__)


def process_data(input_path: str, output_path: Optional[str] = None) -> dict[str, int]:
    """
    Process data from input file and optionally save to output.

    Args:
        input_path: Path to input data file
        output_path: Optional path to save processed data

    Returns:
        Dictionary with processing statistics

    Raises:
        FileNotFoundError: If input_path doesn't exist
    """
    logger.info(f"Processing data from {input_path}")
    # Implementation here
    return {"processed": 42}
```

### Configuration Files

- Use YAML for configuration (not shell scripts for new configs)
- Add validation with Pydantic models

### Documentation

- Update README.md for user-facing changes
- Keep documentation concise and practical
- Use code examples where helpful

## Testing

### Writing Tests

All new code should include tests:

```bash
# Create test file in appropriate directory
tests/unit/test_your_feature.py
```

### Test Structure

```python
"""Tests for your feature."""

import pytest
from fluxflow.your_module import YourClass


class TestYourClass:
    """Tests for YourClass."""

    def test_basic_functionality(self):
        """Test basic functionality works."""
        obj = YourClass()
        result = obj.method()
        assert result == expected

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            obj = YourClass(invalid_param="bad")
```

### Running Tests

```bash
# All tests
make test

# Specific test file
pytest tests/unit/test_your_feature.py -v

# Specific test function
pytest tests/unit/test_your_feature.py::test_function_name -v

# Specific test method in class
pytest tests/unit/test_your_feature.py::TestClass::test_method -v
```

## Project Structure

```
fluxflow-core/
├── src/fluxflow/
│   ├── __init__.py           # Package init, version
│   ├── config.py             # Pydantic-based configuration
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── types.py              # Type definitions
│   ├── models/               # Model architectures
│   │   ├── activations.py    # Bezier activations
│   │   ├── conditioning.py   # FiLM, SPADE, context modules
│   │   ├── discriminators.py # PatchDiscriminator
│   │   ├── encoders.py       # BertTextEncoder, ImageEncoder
│   │   ├── flow.py           # FluxFlowProcessor
│   │   ├── pipeline.py       # FluxPipeline
│   │   ├── diffusion_pipeline.py  # FluxFlowPipeline
│   │   └── vae.py            # FluxCompressor, FluxExpander
│   └── utils/                # Utilities
│       ├── io.py             # Checkpoint save/load
│       ├── logger.py         # Logging setup
│       └── visualization.py  # Sample generation
├── tests/
│   ├── conftest.py           # Shared pytest fixtures
│   └── unit/                 # Unit tests
├── docs/
│   └── ARCHITECTURE.md       # Architecture documentation
├── pyproject.toml            # Project configuration
├── Makefile                  # Build automation
└── .pre-commit-config.yaml   # Pre-commit hooks
```

Related repositories (separate packages):
- **fluxflow-training**: Training tools and scripts
- **fluxflow-ui**: Web interface
- **fluxflow-comfyui**: ComfyUI integration

## Pull Request Process

1. **Create a branch** from `develop`
2. **Make your changes** with tests and documentation
3. **Run checks**: `make lint && make test && mypy src/`
4. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description

   Detailed explanation of what changed and why.

   - Specific change 1
   - Specific change 2"
   ```
5. **Push and create PR** on GitHub
6. **Address review feedback** if any

## Common Tasks

### Adding a New Feature

1. Create feature branch: `git checkout -b feature/name`
2. Add implementation in appropriate module
3. Add type hints and docstrings
4. Add tests for the feature
5. Update documentation if user-facing
6. Run `make lint && make test` to verify quality
7. Commit and create PR

### Fixing a Bug

1. Create bugfix branch: `git checkout -b fix/issue-description`
2. Add a test that reproduces the bug
3. Fix the bug
4. Verify the test passes
5. Run `make lint && make test`
6. Commit and create PR

### Adding Dependencies

1. Add to appropriate section in `pyproject.toml`:
   - Core dependencies -> `dependencies`
   - Dev dependencies -> `optional-dependencies.dev`
2. Update requirements.txt if needed for backward compatibility
3. Document why the dependency is needed in your PR

## Requesting Contributor Access

### Public Contributors (No Access Needed)

Anyone can contribute via pull requests! Just:

1. Fork the repository
2. Make changes in your fork
3. Submit a pull request to the `develop` branch

No special permissions needed!

### Requesting Direct Repository Access

If you want to contribute regularly and need direct access to the repository:

1. **Start by contributing** - Submit 1-3 quality pull requests first to demonstrate:
   - Code quality and adherence to standards
   - Understanding of the project
   - Commitment to contributing

2. **Open a discussion** at https://github.com/danny-mio/fluxflow-core/discussions
   
3. **Use the title**: `Request: Contributor Access for [Your Name]`

4. **Include in your request**:
   - Your GitHub username
   - Links to your merged PRs or planned contributions
   - Areas you want to contribute to (e.g., models, utilities, documentation)
   - Your availability (approximate hours per week)
   - Your experience with Python/PyTorch/ML
   - Why you want direct access vs continuing with forks

**Review process:**
- Repository owner will review your request
- Typically reviewed within 7 days
- May include a brief video call to discuss contribution plans and expectations

**What you'll get with contributor access:**
- ✅ Write access to create branches directly (no fork needed)
- ✅ Ability to label and assign issues
- ✅ Ability to merge PRs to `develop` (after CI passes)
- ✅ Listed as a project contributor

**Note**: Only the repository owner can merge `develop` → `main` for production releases

## Security Issues

For security vulnerabilities, please follow our [Security Policy](SECURITY.md) instead of opening public issues.

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open a GitHub Issue to discuss before implementing
- **Security**: See [SECURITY.md](SECURITY.md)

## Code of Conduct

- Be respectful and professional
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Keep discussions focused on technical merit

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to FluxFlow!**
