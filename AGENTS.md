# AGENTS.md - FluxFlow Core

## Cross-Project References (CRITICAL)
- **NEVER** reference files in other projects using relative filesystem paths (e.g., `../fluxflow-training/`)
- **ALWAYS** use GitHub URLs when referencing other projects' documentation/code
- **NEVER** include local filesystem paths (e.g., `/Users/`, `/Volumes/`) in any committed documentation
- Each project is a standalone repository; cross-references must use public URLs only
- Example: `See [fluxflow-training TRAINING_GUIDE.md](https://github.com/danny-mio/fluxflow-training/blob/develop/docs/TRAINING_GUIDE.md)`

## Build & Test Commands
```bash
make install-dev          # Install with dev dependencies
make test                 # Run all tests
pytest tests/unit/test_foo.py -v                    # Single test file
pytest tests/unit/test_foo.py::test_bar -v          # Single test function
pytest tests/unit/test_foo.py::TestClass::test_bar  # Single method
make lint                 # Run flake8, black --check, isort --check
make format               # Format with black + isort
```

## Code Style
- **Python >= 3.10** with type hints on public APIs
- **Black** formatting (line-length=100), **isort** (profile=black)
- **Imports**: stdlib, third-party, local (each separated by blank line)
- **Docstrings**: Google-style for all public functions/classes
- **Naming**: snake_case (functions/vars), PascalCase (classes), UPPER_SNAKE (constants)
- **Error handling**: Use custom exceptions from `fluxflow.exceptions`; always log errors
- **Tests**: pytest with `Test*` classes and `test_*` functions; markers: `@pytest.mark.slow`, `@pytest.mark.gpu`
- **Max complexity**: 15 (flake8); keep functions < 50 lines
- Run `pre-commit run --all-files` before committing
