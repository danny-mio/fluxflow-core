# FluxFlow Core - Makefile

.PHONY: install install-dev test lint format clean build

# Install package
install:
	pip install -e .

# Install with development dependencies
install-dev:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v --tb=short

# Run linting
lint:
	flake8 src/
	black --check src/
	isort --check-only src/

# Format code
format:
	black src/
	isort src/

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Help
help:
	@echo "FluxFlow Core - Available targets:"
	@echo "  install      - Install package"
	@echo "  install-dev  - Install with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build distribution package"
