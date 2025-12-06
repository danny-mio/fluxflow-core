#!/bin/bash
# FluxFlow Core - Setup Script
# Sets up the core package for development or production use

set -e

echo "=== FluxFlow Core Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package
if [ "$1" == "--dev" ]; then
    echo "Installing in development mode with dev dependencies..."
    pip install -e ".[dev]"
else
    echo "Installing package..."
    pip install -e .
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To use the package:"
echo "  from fluxflow.models.pipeline import FluxPipeline"
echo "  from fluxflow.models.vae import FluxCompressor, FluxExpander"
echo ""
