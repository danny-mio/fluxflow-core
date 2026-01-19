"""
FluxFlow v0.6.0 models (current optimizations).

This version contains the optimized activation functions and architecture
with improved VAE reconstruction and flow processing.
"""

# Auto-register this version using metaclass
from ..registry import VersionedModelMeta
from .flow import FluxFlowProcessor
from .vae import FluxCompressor, FluxExpander


class VersionRegistrar(metaclass=VersionedModelMeta):
    """Auto-registers v0.6.0 models when this class is defined."""

    VERSION = "0.6.0"
    COMPONENTS = {
        "FluxCompressor": FluxCompressor,
        "FluxExpander": FluxExpander,
        "FluxFlowProcessor": FluxFlowProcessor,
    }
