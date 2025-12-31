"""
FluxFlow v0.3.0 models (develop branch).

This version contains the original activation functions and architecture
from the develop branch before the v0.6.0 optimizations.
"""

# Auto-register this version using metaclass
from ..registry import VersionedModelMeta
from .flow import FluxFlowProcessor
from .vae import FluxCompressor, FluxExpander


class VersionRegistrar(metaclass=VersionedModelMeta):
    """Auto-registers v0.3.0 models when this class is defined."""

    VERSION = "0.3.0"
    COMPONENTS = {
        "FluxCompressor": FluxCompressor,
        "FluxExpander": FluxExpander,
        "FluxFlowProcessor": FluxFlowProcessor,
    }
