"""
FluxFlow v0.7.0 models (context-enhanced architecture).

This version introduces separate context generation in the compressor,
context-aware SPADE conditioning in the expander, and unified flow processing
of VAE dimensions + context dimensions as a single entity.
"""

# Auto-register this version using metaclass
from ..registry import VersionedModelMeta
from .flow import FluxFlowProcessor
from .vae import FluxCompressor, FluxExpander


class VersionRegistrar(metaclass=VersionedModelMeta):
    """Auto-registers v0.7.0 models when this class is defined."""

    VERSION = "0.7.0"
    COMPONENTS = {
        "FluxCompressor": FluxCompressor,
        "FluxExpander": FluxExpander,
        "FluxFlowProcessor": FluxFlowProcessor,
    }
