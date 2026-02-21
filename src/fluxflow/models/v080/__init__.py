"""
FluxFlow v0.8.0 models (pillar-attention architecture).

This version adds direct text attention and FiLM conditioning to each pillar,
addressing the lossy text signal path in v0.7.0.
VAE is unchanged (imported from v070).
"""

# Auto-register this version using metaclass
from ..registry import VersionedModelMeta
from ..v070.vae import FluxCompressor, FluxExpander  # VAE unchanged
from .flow import FluxFlowProcessor_v080 as FluxFlowProcessor


class VersionRegistrar(metaclass=VersionedModelMeta):
    """Auto-registers v0.8.0 models when this class is defined."""

    VERSION = "0.8.0"
    COMPONENTS = {
        "FluxCompressor": FluxCompressor,
        "FluxExpander": FluxExpander,
        "FluxFlowProcessor": FluxFlowProcessor,
    }
