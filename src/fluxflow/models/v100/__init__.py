"""FluxFlow v0.10.0 models (independent context branch, D-dimensional context)."""

from ..registry import VersionedModelMeta
from .flow import FluxFlowProcessor_v100 as FluxFlowProcessor
from .vae import FluxCompressor_v100 as FluxCompressor
from .vae import FluxExpander_v100 as FluxExpander


class VersionRegistrar(metaclass=VersionedModelMeta):
    """Auto-registers v0.10.0 models when this class is defined."""

    VERSION = "0.10.0"
    COMPONENTS = {
        "FluxCompressor": FluxCompressor,
        "FluxExpander": FluxExpander,
        "FluxFlowProcessor": FluxFlowProcessor,
    }
