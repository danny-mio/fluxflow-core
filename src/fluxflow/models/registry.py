"""
Model class registry for self-registering versions.

This module contains the registry that version modules use to register themselves,
avoiding circular import issues.
"""

from typing import Dict, Optional, Type

import torch.nn as nn

# Global registry instance to avoid import issues
_registry: Dict[str, Dict[str, Type[nn.Module]]] = {}


def register_version(
    version: str,
    compressor_class: Type[nn.Module],
    expander_class: Type[nn.Module],
    flow_processor_class: Type[nn.Module],
) -> None:
    """Register model classes for a specific version."""
    _registry[version] = {
        "FluxCompressor": compressor_class,
        "FluxExpander": expander_class,
        "FluxFlowProcessor": flow_processor_class,
    }


def get_version_classes(version: str) -> Optional[Dict[str, Type[nn.Module]]]:
    """Get model classes for a specific version."""
    return _registry.get(version)


def list_versions() -> list[str]:
    """List all registered versions."""
    return sorted(_registry.keys())


class ModelClassRegistry:
    """Compatibility class for existing code."""

    @classmethod
    def register(cls, *args, **kwargs):
        register_version(*args, **kwargs)

    @classmethod
    def get_classes(cls, version: str):
        return get_version_classes(version)

    @classmethod
    def list_versions(cls):
        return list_versions()


class VersionedModelMeta(type):
    """
    Metaclass for automatic version registration.

    Classes using this metaclass will automatically register themselves
    when they're defined, if they have a VERSION class attribute.
    """

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)

        # Check if this class should be auto-registered
        if hasattr(cls, "VERSION") and hasattr(cls, "COMPONENTS"):
            # This is a versioned model class, register it
            components = getattr(cls, "COMPONENTS")
            if all(
                key in components for key in ["FluxCompressor", "FluxExpander", "FluxFlowProcessor"]
            ):
                register_version(
                    version=cls.VERSION,
                    compressor_class=components["FluxCompressor"],
                    expander_class=components["FluxExpander"],
                    flow_processor_class=components["FluxFlowProcessor"],
                )
