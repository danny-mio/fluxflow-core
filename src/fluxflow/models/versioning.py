"""
Model versioning system for FluxFlow.

This module provides a robust versioning system for loading and saving FluxFlow models
with explicit version metadata, enabling smooth model evolution and backward compatibility.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type

import safetensors.torch
import torch

from ..exceptions import CheckpointError

logger = logging.getLogger(__name__)


# ============================================================
# Version Metadata Schema
# ============================================================


class ModelMetadata:
    """Model version and architecture metadata.

    Attributes:
        model_version: Model version string (semantic versioning)
        library_version: FluxFlow library version used to create the model
        architecture: Dict containing model architecture parameters
        components: Dict mapping component names to class names
        training_info: Optional dict with training metadata
        checksum: Optional dict with file integrity checksums
    """

    def __init__(
        self,
        model_version: str,
        library_version: str,
        architecture: Dict[str, Any],
        components: Dict[str, str],
        training_info: Optional[Dict[str, Any]] = None,
        checksum: Optional[Dict[str, str]] = None,
    ):
        self.model_version = model_version
        self.library_version = library_version
        self.architecture = architecture
        self.components = components
        self.training_info = training_info or {}
        self.checksum = checksum or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "model_version": self.model_version,
            "library_version": self.library_version,
            "architecture": self.architecture,
            "components": self.components,
            "training_info": self.training_info,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Load metadata from dictionary."""
        return cls(
            model_version=data["model_version"],
            library_version=data["library_version"],
            architecture=data["architecture"],
            components=data["components"],
            training_info=data.get("training_info"),
            checksum=data.get("checksum"),
        )

    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ModelMetadata":
        """Load metadata from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================
# Version Loader Interface
# ============================================================


class ModelVersionLoader(ABC):
    """Abstract base class for version-specific model loaders."""

    VERSION: str = "0.0.0"  # Override in subclasses
    COMPATIBLE_VERSIONS: list = []  # Minor versions this loader supports

    @abstractmethod
    def load_checkpoint(
        self, checkpoint_path: Path, metadata: ModelMetadata, device: str, **kwargs
    ) -> Any:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Model metadata
            device: Target device
            **kwargs: Additional arguments

        Returns:
            Loaded model (FluxPipeline or FluxFlowPipeline)
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self, model: Any, output_path: Path, metadata: ModelMetadata, **kwargs
    ) -> None:
        """Save model to checkpoint."""
        pass

    def can_load(self, version: str) -> bool:
        """Check if this loader can load the given version."""
        return version == self.VERSION or version in self.COMPATIBLE_VERSIONS

    def migrate_from(self, prev_version: str, state_dict: Dict) -> Dict:
        """
        Migrate state dict from previous version.

        Override in subclasses to handle version-specific migrations.
        """
        return state_dict


# ============================================================
# Version Registry
# ============================================================


class ModelVersionRegistry:
    """Registry mapping model versions to loader classes."""

    _loaders: Dict[str, Type[ModelVersionLoader]] = {}

    @classmethod
    def register(cls, loader_class: Type[ModelVersionLoader]) -> None:
        """Register a version loader."""
        version = loader_class.VERSION
        if version in cls._loaders:
            logger.warning(f"Overwriting loader for version {version}")
        cls._loaders[version] = loader_class
        logger.debug(f"Registered loader for version {version}: {loader_class.__name__}")

    @classmethod
    def get_loader(cls, version: str) -> Optional[Type[ModelVersionLoader]]:
        """Get loader for specific version."""
        # Exact match
        if version in cls._loaders:
            return cls._loaders[version]

        # Check compatible versions (e.g., 0.3.1 can use 0.3.0 loader)
        for loader_version, loader_class in cls._loaders.items():
            if loader_class().can_load(version):
                return loader_class

        return None

    @classmethod
    def list_versions(cls) -> list:
        """List all registered versions."""
        return sorted(cls._loaders.keys())


# ============================================================
# Current Version (0.3.x) Loader
# ============================================================


class ModelLoaderV03(ModelVersionLoader):
    """Loader for FluxFlow model version 0.3.x."""

    VERSION = "0.3.0"
    COMPATIBLE_VERSIONS = ["0.3.1", "0.3.2"]  # Patch versions

    def load_checkpoint(
        self, checkpoint_path: Path, metadata: ModelMetadata, device: str, **kwargs
    ) -> Any:
        """Load v0.3.x checkpoint."""
        from .flow import FluxFlowProcessor
        from .pipeline import FluxPipeline
        from .vae import FluxCompressor, FluxExpander

        config = metadata.architecture

        # Initialize models with explicit config
        compressor = FluxCompressor(
            in_channels=config.get("in_channels", 3),
            d_model=config["vae_dim"],
            downscales=config["downscales"],
            max_hw=config.get("max_hw", 1024),
            use_attention=True,
            attn_layers=config.get("vae_attn_layers", 2),
        )

        flow_processor = FluxFlowProcessor(
            d_model=config["flow_dim"],
            vae_dim=config["vae_dim"],
            embedding_size=config.get("text_embed_dim", 768),
            n_head=config.get("flow_attn_heads", 8),
            n_layers=config.get("flow_transformer_layers", 10),
            max_hw=config.get("max_hw", 1024),
        )

        expander = FluxExpander(
            d_model=config["vae_dim"],
            upscales=config.get("upscales", config["downscales"]),
            max_hw=config.get("max_hw", 1024),
        )

        pipeline = FluxPipeline(compressor, flow_processor, expander)

        # Load weights
        if checkpoint_path.suffix == ".safetensors":
            state_dict = safetensors.torch.load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Strip 'diffuser.' prefix if present
        diffuser_state = {
            k.replace("diffuser.", ""): v
            for k, v in state_dict.items()
            if k.startswith("diffuser.")
        }
        if not diffuser_state:
            diffuser_state = state_dict

        pipeline.load_state_dict(diffuser_state, strict=False)
        pipeline.to(device)
        pipeline.eval()

        return pipeline

    def save_checkpoint(
        self, model: Any, output_path: Path, metadata: ModelMetadata, **kwargs
    ) -> None:
        """Save v0.3.x checkpoint."""
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_path = output_path / "model_metadata.json"
        metadata.save(metadata_path)

        # Save weights
        checkpoint_path = output_path / "model.safetensors"
        state_dict = {f"diffuser.{k}": v.cpu() for k, v in model.state_dict().items()}

        # Compute checksum
        checkpoint_bytes = safetensors.torch.save(state_dict)
        checksum = hashlib.sha256(checkpoint_bytes).hexdigest()
        metadata.checksum["weights_hash"] = checksum
        metadata.checksum["algorithm"] = "sha256"
        metadata.save(metadata_path)  # Re-save with checksum

        # Write checkpoint
        with open(checkpoint_path, "wb") as f:
            f.write(checkpoint_bytes)


# Register v0.3.x loader
ModelVersionRegistry.register(ModelLoaderV03)


# ============================================================
# Legacy Loader (pre-0.3.0, no version metadata)
# ============================================================


class ModelLoaderLegacy(ModelVersionLoader):
    """Loader for legacy FluxFlow models without version metadata."""

    VERSION = "0.2.0"  # Assumed version for legacy models
    COMPATIBLE_VERSIONS = ["0.2.x", "legacy"]

    def load_checkpoint(
        self, checkpoint_path: Path, metadata: Optional[ModelMetadata], device: str, **kwargs
    ) -> Any:
        """Load legacy checkpoint using config detection."""
        from .pipeline import FluxPipeline

        logger.warning(
            f"Loading legacy checkpoint without version metadata: {checkpoint_path}. "
            "Architecture will be inferred from checkpoint structure. "
            "Consider re-saving with 'save_versioned_checkpoint()' to add metadata."
        )

        # Use existing _detect_config logic
        if checkpoint_path.suffix == ".safetensors":
            state_dict = safetensors.torch.load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        config = FluxPipeline._detect_config(state_dict)

        # Create metadata for future use
        if metadata is None:
            metadata = ModelMetadata(
                model_version="0.2.0",
                library_version="0.3.1",
                architecture=config,
                components={
                    "compressor": "FluxCompressor",
                    "flow_processor": "FluxFlowProcessor",
                    "expander": "FluxExpander",
                },
            )

        # Delegate to v0.3 loader (backward compatible)
        v03_loader = ModelLoaderV03()
        return v03_loader.load_checkpoint(checkpoint_path, metadata, device, **kwargs)

    def save_checkpoint(
        self, model: Any, output_path: Path, metadata: ModelMetadata, **kwargs
    ) -> None:
        """Upgrade legacy model to v0.3.x format."""
        logger.info("Upgrading legacy model to v0.3.x format with metadata")
        v03_loader = ModelLoaderV03()
        v03_loader.save_checkpoint(model, output_path, metadata, **kwargs)


ModelVersionRegistry.register(ModelLoaderLegacy)


# ============================================================
# Unified Loading Interface
# ============================================================


def load_versioned_checkpoint(checkpoint_path: Path, device: Optional[str] = None, **kwargs) -> Any:
    """
    Load FluxFlow checkpoint with automatic version detection and routing.

    Args:
        checkpoint_path: Path to checkpoint file or directory
        device: Target device ('cuda', 'cpu', 'mps', or None for auto)
        **kwargs: Additional arguments passed to version loader

    Returns:
        Loaded model (FluxPipeline or FluxFlowPipeline)

    Raises:
        CheckpointError: If checkpoint is incompatible or corrupted

    Example:
        >>> from fluxflow.models.versioning import load_versioned_checkpoint
        >>> pipeline = load_versioned_checkpoint("checkpoints/model_v0.3.0/")
    """
    checkpoint_path = Path(checkpoint_path)

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Handle directory vs file
    if checkpoint_path.is_dir():
        metadata_path = checkpoint_path / "model_metadata.json"
        weights_path = checkpoint_path / "model.safetensors"
        if not weights_path.exists():
            weights_path = checkpoint_path / "flxflow_final.safetensors"
        if not weights_path.exists():
            raise CheckpointError(f"No checkpoint found in {checkpoint_path}")
    else:
        metadata_path = checkpoint_path.parent / "model_metadata.json"
        weights_path = checkpoint_path

    # Load metadata
    if metadata_path.exists():
        metadata = ModelMetadata.load(metadata_path)
        version = metadata.model_version
        logger.info(f"Loading model version {version} from {checkpoint_path}")
    else:
        # Legacy checkpoint without metadata
        logger.warning("No metadata found, treating as legacy checkpoint")
        metadata = None
        version = "legacy"

    # Get appropriate loader
    loader_class = ModelVersionRegistry.get_loader(version)
    if loader_class is None:
        # Try to determine if it's a newer version
        if metadata and _is_newer_version(version, "0.3.1"):
            raise CheckpointError(
                f"Checkpoint version {version} is newer than library version 0.3.1. "
                "Please upgrade fluxflow: pip install --upgrade fluxflow"
            )
        else:
            raise CheckpointError(
                f"No loader found for checkpoint version {version}. "
                f"Supported versions: {ModelVersionRegistry.list_versions()}"
            )

    # Load checkpoint (metadata can be None for legacy loaders)
    loader = loader_class()
    # Type check: metadata is Optional[ModelMetadata] for legacy loader compatibility
    model = loader.load_checkpoint(weights_path, metadata, device, **kwargs)  # type: ignore[arg-type]

    logger.info(f"Successfully loaded model version {version}")
    return model


def save_versioned_checkpoint(
    model: Any,
    output_path: Path,
    model_version: str = "0.3.0",
    architecture: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Save FluxFlow checkpoint with version metadata.

    Args:
        model: Model to save (FluxPipeline or FluxFlowPipeline)
        output_path: Directory to save checkpoint
        model_version: Model version string (semantic versioning)
        architecture: Architecture config (auto-detected if None)
        training_info: Optional training metadata
        **kwargs: Additional arguments

    Example:
        >>> from fluxflow.models.versioning import save_versioned_checkpoint
        >>> save_versioned_checkpoint(
        ...     pipeline,
        ...     "outputs/model_v0.3.0/",
        ...     model_version="0.3.0",
        ...     training_info={"total_steps": 50000, "dataset": "COCO"}
        ... )
    """
    output_path = Path(output_path)

    # Auto-detect architecture if not provided
    if architecture is None:
        architecture = _detect_architecture(model)

    # Create metadata
    from fluxflow import __version__ as library_version

    metadata = ModelMetadata(
        model_version=model_version,
        library_version=library_version,
        architecture=architecture,
        components=_detect_components(model),
        training_info=training_info,
    )

    # Get appropriate loader
    loader_class = ModelVersionRegistry.get_loader(model_version)
    if loader_class is None:
        raise ValueError(f"No loader found for version {model_version}")

    loader = loader_class()
    loader.save_checkpoint(model, output_path, metadata, **kwargs)

    logger.info(f"Saved model version {model_version} to {output_path}")


# ============================================================
# Utility Functions
# ============================================================


def _is_newer_version(v1: str, v2: str) -> bool:
    """Compare semantic versions."""

    def parse(v: str) -> tuple:
        return tuple(int(x) for x in v.split("."))

    try:
        return bool(parse(v1) > parse(v2))
    except Exception:
        return False


def _detect_architecture(model: Any) -> Dict[str, Any]:
    """Auto-detect model architecture from instance."""
    config = {}

    if hasattr(model, "compressor"):
        config["vae_dim"] = model.compressor.d_model
        config["downscales"] = model.compressor.downscales
        config["max_hw"] = model.compressor.max_hw
        config["vae_attn_layers"] = model.compressor.attn_layers
        config["in_channels"] = 3  # Default

    if hasattr(model, "flow_processor"):
        # FluxFlowProcessor doesn't store d_model, infer from linear layer
        if hasattr(model.flow_processor, "dmodel_to_vae"):
            config["flow_dim"] = model.flow_processor.dmodel_to_vae.in_features
        config["flow_transformer_layers"] = len(model.flow_processor.transformer_blocks)
        # Infer n_head from self_attn layer
        if len(model.flow_processor.transformer_blocks) > 0:
            first_block = model.flow_processor.transformer_blocks[0]
            if hasattr(first_block, "self_attn") and hasattr(first_block.self_attn, "n_head"):
                config["flow_attn_heads"] = first_block.self_attn.n_head
            else:
                config["flow_attn_heads"] = 8  # Default
        # Infer text_embed_dim from text_proj layer
        if hasattr(model.flow_processor, "text_proj"):
            config["text_embed_dim"] = model.flow_processor.text_proj.in_features
        else:
            config["text_embed_dim"] = 768  # Default

    if hasattr(model, "expander"):
        config["upscales"] = len(model.expander.upscale.layers)

    return config


def _detect_components(model: Any) -> Dict[str, str]:
    """Detect model component class names."""
    components = {}

    if hasattr(model, "compressor"):
        components["compressor"] = model.compressor.__class__.__name__
    if hasattr(model, "flow_processor"):
        components["flow_processor"] = model.flow_processor.__class__.__name__
    if hasattr(model, "expander"):
        components["expander"] = model.expander.__class__.__name__
    if hasattr(model, "text_encoder"):
        components["text_encoder"] = model.text_encoder.__class__.__name__

    return components
