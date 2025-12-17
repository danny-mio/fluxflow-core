"""
FluxPipeline: Main wrapper combining VAE and flow models.
"""

import logging
import os
from typing import Optional, cast

import safetensors.torch
import torch
import torch.nn as nn

from .flow import FluxFlowProcessor
from .vae import FluxCompressor, FluxExpander

logger = logging.getLogger(__name__)

# Checkpoint format versions
CHECKPOINT_VERSION = "1.0"
SUPPORTED_CHECKPOINT_VERSIONS = ["1.0"]


class FluxPipeline(nn.Module):
    """
    Complete FluxFlow pipeline combining compressor, flow processor, and expander.

    This is the main model wrapper that orchestrates:
    1. Compression: Image → Latent tokens (VAE encoder)
    2. Flow processing: Latent diffusion/flow prediction (optional)
    3. Expansion: Latent tokens → Image (VAE decoder)

    Args:
        compressor: FluxCompressor instance
        flow_processor: FluxFlowProcessor instance
        expander: FluxExpander instance
    """

    def __init__(self, compressor, flow_processor, expander):
        super().__init__()
        self.compressor = compressor
        self.flow_processor = flow_processor
        self.expander = expander

        # Keep ctx_tokens consistent across submodules
        K = getattr(self.compressor, "ctx_tokens", None) or getattr(
            self.compressor, "downscales", 4
        )
        if hasattr(self.flow_processor, "ctx_tokens"):
            self.flow_processor.ctx_tokens = K
        if hasattr(self.expander, "ctx_tokens"):
            self.expander.ctx_tokens = K

    def forward(self, img, text_embeddings=None, timesteps=None, use_flow=True):
        """
        Args:
            img: Input image [B, C, H, W]
            text_embeddings: Text conditioning [B, D] (required if use_flow=True)
            timesteps: Diffusion timesteps [B] (required if use_flow=True)
            use_flow: Enable flow processing (default: True)

        Returns:
            Generated/reconstructed image [B, C, H, W]
        """
        packed = self.compressor(img)

        if use_flow:
            if text_embeddings is None or timesteps is None:
                raise ValueError("Missing text_embeddings or timesteps when use_flow=True")
            packed = self.flow_processor(packed, text_embeddings, timesteps)

        return self.expander(packed)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_versioning: bool = False,
        **kwargs,
    ) -> "FluxPipeline":
        """
        Load a FluxPipeline from a pretrained checkpoint.

        Args:
            checkpoint_path: Path to .safetensors or .pt checkpoint file
            device: Device to load model on ('cuda', 'cpu', 'mps', or None for auto)
            use_versioning: Use versioned loading system (default: False for backward compatibility)
            **kwargs: Additional arguments for model initialization

        Returns:
            FluxPipeline: Loaded model ready for inference

        Example:
            >>> pipeline = FluxPipeline.from_pretrained("path/to/checkpoint.safetensors")
            >>> image = pipeline(img, text_embeddings, timesteps)

            >>> # Use versioned loading for better compatibility
            >>> pipeline = FluxPipeline.from_pretrained(
            ...     "path/to/checkpoint/",
            ...     use_versioning=True
            ... )
        """
        # Use new versioned loading system if requested
        if use_versioning:
            from pathlib import Path

            from .versioning import load_versioned_checkpoint

            # Type assertion: load_versioned_checkpoint returns FluxPipeline
            return cast(
                "FluxPipeline", load_versioned_checkpoint(Path(checkpoint_path), device, **kwargs)
            )

        # Legacy loading path (default for backward compatibility)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Load checkpoint
        if checkpoint_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Validate checkpoint version
        cls._validate_checkpoint_version(state_dict)

        # Detect configuration from checkpoint
        config = cls._detect_config(state_dict)

        # Override with any provided kwargs
        config.update(kwargs)

        # Initialize models
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

        # Create pipeline
        pipeline = cls(compressor, flow_processor, expander)

        # Load weights
        diffuser_state = {
            k.replace("diffuser.", ""): v
            for k, v in state_dict.items()
            if k.startswith("diffuser.")
        }

        # If no 'diffuser.' prefix, try loading directly
        if not diffuser_state:
            diffuser_state = state_dict

        pipeline.load_state_dict(diffuser_state, strict=False)
        pipeline.to(device)
        pipeline.eval()

        return pipeline

    @staticmethod
    def _detect_config(state_dict: dict) -> dict:
        """
        Detect model configuration from checkpoint state dict.

        Args:
            state_dict: Model state dictionary

        Returns:
            dict: Configuration parameters
        """
        config = {}
        keys = list(state_dict.keys())

        # Detect VAE dimension from compressor.latent_proj
        for key in keys:
            if "compressor.latent_proj" in key and ".0.0.weight" in key:
                shape = state_dict[key].shape
                if len(shape) == 4:
                    config["vae_dim"] = shape[0] // 5  # Bezier activation outputs 5x
                    break

        # Detect flow dimension from vae_to_dmodel
        for key in keys:
            if "flow_processor.vae_to_dmodel.weight" in key:
                shape = state_dict[key].shape
                config["flow_dim"] = shape[0]
                if "vae_dim" not in config:
                    config["vae_dim"] = shape[1]
                break

        # Detect text embedding dimension
        for key in keys:
            if "flow_processor.text_proj.weight" in key:
                shape = state_dict[key].shape
                config["text_embed_dim"] = shape[1]
                break

        # Default text_embed_dim if not found
        if "text_embed_dim" not in config:
            config["text_embed_dim"] = 768

        # Detect downscales
        encoder_stages = [k for k in keys if "compressor.encoder_z." in k and ".0.weight" in k]
        config["downscales"] = len(encoder_stages) if encoder_stages else 4

        # Detect upscales
        upscale_layers = [
            k for k in keys if "expander.upscale.layers." in k and ".conv1.0.weight" in k
        ]
        config["upscales"] = len(upscale_layers) if upscale_layers else config["downscales"]

        # Detect attention layers
        attn_layers = [
            k for k in keys if "compressor.token_attn." in k and ".attn.in_proj_weight" in k
        ]
        config["vae_attn_layers"] = len(attn_layers) if attn_layers else 2

        # Detect transformer blocks
        transformer_blocks = [
            k
            for k in keys
            if "flow_processor.transformer_blocks." in k and ".self_attn.q_proj.weight" in k
        ]
        config["flow_transformer_layers"] = len(transformer_blocks) if transformer_blocks else 10

        # Detect number of attention heads from rotary PE buffer
        for key in keys:
            if "flow_processor.transformer_blocks.0.rotary_pe.inv_freq" in key:
                inv_freq_size = state_dict[key].shape[0]
                head_dim = inv_freq_size * 2  # inv_freq is dim // 2
                config["flow_attn_heads"] = config["flow_dim"] // head_dim
                break

        # Set default max_hw
        config["max_hw"] = 1024

        # Validate required config
        if "vae_dim" not in config:
            raise ValueError("Could not detect VAE dimension from checkpoint")
        if "flow_dim" not in config:
            raise ValueError("Could not detect flow dimension from checkpoint")

        return config

    @classmethod
    def _validate_checkpoint_version(cls, state_dict: dict) -> None:
        """
        Validate checkpoint version compatibility.

        Args:
            state_dict: Loaded checkpoint state dictionary

        Raises:
            ValueError: If checkpoint version is incompatible

        Note:
            Logs warning for unknown versions but allows loading.
            Future versions may be incompatible and will raise an error.
        """
        version = state_dict.get("__version__") or state_dict.get("_version")

        if version is None:
            # Legacy checkpoint without version - assume compatible
            logger.debug("No version info in checkpoint, assuming compatible format")
            return

        if isinstance(version, torch.Tensor):
            # Handle case where version might be stored as tensor
            version = str(version.item()) if version.numel() == 1 else str(version.tolist())

        version = str(version)

        if version in SUPPORTED_CHECKPOINT_VERSIONS:
            logger.debug(f"Checkpoint version {version} is supported")
            return

        # Check if it's a newer version (might be forward-incompatible)
        try:
            current_major = int(CHECKPOINT_VERSION.split(".")[0])
            checkpoint_major = int(version.split(".")[0])

            if checkpoint_major > current_major:
                raise ValueError(
                    f"Checkpoint version {version} is newer than supported version "
                    f"{CHECKPOINT_VERSION}. Please update fluxflow to load this checkpoint."
                )
        except (ValueError, IndexError):
            pass

        # Unknown version - warn but try to load
        logger.warning(
            f"Unknown checkpoint version {version}. "
            f"Supported versions: {SUPPORTED_CHECKPOINT_VERSIONS}. "
            f"Loading may fail or produce unexpected results."
        )
