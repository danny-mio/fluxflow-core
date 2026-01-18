"""
Model factory for creating Bezier or Baseline FluxFlow models.

This module provides a unified interface for building either:
- Bezier models (original architecture with BezierActivation)
- Baseline models (standard activations: SiLU/GELU/ReLU)

The factory ensures proper component compatibility and parameter matching.
"""

import importlib
from pathlib import Path
from typing import Any, Literal, Optional

import torch.nn as nn

from . import registry
from .encoders import BertTextEncoder
from .v060.flow import BaselineFluxFlowProcessor
from .v060.vae import BaselineFluxExpander, BaselineResidualUpsampleBlock

# Import the registry to ensure it's available for version modules
ModelClassRegistry = registry.ModelClassRegistry


# Automatically discover and import all version modules
# This enables truly self-registering versions - no code changes needed for new versions


def _auto_discover_versions():
    """Automatically discover and import all version modules."""
    # Get the directory containing this factory.py file
    models_dir = Path(__file__).parent

    # Find all directories that match version pattern (vXXX)
    for item in models_dir.iterdir():
        if (
            item.is_dir()
            and item.name.startswith("v")
            and len(item.name) >= 4  # vXXX minimum
            and item.name[1:].isdigit()
            and (item / "__init__.py").exists()
        ):  # Must have __init__.py

            try:
                # Import the version module to trigger self-registration
                # __name__ is 'fluxflow.models.factory', parent is 'fluxflow.models'
                parent_package = ".".join(__name__.split(".")[:-1])
                version_module_name = item.name
                full_module_name = f"{parent_package}.{version_module_name}"
                importlib.import_module(full_module_name)
            except ImportError:
                # Skip modules that can't be imported
                continue


# Auto-discover all versions
_auto_discover_versions()

ModelType = Literal["bezier", "baseline"]
ActivationType = Literal["silu", "gelu", "relu"]


class ModelFactory:
    """
    Factory for creating Bezier or Baseline FluxFlow models with versioning support.

    Usage:
        # Create Bezier models (original) - defaults to v0.6.0
        factory = ModelFactory(model_type="bezier")
        vae_encoder = factory.create_vae_encoder()
        vae_decoder = factory.create_vae_decoder()
        flow = factory.create_flow_processor()

        # Create versioned Bezier models
        factory = ModelFactory(model_type="bezier", model_version="0.3.0")
        # or
        factory = ModelFactory(model_type="bezier", model_version="0.6.0")

        # Create Baseline models (for comparison)
        factory = ModelFactory(
            model_type="baseline",
            baseline_activation="silu",
            baseline_vae_width_mult=4.5,
            baseline_vae_depth_mult=1.0,
            baseline_flow_blocks=17,
        )
        vae_encoder = factory.create_vae_encoder()
        vae_decoder = factory.create_vae_decoder()
        flow = factory.create_flow_processor()
    """

    def __init__(
        self,
        model_type: ModelType = "bezier",
        model_version: str = "0.6.0",
        # Common config
        vae_dim: int = 128,
        feature_maps_dim: int = 128,
        flow_d_model: int = 512,
        flow_embedding_size: int = 1024,
        # Baseline-specific config
        baseline_activation: ActivationType = "silu",
        baseline_vae_width_mult: float = 4.5,
        baseline_vae_depth_mult: float = 1.0,
        baseline_flow_blocks: int = 17,
        baseline_flow_ffn_expansion: float = 4.0,
        # Bezier-specific config (uses defaults)
        bezier_flow_blocks: int = 10,
        # Version-specific config
        v060_downscales: int = 3,  # v0.6.0: Reduced from 4 (8x instead of 16x compression)
        v030_downscales: int = 4,  # v0.3.0: Original compression (16x)
    ):
        """
        Initialize model factory.

        Args:
            model_type: "bezier" or "baseline"
            vae_dim: Latent dimension (MUST be same for both model types)
            feature_maps_dim: VAE feature map dimension
            flow_d_model: Flow transformer internal dimension
            flow_embedding_size: Text embedding size (from BertTextEncoder)

            # Baseline-specific:
            baseline_activation: Activation function ("silu", "gelu", "relu")
            baseline_vae_width_mult: VAE width multiplier (4.5 matches Bezier 5.0)
            baseline_vae_depth_mult: VAE depth multiplier (1.0 = single layer)
            baseline_flow_blocks: Number of transformer blocks (17 vs Bezier's 12)
            baseline_flow_ffn_expansion: FFN expansion factor (4.0 standard)

            # Bezier-specific:
            bezier_flow_blocks: Number of transformer blocks (12 default)
        """
        self.model_type = model_type
        self.model_version = model_version
        self.vae_dim = vae_dim
        self.feature_maps_dim = feature_maps_dim
        self.flow_d_model = flow_d_model
        self.flow_embedding_size = flow_embedding_size

        # Baseline config
        self.baseline_activation = baseline_activation
        self.baseline_vae_width_mult = baseline_vae_width_mult
        self.baseline_vae_depth_mult = baseline_vae_depth_mult
        self.baseline_flow_blocks = baseline_flow_blocks
        self.baseline_flow_ffn_expansion = baseline_flow_ffn_expansion

        # Bezier config
        self.bezier_flow_blocks = bezier_flow_blocks

    def create_vae_encoder(
        self,
        in_channels: int = 3,
        downscales: int = 4,
        max_hw: int = 1024,
        use_attention: bool = True,
        attn_layers: int = 4,
        attn_heads: int = 8,
        attn_ff_mult: int = 2,
        attn_dropout: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ) -> nn.Module:
        """
        Create VAE encoder (FluxCompressor).

        For versioned bezier models, uses the appropriate versioned FluxCompressor.
        For baseline models, uses the current implementation.

        Returns:
            FluxCompressor instance
        """
        # Get the appropriate class based on model type and version
        classes = self._get_versioned_classes()
        FluxCompressor = classes["FluxCompressor"]

        return FluxCompressor(  # type: ignore
            in_channels=in_channels,
            d_model=self.vae_dim,  # Latent dimension
            downscales=downscales,
            max_hw=max_hw,
            use_attention=use_attention,
            attn_layers=attn_layers,
            attn_heads=attn_heads,
            attn_ff_mult=attn_ff_mult,
            attn_dropout=attn_dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def create_vae_decoder(
        self,
        upscales: int = 4,
        max_hw: int = 1024,
        ctx_tokens: int = 4,
        use_gradient_checkpointing: bool = True,
    ) -> nn.Module:
        """
        Create VAE decoder (FluxExpander).

        For versioned bezier models, uses the appropriate versioned FluxExpander.
        For Baseline models, creates a modified expander with BaselineResidualUpsampleBlock.

        Returns:
            FluxExpander instance (Bezier or Baseline variant)
        """
        if self.model_type == "bezier":
            # Use versioned FluxExpander for bezier models
            classes = self._get_versioned_classes()
            FluxExpander = classes["FluxExpander"]
            return FluxExpander(  # type: ignore
                d_model=self.vae_dim,
                upscales=upscales,
                max_hw=max_hw,
                ctx_tokens=ctx_tokens,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            # Baseline: Create custom expander with baseline blocks
            return self._create_baseline_expander(
                d_model=self.vae_dim,
                upscales=upscales,
                max_hw=max_hw,
                ctx_tokens=ctx_tokens,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

    def _create_baseline_expander(
        self,
        d_model: int,
        upscales: int,
        max_hw: int,
        ctx_tokens: int,
        use_gradient_checkpointing: bool,
    ) -> nn.Module:
        """
        Create baseline variant of FluxExpander.

        Uses BaselineFluxExpander with BaselineResidualUpsampleBlock.
        """
        return BaselineFluxExpander(
            d_model=d_model,
            upscales=upscales,
            max_hw=max_hw,
            ctx_tokens=ctx_tokens,
            baseline_activation=self.baseline_activation,
            width_multiplier=self.baseline_vae_width_mult,
            depth_multiplier=self.baseline_vae_depth_mult,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def create_baseline_upsampler(
        self,
        channels: int,
        steps: int,
        context_size: int = 1024,
        use_gradient_checkpointing: bool = True,
    ) -> nn.Module:
        """
        Create baseline upsampler using BaselineResidualUpsampleBlock.

        This is a helper for building baseline decoders.

        Args:
            channels: Number of channels
            steps: Number of upsampling steps
            context_size: Context dimension for SPADE
            use_gradient_checkpointing: Enable gradient checkpointing

        Returns:
            ProgressiveUpscaler with baseline blocks
        """
        # Create progressive upscaler with baseline blocks
        from functools import partial

        from torch.utils.checkpoint import checkpoint

        class BaselineProgressiveUpscaler(nn.Module):
            """Progressive upscaler using baseline blocks."""

            def __init__(
                self, channels, steps, context_size, use_spade, use_gradient_checkpointing
            ):
                super().__init__()
                self.use_gradient_checkpointing = use_gradient_checkpointing
                self.layers = nn.ModuleList(
                    [
                        BaselineResidualUpsampleBlock(
                            channels=channels,
                            context_size=context_size,
                            use_spade=use_spade,
                            baseline_activation=baseline_activation,
                            width_multiplier=baseline_vae_width_mult,
                            depth_multiplier=baseline_vae_depth_mult,
                        )
                        for _ in range(steps)
                    ]
                )

            def forward(self, x, context=None):
                def upscale_all(x, context):
                    for layer in self.layers:
                        x = layer(x, context)
                    return x

                if self.use_gradient_checkpointing:
                    return checkpoint(partial(upscale_all), x, context, use_reentrant=False)
                else:
                    return upscale_all(x, context)

        baseline_activation = self.baseline_activation
        baseline_vae_width_mult = self.baseline_vae_width_mult
        baseline_vae_depth_mult = self.baseline_vae_depth_mult

        return BaselineProgressiveUpscaler(
            channels=channels,
            steps=steps,
            context_size=context_size,
            use_spade=True,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def create_flow_processor(
        self,
        n_head: int = 8,
        max_hw: int = 1024,
        ctx_tokens: int = 4,
    ) -> nn.Module:
        """
        Create flow processor (FluxFlowProcessor).

        For versioned bezier models, uses the appropriate versioned FluxFlowProcessor.
        For Baseline models, creates a modified processor with BaselineFluxTransformerBlock.

        Returns:
            FluxFlowProcessor instance (Bezier or Baseline variant)
        """
        n_layers = (
            self.baseline_flow_blocks if self.model_type == "baseline" else self.bezier_flow_blocks
        )

        if self.model_type == "bezier":
            # Use versioned FluxFlowProcessor for bezier models
            classes = self._get_versioned_classes()
            FluxFlowProcessor = classes["FluxFlowProcessor"]
            return FluxFlowProcessor(  # type: ignore
                d_model=self.flow_d_model,
                vae_dim=self.vae_dim,
                embedding_size=self.flow_embedding_size,
                n_head=n_head,
                n_layers=n_layers,
                max_hw=max_hw,
                ctx_tokens=ctx_tokens,
            )
        else:
            # Baseline: Create custom processor with baseline blocks
            return self._create_baseline_flow_processor(
                d_model=self.flow_d_model,
                vae_dim=self.vae_dim,
                embedding_size=self.flow_embedding_size,
                n_head=n_head,
                n_layers=n_layers,
                max_hw=max_hw,
                ctx_tokens=ctx_tokens,
            )

    def _create_baseline_flow_processor(
        self,
        d_model: int,
        vae_dim: int,
        embedding_size: int,
        n_head: int,
        n_layers: int,
        max_hw: int,
        ctx_tokens: int,
    ) -> nn.Module:
        """
        Create baseline variant of FluxFlowProcessor.

        Uses BaselineFluxFlowProcessor with BaselineFluxTransformerBlock.
        """
        return BaselineFluxFlowProcessor(
            d_model=d_model,
            vae_dim=vae_dim,
            embedding_size=embedding_size,
            n_head=n_head,
            n_layers=n_layers,
            max_hw=max_hw,
            ctx_tokens=ctx_tokens,
            baseline_activation=self.baseline_activation,
            ffn_expansion=self.baseline_flow_ffn_expansion,
        )

    def create_text_encoder(
        self,
        embed_dim: int = 1024,
        pretrain_model: Optional[str] = None,
        frozen: bool = True,
    ) -> nn.Module:
        """
        Create text encoder (BertTextEncoder).

        CRITICAL: Text encoder is ALWAYS the same (Bezier with frozen weights)
        for both Bezier and Baseline experiments. This isolates architectural
        differences to VAE + Flow only.

        Args:
            embed_dim: Output embedding dimension (1024 default)
            pretrain_model: Pretrained model name or None
            frozen: Freeze all parameters (default: True)

        Returns:
            BertTextEncoder with Bezier activations (frozen)
        """
        encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=pretrain_model)

        if frozen:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

    def _get_versioned_classes(self) -> dict[str, Any]:
        """
        Get versioned classes for VAE and Flow components.

        Uses the self-registering ModelClassRegistry for bezier models.
        For baseline or unspecified versions, uses current implementations.

        Returns:
            Dict with 'FluxCompressor', 'FluxExpander', 'FluxFlowProcessor' classes
        """
        if self.model_type == "bezier":
            # Try to get registered classes for this version
            registered_classes = ModelClassRegistry.get_classes(self.model_version)
            if registered_classes:
                return registered_classes  # type: ignore

            # Fallback: try to dynamically import version module
            try:
                version_module = f".v{self.model_version.replace('.', '')}"
                _ = __import__(version_module, fromlist=[""], globals=globals())
                # If the module has registered itself, it should be in the registry now
                registered_classes = ModelClassRegistry.get_classes(self.model_version)
                if registered_classes:
                    return registered_classes  # type: ignore
            except ImportError:
                pass

        # Fallback to current implementations (baseline or default bezier)
        from .v060.flow import FluxFlowProcessor as FFP_v060
        from .v060.vae import FluxCompressor as FC_v060
        from .v060.vae import FluxExpander as FE_v060

        return {
            "FluxCompressor": FC_v060,
            "FluxExpander": FE_v060,
            "FluxFlowProcessor": FFP_v060,
        }

    def get_config(self) -> dict[str, Any]:
        """
        Get current factory configuration.

        Returns:
            Configuration dictionary
        """
        config: dict[str, Any] = {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "vae_dim": self.vae_dim,
            "feature_maps_dim": self.feature_maps_dim,
            "flow_d_model": self.flow_d_model,
            "flow_embedding_size": self.flow_embedding_size,
        }

        if self.model_type == "baseline":
            config.update(
                {
                    "baseline_activation": self.baseline_activation,
                    "baseline_vae_width_mult": self.baseline_vae_width_mult,
                    "baseline_vae_depth_mult": self.baseline_vae_depth_mult,
                    "baseline_flow_blocks": self.baseline_flow_blocks,
                    "baseline_flow_ffn_expansion": self.baseline_flow_ffn_expansion,
                }
            )
        else:
            config.update(
                {
                    "bezier_flow_blocks": self.bezier_flow_blocks,
                }
            )

        return config

    @staticmethod
    def from_config(config: dict[str, Any]) -> "ModelFactory":
        """
        Create factory from configuration dictionary.

        Args:
            config: Configuration dict (from get_config())

        Returns:
            ModelFactory instance
        """
        return ModelFactory(**config)


def create_bezier_models(
    vae_dim: int = 128,
    flow_d_model: int = 512,
    flow_embedding_size: int = 1024,
    model_version: str = "0.7.0",
    downscales: Optional[int] = None,
) -> tuple:
    """
    Convenience function to create full Bezier model set.

    Uses self-registering version system - no if statements needed for new versions.

    Returns:
        (vae_encoder, vae_decoder, flow_processor, text_encoder)
    """
    from .encoders import BertTextEncoder

    # Get classes from registry (will import and register the version module if needed)
    classes = ModelClassRegistry.get_classes(model_version)
    if classes is None:
        # Try to dynamically import the version module to trigger registration
        try:
            version_module = f"fluxflow.models.v{model_version.replace('.', '')}"
            __import__(version_module)
            classes = ModelClassRegistry.get_classes(model_version)
        except ImportError:
            pass

    if classes is None:
        raise ValueError(
            f"Unsupported model version: {model_version}. Available: {ModelClassRegistry.list_versions()}"
        )

    Compressor = classes["FluxCompressor"]
    Expander = classes["FluxExpander"]
    FlowProcessor = classes["FluxFlowProcessor"]

    # Get version-specific downscales
    if downscales is None:
        # All versions use 4 downscales for compatibility
        downscales = 4

    compressor = Compressor(
        in_channels=3,
        d_model=vae_dim,
        downscales=downscales,
        max_hw=1024,
        use_attention=True,
        attn_layers=4,
        attn_heads=8,
        attn_ff_mult=2,
        attn_dropout=0.0,
        use_gradient_checkpointing=True,
    )

    expander = Expander(
        d_model=vae_dim,
        upscales=downscales,
        max_hw=1024,
        ctx_tokens=4,
        use_gradient_checkpointing=True,
    )

    flow_processor = FlowProcessor(
        d_model=flow_d_model,
        vae_dim=vae_dim,
        embedding_size=flow_embedding_size,
        n_head=8,
        n_layers=10,
        max_hw=1024,
        ctx_tokens=4,
    )

    text_encoder = BertTextEncoder(embed_dim=flow_embedding_size)

    return compressor, expander, flow_processor, text_encoder


def create_baseline_models(
    vae_dim: int = 128,
    flow_d_model: int = 512,
    flow_embedding_size: int = 1024,
    activation: ActivationType = "silu",
    vae_width_mult: float = 4.5,
    vae_depth_mult: float = 1.0,
    flow_blocks: int = 17,
    model_version: str = "0.3.0",
) -> tuple:
    """
    Convenience function to create full Baseline model set.

    NOTE: Flow processor still raises NotImplementedError (Phase 1.2).
    VAE encoder and decoder are fully functional.

    Args:
        model_version: Version for compatibility (always uses 4 downscales)

    Returns:
        (vae_encoder, vae_decoder, flow_processor, text_encoder)
    """
    # Use consistent downscales for all versions
    downscales = 4
    upscales = downscales

    factory = ModelFactory(
        model_type="baseline",
        vae_dim=vae_dim,
        flow_d_model=flow_d_model,
        flow_embedding_size=flow_embedding_size,
        baseline_activation=activation,
        baseline_vae_width_mult=vae_width_mult,
        baseline_vae_depth_mult=vae_depth_mult,
        baseline_flow_blocks=flow_blocks,
    )

    vae_encoder = factory.create_vae_encoder(downscales=downscales)
    vae_decoder = factory.create_vae_decoder(upscales=upscales)
    flow = factory.create_flow_processor()  # Now implemented!
    text_encoder = factory.create_text_encoder(embed_dim=flow_embedding_size)

    return vae_encoder, vae_decoder, flow, text_encoder


def create_models_from_config(model_config) -> tuple:
    """
    Create models from a ModelConfig object (from fluxflow.config).

    Args:
        model_config: ModelConfig instance from fluxflow.config

    Returns:
        (vae_encoder, vae_decoder, flow_processor, text_encoder)

    Example:
        >>> from fluxflow.config import load_config
        >>> config = load_config("config.yaml")
        >>> models = create_models_from_config(config.model)
    """
    if model_config.model_type == "bezier":
        return create_bezier_models(
            vae_dim=model_config.vae_dim,
            flow_d_model=model_config.feature_maps_dim,
            flow_embedding_size=model_config.text_embedding_dim,
            model_version=model_config.model_version,
        )
    elif model_config.model_type == "baseline":
        return create_baseline_models(
            vae_dim=model_config.vae_dim,
            flow_d_model=model_config.feature_maps_dim,
            flow_embedding_size=model_config.text_embedding_dim,
            activation=model_config.baseline_activation,
            vae_width_mult=model_config.baseline_vae_width_mult,
            vae_depth_mult=model_config.baseline_vae_depth_mult,
            flow_blocks=model_config.baseline_flow_blocks,
            model_version=model_config.model_version,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_config.model_type}")
