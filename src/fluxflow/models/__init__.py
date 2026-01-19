"""FluxFlow model components."""

# Backward compatibility: allow imports like 'from fluxflow.models.vae import ...'
# by creating module aliases
import sys

from .activations import (
    BezierActivation,
    BezierActivationModule,
    Flip,
    Rot90,
    TrainableBezier,
    xavier_init,
)
from .bezier_jit import get_jit_bezier_function
from .bezier_power_cache import get_cache_stats, get_power_computation_fn, reset_cache_stats
from .conditioning import (
    DEFAULT_CONFIG_VALUE,
    SPADE,
    ContextAttentionMixer,
    FiLM,
    GatedContextInjection,
    LeanContext1D,
    LeanContext2D,
    LeanContextModule,
    stable_scale_text_embeddings,
)
from .diffusion_pipeline import FluxFlowPipeline, FluxFlowPipelineOutput
from .discriminators import DBlock, PatchDiscriminator
from .encoders import BertTextEncoder, ImageEncoder
from .factory import (
    ModelFactory,
    create_baseline_models,
    create_bezier_models,
    create_models_from_config,
)
from .pipeline import FluxPipeline

# Import from default version (v0.6.0) for backward compatibility
from .v060.flow import (
    FluxFlowProcessor,
    FluxTransformerBlock,
    ParallelAttention,
    RotaryPositionalEmbedding,
    pillarLayer,
)

# Import from default version (v0.6.0) for backward compatibility
from .v060.vae import (
    Clamp,
    FluxCompressor,
    FluxExpander,
    ProgressiveUpscaler,
    ResidualUpsampleBlock,
)

sys.modules["fluxflow.models.vae"] = sys.modules["fluxflow.models.v060.vae"]
sys.modules["fluxflow.models.flow"] = sys.modules["fluxflow.models.v060.flow"]

__all__ = [
    # Activations
    "BezierActivation",
    "BezierActivationModule",
    "TrainableBezier",
    "Flip",
    "Rot90",
    "xavier_init",
    # JIT and optimizations
    "get_jit_bezier_function",
    "get_power_computation_fn",
    "get_cache_stats",
    "reset_cache_stats",
    # Conditioning
    "FiLM",
    "SPADE",
    "GatedContextInjection",
    "LeanContextModule",
    "LeanContext2D",
    "LeanContext1D",
    "ContextAttentionMixer",
    "stable_scale_text_embeddings",
    "DEFAULT_CONFIG_VALUE",
    # VAE
    "FluxCompressor",
    "FluxExpander",
    "ResidualUpsampleBlock",
    "ProgressiveUpscaler",
    "Clamp",
    # Flow
    "FluxFlowProcessor",
    "FluxTransformerBlock",
    "RotaryPositionalEmbedding",
    "ParallelAttention",
    "pillarLayer",
    # Discriminators
    "PatchDiscriminator",
    "DBlock",
    # Encoders
    "BertTextEncoder",
    "ImageEncoder",
    # Pipeline
    "FluxPipeline",
    "FluxFlowPipeline",
    "FluxFlowPipelineOutput",
    # Factory
    "ModelFactory",
    "create_baseline_models",
    "create_bezier_models",
    "create_models_from_config",
]
