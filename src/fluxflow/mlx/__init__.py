"""
Apple Silicon native inference via MLX.

Install with: pip install fluxflow[mlx]
"""

try:
    import mlx.core as mx  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Apple Silicon inference requires MLX. " "Install with: pip install fluxflow[mlx]"
    ) from e

from fluxflow.mlx.pipeline import FluxFlowPipelineMLX  # noqa: F401
from fluxflow.mlx.convert import convert_checkpoint  # noqa: F401

__all__ = ["FluxFlowPipelineMLX", "convert_checkpoint"]
