"""MLX inference pipeline for FluxFlow — Apple Silicon native."""

from __future__ import annotations

import os

import mlx.core as mx

from fluxflow.mlx.convert import convert_checkpoint
from fluxflow.mlx.layers.vae import FluxExpander


class FluxFlowPipelineMLX:
    """
    Inference-only pipeline for Apple Silicon.

    Usage:
        pipe = FluxFlowPipelineMLX.from_checkpoint("path/to/model.safetensors")
        image = pipe.decode(packed)  # packed: mx.array [B, T+1, D]
    """

    def __init__(self, expander: FluxExpander) -> None:
        self.expander = expander

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        d_model: int = 128,
        upscales: int = 4,
        max_hw: int = 1024,
    ) -> "FluxFlowPipelineMLX":
        """
        Load FluxFlowPipelineMLX from a .safetensors checkpoint.

        Converts to MLX .npz on first call; subsequent calls reuse the cached file.

        Args:
            checkpoint_path: Path to .safetensors checkpoint.
            d_model: Latent dimension (must match checkpoint).
            upscales: Number of upscale blocks (must match checkpoint).
            max_hw: Maximum spatial dimension used during training.

        Returns:
            Loaded FluxFlowPipelineMLX instance.
        """
        base, _ = os.path.splitext(checkpoint_path)
        npz_path = base + ".npz"
        npz_stale = not os.path.exists(npz_path) or (
            os.path.getmtime(checkpoint_path) > os.path.getmtime(npz_path)
        )
        if npz_stale:
            convert_checkpoint(checkpoint_path, npz_path)

        arrays = dict(mx.load(npz_path))
        expander = FluxExpander(d_model=d_model, upscales=upscales, max_hw=max_hw)

        # Strip "expander." prefix and load weights
        exp_weights = [
            (k[len("expander.") :], v) for k, v in arrays.items() if k.startswith("expander.")
        ]
        expander.load_weights(exp_weights)
        mx.eval(expander.parameters())
        return cls(expander)

    def decode(self, packed: mx.array) -> mx.array:
        """
        Decode packed latent representation to image.

        Args:
            packed: [B, T+1, d_model+CONTEXT_DIMS] packed latent tensor.

        Returns:
            Image tensor [B, 3, H, W] in [-1, 1].
        """
        return self.expander(packed)
