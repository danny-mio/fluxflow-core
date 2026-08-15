"""MLX inference pipeline for FluxFlow — Apple Silicon native."""

from __future__ import annotations

import os

import mlx.core as mx

from fluxflow.exceptions import ModelArchitectureError
from fluxflow.mlx.convert import convert_checkpoint
from fluxflow.mlx.layers.vae import FluxExpander

# Substrings unique to the v0.10.0 torch decoder (SPADE_v100b heads +
# seam-smoother). None of these appear anywhere in the MLX FluxExpander's
# v0.7.0/v0.8.0-era single-head SPADE, so any match means the checkpoint
# being loaded is architecturally newer than this backend.
_V100_ONLY_KEY_MARKERS = (
    "beta_low",
    "beta_mid",
    "beta_hi",
    "gamma_head",
    "gamma_scale",
    "seam_smoother",
)


def _check_not_v100_only_keys(arrays: dict[str, mx.array]) -> None:
    """Reject array dicts that carry v0.10.0-only decoder keys.

    The MLX decoder (`FluxExpander`) has not been ported past its
    v0.7.0/v0.8.0-era single-head SPADE architecture — it has no
    `SPADE_v100b` heads (`beta_low`/`beta_mid`/`beta_hi`/`gamma_head`/
    `gamma_scale`) and no `seam_smoother`/`seam_smoother_ctx`. Loading a
    genuine v0.10.0 checkpoint here is expected to fail loudly at
    `load_weights` (unmatched keys) rather than silently corrupt — this is
    analytically well-supported by MLX's default strict key-matching but has
    NOT been empirically confirmed (no MLX runtime available in the Linux
    review environment this guard was written in). This check exists to
    fail fast with a clearer message before `load_weights` is ever reached.

    Args:
        arrays: Flat dict of checkpoint keys to MLX arrays, as returned by
            `mx.load()` on the converted `.npz`.

    Raises:
        ModelArchitectureError: If any key contains a v0.10.0-only marker
            substring.
    """
    for key in arrays:
        for marker in _V100_ONLY_KEY_MARKERS:
            if marker in key:
                raise ModelArchitectureError(
                    f"Checkpoint key '{key}' matches v0.10.0-only marker "
                    f"'{marker}' (SPADE_v100b / seam-smoother). This looks "
                    "like a v0.10.0 checkpoint, but the MLX backend "
                    "(FluxExpander) has not been ported past its "
                    "v0.7.0/v0.8.0 single-head SPADE architecture. Loading "
                    "it is expected to fail loudly rather than silently "
                    "produce corrupt output — this MLX guard just fails "
                    "fast with a clearer message before weight loading."
                )


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

        Raises:
            ModelArchitectureError: If the checkpoint contains v0.10.0-only
                SPADE_v100b/seam-smoother keys — this MLX backend has not
                been ported past v0.7.0/v0.8.0 and cannot load it.
        """
        base, _ = os.path.splitext(checkpoint_path)
        npz_path = base + ".npz"
        npz_stale = not os.path.exists(npz_path) or (
            os.path.getmtime(checkpoint_path) > os.path.getmtime(npz_path)
        )
        if npz_stale:
            convert_checkpoint(checkpoint_path, npz_path)

        arrays = dict(mx.load(npz_path))
        _check_not_v100_only_keys(arrays)
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
