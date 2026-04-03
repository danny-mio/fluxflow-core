"""Convert FluxFlow PyTorch .safetensors checkpoint to MLX .npz format."""

from __future__ import annotations

import os

import mlx.core as mx
from safetensors.torch import load_file

# Key patterns that identify ConvTranspose2d weights in the FluxFlow architecture.
#
# From v070/vae.py:
#   ResidualUpsampleBlock.conv1 = nn.Sequential(
#       nn.ConvTranspose2d(...),  # index 0  → key fragment "conv1.0"
#       BezierActivation(...),
#       nn.Conv2d(...),           # index 2  → regular Conv2d
#       ...
#   )
#   BaselineResidualUpsampleBlock.conv_sequence[0] = nn.ConvTranspose2d(...)
#       → key fragment "conv_sequence.0"
#
# nn.Upsample (skip_upsample) has no learnable weights — excluded automatically.
_CONV_TRANSPOSE_PATTERNS = ("conv1.0.", "conv_sequence.0.")


def _is_conv2d_weight(key: str, shape: tuple) -> bool:
    """Detect regular Conv2d 4D weight tensors [out, in, kH, kW]."""
    return (
        len(shape) == 4
        and key.endswith(".weight")
        and not any(pat in key for pat in _CONV_TRANSPOSE_PATTERNS)
    )


def _is_conv_transpose_weight(key: str, shape: tuple) -> bool:
    """Detect ConvTranspose2d 4D weight tensors [in, out, kH, kW]."""
    return (
        len(shape) == 4
        and key.endswith(".weight")
        and any(pat in key for pat in _CONV_TRANSPOSE_PATTERNS)
    )


def convert_checkpoint(src_path: str, dst_path: str | None = None) -> str:
    """
    Convert a FluxFlow .safetensors checkpoint to MLX .npz format.

    Conv2d weights are transposed from PyTorch [out, in, kH, kW] to
    MLX [out, kH, kW, in]. ConvTranspose2d weights are transposed from
    [in, out, kH, kW] to [out, kH, kW, in]. All other tensors are
    layout-compatible and copied without transposition.

    The original .safetensors file is never modified.

    Args:
        src_path: Path to existing .safetensors file.
        dst_path: Output path. Defaults to src_path with .npz extension.

    Returns:
        Path to the written .npz file.
    """
    if dst_path is None:
        base, _ = os.path.splitext(src_path)
        dst_path = base + ".npz"

    pt_state = load_file(src_path)
    mlx_arrays: dict[str, mx.array] = {}

    for key, tensor in pt_state.items():
        np_arr = tensor.float().numpy()
        shape = np_arr.shape

        if _is_conv_transpose_weight(key, shape):
            # PyTorch ConvTranspose2d [in, out, kH, kW] → MLX [out, kH, kW, in]
            np_arr = np_arr.transpose(1, 2, 3, 0)
        elif _is_conv2d_weight(key, shape):
            # PyTorch Conv2d [out, in, kH, kW] → MLX [out, kH, kW, in]
            np_arr = np_arr.transpose(0, 2, 3, 1)

        mlx_arrays[key] = mx.array(np_arr)

    mx.savez(dst_path, **mlx_arrays)
    return dst_path
