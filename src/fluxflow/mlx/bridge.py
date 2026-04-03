"""
Experimental: PyTorch ↔ MLX tensor bridge for gradient-free hybrid routing.

Use only for operations that do not need gradients (e.g. text encoder forward
pass during training, sample generation). Autograd is NOT supported.

On Apple Silicon, the PyTorch → numpy → MLX transfer is a metadata-only
operation (unified memory, zero-copy). On other platforms this falls back
to a CPU copy.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import numpy as np
import torch


def mlx_forward(
    x: torch.Tensor,
    mlx_fn: Callable[[mx.array], mx.array],
) -> torch.Tensor:
    """
    Run a gradient-free operation in MLX on a PyTorch tensor.

    Args:
        x: Input PyTorch tensor (any device). Gradients are detached.
        mlx_fn: Function mapping mx.array → mx.array.

    Returns:
        Output PyTorch tensor on the same device as the input.

    Warning:
        No gradient support. The output has ``requires_grad=False``.
        Use only for inference-time ops within training loops.
    """
    device = x.device
    np_arr = x.detach().cpu().numpy()
    mlx_arr = mx.array(np_arr)
    result = mlx_fn(mlx_arr)
    mx.eval(result)
    out_np = np.array(result)
    return torch.from_numpy(out_np).to(device=device, dtype=x.dtype)
