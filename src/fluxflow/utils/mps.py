"""MPS-safe wrappers for PyTorch ops that have MPS limitations."""

import torch
import torch.nn.functional as F


def mps_safe_pool2d(x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """
    Adaptive average pooling that works on MPS even with non-divisible sizes.

    MPS raises RuntimeError when spatial dims are not divisible by output size.
    Falls back to interpolate + mean for those cases.

    Args:
        x: Input tensor [B, C, H, W]
        output_size: Target spatial size (h, w)

    Returns:
        Pooled tensor [B, C, output_size[0], output_size[1]]
    """
    try:
        return F.adaptive_avg_pool2d(x, output_size)
    except RuntimeError as e:
        if "MPS" in str(e) or "divisible" in str(e):
            # Interpolate to exact size then collapse spatial dims
            x = F.interpolate(
                x.float(),
                size=output_size,
                mode="bilinear",
                align_corners=False,
            ).to(x.dtype)
            return x
        raise
