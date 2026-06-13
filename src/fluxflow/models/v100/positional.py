"""2D axial Rotary Positional Embedding for v0.10.0 flow transformer."""

import torch


def build_axial_rope_2d(
    H: int,
    W: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build axial 2D RoPE sin/cos buffers for an H x W token grid.

    The top half of ``head_dim`` rotates with W-position frequencies, the
    bottom half rotates with H-position frequencies. Each half uses standard
    1D RoPE math (interleaved sin/cos as in the v070
    ``RotaryPositionalEmbedding``).

    Args:
        H: Latent height in tokens.
        W: Latent width in tokens.
        head_dim: Attention head dimension. Must be divisible by 4 so each
            half-axis has an even number of dims for sin/cos pairing.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple ``(sin, cos)``, each of shape ``[H*W, head_dim]``, to feed into
        the existing ``RotaryPositionalEmbedding.apply_rotary`` helper.
    """
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for axial 2D RoPE; got {head_dim}"
    half = head_dim // 2

    def _rope_1d(L: int, C: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Mirrors v070 RotaryPositionalEmbedding.get_embed for one axis.
        inv_freq = 1.0 / (10000 ** (torch.arange(0, C // 2, device=device, dtype=dtype) / (C // 2)))
        pos = torch.arange(L, device=device, dtype=dtype)
        sinusoid = torch.einsum("i,j->ij", pos, inv_freq)  # [L, C/2]
        sin = sinusoid.sin().repeat_interleave(2, dim=-1)  # [L, C]
        cos = sinusoid.cos().repeat_interleave(2, dim=-1)  # [L, C]
        return sin, cos

    sin_w, cos_w = _rope_1d(W, half)  # along columns
    sin_h, cos_h = _rope_1d(H, half)  # along rows

    # Broadcast to H x W grid, then concatenate halves along head_dim.
    sin_w_full = sin_w[None, :, :].expand(H, W, half)  # [H, W, half]
    cos_w_full = cos_w[None, :, :].expand(H, W, half)
    sin_h_full = sin_h[:, None, :].expand(H, W, half)
    cos_h_full = cos_h[:, None, :].expand(H, W, half)

    sin = torch.cat([sin_w_full, sin_h_full], dim=-1).reshape(H * W, head_dim)
    cos = torch.cat([cos_w_full, cos_h_full], dim=-1).reshape(H * W, head_dim)

    return sin, cos
