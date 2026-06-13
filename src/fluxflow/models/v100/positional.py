"""2D axial Rotary Positional Embedding for v0.10.0 flow transformer."""

import math

import torch


def build_axial_rope_2d(
    H: int,
    W: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build axial 2D RoPE sin/cos buffers for an H x W token grid.

    Returns four buffers — ``sin_w``/``cos_w`` for W-axis rotation and
    ``sin_h``/``cos_h`` for H-axis rotation. M4 callers slice the ``head_dim``
    in half and apply each pair independently via ``apply_rotary``::

        half = head_dim // 2
        q_w = apply_rotary(q[..., :half], sin_w, cos_w)
        q_h = apply_rotary(q[..., half:], sin_h, cos_h)
        q_rotated = torch.cat([q_w, q_h], dim=-1)

    This is the standard axial 2D RoPE formulation used in Flux/SD3/Pixtral:
    the W-axis and H-axis rotations are mathematically independent because
    each operates on its own ``head_dim // 2`` slice of the head dimension.

    Note:
        ``head_dim >= 8`` is recommended; with ``head_dim = 4``, each half has
        only one frequency and the embedding loses spatial discriminability
        quickly.

        Callers are responsible for caching the returned buffers across
        forward passes if reuse is desired — this helper allocates fresh
        tensors on every call.

    Args:
        H: Latent height in tokens.
        W: Latent width in tokens.
        head_dim: Attention head dimension. Must be divisible by 4 so each
            half-axis has an even number of dims for sin/cos pairing.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tuple ``(sin_w, cos_w, sin_h, cos_h)``, each of shape
        ``[H*W, head_dim // 2]``.
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

    sin_w_1d, cos_w_1d = _rope_1d(W, half)  # [W, half] — varies along columns
    sin_h_1d, cos_h_1d = _rope_1d(H, half)  # [H, half] — varies along rows

    # Broadcast each 1D buffer over the orthogonal axis to form the H x W grid.
    sin_w = sin_w_1d[None, :, :].expand(H, W, half).reshape(H * W, half)
    cos_w = cos_w_1d[None, :, :].expand(H, W, half).reshape(H * W, half)
    sin_h = sin_h_1d[:, None, :].expand(H, W, half).reshape(H * W, half)
    cos_h = cos_h_1d[:, None, :].expand(H, W, half).reshape(H * W, half)

    return sin_w, cos_w, sin_h, cos_h


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Continuous sinusoidal time embedding (DDPM-style).

    Replaces the v0.10.0 Embedding(1000) discrete time index with a continuous
    embedding so the flow can be conditioned on any t in [0, 1] without
    quantization loss.

    Args:
        t: Time tensor of shape [B] with values typically in [0, 1].
        dim: Embedding dimension (must be even).

    Returns:
        Embedding tensor of shape [B, dim].
    """
    assert dim % 2 == 0, f"sinusoidal_embedding dim must be even; got {dim}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )  # [half]
    # Frequencies already span 4 orders of magnitude (1.0 down to 1e-4), so
    # t in [0, 1] yields enough phase spread across `half` channels to keep
    # nearby timesteps distinguishable without an extra scale factor.
    args = t[:, None] * freqs[None, :]  # [B, half]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
