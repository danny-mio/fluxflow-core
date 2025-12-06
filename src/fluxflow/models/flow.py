"""
Flow-based diffusion model components for FluxFlow.

Contains:
- FluxFlowProcessor: Main flow prediction model with transformers
- FluxTransformerBlock: Transformer block with Bezier activations
- RotaryPositionalEmbedding: Rotary position encoding
- ParallelAttention: Multi-head attention with separate Q and KV
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from .activations import BezierActivation, TrainableBezier, xavier_init
from .conditioning import ContextAttentionMixer, GatedContextInjection


def pillarLayer(
    in_size: int = 32, out_size: int = 32, depth: int = 2, activation=nn.SiLU()
) -> nn.Sequential:
    """
    Create a pillar (multi-layer perceptron) with specified depth.

    Args:
        in_size: Input dimension
        out_size: Output dimension (used for intermediate layers)
        depth: Number of layers
        activation: Activation function

    Returns:
        Sequential module with depth layers
    """
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(
                    (
                        in_size if i == 0 else out_size
                    ),  # First layer takes in_size, rest take out_size
                    (
                        out_size if i < depth - 1 else in_size
                    ),  # Last layer outputs in_size, rest output out_size
                ),
                activation,
            )
            for i in range(depth)
        ]
    )


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for transformers.

    Applies rotary encodings to queries and keys in attention mechanism.

    Args:
        dim: Embedding dimension (should be head dimension)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
        self.apply(xavier_init)

    def get_embed(self, pos_ids):
        """
        Generate sin/cos embeddings for given positions.

        Args:
            pos_ids: Position indices [seq_len]

        Returns:
            tuple: (sin, cos) embeddings
        """
        sinusoid = torch.einsum("i,j->ij", pos_ids, self.inv_freq)
        sin = sinusoid.sin().repeat_interleave(2, dim=-1)
        cos = sinusoid.cos().repeat_interleave(2, dim=-1)
        return sin, cos

    def apply_rotary(self, x, sin, cos):
        """
        Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor [..., dim]
            sin: Sin embeddings
            cos: Cos embeddings

        Returns:
            Rotated tensor
        """
        x1, x2 = x.chunk(2, dim=-1)
        sin = sin.reshape(1, 1, *sin.shape)
        cos = cos.reshape(1, 1, *cos.shape)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class ParallelAttention(nn.Module):
    """
    Parallel multi-head attention with separate query and key-value inputs.

    Supports cross-attention where queries come from one source and
    keys/values from another.

    Args:
        d_model: Model dimensionality
        n_head: Number of attention heads
    """

    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.apply(xavier_init)

    def forward(self, x_q, x_kv, rotary_q, rotary_k):
        """
        Args:
            x_q: Query input [B, S_q, D]
            x_kv: Key/Value input [B, S_kv, D]
            rotary_q: Function to apply rotary encoding to queries
            rotary_k: Function to apply rotary encoding to keys

        Returns:
            Attention output [B, S_q, D]
        """
        q = rearrange(self.q_proj(x_q), "b s (h d) -> b h s d", h=self.n_head)
        k, v = self.kv_proj(x_kv).chunk(2, dim=-1)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_head)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_head)

        q = rotary_q(q)
        k = rotary_k(k)
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * (self.d_head**-0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out_proj(out)


class FluxTransformerBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and Bezier activation.

    Uses rotary position embeddings and pillar layers for enhanced expressiveness.

    Args:
        d_model: Model dimensionality
        n_head: Number of attention heads
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.bezier_activation = BezierActivation()
        self.p_preactivation = nn.SiLU()
        self.self_attn = ParallelAttention(d_model, n_head)
        self.cross_attn = ParallelAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.p0 = pillarLayer(
            in_size=d_model, out_size=d_model, depth=3, activation=self.p_preactivation
        )
        self.p1 = pillarLayer(
            in_size=d_model, out_size=d_model, depth=3, activation=self.p_preactivation
        )
        self.p2 = pillarLayer(
            in_size=d_model, out_size=d_model, depth=3, activation=self.p_preactivation
        )
        self.p3 = pillarLayer(
            in_size=d_model, out_size=d_model, depth=3, activation=self.p_preactivation
        )
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model))
        self.rotary_pe = RotaryPositionalEmbedding(d_model // n_head)
        self.apply(xavier_init)

    def forward(
        self,
        img_seq,
        text_seq,
        sin_img,
        cos_img,
        sin_txt,
        cos_txt,
        p0_x,
        p1_x,
        p2_x,
        p3_x,
    ):
        """
        Args:
            img_seq: Image token sequence [B, T_img, D]
            text_seq: Text token sequence [B, T_txt, D]
            sin_img, cos_img: Rotary embeddings for image tokens
            sin_txt, cos_txt: Rotary embeddings for text tokens
            p0_x, p1_x, p2_x, p3_x: Bezier control point features (or None)

        Returns:
            tuple: (updated img_seq, p0, p1, p2, p3)
        """
        # Self-attention on image tokens
        normed_img_seq = self.norm1(img_seq)
        img_seq = img_seq + self.self_attn(
            normed_img_seq,
            normed_img_seq,
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
        )

        # Cross-attention with text tokens
        img_seq = img_seq + self.cross_attn(
            self.norm2(img_seq),
            self.norm2(text_seq),
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
            lambda x: self.rotary_pe.apply_rotary(x, sin_txt, cos_txt),
        )

        # Gated Bezier control points
        g = torch.sigmoid(img_seq)
        img_p0 = self.p0(g * p0_x if p0_x is not None else g)
        img_p1 = self.p1(g * p1_x if p1_x is not None else g)
        img_p2 = self.p2(g * p2_x if p2_x is not None else g)
        img_p3 = self.p3(g * p3_x if p3_x is not None else g)

        # Feed-forward + Bezier activation
        img_seq = img_seq + self.ffn(self.norm3(img_seq))
        img_seq = self.bezier_activation(
            torch.cat([img_seq, img_p0, img_p1, img_p2, img_p3], dim=-1)
        )

        return img_seq, img_p0, img_p1, img_p2, img_p3


class FluxFlowProcessor(nn.Module):
    """
    Flow prediction model using transformer blocks.

    Processes latent representations and predicts flow for diffusion sampling.
    Uses image tokens themselves as context via attention mixer.

    Args:
        d_model: Model dimensionality (default: 512)
        vae_dim: VAE latent dimension (default: 128)
        embedding_size: Text embedding dimension (default: 1024)
        n_head: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 10)
        max_hw: Maximum spatial dimension (default: 1024)
        ctx_tokens: Number of context tokens to extract (default: 4)
    """

    def __init__(
        self,
        d_model=512,
        vae_dim=128,
        embedding_size=1024,
        n_head=8,
        n_layers=10,
        max_hw=1024,
        ctx_tokens=4,
    ):
        super().__init__()
        self.max_hw = max_hw
        self.ctx_tokens = ctx_tokens

        self.vae_to_dmodel = nn.Linear(vae_dim, d_model)
        self.dmodel_to_vae = nn.Linear(d_model, vae_dim)

        # Use image tokens as context: lightweight mixer over first K tokens
        self.ctx_mixer = ContextAttentionMixer(d_model, n_head=max(1, d_model // 128), use_cls=True)

        self.text_proj = nn.Linear(embedding_size, d_model)
        self.time_embed = nn.Sequential(
            nn.Embedding(1000, d_model),
            nn.LayerNorm(d_model),
            TrainableBezier((d_model,)),
            nn.Linear(d_model, embedding_size),
        )

        self.context_injection = GatedContextInjection(d_model, d_model)
        self.transformer_blocks = nn.ModuleList(
            [FluxTransformerBlock(d_model, n_head) for _ in range(n_layers)]
        )

        self.flow_predictor = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2),
            TrainableBezier((d_model, 1, 1)),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.context_final = nn.Sequential(
            nn.Conv2d(d_model + 2, d_model, kernel_size=7, padding=3),
            TrainableBezier((d_model, 1, 1)),
        )

    def add_coord_channels(self, x):
        """Add normalized coordinate channels to feature map."""
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def forward(self, packed, text_embeddings, timesteps):
        """
        Args:
            packed: Latent representation [B, T+1, Dv] from VAE
                    IMPORTANT: During training, only add noise to image tokens packed[:, :-1, :]
                    The HW token (packed[:, -1, :]) should remain clean as it encodes spatial metadata
            text_embeddings: Text embeddings [B, embedding_size]
            timesteps: Diffusion timesteps [B]

        Returns:
            Updated packed latent [B, T+1, Dv] with preserved HW token
        """
        # Unpack image tokens and HW metadata
        img_seq_v = packed[:, :-1, :].contiguous()  # [B, T, Dv] - image tokens (may be noisy)
        hw_vec = packed[:, -1, :].contiguous()  # [B, Dv] - HW metadata (should be clean)

        B, T, Dv = img_seq_v.shape
        # Extract spatial dimensions from HW token (first 2 channels)
        H = (hw_vec[:, 0] * self.max_hw).round().clamp(min=1).long()
        W = (hw_vec[:, 1] * self.max_hw).round().clamp(min=1).long()

        # VAE→d_model
        img_seq = self.vae_to_dmodel(img_seq_v)  # [B, T, Dm]

        # Derive context from image tokens: take first K tokens
        K = min(self.ctx_tokens, T)
        ctx_tokens = img_seq[:, :K, :]  # [B, K, Dm]
        ctx_agg, ctx_tokens = self.ctx_mixer(ctx_tokens)  # pooled: [B,Dm], tokens:[B,K,Dm]

        # Convert timesteps (float 0-1) to discrete indices (0-999)
        timestep_indices = (timesteps * 999).long().clamp(0, 999)
        cond = text_embeddings + self.time_embed(timestep_indices)
        text_seq = self.text_proj(cond).unsqueeze(1)

        first_block = self.transformer_blocks[0]
        assert isinstance(first_block, FluxTransformerBlock)
        sin_img, cos_img = first_block.rotary_pe.get_embed(torch.arange(T, device=img_seq.device))
        sin_txt, cos_txt = first_block.rotary_pe.get_embed(
            torch.arange(text_seq.size(1), device=img_seq.device)
        )

        # Checkpoint all transformer blocks as one unit instead of per-block
        def transformer_blocks_fn(img_seq, ctx_agg):
            p0 = p1 = p2 = p3 = None
            for block in self.transformer_blocks:
                img_seq = self.context_injection(img_seq, ctx_agg)
                img_seq, p0, p1, p2, p3 = block(
                    img_seq,
                    text_seq,
                    sin_img,
                    cos_img,
                    sin_txt,
                    cos_txt,
                    p0,
                    p1,
                    p2,
                    p3,
                )
                # Evolve pooled context from current features
                ctx_agg = ctx_agg + img_seq.mean(dim=1)
            return img_seq, ctx_agg

        img_seq, ctx_agg = checkpoint(
            partial(transformer_blocks_fn), img_seq, ctx_agg, use_reentrant=False
        )

        # Pre-compute projection to VAE space for all samples (batched)
        img_seq_v_all = self.dmodel_to_vae(img_seq)  # [B, T, Dv]

        # Check if all samples have same H, W for batch optimization
        if B > 1 and (H == H[0]).all() and (W == W[0]).all():
            # Fast path: all samples have same dimensions, can batch process
            h, w = H[0].item(), W[0].item()
            t_valid = min(h * w, T)

            if t_valid < h * w:
                h = int(t_valid**0.5)
                w = t_valid // h

            # Batch process all samples at once
            feat = img_seq[:, :t_valid, :].reshape(B, t_valid, -1)
            feat = rearrange(feat, "b (h w) d -> b d h w", h=h, w=w)

            flow = self.flow_predictor(feat)
            new_context_feat = self.context_final(self.add_coord_channels(flow))
            pooled = F.adaptive_avg_pool2d(new_context_feat, (1, 1)).view(B, -1)  # [B, Dm]

            ctx_update_v = self.dmodel_to_vae(pooled)  # [B, Dv]
            k_i = min(self.ctx_tokens, t_valid)
            if k_i > 0:
                ctx_update_expanded = ctx_update_v.unsqueeze(1).expand(-1, k_i, -1)
                img_seq_v_all = torch.cat(
                    [img_seq_v_all[:, :k_i, :] + ctx_update_expanded, img_seq_v_all[:, k_i:, :]],
                    dim=1,
                )

            # Repack all samples
            return torch.cat([img_seq_v_all, hw_vec.unsqueeze(1)], dim=1)  # [B, T+1, Dv]

        else:
            # Slow path: variable dimensions, must process individually
            outputs = []
            for i in range(B):
                h, w = H[i].item(), W[i].item()
                t_valid = min(h * w, T)

                if t_valid < h * w:
                    h = int(t_valid**0.5)
                    w = t_valid // h

                feat = img_seq[i, :t_valid].reshape(1, t_valid, -1)
                feat = rearrange(feat, "b (h w) d -> b d h w", h=h, w=w)

                flow = self.flow_predictor(feat)
                new_context_feat = self.context_final(self.add_coord_channels(flow))
                pooled = F.adaptive_avg_pool2d(new_context_feat, (1, 1)).view(1, -1)  # [1, Dm]

                ctx_update_v = self.dmodel_to_vae(pooled)  # [1, Dv]
                img_seq_v_i = img_seq_v_all[i : i + 1]  # [1, T, Dv] - use pre-computed
                k_i = min(self.ctx_tokens, t_valid)
                if k_i > 0:
                    ctx_update_expanded = ctx_update_v.unsqueeze(1).expand(-1, k_i, -1)
                    img_seq_v_i = torch.cat(
                        [img_seq_v_i[:, :k_i, :] + ctx_update_expanded, img_seq_v_i[:, k_i:, :]],
                        dim=1,
                    )

                packed_i = torch.cat(
                    [img_seq_v_i, hw_vec[i : i + 1].unsqueeze(1)], dim=1
                )  # [1, T+1, Dv]
                outputs.append(packed_i)

            return torch.cat(outputs, dim=0)
