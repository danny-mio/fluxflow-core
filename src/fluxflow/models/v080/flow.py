"""
Flow-based diffusion model components for FluxFlow v0.8.0 (pillar-attention).

New vs v0.7.0:
- FiLM conditioning per pillar using pooled text+time embedding
- Shared cross-attention block (pillar_cross_attn) attending to text_seq
- VAE components unchanged (imported from v070)
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ..activations import BezierActivation, TrainableBezier, xavier_init
from ..conditioning import ContextAttentionMixer, GatedContextInjection
from ..v070.flow import ParallelAttention, RotaryPositionalEmbedding, pillarLayer
from ..v070.vae import CONTEXT_DIMS


class FluxTransformerBlock_v080(nn.Module):
    """
    Transformer block with pillar-attention architecture (v0.8.0).

    Adds to v0.7.0 FluxTransformerBlock:
    - FiLM per-pillar modulation using pooled text conditioning
    - Shared cross-attention from pillar outputs to text_seq

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

        # Four pillar MLPs (depth=3), same as v0.7.0
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

        # --- v0.8.0 additions ---
        # FiLM: independent per-pillar, projects text_cond [B, D] → [B, 2*D]
        self.film_p0 = nn.Linear(d_model, 2 * d_model)
        self.film_p1 = nn.Linear(d_model, 2 * d_model)
        self.film_p2 = nn.Linear(d_model, 2 * d_model)
        self.film_p3 = nn.Linear(d_model, 2 * d_model)

        # Shared pillar cross-attention (Q=pillar output, KV=text_seq)
        n_pillar_heads = max(1, n_head // 4)
        self.pillar_cross_attn = ParallelAttention(d_model, n_pillar_heads)
        self.norm_pillar = nn.LayerNorm(d_model)

        self.apply(xavier_init)

    def _film_modulate(
        self, gate: torch.Tensor, film_layer: nn.Linear, text_cond: torch.Tensor
    ) -> torch.Tensor:
        """Apply FiLM modulation: g_pi = gate * (1 + gamma) + beta."""
        gamma, beta = film_layer(text_cond).chunk(2, dim=-1)  # each [B, D]
        result: torch.Tensor = gate * (1 + gamma[:, None, :]) + beta[:, None, :]
        return result

    def forward(
        self,
        img_seq: torch.Tensor,
        text_seq: torch.Tensor,
        sin_img: torch.Tensor,
        cos_img: torch.Tensor,
        sin_txt: torch.Tensor,
        cos_txt: torch.Tensor,
        p0_x,
        p1_x,
        p2_x,
        p3_x,
        text_cond: torch.Tensor,
    ):
        """
        Args:
            img_seq: Image token sequence [B, T_img, D]
            text_seq: Text token sequence [B, T_txt, D]
            sin_img, cos_img: Rotary embeddings for image tokens
            sin_txt, cos_txt: Rotary embeddings for text tokens
            p0_x … p3_x: Cross-block pillar features (or None for first block)
            text_cond: Pooled text+time conditioning [B, D]

        Returns:
            tuple: (updated img_seq, img_p0, img_p1, img_p2, img_p3)
        """
        # 1. Self-attention on image tokens (unchanged)
        normed_img_seq = self.norm1(img_seq)
        img_seq = img_seq + self.self_attn(
            normed_img_seq,
            normed_img_seq,
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
        )

        # 2. Cross-attention img_seq × text_seq (unchanged)
        img_seq = img_seq + self.cross_attn(
            self.norm2(img_seq),
            self.norm2(text_seq),
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
            lambda x: self.rotary_pe.apply_rotary(x, sin_txt, cos_txt),
        )

        # 3. Sigmoid gate (unchanged)
        g = torch.sigmoid(img_seq)

        # 4. FiLM per-pillar modulation [NEW]
        g_p0 = self._film_modulate(g, self.film_p0, text_cond)
        g_p1 = self._film_modulate(g, self.film_p1, text_cond)
        g_p2 = self._film_modulate(g, self.film_p2, text_cond)
        g_p3 = self._film_modulate(g, self.film_p3, text_cond)

        # 5. Pillar MLPs on FiLM-modulated gate
        raw_p0 = self.p0(g_p0 * p0_x if p0_x is not None else g_p0)
        raw_p1 = self.p1(g_p1 * p1_x if p1_x is not None else g_p1)
        raw_p2 = self.p2(g_p2 * p2_x if p2_x is not None else g_p2)
        raw_p3 = self.p3(g_p3 * p3_x if p3_x is not None else g_p3)

        # 6. Shared pillar cross-attention (Q=pillar, KV=text_seq) [NEW]
        # Stack pillars on sequence dim, attend jointly, then split
        raw_stacked = torch.cat([raw_p0, raw_p1, raw_p2, raw_p3], dim=1)  # [B, 4T, D]
        normed_stacked = self.norm_pillar(raw_stacked)
        # Identity rotary for pillar queries (no positional meaning needed)
        attended = self.pillar_cross_attn(normed_stacked, text_seq, lambda x: x, lambda x: x)
        raw_stacked = raw_stacked + attended
        T_img = img_seq.shape[1]
        img_p0 = raw_stacked[:, :T_img, :]
        img_p1 = raw_stacked[:, T_img : 2 * T_img, :]
        img_p2 = raw_stacked[:, 2 * T_img : 3 * T_img, :]
        img_p3 = raw_stacked[:, 3 * T_img :, :]

        # 7. FFN + BezierActivation (unchanged)
        img_seq = img_seq + self.ffn(self.norm3(img_seq))
        img_seq = self.bezier_activation(
            torch.cat([img_seq, img_p0, img_p1, img_p2, img_p3], dim=-1)
        )

        return img_seq, img_p0, img_p1, img_p2, img_p3


class FluxFlowProcessor_v080(nn.Module):
    """
    Flow prediction model with pillar-attention architecture (v0.8.0).

    Same as v0.7.0 FluxFlowProcessor but uses FluxTransformerBlock_v080,
    which receives direct text conditioning per pillar via FiLM and
    shared cross-attention to text_seq.

    External interface (forward signature) is identical to v0.7.0.

    Args:
        d_model: Model dimensionality (default: 512)
        vae_dim: VAE latent dimension (default: 128)
        embedding_size: Text embedding dimension (default: 1024)
        n_head: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 10)
        max_hw: Maximum spatial dimension (default: 1024)
        ctx_tokens: Number of context tokens (default: 4)
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

        # Input: vae_dim + CONTEXT_DIMS (same as v0.7.0)
        self.vae_to_dmodel = nn.Linear(vae_dim + CONTEXT_DIMS, d_model)
        self.dmodel_to_vae = nn.Linear(d_model, vae_dim + CONTEXT_DIMS)

        self.ctx_mixer = ContextAttentionMixer(d_model, n_head=max(1, d_model // 128), use_cls=True)

        self.text_proj = nn.Linear(embedding_size, d_model)
        self.time_embed = nn.Sequential(
            nn.Embedding(1000, d_model),
            nn.LayerNorm(d_model),
            TrainableBezier((d_model,)),
            nn.Linear(d_model, embedding_size),
        )

        # Projection from embedding_size → d_model for text_cond passed to blocks
        self.text_cond_proj = nn.Linear(embedding_size, d_model)

        self.context_injection = GatedContextInjection(d_model, d_model)
        self.transformer_blocks = nn.ModuleList(
            [FluxTransformerBlock_v080(d_model, n_head) for _ in range(n_layers)]
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

    def forward(self, packed: torch.Tensor, text_embeddings: torch.Tensor, timesteps: torch.Tensor):
        """
        Args:
            packed: [B, T+1, vae_dim+CONTEXT_DIMS] — identical to v0.7.0
            text_embeddings: [B, embedding_size]
            timesteps: [B] floats in [0, 1]

        Returns:
            [B, T+1, vae_dim+CONTEXT_DIMS] with preserved HW token
        """
        img_seq_v = packed[:, :-1, :].contiguous()
        hw_vec_full = packed[:, -1, :].contiguous()

        B, T, _ = img_seq_v.shape
        H = (hw_vec_full[:, 0] * self.max_hw).round().clamp(min=1).long()
        W = (hw_vec_full[:, 1] * self.max_hw).round().clamp(min=1).long()

        img_seq = self.vae_to_dmodel(img_seq_v)

        K = min(self.ctx_tokens, T)
        ctx_tokens = img_seq[:, :K, :]
        ctx_agg, ctx_tokens = self.ctx_mixer(ctx_tokens)

        # Build text conditioning (identical to v0.7.0)
        timestep_indices = (timesteps * 999).long().clamp(0, 999)
        cond = text_embeddings + self.time_embed(timestep_indices)  # [B, embedding_size]

        # text_seq: [B, 1, D]
        text_seq = self.text_proj(cond).unsqueeze(1)

        # text_cond for FiLM: project cond → d_model [B, D]
        text_cond = self.text_cond_proj(cond)  # [B, D]

        first_block = self.transformer_blocks[0]
        assert isinstance(first_block, FluxTransformerBlock_v080)
        sin_img, cos_img = first_block.rotary_pe.get_embed(torch.arange(T, device=img_seq.device))
        sin_txt, cos_txt = first_block.rotary_pe.get_embed(
            torch.arange(text_seq.size(1), device=img_seq.device)
        )

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
                    text_cond,
                )
                ctx_agg = ctx_agg + img_seq.mean(dim=1)
            return img_seq, ctx_agg

        if torch.is_grad_enabled() and (img_seq.requires_grad or ctx_agg.requires_grad):
            img_seq, ctx_agg = checkpoint(
                partial(transformer_blocks_fn), img_seq, ctx_agg, use_reentrant=False
            )
        else:
            img_seq, ctx_agg = transformer_blocks_fn(img_seq, ctx_agg)

        img_seq_v_all = self.dmodel_to_vae(img_seq)

        if B > 1 and (H == H[0]).all() and (W == W[0]).all():
            h, w = H[0].item(), W[0].item()
            t_valid = min(h * w, T)

            if t_valid < h * w:
                h = int(t_valid**0.5)
                w = t_valid // h

            feat = img_seq[:, :t_valid, :].reshape(B, t_valid, -1)
            feat = rearrange(feat, "b (h w) d -> b d h w", h=h, w=w)

            flow = self.flow_predictor(feat)
            new_context_feat = self.context_final(self.add_coord_channels(flow))
            pooled = F.adaptive_avg_pool2d(new_context_feat, (1, 1)).view(B, -1)

            ctx_update_v = self.dmodel_to_vae(pooled)
            k_i = min(self.ctx_tokens, t_valid)
            if k_i > 0:
                ctx_update_expanded = ctx_update_v.unsqueeze(1).expand(-1, k_i, -1)
                img_seq_v_all = torch.cat(
                    [img_seq_v_all[:, :k_i, :] + ctx_update_expanded, img_seq_v_all[:, k_i:, :]],
                    dim=1,
                )

            return torch.cat([img_seq_v_all, hw_vec_full.unsqueeze(1)], dim=1).contiguous()

        else:
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
                pooled = F.adaptive_avg_pool2d(new_context_feat, (1, 1)).view(1, -1)

                ctx_update_v = self.dmodel_to_vae(pooled)
                img_seq_v_i = img_seq_v_all[i : i + 1]
                k_i = min(self.ctx_tokens, t_valid)
                if k_i > 0:
                    ctx_update_expanded = ctx_update_v.unsqueeze(1).expand(-1, k_i, -1)
                    img_seq_v_i = torch.cat(
                        [img_seq_v_i[:, :k_i, :] + ctx_update_expanded, img_seq_v_i[:, k_i:, :]],
                        dim=1,
                    )

                packed_i = torch.cat([img_seq_v_i, hw_vec_full[i : i + 1].unsqueeze(1)], dim=1)
                outputs.append(packed_i)

            return torch.cat(outputs, dim=0).contiguous()
