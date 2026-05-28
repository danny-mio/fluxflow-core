"""
Flow-based diffusion model components for FluxFlow v0.10.0.

Note: vae_to_dmodel projects the full 2*vae_dim packed token without separating z from context
dims. The context dims are not given special architectural treatment inside the transformer; they
are denoised jointly with z through the standard v-prediction objective. This is an intentional
design choice — see the architectural review in model-0.10.0.md §3.8 for the full rationale.

Changes vs v0.8.0:
- vae_to_dmodel: nn.Linear(2*vae_dim, d_model)  (was vae_dim + CONTEXT_DIMS = vae_dim + 5)
- dmodel_to_vae: nn.Linear(d_model, 2*vae_dim)  (was d_model → vae_dim + CONTEXT_DIMS)
- context_dims instance attribute: defaults to vae_dim; inspectable by downstream code.
- No CONTEXT_DIMS import from v070.
- All other internals (FiLM, pillar_cross_attn, rotary PE, etc.) are unchanged from v0.8.0.
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


def _valid_pillar_heads(d_model: int, preferred: int) -> int:
    """Return largest value <= preferred that evenly divides d_model."""
    for h in range(max(preferred, 1), 0, -1):
        if d_model % h == 0:
            return h
    return 1


class FluxTransformerBlock_v100(nn.Module):
    """
    Transformer block with pillar-attention architecture (v0.10.0).

    Corrects the FiLM ordering from v0.8.0:
        v0.8.0: FiLM(g) → PillarMLP(FiLM(g))      [low-variance FiLM input, attenuated gradients]
        v0.10.0: PillarMLP(g) → FiLM(PillarMLP(g)) [full-variance FiLM input, clean gradient flow]

    Args:
        d_model: Model dimensionality
        n_head: Number of attention heads
    """

    def __init__(self, d_model: int, n_head: int) -> None:
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

        # FiLM: independent per-pillar, projects text_cond [B, D] → [B, 2*D]
        self.film_p0 = nn.Linear(d_model, 2 * d_model)
        self.film_p1 = nn.Linear(d_model, 2 * d_model)
        self.film_p2 = nn.Linear(d_model, 2 * d_model)
        self.film_p3 = nn.Linear(d_model, 2 * d_model)

        n_pillar_heads = _valid_pillar_heads(d_model, max(1, n_head // 4))
        self.pillar_cross_attn = ParallelAttention(d_model, n_pillar_heads)
        self.norm_pillar = nn.LayerNorm(d_model)

        self.apply(xavier_init)

    def _film_modulate(
        self, gate: torch.Tensor, film_layer: nn.Linear, text_cond: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = film_layer(text_cond).chunk(2, dim=-1)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            img_seq: [B, T_img, D]
            text_seq: [B, T_txt, D]
            sin_img, cos_img: Rotary embeddings for image tokens
            sin_txt, cos_txt: Rotary embeddings for text tokens
            p0_x … p3_x: Cross-block pillar features (None for first block)
            text_cond: Pooled text+time conditioning [B, D]

        Returns:
            tuple: (updated img_seq, img_p0, img_p1, img_p2, img_p3)
        """
        # 1. Self-attention
        normed = self.norm1(img_seq)
        img_seq = img_seq + self.self_attn(
            normed,
            normed,
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
        )

        # 2. Cross-attention img_seq × text_seq
        img_seq = img_seq + self.cross_attn(
            self.norm2(img_seq),
            self.norm2(text_seq),
            lambda x: self.rotary_pe.apply_rotary(x, sin_img, cos_img),
            lambda x: self.rotary_pe.apply_rotary(x, sin_txt, cos_txt),
        )

        # 3. Sigmoid gate
        g = torch.sigmoid(img_seq)

        # 4. Pillar MLPs FIRST on raw gate (corrected vs v0.8.0)
        g_p0 = self.p0(g * p0_x if p0_x is not None else g)
        g_p1 = self.p1(g * p1_x if p1_x is not None else g)
        g_p2 = self.p2(g * p2_x if p2_x is not None else g)
        g_p3 = self.p3(g * p3_x if p3_x is not None else g)

        # 5. FiLM modulation on Pillar MLP output (corrected vs v0.8.0)
        raw_p0 = self._film_modulate(g_p0, self.film_p0, text_cond)
        raw_p1 = self._film_modulate(g_p1, self.film_p1, text_cond)
        raw_p2 = self._film_modulate(g_p2, self.film_p2, text_cond)
        raw_p3 = self._film_modulate(g_p3, self.film_p3, text_cond)

        # 6. Shared pillar cross-attention (Q=pillar, KV=text_seq)
        raw_stacked = torch.cat([raw_p0, raw_p1, raw_p2, raw_p3], dim=1)
        normed_stacked = self.norm_pillar(raw_stacked)
        attended = self.pillar_cross_attn(normed_stacked, text_seq, lambda x: x, lambda x: x)
        raw_stacked = raw_stacked + attended
        T_img = img_seq.shape[1]
        img_p0 = raw_stacked[:, :T_img, :]
        img_p1 = raw_stacked[:, T_img : 2 * T_img, :]
        img_p2 = raw_stacked[:, 2 * T_img : 3 * T_img, :]
        img_p3 = raw_stacked[:, 3 * T_img :, :]

        # 7. FFN + BezierActivation
        img_seq = img_seq + self.ffn(self.norm3(img_seq))
        img_seq = self.bezier_activation(
            torch.cat([img_seq, img_p0, img_p1, img_p2, img_p3], dim=-1)
        )

        return img_seq, img_p0, img_p1, img_p2, img_p3


class FluxFlowProcessor_v100(nn.Module):
    """
    Flow prediction model for FluxFlow v0.10.0 (2*vae_dim packed tokens).

    Extends the v0.8.0 pillar-attention design with corrected Pillar→FiLM ordering:
    - vae_to_dmodel accepts 2*vae_dim inputs (not vae_dim + CONTEXT_DIMS).
    - dmodel_to_vae produces 2*vae_dim outputs.
    - context_dims attribute is explicitly stored and inspectable.

    External forward signature is identical to v0.8.0:
        forward(packed, text_embeddings, timesteps) -> packed

    Args:
        d_model: Model dimensionality (default: 512)
        vae_dim: VAE latent dimension (default: 128)
        embedding_size: Text embedding dimension (default: 1024)
        n_head: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 10)
        max_hw: Maximum spatial dimension (default: 1024)
        ctx_tokens: Number of context tokens for ContextAttentionMixer (default: 4)
        context_dims: Context dimensionality (default: None = vae_dim).
            Set explicitly only if context dims are decoupled from vae_dim in future versions.
    """

    def __init__(
        self,
        d_model: int = 512,
        vae_dim: int = 128,
        embedding_size: int = 1024,
        n_head: int = 8,
        n_layers: int = 10,
        max_hw: int = 1024,
        ctx_tokens: int = 4,
        context_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.max_hw = max_hw
        self.ctx_tokens = ctx_tokens
        # context_dims defaults to vae_dim; full packed token width = 2 * vae_dim
        self.context_dims = context_dims if context_dims is not None else vae_dim

        packed_width = vae_dim + self.context_dims  # = 2*vae_dim when context_dims == vae_dim

        self.vae_to_dmodel = nn.Linear(packed_width, d_model)
        self.dmodel_to_vae = nn.Linear(d_model, packed_width)

        self.ctx_mixer = ContextAttentionMixer(d_model, n_head=max(1, d_model // 128), use_cls=True)

        self.text_proj = nn.Linear(embedding_size, d_model)
        self.time_embed = nn.Sequential(
            nn.Embedding(1000, d_model),
            nn.LayerNorm(d_model),
            TrainableBezier((d_model,)),
            nn.Linear(d_model, embedding_size),
        )
        self.text_cond_proj = nn.Linear(embedding_size, d_model)

        self.context_injection = GatedContextInjection(d_model, d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.transformer_blocks = nn.ModuleList(
            [FluxTransformerBlock_v100(d_model, n_head) for _ in range(n_layers)]
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

    def add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Add normalized coordinate channels to feature map."""
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, coords], dim=1)

    def forward(
        self,
        packed: torch.Tensor,
        text_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict flow velocity for packed latent tokens.

        Args:
            packed: [B, T+1, 2*vae_dim] — packed z + context tokens + HW token
            text_embeddings: [B, embedding_size]
            timesteps: [B] floats in [0, 1]

        Returns:
            torch.Tensor: [B, T+1, 2*vae_dim] with preserved HW token
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

        timestep_indices = (timesteps * 999).long().clamp(0, 999)
        cond = text_embeddings + self.time_embed(timestep_indices)

        text_seq = self.text_proj(cond).unsqueeze(1)
        text_cond = self.text_cond_proj(cond)

        first_block = self.transformer_blocks[0]
        assert isinstance(first_block, FluxTransformerBlock_v100)
        sin_img, cos_img = first_block.rotary_pe.get_embed(torch.arange(T, device=img_seq.device))
        sin_txt, cos_txt = first_block.rotary_pe.get_embed(
            torch.arange(text_seq.size(1), device=img_seq.device)
        )

        def transformer_blocks_fn(
            img_seq: torch.Tensor, ctx_agg: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            p0 = p1 = p2 = p3 = None
            for block in self.transformer_blocks:
                img_seq = self.context_injection(img_seq, self.norm_ctx(ctx_agg))
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
            h, w = int(H[0].item()), int(W[0].item())
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
                h, w = int(H[i].item()), int(W[i].item())
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
