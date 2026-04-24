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
from ..v080.flow import FluxTransformerBlock_v080, _valid_pillar_heads


class FluxFlowProcessor_v100(nn.Module):
    """
    Flow prediction model for FluxFlow v0.10.0 (2*vae_dim packed tokens).

    Identical to FluxFlowProcessor_v080 (pillar-attention) except:
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
        assert isinstance(first_block, FluxTransformerBlock_v080)
        sin_img, cos_img = first_block.rotary_pe.get_embed(torch.arange(T, device=img_seq.device))
        sin_txt, cos_txt = first_block.rotary_pe.get_embed(
            torch.arange(text_seq.size(1), device=img_seq.device)
        )

        def transformer_blocks_fn(
            img_seq: torch.Tensor, ctx_agg: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
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
