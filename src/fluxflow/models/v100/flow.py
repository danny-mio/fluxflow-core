"""
Flow-based diffusion model components for FluxFlow v0.10.0-bezier-coupled.

Note: vae_to_dmodel projects the full 2*vae_dim packed token without separating z from context
dims. The context dims are not given special architectural treatment inside the transformer; they
are denoised jointly with z through the standard v-prediction objective. This is an intentional
design choice — see the architectural review in model-0.10.0.md §3.8 for the full rationale.

Key changes vs v0.8.0 (M4 bezier-coupled redesign):
- vae_to_dmodel: nn.Linear(2*vae_dim, d_model)  (was vae_dim + CONTEXT_DIMS = vae_dim + 5)
- dmodel_to_vae: nn.Linear(d_model, 2*vae_dim)  (was d_model → vae_dim + CONTEXT_DIMS)
- context_dims instance attribute: defaults to vae_dim; inspectable by downstream code.
- No CONTEXT_DIMS import from v070.
- FluxTransformerBlock_v100 now consumes per-token (text_seq, text_mask) with
  2D axial RoPE, split norm2_q/norm2_kv, widened pillars, and dual FiLM.
- FluxFlowProcessor_v100 forward signature is
  ``forward(packed, text_seq, text_mask, timesteps)``: per-token text +
  continuous sinusoidal time (replaces the legacy ``Embedding(1000)``) +
  GRU-style gated ctx_agg residual (replaces the running sum accumulator).
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from ..activations import BezierActivation, TrainableBezier, xavier_init
from ..conditioning import ContextAttentionMixer, GatedContextInjection
from ..v070.flow import ParallelAttention, RotaryPositionalEmbedding
from .pillar import pillarLayerWide


class FluxTransformerBlock_v100(nn.Module):
    """
    Redesigned transformer block for v0.10.0-bezier-coupled.

    Differences vs the predecessor:
    - Accepts (text_seq, text_mask) with T_txt > 1; cross_attn is now real
      attention over text tokens (M1.5 mask-aware ParallelAttention).
    - 2D axial RoPE on image tokens via M1.2 build_axial_rope_2d (caller
      supplies sin_w/cos_w/sin_h/cos_h buffers; each half of head_dim is
      rotated independently).
    - norm2 split into norm2_q (img-Q) and norm2_kv (text-KV).
    - Pillars use M1.4 pillarLayerWide (D → 2D → 2D → D).
    - Dual FiLM per pillar: separate text_cond and time_cond channels with
      additive scales and biases.
    - pillar_cross_attn / norm_pillar removed (length-1 degenerate in the
      predecessor; no longer needed now that text is per-token).

    Args:
        d_model: Model dimensionality.
        n_head: Number of attention heads (must divide d_model).
    """

    def __init__(self, d_model: int, n_head: int, attn_backend: str = "einsum") -> None:
        super().__init__()
        assert d_model % n_head == 0
        head_dim = d_model // n_head
        assert (
            head_dim % 4 == 0
        ), f"head_dim must be divisible by 4 for 2D axial RoPE; got {head_dim}"

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim
        self.bezier_activation = BezierActivation()

        self.self_attn = ParallelAttention(d_model, n_head, attn_backend=attn_backend)
        self.cross_attn = ParallelAttention(d_model, n_head, attn_backend=attn_backend)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2_q = nn.LayerNorm(d_model)
        self.norm2_kv = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.p0 = pillarLayerWide(d_model)
        self.p1 = pillarLayerWide(d_model)
        self.p2 = pillarLayerWide(d_model)
        self.p3 = pillarLayerWide(d_model)

        self.ffn = nn.Sequential(nn.Linear(d_model, d_model))

        # Text path: 1D RoPE on text tokens (full head_dim).
        self.rotary_pe_txt = RotaryPositionalEmbedding(head_dim)

        # Dual FiLM per pillar.
        self.film_p0_text = nn.Linear(d_model, 2 * d_model)
        self.film_p1_text = nn.Linear(d_model, 2 * d_model)
        self.film_p2_text = nn.Linear(d_model, 2 * d_model)
        self.film_p3_text = nn.Linear(d_model, 2 * d_model)
        self.film_p0_time = nn.Linear(d_model, 2 * d_model)
        self.film_p1_time = nn.Linear(d_model, 2 * d_model)
        self.film_p2_time = nn.Linear(d_model, 2 * d_model)
        self.film_p3_time = nn.Linear(d_model, 2 * d_model)

        self.apply(xavier_init)

    def _film_dual(
        self,
        gate: torch.Tensor,
        film_text: nn.Linear,
        film_time: nn.Linear,
        text_cond: torch.Tensor,
        time_cond: torch.Tensor,
    ) -> torch.Tensor:
        gt, bt = film_text(text_cond).chunk(2, dim=-1)
        gtau, btau = film_time(time_cond).chunk(2, dim=-1)
        out: torch.Tensor = (
            gate * (1.0 + gt[:, None, :] + gtau[:, None, :]) + bt[:, None, :] + btau[:, None, :]
        )
        return out

    def _apply_axial_rope_2d(
        self,
        x: torch.Tensor,
        sin_w: torch.Tensor,
        cos_w: torch.Tensor,
        sin_h: torch.Tensor,
        cos_h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply axial 2D RoPE to a Q or K tensor of shape [B, H, T, head_dim].

        Top half of head_dim rotates with W frequencies; bottom half with H.
        Uses v070's apply_rotary on each half independently.
        """
        half = self.head_dim // 2
        x_w = x[..., :half]
        x_h = x[..., half:]
        # The v070 apply_rotary is a static method on the class; invoke via
        # the bound helper that lives on rotary_pe_txt (it's stateless math).
        x_w_rot = self.rotary_pe_txt.apply_rotary(x_w, sin_w, cos_w)
        x_h_rot = self.rotary_pe_txt.apply_rotary(x_h, sin_h, cos_h)
        return torch.cat([x_w_rot, x_h_rot], dim=-1)

    def forward(
        self,
        img_seq: torch.Tensor,
        text_seq: torch.Tensor,
        text_mask: torch.Tensor,
        sin_w_img: torch.Tensor,
        cos_w_img: torch.Tensor,
        sin_h_img: torch.Tensor,
        cos_h_img: torch.Tensor,
        sin_txt: torch.Tensor,
        cos_txt: torch.Tensor,
        p0_x,
        p1_x,
        p2_x,
        p3_x,
        text_cond: torch.Tensor,
        time_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Self-attention with axial 2D RoPE on img.
        normed = self.norm1(img_seq)
        img_seq = img_seq + self.self_attn(
            normed,
            normed,
            lambda q: self._apply_axial_rope_2d(q, sin_w_img, cos_w_img, sin_h_img, cos_h_img),
            lambda k: self._apply_axial_rope_2d(k, sin_w_img, cos_w_img, sin_h_img, cos_h_img),
        )

        # 2. Cross-attention img → text (T_txt > 1 now genuine).
        img_seq = img_seq + self.cross_attn(
            self.norm2_q(img_seq),
            self.norm2_kv(text_seq),
            lambda q: self._apply_axial_rope_2d(q, sin_w_img, cos_w_img, sin_h_img, cos_h_img),
            lambda k: self.rotary_pe_txt.apply_rotary(k, sin_txt, cos_txt),
            attn_mask=text_mask,
        )

        # 3. Sigmoid gate.
        g = torch.sigmoid(img_seq)

        # 4. Pillar MLPs first.
        g_p0 = self.p0(g * p0_x if p0_x is not None else g)
        g_p1 = self.p1(g * p1_x if p1_x is not None else g)
        g_p2 = self.p2(g * p2_x if p2_x is not None else g)
        g_p3 = self.p3(g * p3_x if p3_x is not None else g)

        # 5. Dual FiLM (text + time additive).
        img_p0 = self._film_dual(g_p0, self.film_p0_text, self.film_p0_time, text_cond, time_cond)
        img_p1 = self._film_dual(g_p1, self.film_p1_text, self.film_p1_time, text_cond, time_cond)
        img_p2 = self._film_dual(g_p2, self.film_p2_text, self.film_p2_time, text_cond, time_cond)
        img_p3 = self._film_dual(g_p3, self.film_p3_text, self.film_p3_time, text_cond, time_cond)

        # 6. (Removed) pillar_cross_attn — was length-1 degenerate.

        # 7. FFN + Bezier-combine across pillars.
        img_seq = img_seq + self.ffn(self.norm3(img_seq))
        img_seq = self.bezier_activation(
            torch.cat([img_seq, img_p0, img_p1, img_p2, img_p3], dim=-1)
        )

        return img_seq, img_p0, img_p1, img_p2, img_p3


class FluxFlowProcessor_v100(nn.Module):
    """
    Redesigned flow processor for v0.10.0-bezier-coupled.

    Key changes vs the predecessor:
    - forward(packed, text_seq, text_mask, timesteps): per-token text + mask.
    - Continuous sinusoidal time embedding, separate from text via a dedicated
      time_mlp; text_cond and time_cond fed to dual FiLM in each block.
    - 2D axial RoPE on image tokens (built per H, W).
    - GRU-style gated ctx_agg residual (replaces running mean accumulator).
    - pillar_cross_attn / norm_pillar removed (length-1 degenerate).

    External shape contract: packed_in shape == packed_out shape.

    Args:
        d_model: Model dimensionality (default: 512).
        vae_dim: VAE latent dimension (default: 128).
        embedding_size: Text embedding dimension (default: 1024).
        n_head: Number of attention heads (default: 8).
        n_layers: Number of transformer layers (default: 10).
        max_hw: Maximum spatial dimension for HW token decoding (default: 1024).
        ctx_tokens: Number of context tokens for ContextAttentionMixer (default: 4).
        context_dims: Context dim (default: None = vae_dim).
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
        attn_backend: str = "einsum",
    ) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.vae_dim = vae_dim
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.max_hw = max_hw
        self.ctx_tokens = ctx_tokens
        self.context_dims = context_dims if context_dims is not None else vae_dim

        packed_width = vae_dim + self.context_dims

        self.vae_to_dmodel = nn.Linear(packed_width, d_model)
        self.dmodel_to_vae = nn.Linear(d_model, packed_width)

        self.ctx_mixer = ContextAttentionMixer(d_model, n_head=max(1, d_model // 128), use_cls=True)

        # Text projection — applied per-token (broadcasts over T_txt).
        self.text_proj = nn.Linear(embedding_size, d_model)
        self.text_cond_proj = nn.Linear(embedding_size, d_model)

        # Continuous sinusoidal time → MLP → time_cond.
        # Replaces the old Embedding(1000) + Bezier + Linear chain.
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 5),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Linear(d_model, d_model),
        )

        self.context_injection = GatedContextInjection(d_model, d_model)
        self.norm_ctx = nn.LayerNorm(d_model)

        # GRU-style gated ctx_agg residual.
        self.ctx_gate_proj = nn.Linear(d_model, d_model)
        self.ctx_delta_proj = nn.Linear(d_model, d_model)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock_v100(d_model, n_head, attn_backend=attn_backend)
                for _ in range(n_layers)
            ]
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
        text_seq: torch.Tensor,
        text_mask: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict flow velocity for packed latent tokens.

        Args:
            packed: [B, T+1, 2*vae_dim] from compressor (z || ctx + HW token).
            text_seq: [B, T_txt, embedding_size] per-token text embeddings.
            text_mask: [B, T_txt] bool mask over text tokens.
            timesteps: [B] continuous timesteps in [0, 1].

        Returns:
            packed_out: [B, T+1, 2*vae_dim] flow-modulated tokens (HW token preserved).
        """
        # Local imports to avoid potential import cycles with the positional module.
        from .positional import build_axial_rope_2d, sinusoidal_embedding

        if text_seq.size(0) != packed.size(0):
            raise ValueError(
                f"Batch mismatch: packed latents have batch {packed.size(0)} but "
                f"text_seq has batch {text_seq.size(0)}. Cross-attention would "
                f"silently broadcast and corrupt downstream shapes."
            )

        img_seq_v = packed[:, :-1, :].contiguous()
        hw_vec_full = packed[:, -1, :].contiguous()

        B, T, _ = img_seq_v.shape
        H = (hw_vec_full[:, 0] * self.max_hw).round().clamp(min=1).long()
        W = (hw_vec_full[:, 1] * self.max_hw).round().clamp(min=1).long()

        img_seq = self.vae_to_dmodel(img_seq_v)

        K = min(self.ctx_tokens, T)
        ctx_tokens = img_seq[:, :K, :]
        ctx_agg, _ = self.ctx_mixer(ctx_tokens)

        # Per-token text + CLS-pooled text_cond.
        text_seq_proj = self.text_proj(text_seq)
        text_cond = self.text_cond_proj(text_seq[:, 0, :])

        # Continuous sinusoidal time → time_cond.
        time_emb = sinusoidal_embedding(timesteps, self.d_model)
        time_cond = self.time_mlp(time_emb)

        # 2D axial RoPE on image tokens — assumes same-H/W in batch.
        same_hw = bool((H == H[0]).all().item() and (W == W[0]).all().item())
        block0 = self.transformer_blocks[0]
        assert isinstance(block0, FluxTransformerBlock_v100)
        T_txt = text_seq_proj.size(1)

        if same_hw:
            # Mirror the spatial post-processing fallback: if H*W != T (e.g.
            # the caller's hw_vec is inconsistent with the actual token count),
            # clamp h*w to T by taking sqrt(T).
            h_eff = int(H[0].item())
            w_eff = int(W[0].item())
            if h_eff * w_eff != T:
                h_eff = int(T**0.5)
                w_eff = T // h_eff
            sin_w_img, cos_w_img, sin_h_img, cos_h_img = build_axial_rope_2d(
                h_eff,
                w_eff,
                self.head_dim,
                device=img_seq.device,
                dtype=img_seq.dtype,
            )
            # If h_eff * w_eff < T (non-perfect-square T), pad the RoPE buffer
            # by zero-sin/one-cos for the trailing positions so it covers T.
            extra = T - h_eff * w_eff
            if extra > 0:
                half = self.head_dim // 2
                pad_sin = torch.zeros(extra, half, device=img_seq.device, dtype=img_seq.dtype)
                pad_cos = torch.ones(extra, half, device=img_seq.device, dtype=img_seq.dtype)
                sin_w_img = torch.cat([sin_w_img, pad_sin], dim=0)
                cos_w_img = torch.cat([cos_w_img, pad_cos], dim=0)
                sin_h_img = torch.cat([sin_h_img, pad_sin], dim=0)
                cos_h_img = torch.cat([cos_h_img, pad_cos], dim=0)
            sin_txt, cos_txt = block0.rotary_pe_txt.get_embed(
                torch.arange(T_txt, device=img_seq.device, dtype=img_seq.dtype)
            )
        else:
            # Mixed-H/W fallback: punt to legacy 1D rotary on the flattened
            # token index. Adjacent tokens won't have axial structure, but the
            # forward will at least run. M4 same-H/W is the common case.
            half = self.head_dim // 2
            sin1d, cos1d = block0.rotary_pe_txt.get_embed(
                torch.arange(T, device=img_seq.device, dtype=img_seq.dtype)
            )
            sin_w_img = sin1d[:, :half]
            cos_w_img = cos1d[:, :half]
            sin_h_img = sin1d[:, half:]
            cos_h_img = cos1d[:, half:]
            sin_txt, cos_txt = block0.rotary_pe_txt.get_embed(
                torch.arange(T_txt, device=img_seq.device, dtype=img_seq.dtype)
            )

        def transformer_blocks_fn(
            img_seq: torch.Tensor, ctx_agg: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            p0 = p1 = p2 = p3 = None
            for block in self.transformer_blocks:
                img_seq = self.context_injection(img_seq, self.norm_ctx(ctx_agg))
                img_seq, p0, p1, p2, p3 = block(
                    img_seq,
                    text_seq_proj,
                    text_mask,
                    sin_w_img,
                    cos_w_img,
                    sin_h_img,
                    cos_h_img,
                    sin_txt,
                    cos_txt,
                    p0,
                    p1,
                    p2,
                    p3,
                    text_cond,
                    time_cond,
                )
                # GRU-style gated residual update of ctx_agg.
                delta = img_seq.mean(dim=1)
                gate = torch.sigmoid(self.ctx_gate_proj(ctx_agg))
                ctx_agg = gate * ctx_agg + (1.0 - gate) * self.ctx_delta_proj(delta)
            return img_seq, ctx_agg

        if torch.is_grad_enabled() and (img_seq.requires_grad or ctx_agg.requires_grad):
            img_seq, ctx_agg = checkpoint(
                partial(transformer_blocks_fn), img_seq, ctx_agg, use_reentrant=False
            )
        else:
            img_seq, ctx_agg = transformer_blocks_fn(img_seq, ctx_agg)

        img_seq_v_all = self.dmodel_to_vae(img_seq)

        # Spatial post-processing — same-H/W fast path.
        if same_hw:
            h, w = int(H[0].item()), int(W[0].item())
            t_valid = min(h * w, T)
            if t_valid < h * w:
                h = int(t_valid**0.5)
                w = t_valid // h
                # h*w may undershoot t_valid for non-perfect squares; shrink
                # t_valid to the grid so the rearrange below stays valid.
                t_valid = h * w
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
                    [
                        img_seq_v_all[:, :k_i, :] + ctx_update_expanded,
                        img_seq_v_all[:, k_i:, :],
                    ],
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
                    # Same non-perfect-square guard as the fast path above.
                    t_valid = h * w
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
                        [
                            img_seq_v_i[:, :k_i, :] + ctx_update_expanded,
                            img_seq_v_i[:, k_i:, :],
                        ],
                        dim=1,
                    )
                packed_i = torch.cat([img_seq_v_i, hw_vec_full[i : i + 1].unsqueeze(1)], dim=1)
                outputs.append(packed_i)
            return torch.cat(outputs, dim=0).contiguous()
