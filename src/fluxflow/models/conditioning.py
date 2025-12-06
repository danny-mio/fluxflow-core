"""
Conditioning and context modules for FluxFlow models.

Contains:
- FiLM: Feature-wise Linear Modulation
- SPADE: Spatially-Adaptive Denormalization
- LeanContext modules: Self-attention context processing
- GatedContextInjection: Gated context conditioning
- ContextAttentionMixer: Multi-head attention mixer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import BezierActivation, xavier_init

# Default config value for text embedding scaling
DEFAULT_CONFIG_VALUE = 10.0


def stable_scale_text_embeddings(text_embeddings, config_value):
    """L2 normalize and scale text embeddings."""
    normed = F.normalize(text_embeddings, dim=1)
    return normed * config_value / DEFAULT_CONFIG_VALUE


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.

    Learns scale (gamma) and bias (beta) from text embeddings,
    then applies them to features: out = x * (1 + gamma) + beta

    Args:
        text_dim: Dimensionality of text embeddings
        num_channels: Number of feature channels to modulate
    """

    def __init__(self, text_dim, num_channels):
        super(FiLM, self).__init__()
        self.linear = nn.Linear(text_dim, 2 * num_channels)

    def forward(self, x, text_emb):
        """
        Args:
            x: Feature tensor [B, num_channels, H, W]
            text_emb: Text embeddings [B, text_dim]

        Returns:
            Modulated features [B, num_channels, H, W]
        """
        gamma_beta = self.linear(text_emb)  # [B, 2 * num_channels]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, num_channels, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class SPADE(nn.Module):
    """
    Spatially-Adaptive Denormalization (SPADE).

    Applies context-dependent spatial modulation:
    out = (1 + gamma) * GroupNorm(x) + beta
    where gamma and beta are spatially varying and predicted from context.

    Args:
        context_nc: Number of channels in the context map
        num_features: Number of channels in the feature x
    """

    def __init__(self, context_nc, num_features):
        super().__init__()
        # GroupNorm instead of BatchNorm (works with batch_size=1)
        # Use 32 groups or num_features if smaller
        num_groups = min(32, num_features)
        self.bn = nn.GroupNorm(num_groups, num_features, affine=False)

        # Conv block to produce gamma/beta from context
        hidden_dim = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(context_nc, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_dim, num_features, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_dim, num_features, kernel_size=3, padding=1)

    def forward(self, x, context):
        """
        Args:
            x: Feature tensor [B, num_features, H, W]
            context: Context map [B, context_nc, Hc, Wc]

        Returns:
            Modulated features [B, num_features, H, W]
        """
        # Normalize x
        normalized = self.bn(x)

        # Upsample context if needed to match x's spatial dimensions
        if context.size(2) != x.size(2) or context.size(3) != x.size(3):
            context = F.interpolate(context, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Produce gamma, beta from context
        actv = self.mlp_shared(context)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Apply affine transform
        out = normalized * (1 + gamma) + beta
        return out


class GatedContextInjection(nn.Module):
    """
    Gated context injection using learned gates.

    out = x * gate(context) + bias(context)

    Args:
        context_dim: Dimensionality of context vectors
        target_dim: Dimensionality of target features
    """

    def __init__(self, context_dim, target_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(context_dim, target_dim), nn.Sigmoid())
        self.bias = nn.Linear(context_dim, target_dim)

    def forward(self, x, context):
        """
        Args:
            x: Input tensor [B, T, target_dim]
            context: Context vectors [B, context_dim]

        Returns:
            Gated output [B, T, target_dim]
        """
        g = self.gate(context)[:, None, :]
        b = self.bias(context)[:, None, :]
        return x * g + b


class LeanContextModule(nn.Module):
    """
    Lean 2D self-attention context module.

    Processes feature maps to produce context through self-attention,
    without relying on text embeddings.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (default: 64)
    """

    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 5, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels * 5),
            BezierActivation(),
        )
        self.apply(xavier_init)

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, in_channels, H, W]

        Returns:
            Context features [B, out_channels, H//2, W//2]
        """
        B, C, H, W = x.shape

        # Compute query, key, value
        q = self.query(x).view(B, -1, H * W)  # [B, C//8, HW]
        k = self.key(x).view(B, -1, H * W)  # [B, C//8, HW]
        v = self.value(x).view(B, -1, H * W)  # [B, C, HW]

        # Self-attention
        attn = torch.bmm(q.permute(0, 2, 1), k)  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, HW]
        out = out.view(B, C, H, W)
        out = self.gamma * out + x

        return self.conv(out)


class LeanContext2D(nn.Module):
    """
    Multi-head 2D self-attention context module.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (default: 64)
        num_heads: Number of attention heads (default: 1)
    """

    def __init__(self, in_channels, out_channels=64, num_heads=1):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        self.num_heads = num_heads

        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels),
        )

        self.apply(xavier_init)

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, in_channels, H, W]

        Returns:
            Context features [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        q = self.query(x).view(B, self.num_heads, self.head_dim, -1)  # [B, heads, head_dim, HW]
        k = self.key(x).view(B, self.num_heads, self.head_dim, -1)
        v = self.value(x).view(B, self.num_heads, self.head_dim, -1)

        attn = torch.einsum("bncd,bnce->bnde", q, k)  # [B, heads, HW, HW]
        attn = F.softmax(attn / self.head_dim**0.5, dim=-1)

        out = torch.einsum("bnce,bnde->bncd", v, attn).reshape(B, C, H, W)
        out = self.gamma * out + x
        return self.projection(out)


class LeanContext1D(nn.Module):
    """
    Multi-head 1D self-attention context module for sequence data.

    Supports both [B, C] and [B, T, C] inputs.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (default: 64)
        num_heads: Number of attention heads (default: 1)
    """

    def __init__(self, in_channels, out_channels=64, num_heads=1):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        self.num_heads = num_heads

        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(0.0)

        self.projection = nn.Sequential(
            nn.Linear(in_channels, out_channels * 5),
            nn.LayerNorm(out_channels * 5),
            BezierActivation(),
        )

        self.apply(xavier_init)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C] or [B, T, C]

        Returns:
            Context features [B, out_channels] or [B, T, out_channels]
        """
        # Support both (B, C) and (B, T, C)
        is_batched_vector = x.dim() == 2
        if is_batched_vector:
            x = x.unsqueeze(1)  # [B, 1, C]

        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.einsum("bhtd,bhsd->bhts", q, k) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.einsum("bhts,bhsd->bhtd", attn, v).transpose(1, 2).contiguous().view(B, T, C)

        out = self.gamma * out + x  # residual
        out = self.projection(out)

        if is_batched_vector:
            out = out.squeeze(1)  # Back to [B, C_out]

        return out


class ContextAttentionMixer(nn.Module):
    """
    Multi-head self-attention mixer for context tokens with optional CLS pooling.

    Args:
        d_model: Model dimensionality
        n_head: Number of attention heads (default: 4)
        use_cls: Use CLS token for pooling (default: True)
        attn_drop: Attention dropout rate (default: 0.0)
        proj_drop: Projection dropout rate (default: 0.0)

    Returns:
        pooled: [B, D] - CLS-pooled or mean-pooled representation
        tokens: [B, K, D] - Updated token representations
    """

    def __init__(self, d_model, n_head=4, use_cls=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.use_cls = use_cls

        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_in = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(xavier_init)

    def forward(self, x):
        """
        Args:
            x: Input tokens [B, K, D]

        Returns:
            tuple: (pooled [B, D], tokens [B, K, D])
        """
        B, K, D = x.shape
        y = self.norm_in(x)

        if self.use_cls:
            cls_tok = self.cls.expand(B, -1, -1)  # [B, 1, D]
            y = torch.cat([cls_tok, y], dim=1)  # [B, 1+K, D]

        Q = self.q_proj(y)
        Kq = self.k_proj(y)
        V = self.v_proj(y)

        # Reshape to heads
        Q = Q.view(B, -1, self.n_head, self.d_head).transpose(1, 2)  # [B, H, S, Dh]
        Kq = Kq.view(B, -1, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(B, -1, self.n_head, self.d_head).transpose(1, 2)

        attn = torch.matmul(Q, Kq.transpose(-2, -1)) / (self.d_head**0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)  # [B, H, S, Dh]
        out = out.transpose(1, 2).contiguous().view(B, -1, D)  # [B, S, D]
        out = self.proj_drop(self.out_proj(out))
        out = self.norm_out(out)

        if self.use_cls:
            pooled = out[:, 0]  # [B, D]
            tokens = out[:, 1:]  # [B, K, D]
        else:
            pooled = out.mean(dim=1)  # [B, D]
            tokens = out  # [B, K, D]

        return pooled, tokens
