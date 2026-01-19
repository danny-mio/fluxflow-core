"""
v0.7.0 specific conditioning modules.

Enhanced SPADE that supports context=None for identity modulation.
"""

import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    Spatially-Adaptive Denormalization (SPADE) - v0.7.0 enhanced version.

    Supports context=None for identity modulation (normalization only).
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
        # Find largest num_groups <= 32 that divides num_features
        num_groups = 1  # fallback
        for ng in range(min(32, num_features), 0, -1):
            if num_features % ng == 0:
                num_groups = ng
                break
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
            context: Context map [B, context_nc, Hc, Wc], or None for identity modulation

        Returns:
            Modulated features [B, num_features, H, W]
        """
        # Normalize x
        normalized = self.bn(x)

        if context is not None:
            # Upsample context if needed to match x's spatial dimensions
            if context.size(2) != x.size(2) or context.size(3) != x.size(3):
                context = F.interpolate(
                    context, size=x.shape[2:], mode="bilinear", align_corners=False
                )

            # Produce gamma, beta from context
            actv = self.mlp_shared(context)
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)

            # Apply affine transform
            out = normalized * (1 + gamma) + beta
        else:
            # Identity modulation: just return normalized features
            out = normalized

        return out
