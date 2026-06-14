"""
v0.10.0 conditioning modules.

Contains SPADE_v100b — a multi-scale SPADE variant with three additive β heads
(1×1 low-freq, 3×3 mid-freq, 3×3 dilated high-freq — effective 7×7 receptive
field via dilation=3) and a bounded multiplicative γ branch.  Both scales
start at 0 so the layer is exactly identity at init.

``SPADEWithLearnableScale`` is kept as a deprecated alias for one release to
ease migration of any external code; the v100 callsites are updated in M2.4.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import BezierActivation
from ..v070.conditioning import SPADE


class SPADE_v100b(SPADE):
    """
    Multi-scale SPADE with bounded multiplicative γ.

    The single beta head of v070's SPADE is replaced by three additive heads
    operating at different receptive fields (1×1, 3×3, and 3×3 dilated with
    effective 7×7 receptive field via dilation=3). A new
    multiplicative γ branch produces γ = 1 + softplus(scale·raw) - softplus(0),
    which is strictly > 0, exactly 1 at init, and can only amplify never null
    features — preserving the beta-only design's MSE-blur safety.

    Two learnable scalars ``beta_scale`` and ``gamma_scale`` start at 0 so the
    layer is exactly identity at init (out == GroupNorm(x)). The shared MLP
    is two Conv3×3 + Bezier(sig, silu) pairs, ethos-aligned.

    Args:
        context_nc: Channels of the context tensor.
        num_features: Channels of the feature tensor being normalised.
    """

    def __init__(self, context_nc: int, num_features: int) -> None:
        super().__init__(context_nc, num_features)
        # Replace the inherited 1-layer mlp_shared / mlp_beta with the new heads.
        hidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(context_nc, hidden * 5, kernel_size=3, padding=1),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Conv2d(hidden, hidden * 5, kernel_size=3, padding=1),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
        )
        # Three β heads at increasing receptive field.
        self.beta_low = nn.Conv2d(hidden, num_features, kernel_size=1)
        self.beta_mid = nn.Conv2d(hidden, num_features, kernel_size=3, padding=1)
        self.beta_hi = nn.Conv2d(hidden, num_features, kernel_size=3, padding=3, dilation=3)
        # Multiplicative γ head.
        self.gamma_head = nn.Conv2d(hidden, num_features, kernel_size=3, padding=1)
        # Drop the inherited single mlp_beta; explicit del so loaded state_dicts
        # from old SPADEWithLearnableScale cleanly fail at load time.
        if hasattr(self, "mlp_beta"):
            del self.mlp_beta
        # Learnable scalars — both start at 0 → identity at init.
        self.beta_scale = nn.Parameter(torch.zeros(1))
        self.gamma_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, num_features, H, W].
            context: Context tensor [B, context_nc, Hc, Wc] or None.

        Returns:
            Modulated feature tensor [B, num_features, H, W].
        """
        normalized = self.bn(x)
        if context is None:
            return cast(torch.Tensor, normalized)

        if context.size(2) != x.size(2) or context.size(3) != x.size(3):
            context = F.interpolate(context, size=x.shape[2:], mode="bilinear", align_corners=False)
        actv = self.mlp_shared(context)

        beta = self.beta_low(actv) + self.beta_mid(actv) + self.beta_hi(actv)
        beta = self.beta_scale * beta

        gamma_raw = self.gamma_head(actv)
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        gamma = 1.0 + F.softplus(self.gamma_scale * gamma_raw) - F.softplus(zero)

        return cast(torch.Tensor, gamma * normalized + beta)


# Deprecated alias — kept for one release; will be removed after M9.
SPADEWithLearnableScale = SPADE_v100b
