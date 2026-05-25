"""
v0.10.0 conditioning modules.

Contains SPADEWithLearnableScale: a beta-scale extension of v070's SPADE.
The beta_scale learnable scalar starts at 0 so SPADE acts as plain GroupNorm at init,
preventing large magnitude context signals from destabilising early training.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..v070.conditioning import SPADE


class SPADEWithLearnableScale(SPADE):
    """
    SPADE with a learnable beta_scale scalar that starts at 0 (identity at init).

    The plain SPADE formula is: out = GroupNorm(x) + beta
    This subclass scales beta by a learnable scalar:
        out = GroupNorm(x) + beta_scale * beta

    At init, beta_scale == 0 so the layer acts as plain GroupNorm regardless of the
    context signal magnitude.  The scale grows freely through training via the
    compressor/expander optimizer parameter groups.

    No gamma branch is added; the beta-only (additive) design of SPADE is preserved.

    Args:
        context_nc: Number of channels in the context map
        num_features: Number of channels in the feature map x
    """

    def __init__(self, context_nc: int, num_features: int) -> None:
        super().__init__(context_nc, num_features)
        # Learnable scalar: starts at 0 so SPADE is identity at init.
        self.beta_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            x: Feature tensor [B, num_features, H, W]
            context: Context map [B, context_nc, Hc, Wc], or None for identity modulation

        Returns:
            torch.Tensor: Modulated features [B, num_features, H, W]
        """
        normalized = self.bn(x)

        if context is not None:
            if context.size(2) != x.size(2) or context.size(3) != x.size(3):
                context = F.interpolate(
                    context, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            actv = self.mlp_shared(context)
            raw_beta = self.mlp_beta(actv)
            beta = self.beta_scale * raw_beta
            return cast(torch.Tensor, normalized + beta)

        return cast(torch.Tensor, normalized)
