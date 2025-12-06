"""
Discriminator models for FluxFlow GAN training.

Contains:
- PatchDiscriminator: PatchGAN discriminator with optional projection conditioning
- DBlock: Downsampling discriminator block with spectral normalization
"""

import torch.nn as nn
from torch.nn.utils import spectral_norm as _sn


def snconv(in_ch, out_ch, k, s=1, p=0):
    """Spectral-normalized convolution block with LeakyReLU."""
    return nn.Sequential(
        _sn(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)),
        nn.LeakyReLU(0.2, inplace=True),
        _sn(nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=s, padding=p)),
    )


def snlinear(in_f, out_f):
    """Spectral-normalized linear block with LeakyReLU."""
    return nn.Sequential(
        _sn(nn.Linear(in_f, out_f)),
        nn.LeakyReLU(0.2, inplace=True),
        _sn(nn.Linear(out_f, out_f)),
    )


class DBlock(nn.Module):
    """
    Downsampling discriminator block with spectral normalization.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        down: Enable 2x downsampling via average pooling (default: True)
    """

    def __init__(self, in_ch, out_ch, down=True):
        super().__init__()
        self.conv1 = snconv(in_ch, out_ch, 3, 1, 1)
        self.conv2 = snconv(out_ch, out_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.down = down
        if down:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, in_ch, H, W]

        Returns:
            Output tensor [B, out_ch, H//2, W//2] if down=True, else [B, out_ch, H, W]
        """
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        if self.down:
            h = self.pool(h)
        return h


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator with optional Miyato-style projection conditioning.

    Discriminates image patches and optionally conditions on context vectors
    using projection-based conditioning.

    Args:
        in_channels: Input image channels (default: 3)
        base_ch: Base number of channels (default: 64)
        depth: Number of DBlocks (each downsamples by 2x) (default: 4)
        ctx_dim: Context vector dimension; 0 disables projection (default: 0)
    """

    def __init__(self, in_channels=3, base_ch=64, depth=4, ctx_dim=0):
        super().__init__()
        ch = base_ch
        blocks = [snconv(in_channels, ch, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        c = ch
        for _ in range(depth):
            blocks.append(DBlock(c, c * 2, down=True))
            c *= 2
        self.backbone = nn.Sequential(*blocks)
        # Head: output patch logits without activation (for single channel output)
        self.head = _sn(nn.Conv2d(c, 1, kernel_size=3, stride=1, padding=1))
        self.ctx_dim = ctx_dim
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        if ctx_dim and ctx_dim > 0:
            # Projection: <phi(h_pool), Wc * c>
            self.ctx_proj = snlinear(ctx_dim, c)
            self.feat_proj = nn.Identity()  # Can replace with snlinear(c, c) for learnable mapping

    def forward(self, x, ctx_vec=None, return_feats=False):
        """
        Args:
            x: Input images [B, in_channels, H, W] in [-1, 1]
            ctx_vec: Optional context vector [B, ctx_dim]
            return_feats: If True, returns (features, patch_logits); else logits only

        Returns:
            If return_feats=False: patch_logits [B, 1, H', W']
            If return_feats=True: tuple of (features [B, C], patch_logits [B, 1, H', W'])
        """
        h = self.backbone(x)  # [B, C, H', W']
        patch_logits = self.head(h)  # [B, 1, H', W']

        if self.ctx_dim and ctx_vec is not None:
            h_pool = self.pool(h).flatten(1)  # [B, C]
            c_proj = self.ctx_proj(ctx_vec)  # [B, C]
            proj_term = (h_pool * c_proj).sum(dim=1, keepdim=True)  # [B, 1]
            patch_logits = patch_logits + proj_term.unsqueeze(-1).unsqueeze(-1)

        if return_feats:
            # Return pooled features for optional feature-matching
            feats = self.pool(h).flatten(1)  # [B, C]
            return feats, patch_logits

        return patch_logits
