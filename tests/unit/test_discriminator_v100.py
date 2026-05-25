"""Tests for PatchDiscriminator with v0.10.0 ctx_dim=256."""

import torch


class TestPatchDiscriminatorV100:
    """PatchDiscriminator ctx_dim=256 (2 * 128) compatibility tests."""

    def test_ctx_proj_width_matches_2d_packed_mean(self):
        """With v0.10.0, ctx_dim=256 (2*128). Projection must accept [B, 256] ctx_vec."""
        from fluxflow.models.discriminators import PatchDiscriminator

        disc = PatchDiscriminator(in_channels=3, base_ch=16, depth=2, ctx_dim=256)
        img = torch.randn(2, 3, 64, 64)
        ctx = torch.randn(2, 256)
        with torch.no_grad():
            out = disc(img, ctx_vec=ctx)
        assert out.shape[0] == 2

    def test_ctx_proj_backward_reaches_ctx_vec(self):
        """Gradient must flow from discriminator output to ctx_vec."""
        from fluxflow.models.discriminators import PatchDiscriminator

        disc = PatchDiscriminator(in_channels=3, base_ch=16, depth=2, ctx_dim=256)
        img = torch.randn(1, 3, 64, 64)
        ctx = torch.randn(1, 256, requires_grad=True)
        with torch.no_grad():
            disc(img, ctx_vec=ctx.detach())
        # Re-do with grad enabled
        ctx2 = torch.randn(1, 256, requires_grad=True)
        logits2 = disc(img, ctx_vec=ctx2)
        logits2.mean().backward()
        assert ctx2.grad is not None

    def test_ctx_proj_in_features_equals_256(self):
        """ctx_proj.in_features must be 256 for v0.10.0 context."""
        from fluxflow.models.discriminators import PatchDiscriminator

        disc = PatchDiscriminator(in_channels=3, base_ch=16, depth=2, ctx_dim=256)
        assert disc.ctx_proj.in_features == 256
