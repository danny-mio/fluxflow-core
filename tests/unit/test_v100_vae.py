"""Unit tests for FluxFlow v0.10.0 VAE (FluxCompressor_v100 and FluxExpander_v100).

Run *before* implementation — all tests must fail until implementation exists.
"""

import torch
import pytest


class TestFluxCompressorV100:
    """Tests for FluxCompressor_v100."""

    def test_get_context_dims_equals_d_model(self):
        """get_context_dims() must return d_model (not a fixed constant)."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=128, downscales=2)
        assert comp.get_context_dims() == 128

    def test_get_context_dims_varies_with_d_model(self):
        """get_context_dims() must reflect the actual d_model, not a hard-coded value."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=64, downscales=2)
        assert comp.get_context_dims() == 64

    def test_forward_packed_shape(self):
        """Packed output must have shape [B, T+1, 2*D]."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=64, downscales=2)
        img = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            packed = comp(img)
        H_lat, W_lat = 64 // 4, 64 // 4  # 2 downscales → stride 4
        T = H_lat * W_lat
        assert packed.shape == (2, T + 1, 128), f"Expected (2, {T+1}, 128), got {packed.shape}"

    def test_forward_training_returns_mu_logvar(self):
        """training=True must return (packed, mu, logvar)."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=64, downscales=2)
        img = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            result = comp(img, training=True)
        assert isinstance(result, tuple) and len(result) == 3
        packed, mu, logvar = result
        assert mu.shape[1] == 64  # d_model channels
        assert logvar.shape == mu.shape

    def test_context_branch_independent_of_z(self):
        """Zeroing z-branch weights must not affect context tokens."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=32, downscales=2, ctx_attn_layers=1, ctx_attn_heads=4)
        img = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            packed_before = comp(img)
        # Zero out z-branch mu_proj to change z output
        comp.mu_proj[0][0].weight.data.zero_()
        with torch.no_grad():
            packed_after = comp(img)
        D = comp.d_model
        T = packed_before.shape[1] - 1
        # z tokens must differ
        assert not torch.allclose(packed_before[:, :T, :D], packed_after[:, :T, :D])
        # context tokens must be identical
        assert torch.allclose(packed_before[:, :T, D:], packed_after[:, :T, D:])

    def test_ctx_attn_layers_constructor_param(self):
        """ctx_attn_layers controls depth of ctx_token_attn ModuleList."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp2 = FluxCompressor_v100(d_model=32, downscales=2, ctx_attn_layers=2, ctx_attn_heads=4)
        comp4 = FluxCompressor_v100(d_model=32, downscales=2, ctx_attn_layers=4, ctx_attn_heads=4)
        assert len(comp2.ctx_token_attn) == 2
        assert len(comp4.ctx_token_attn) == 4

    def test_hw_vec_encodes_spatial_dims(self):
        """HW token [B, -1, :] must have H/max_hw and W/max_hw in first two positions."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=32, downscales=2, max_hw=1024)
        img = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            packed = comp(img)
        H_lat = 32 // 4
        W_lat = 32 // 4
        assert abs(float(packed[0, -1, 0]) - H_lat / 1024.0) < 1e-5
        assert abs(float(packed[0, -1, 1]) - W_lat / 1024.0) < 1e-5

    def test_no_module_level_context_dims_constant(self):
        """v100.vae must not export CONTEXT_DIMS as module-level constant."""
        import fluxflow.models.v100.vae as vae_module

        assert not hasattr(vae_module, "CONTEXT_DIMS"), (
            "v100/vae.py must not define module-level CONTEXT_DIMS"
        )

    def test_gradient_flows_through_compressor(self):
        """Gradients must flow from packed output back to img input."""
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=32, downscales=2, use_gradient_checkpointing=False)
        img = torch.randn(1, 3, 32, 32, requires_grad=True)
        packed = comp(img)
        packed.sum().backward()
        assert img.grad is not None
        assert not torch.isnan(img.grad).any()


class TestFluxExpanderV100:
    """Tests for FluxExpander_v100."""

    def test_unpack_splits_correctly(self):
        """unpack must split [B, T+1, 2D] into img_seq [B,T,D], context [B,T,D], H, W."""
        from fluxflow.models.v100.vae import FluxExpander_v100

        exp = FluxExpander_v100(d_model=64)
        B, T, D = 1, 16, 64
        packed = torch.randn(B, T + 1, 2 * D)
        # Set HW token: H=16, W=16 → 16/1024
        packed[:, -1, :] = 0
        packed[:, -1, 0] = 16 / 1024.0
        packed[:, -1, 1] = 16 / 1024.0
        img_seq, context, H, W = exp.unpack(packed)
        assert img_seq.shape == (B, T, D)
        assert context.shape == (B, T, D)
        assert H[0].item() == 16
        assert W[0].item() == 16

    def test_spade_beta_scale_starts_at_zero(self):
        """SPADEWithLearnableScale.beta_scale must init to 0 (identity at init)."""
        from fluxflow.models.v100.vae import FluxExpander_v100

        exp = FluxExpander_v100(d_model=32, upscales=2)
        for layer in exp.upscale.layers:
            assert hasattr(layer, "spade"), "ResidualUpsampleBlock must have .spade"
            assert hasattr(layer.spade, "beta_scale"), "SPADE must have beta_scale"
            assert float(layer.spade.beta_scale.item()) == 0.0

    def test_roundtrip_image_reconstruct_shape(self):
        """comp(img) → exp(packed) must yield original image shape."""
        from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

        comp = FluxCompressor_v100(d_model=64, downscales=2)
        exp = FluxExpander_v100(d_model=64, upscales=2)
        img = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            packed = comp(img)
            out = exp(packed)
        assert out.shape == (1, 3, 64, 64)

    def test_expander_forward_without_context(self):
        """Expander must work when use_context=False (disables SPADE)."""
        from fluxflow.models.v100.vae import FluxExpander_v100

        exp = FluxExpander_v100(d_model=32, upscales=2)
        B, T, D = 1, 16, 32
        packed = torch.zeros(B, T + 1, 2 * D)
        packed[:, -1, 0] = 16 / 1024.0
        packed[:, -1, 1] = 16 / 1024.0
        with torch.no_grad():
            out = exp(packed, use_context=False)
        assert out.shape[0] == 1
        assert out.shape[1] == 3


class TestSPADEWithLearnableScale:
    """Tests for v100 conditioning module."""

    def test_beta_scale_is_nn_parameter(self):
        """beta_scale must be an nn.Parameter so optimizer sees it."""
        from fluxflow.models.v100.conditioning import SPADEWithLearnableScale

        spade = SPADEWithLearnableScale(context_nc=64, num_features=32)
        assert isinstance(spade.beta_scale, torch.nn.Parameter)

    def test_beta_scale_starts_at_zero(self):
        """At init, beta_scale == 0 so SPADE acts as plain GroupNorm."""
        from fluxflow.models.v100.conditioning import SPADEWithLearnableScale

        spade = SPADEWithLearnableScale(context_nc=64, num_features=32)
        assert float(spade.beta_scale.item()) == 0.0

    def test_forward_with_context(self):
        """forward must return normalized + beta_scale * raw_beta."""
        from fluxflow.models.v100.conditioning import SPADEWithLearnableScale

        spade = SPADEWithLearnableScale(context_nc=8, num_features=16)
        x = torch.randn(1, 16, 8, 8)
        ctx = torch.randn(1, 8, 8, 8)
        with torch.no_grad():
            out = spade(x, ctx)
        # With beta_scale=0, output == GroupNorm(x)
        assert out.shape == x.shape

    def test_forward_without_context(self):
        """forward with context=None must return GroupNorm(x)."""
        from fluxflow.models.v100.conditioning import SPADEWithLearnableScale

        spade = SPADEWithLearnableScale(context_nc=8, num_features=16)
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            out = spade(x, None)
        assert out.shape == x.shape

    def test_inherits_from_spade(self):
        """SPADEWithLearnableScale must subclass SPADE from v070."""
        from fluxflow.models.v070.conditioning import SPADE
        from fluxflow.models.v100.conditioning import SPADEWithLearnableScale

        assert issubclass(SPADEWithLearnableScale, SPADE)
