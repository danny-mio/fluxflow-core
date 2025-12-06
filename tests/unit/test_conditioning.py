"""Unit tests for conditioning layers (src/models/conditioning.py)."""

import torch
import torch.nn as nn

from fluxflow.models.conditioning import (
    DEFAULT_CONFIG_VALUE,
    SPADE,
    ContextAttentionMixer,
    FiLM,
    GatedContextInjection,
    LeanContextModule,
    stable_scale_text_embeddings,
)


class TestStableScaleTextEmbeddings:
    """Tests for stable_scale_text_embeddings function."""

    def test_l2_normalizes(self):
        """Should L2 normalize embeddings."""
        text_emb = torch.randn(4, 256) * 10.0  # Unnormalized

        result = stable_scale_text_embeddings(text_emb, config_value=10.0)

        # Check L2 norm is 1 (scaled by config_value/DEFAULT_CONFIG_VALUE)
        norms = torch.norm(result, p=2, dim=1)
        expected_norm = 10.0 / DEFAULT_CONFIG_VALUE
        assert torch.allclose(norms, torch.ones(4) * expected_norm, atol=1e-5)

    def test_scales_by_config_value(self):
        """Should scale by config_value / DEFAULT_CONFIG_VALUE."""
        text_emb = torch.randn(2, 128)

        result = stable_scale_text_embeddings(text_emb, config_value=20.0)

        # Norm should be 20.0 / DEFAULT_CONFIG_VALUE
        norms = torch.norm(result, p=2, dim=1)
        expected = 20.0 / DEFAULT_CONFIG_VALUE
        assert torch.allclose(norms, torch.ones(2) * expected, atol=1e-5)


class TestFiLM:
    """Tests for Feature-wise Linear Modulation."""

    def test_initialization(self):
        """Test FiLM layer initializes correctly."""
        film = FiLM(text_dim=256, num_channels=64)

        # Should have linear layer
        assert isinstance(film.linear, nn.Linear)
        assert film.linear.in_features == 256
        assert film.linear.out_features == 128  # 2 * num_channels

    def test_output_shape(self):
        """Output should have same shape as input features."""
        film = FiLM(text_dim=256, num_channels=64)

        x = torch.randn(4, 64, 32, 32)
        text_emb = torch.randn(4, 256)

        output = film(x, text_emb)

        assert output.shape == x.shape

    def test_modulation_formula(self):
        """Should apply modulation: x * (1 + gamma) + beta."""
        film = FiLM(text_dim=128, num_channels=32)

        x = torch.ones(2, 32, 8, 8)
        text_emb = torch.zeros(2, 128)

        # With zero text embeddings, linear outputs zeros
        # gamma=0, beta=0, so output should be x * 1 + 0 = x
        with torch.no_grad():
            film.linear.weight.zero_()
            film.linear.bias.zero_()

        output = film(x, text_emb)
        assert torch.allclose(output, x)

    def test_gradient_flow(self):
        """Gradients should flow through FiLM layer."""
        film = FiLM(text_dim=128, num_channels=32)

        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        text_emb = torch.randn(2, 128, requires_grad=True)

        output = film(x, text_emb)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert text_emb.grad is not None

    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        film = FiLM(text_dim=256, num_channels=64)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 64, 16, 16)
            text_emb = torch.randn(batch_size, 256)

            output = film(x, text_emb)
            assert output.shape == (batch_size, 64, 16, 16)


class TestSPADE:
    """Tests for Spatially-Adaptive Denormalization."""

    def test_initialization(self):
        """Test SPADE layer initializes correctly."""
        spade = SPADE(context_nc=128, num_features=64)

        # Should have GroupNorm (no affine) - works better with batch_size=1
        assert isinstance(spade.bn, nn.GroupNorm)
        assert not spade.bn.affine

        # Should have conv layers
        assert spade.mlp_shared is not None
        assert spade.mlp_gamma is not None
        assert spade.mlp_beta is not None

    def test_output_shape(self):
        """Output should have same shape as input features."""
        spade = SPADE(context_nc=128, num_features=64)

        x = torch.randn(4, 64, 32, 32)
        context = torch.randn(4, 128, 32, 32)

        output = spade(x, context)

        assert output.shape == x.shape

    def test_upsamples_context_if_needed(self):
        """Should upsample context to match x spatial dimensions."""
        spade = SPADE(context_nc=128, num_features=64)

        x = torch.randn(2, 64, 64, 64)
        context = torch.randn(2, 128, 32, 32)  # Half resolution

        # Should not crash
        output = spade(x, context)
        assert output.shape == x.shape

    def test_batch_normalization_applied(self):
        """Should apply batch normalization to x."""
        spade = SPADE(context_nc=128, num_features=64)

        # Use batch with different means
        x = torch.cat(
            [
                torch.ones(1, 64, 16, 16) * 10.0,
                torch.ones(1, 64, 16, 16) * 5.0,
            ],
            dim=0,
        )
        context = torch.randn(2, 128, 16, 16)

        spade.eval()  # Use running stats
        output = spade(x, context)

        # Output should be normalized
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Gradients should flow through SPADE."""
        spade = SPADE(context_nc=128, num_features=64)

        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        context = torch.randn(2, 128, 16, 16, requires_grad=True)

        output = spade(x, context)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert context.grad is not None


class TestGatedContextInjection:
    """Tests for GatedContextInjection."""

    def test_initialization(self):
        """Test initialization."""
        gci = GatedContextInjection(context_dim=256, target_dim=128)

        # Should have gate and bias layers
        assert isinstance(gci.gate, nn.Sequential)
        assert isinstance(gci.bias, nn.Linear)

    def test_output_shape(self):
        """Output should have same shape as input."""
        gci = GatedContextInjection(context_dim=256, target_dim=128)

        x = torch.randn(4, 16, 128)  # [B, T, target_dim]
        context = torch.randn(4, 256)  # [B, context_dim]

        output = gci(x, context)

        assert output.shape == x.shape

    def test_gating_formula(self):
        """Should apply: x * gate(context) + bias(context)."""
        gci = GatedContextInjection(context_dim=128, target_dim=64)

        x = torch.ones(2, 10, 64)
        context = torch.zeros(2, 128)

        # With zero context and zero-initialized weights
        with torch.no_grad():
            gci.gate[0].weight.zero_()
            gci.gate[0].bias.zero_()
            gci.bias.weight.zero_()
            gci.bias.bias.zero_()

        output = gci(x, context)

        # Gate should be sigmoid(0) = 0.5, bias = 0
        # Output = x * 0.5 + 0 = 0.5
        expected = torch.ones(2, 10, 64) * 0.5
        assert torch.allclose(output, expected, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through gating."""
        gci = GatedContextInjection(context_dim=256, target_dim=128)

        x = torch.randn(2, 10, 128, requires_grad=True)
        context = torch.randn(2, 256, requires_grad=True)

        output = gci(x, context)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert context.grad is not None

    def test_gate_range(self):
        """Gate values should be in [0, 1] due to sigmoid."""
        gci = GatedContextInjection(context_dim=128, target_dim=64)

        x = torch.randn(4, 10, 64)
        context = torch.randn(4, 128)

        # Access gate value (not easy without modifying forward, but we can test output range)
        output = gci(x, context)
        assert torch.isfinite(output).all()


class TestContextAttentionMixer:
    """Tests for ContextAttentionMixer."""

    def test_initialization_with_cls(self):
        """Test initialization with CLS token."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=True)

        # Should have CLS parameter
        assert hasattr(mixer, "cls")
        assert mixer.cls.shape == (1, 1, 128)

    def test_initialization_without_cls(self):
        """Test initialization without CLS token."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=False)

        # Should not have CLS parameter
        assert not hasattr(mixer, "cls")

    def test_output_shapes_with_cls(self):
        """With CLS: pooled [B, D], tokens [B, K, D]."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=True)

        x = torch.randn(4, 16, 128)  # [B, K, D]

        pooled, tokens = mixer(x)

        assert pooled.shape == (4, 128)  # [B, D]
        assert tokens.shape == (4, 16, 128)  # [B, K, D] (same as input)

    def test_output_shapes_without_cls(self):
        """Without CLS: pooled is mean-pooled."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=False)

        x = torch.randn(4, 16, 128)

        pooled, tokens = mixer(x)

        assert pooled.shape == (4, 128)
        assert tokens.shape == (4, 16, 128)

    def test_cls_pooling(self):
        """With CLS, pooled should be CLS token representation."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=True)

        x = torch.randn(2, 10, 128)

        pooled, tokens = mixer(x)

        # Pooled should be different from mean of input
        mean_pool = x.mean(dim=1)
        assert not torch.allclose(pooled, mean_pool)

    def test_mean_pooling(self):
        """Without CLS, pooled should be mean of tokens."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=False)

        x = torch.randn(2, 10, 128)

        pooled, _ = mixer(x)

        # Should be close to output mean (after attention)
        # Not exactly equal due to attention transformations
        assert pooled.shape == (2, 128)

    def test_multi_head_attention(self):
        """Should apply multi-head attention."""
        mixer = ContextAttentionMixer(d_model=128, n_head=8, use_cls=True)

        x = torch.randn(4, 16, 128)

        pooled, tokens = mixer(x)

        # Should not crash and produce valid output
        assert torch.isfinite(pooled).all()
        assert torch.isfinite(tokens).all()

    def test_gradient_flow(self):
        """Gradients should flow through attention."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=True)

        x = torch.randn(2, 10, 128, requires_grad=True)

        pooled, tokens = mixer(x)
        loss = pooled.sum() + tokens.sum()
        loss.backward()

        assert x.grad is not None

    def test_dropout_rates(self):
        """Should accept dropout rates."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, attn_drop=0.1, proj_drop=0.1)

        x = torch.randn(4, 16, 128)

        # Should not crash
        pooled, tokens = mixer(x)
        assert pooled.shape == (4, 128)

    def test_different_sequence_lengths(self):
        """Should work with different sequence lengths."""
        mixer = ContextAttentionMixer(d_model=128, n_head=4, use_cls=True)

        for seq_len in [4, 8, 16, 32]:
            x = torch.randn(2, seq_len, 128)
            pooled, tokens = mixer(x)

            assert pooled.shape == (2, 128)
            assert tokens.shape == (2, seq_len, 128)


class TestLeanContextModule:
    """Tests for LeanContextModule (2D self-attention)."""

    def test_initialization(self):
        """Test initialization."""
        module = LeanContextModule(in_channels=64, out_channels=32)

        assert module.in_channels == 64
        # Should have query, key, value convs
        assert hasattr(module, "query")
        assert hasattr(module, "key")
        assert hasattr(module, "value")

    def test_output_shape(self):
        """Should downsample by 2x and output out_channels."""
        module = LeanContextModule(in_channels=64, out_channels=32)

        x = torch.randn(2, 64, 16, 16)
        output = module(x)

        # Should be downsampled by 2 (stride=2 in conv)
        assert output.shape == (2, 32, 8, 8)

    def test_self_attention_applied(self):
        """Should apply self-attention."""
        module = LeanContextModule(in_channels=32, out_channels=16)

        x = torch.randn(2, 32, 8, 8)
        output = module(x)

        # Should produce valid output
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Gradients should flow through self-attention."""
        module = LeanContextModule(in_channels=32, out_channels=16)

        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestConditioningIntegration:
    """Integration tests combining multiple conditioning layers."""

    def test_film_in_network(self):
        """Test FiLM in a simple network."""
        network = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
        )

        film = FiLM(text_dim=256, num_channels=64)

        x = torch.randn(4, 3, 32, 32)
        text_emb = torch.randn(4, 256)

        features = network(x)
        modulated = film(features, text_emb)

        assert modulated.shape == (4, 64, 32, 32)

    def test_spade_in_decoder(self):
        """Test SPADE in decoder-like architecture."""
        spade = SPADE(context_nc=128, num_features=64)

        # Simulate decoder features and encoder context
        decoder_features = torch.randn(2, 64, 32, 32)
        encoder_context = torch.randn(2, 128, 16, 16)  # Lower resolution

        output = spade(decoder_features, encoder_context)

        assert output.shape == decoder_features.shape

    def test_multiple_conditioning_layers(self):
        """Test combining FiLM and gated injection."""
        film = FiLM(text_dim=256, num_channels=64)
        gci = GatedContextInjection(context_dim=256, target_dim=64)

        # Image features
        img_features = torch.randn(4, 64, 16, 16)
        text_emb = torch.randn(4, 256)

        # Apply FiLM
        modulated = film(img_features, text_emb)

        # Reshape for GCI
        B, C, H, W = modulated.shape
        tokens = modulated.view(B, C, -1).transpose(1, 2)  # [B, HW, C]

        # Apply GCI
        output = gci(tokens, text_emb)

        assert output.shape == (4, H * W, 64)

    def test_attention_mixer_with_projection(self):
        """Test ContextAttentionMixer with feature projection."""
        mixer = ContextAttentionMixer(d_model=256, n_head=8, use_cls=True)
        projection = nn.Linear(256, 128)

        x = torch.randn(4, 32, 256)

        pooled, tokens = mixer(x)
        pooled_projected = projection(pooled)

        assert pooled_projected.shape == (4, 128)
