"""
Test that all code examples in documentation are correct.

This ensures documentation stays in sync with implementation.
"""

import torch
import torch.nn as nn
import pytest
from fluxflow.models.activations import BezierActivation, TrainableBezier
from fluxflow.models.flow import pillarLayer


class TestInputBasedBezier:
    """Test Input-Based BezierActivation examples from documentation."""

    def test_basic_architecture(self):
        """Verify basic Input-Based BezierActivation pattern."""
        # Example from docs: Linear(256, 640) → BezierActivation() → 128 outputs
        linear = nn.Linear(256, 640)
        bezier = BezierActivation()

        x = torch.randn(2, 256)
        out_linear = linear(x)
        assert out_linear.shape == (2, 640), f"Expected (2, 640), got {out_linear.shape}"

        out_bezier = bezier(out_linear)
        assert out_bezier.shape == (2, 128), f"Expected (2, 128), got {out_bezier.shape}"

    def test_parameter_count(self):
        """Verify parameter counts match documentation."""
        linear_layer = nn.Linear(256, 640)
        bezier_act = BezierActivation()

        params_linear = sum(p.numel() for p in linear_layer.parameters())
        params_bezier = sum(p.numel() for p in bezier_act.parameters())

        # From docs: 164,480 params in Linear, 0 in Bezier
        assert params_linear == 164480, f"Expected 164480 Linear params, got {params_linear}"
        assert params_bezier == 0, f"Expected 0 Bezier params, got {params_bezier}"

    def test_conv_architecture(self):
        """Verify Conv2d pattern from documentation."""
        # Pattern: Conv2d(C, 5C) → BezierActivation() → C outputs
        conv = nn.Conv2d(128, 640, kernel_size=3, padding=1)
        bezier = BezierActivation()

        x = torch.randn(2, 128, 32, 32)
        out_conv = conv(x)
        assert out_conv.shape == (2, 640, 32, 32), f"Conv output shape wrong: {out_conv.shape}"

        out_bezier = bezier(out_conv)
        expected_shape = (2, 128, 32, 32)  # 640/5 = 128
        assert (
            out_bezier.shape == expected_shape
        ), f"Expected {expected_shape}, got {out_bezier.shape}"


class TestTrainableBezier:
    """Test TrainableBezier examples from documentation."""

    def test_basic_architecture(self):
        """Verify TrainableBezier pattern from docs."""
        latent_dim = 256
        layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), TrainableBezier((latent_dim,), channel_only=True)
        )

        x = torch.randn(2, 256)
        output = layer(x)
        assert output.shape == (2, 256), f"Expected (2, 256), got {output.shape}"

    def test_parameter_count(self):
        """Verify parameter counts match documentation."""
        linear_layer = nn.Linear(256, 128)
        trainable_bezier = TrainableBezier((128,), channel_only=True)

        params_linear = sum(p.numel() for p in linear_layer.parameters())
        params_tb = sum(p.numel() for p in trainable_bezier.parameters())

        # From docs: Linear=32,896, TrainableBezier=512 (4×128)
        assert params_linear == 32896, f"Expected 32896 Linear params, got {params_linear}"
        assert params_tb == 512, f"Expected 512 TrainableBezier params, got {params_tb}"

    def test_mu_logvar_pattern(self):
        """Test VAE latent bottleneck pattern."""
        latent_dim = 256

        # mu/logvar layers as documented
        mu_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), TrainableBezier((latent_dim,), channel_only=True)
        )
        logvar_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), TrainableBezier((latent_dim,), channel_only=True)
        )

        x = torch.randn(4, 256)
        mu = mu_layer(x)
        logvar = logvar_layer(x)

        assert mu.shape == (4, 256), f"mu shape wrong: {mu.shape}"
        assert logvar.shape == (4, 256), f"logvar shape wrong: {logvar.shape}"

    def test_rgb_output_pattern(self):
        """Test RGB output layer pattern (3 channels)."""
        # 3 channels, 4 params each = 12 total params
        rgb_layer = TrainableBezier((3,), channel_only=True)

        params = sum(p.numel() for p in rgb_layer.parameters())
        assert params == 12, f"Expected 12 RGB params (4×3), got {params}"


class TestPillarBased:
    """Test Pillar-Based examples from documentation."""

    def test_pillar_architecture(self):
        """Verify pillarLayer structure."""
        d_model = 128
        depth = 3

        pillar = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())

        # Test forward pass
        x = torch.randn(2, 16, 128)
        output_tensor = x
        for module in pillar:
            output_tensor = module(output_tensor)

        assert output_tensor.shape == (
            2,
            16,
            128,
        ), f"Pillar output shape wrong: {output_tensor.shape}"

    def test_pillar_parameter_count(self):
        """Verify pillar parameter counts match documentation."""
        d_model = 128
        depth = 3

        pillar = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())
        params_pillar = sum(p.numel() for p in pillar.parameters())

        # From docs: 49,536 params per pillar (3 × 16,512)
        assert params_pillar == 49536, f"Expected 49536 params per pillar, got {params_pillar}"

    def test_four_pillars_architecture(self):
        """Test 4 pillars + BezierActivation pattern."""
        d_model = 128
        depth = 3

        # 4 separate pillar networks
        p0 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())
        p1 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())
        p2 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())
        p3 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())

        bezier_activation = BezierActivation()

        # Forward pass
        img_seq = torch.randn(2, 16, 128)  # [batch, seq_len, d_model]
        g = torch.sigmoid(img_seq)

        # Apply pillars
        img_p0 = g
        for module in p0:
            img_p0 = module(img_p0)

        img_p1 = g
        for module in p1:
            img_p1 = module(img_p1)

        img_p2 = g
        for module in p2:
            img_p2 = module(img_p2)

        img_p3 = g
        for module in p3:
            img_p3 = module(img_p3)

        assert img_p0.shape == img_seq.shape, "Pillar p0 output shape mismatch"
        assert img_p1.shape == img_seq.shape, "Pillar p1 output shape mismatch"
        assert img_p2.shape == img_seq.shape, "Pillar p2 output shape mismatch"
        assert img_p3.shape == img_seq.shape, "Pillar p3 output shape mismatch"

        # Concatenate [img_seq, p0, p1, p2, p3]
        concatenated = torch.cat([img_seq, img_p0, img_p1, img_p2, img_p3], dim=-1)
        expected_cat_shape = (2, 16, 640)  # 128 * 5
        assert (
            concatenated.shape == expected_cat_shape
        ), f"Expected {expected_cat_shape}, got {concatenated.shape}"

        # Apply BezierActivation (5→1 reduction)
        output = bezier_activation(concatenated)
        expected_output_shape = (2, 16, 128)
        assert (
            output.shape == expected_output_shape
        ), f"Expected {expected_output_shape}, got {output.shape}"

    def test_four_pillars_parameter_count(self):
        """Verify 4 pillars total parameter count."""
        d_model = 128
        depth = 3

        p0 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())
        p1 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())
        p2 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())
        p3 = pillarLayer(d_model, d_model, depth=depth, activation=nn.SiLU())

        params_total = (
            sum(p.numel() for p in p0.parameters())
            + sum(p.numel() for p in p1.parameters())
            + sum(p.numel() for p in p2.parameters())
            + sum(p.numel() for p in p3.parameters())
        )

        # From docs: 198,144 params (4 × 49,536)
        assert params_total == 198144, f"Expected 198144 total params, got {params_total}"


class TestParameterComparisons:
    """Test parameter comparisons from documentation."""

    def test_two_layer_relu_baseline(self):
        """Verify ReLU baseline calculation."""
        layer1 = nn.Linear(256, 256)
        layer2 = nn.Linear(256, 256)

        params_total = sum(p.numel() for p in layer1.parameters()) + sum(
            p.numel() for p in layer2.parameters()
        )

        # From docs: 131,584 params (2 layers)
        expected = 2 * (256 * 256 + 256)  # weight + bias per layer
        assert params_total == expected, f"Expected {expected}, got {params_total}"
        assert params_total == 131584, f"Expected 131584, got {params_total}"

    def test_input_based_vs_relu(self):
        """Compare Input-Based to ReLU baseline per layer."""
        # From docs example: Linear(256, 640) for Input-Based vs Linear(256, 256) for ReLU
        # For SAME final output (128), Input-Based uses 2.5× more params per layer

        # ReLU baseline per layer: Linear(256, 256)
        relu_layer = nn.Linear(256, 256)
        relu_params_per_layer = sum(p.numel() for p in relu_layer.parameters())

        # Input-Based per layer: Linear(256, 640) → outputs 128 after Bezier
        bezier_layer = nn.Linear(256, 640)
        bezier_params_per_layer = sum(p.numel() for p in bezier_layer.parameters())

        # From docs verification script: Input-Based uses 2.50× ReLU per layer
        ratio = bezier_params_per_layer / relu_params_per_layer
        assert abs(ratio - 2.5) < 0.01, f"Expected ~2.5× ratio, got {ratio:.2f}"

    def test_trainable_vs_relu(self):
        """Compare TrainableBezier to ReLU baseline."""
        # ReLU
        linear1 = nn.Linear(256, 128)
        relu_params = sum(p.numel() for p in linear1.parameters())

        # TrainableBezier
        linear2 = nn.Linear(256, 128)
        tb = TrainableBezier((128,), channel_only=True)
        tb_total_params = sum(p.numel() for p in linear2.parameters()) + sum(
            p.numel() for p in tb.parameters()
        )

        # From docs: 1.02× more than ReLU
        ratio = tb_total_params / relu_params
        assert 1.01 < ratio < 1.03, f"Expected ~1.02× ratio, got {ratio:.3f}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
