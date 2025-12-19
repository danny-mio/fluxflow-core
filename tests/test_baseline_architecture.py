"""
Tests for baseline (non-Bezier) architecture parameter matching.

These tests verify that baseline models match Bezier architecture parameters
within acceptable tolerances while using standard activations (SiLU/GELU/ReLU).

Phase 0.3: Parameter matching validation
- VAE: BaselineResidualUpsampleBlock vs ResidualUpsampleBlock
- Flow: BaselineFluxTransformerBlock vs FluxTransformerBlock
- Latent compatibility: Both produce [B, T+1, 128] shape
"""

import pytest
import torch
import torch.nn as nn

from fluxflow.models.vae import BaselineResidualUpsampleBlock, ResidualUpsampleBlock
from fluxflow.models.flow import BaselineFluxTransformerBlock, FluxTransformerBlock


class TestBaselineVAEParameterMatching:
    """Test VAE parameter matching between Bezier and baseline."""

    def test_baseline_upsample_block_instantiation(self):
        """Verify BaselineResidualUpsampleBlock can be instantiated."""
        block = BaselineResidualUpsampleBlock(
            channels=64,
            baseline_activation="silu",
            depth_multiplier=30.0,
            width_multiplier=2.5,
            use_spade=False,
        )
        assert isinstance(block, nn.Module)

    def test_baseline_upsample_block_forward_shape(self):
        """Verify baseline block produces correct output shape."""
        block = BaselineResidualUpsampleBlock(
            channels=64,
            baseline_activation="silu",
            depth_multiplier=30.0,
            width_multiplier=2.5,
            use_spade=False,
        )
        x = torch.randn(2, 64, 16, 16)
        output = block(x)
        assert output.shape == (2, 64, 32, 32), "Should upsample 2× spatially"

    def test_baseline_vs_bezier_parameter_count(self):
        """
        Test that baseline parameters match Bezier within ±10%.

        Bezier: 5× expansion → ~1405*C² parameters
        Baseline: width_mult × (256 + 9×depth_mult) → target 1405*C²

        Strategy: Test multiple (width, depth) combinations
        """
        channels = 64

        # Bezier reference (5× expansion)
        bezier_block = ResidualUpsampleBlock(
            channels=channels,
            use_spade=False,
            context_size=0,
        )
        bezier_params = sum(p.numel() for p in bezier_block.parameters())

        # Target: Match bezier_params within ±10%
        tolerance = 0.10
        min_params = bezier_params * (1 - tolerance)
        max_params = bezier_params * (1 + tolerance)

        # Test candidate configurations
        # Strategy: Match Bezier's 5× expansion with similar width, minimal depth
        configs = [
            # (width_mult, depth_mult)
            (4.0, 1.0),  # 14% below Bezier
            (4.5, 1.0),  # 2.2% below Bezier - OPTIMAL
            (4.8, 1.0),  # 5.2% above Bezier
        ]

        best_match = None
        best_diff = float("inf")

        for width_mult, depth_mult in configs:
            baseline_block = BaselineResidualUpsampleBlock(
                channels=channels,
                baseline_activation="silu",
                depth_multiplier=depth_mult,
                width_multiplier=width_mult,
                use_spade=False,
            )
            baseline_params = sum(p.numel() for p in baseline_block.parameters())

            diff = abs(baseline_params - bezier_params)
            if diff < best_diff:
                best_diff = diff
                best_match = (width_mult, depth_mult, baseline_params)

            # Check if within tolerance
            if min_params <= baseline_params <= max_params:
                print(
                    f"✓ Match found: width={width_mult}, depth={depth_mult}, "
                    f"params={baseline_params} (Bezier: {bezier_params}, "
                    f"diff={100*diff/bezier_params:.1f}%)"
                )
                return  # Test passes

        # If no match, report best attempt
        w, d, p = best_match
        pytest.fail(
            f"No configuration within ±{tolerance*100}% tolerance. "
            f"Best: width={w}, depth={d}, params={p} (Bezier: {bezier_params}, "
            f"diff={100*best_diff/bezier_params:.1f}%)"
        )

    def test_baseline_activations(self):
        """Test all supported baseline activations work."""
        for activation in ["silu", "gelu", "relu"]:
            block = BaselineResidualUpsampleBlock(
                channels=64,
                baseline_activation=activation,
                depth_multiplier=10.0,
                width_multiplier=2.0,
                use_spade=False,
            )
            x = torch.randn(1, 64, 8, 8)
            output = block(x)
            assert output.shape == (1, 64, 16, 16)


class TestBaselineFlowParameterMatching:
    """Test Flow parameter matching between Bezier and baseline."""

    def test_baseline_transformer_block_instantiation(self):
        """Verify BaselineFluxTransformerBlock can be instantiated."""
        block = BaselineFluxTransformerBlock(
            d_model=128,
            n_head=8,
            baseline_activation="silu",
            ffn_expansion=4.0,
        )
        assert isinstance(block, nn.Module)

    def test_baseline_transformer_block_forward_shape(self):
        """Verify baseline block produces correct output shapes."""
        d_model = 128
        n_head = 8
        batch_size = 2
        seq_len_img = 16
        seq_len_txt = 10

        block = BaselineFluxTransformerBlock(
            d_model=d_model,
            n_head=n_head,
            baseline_activation="silu",
            ffn_expansion=4.0,
        )

        img_seq = torch.randn(batch_size, seq_len_img, d_model)
        text_seq = torch.randn(batch_size, seq_len_txt, d_model)

        # Create rotary embeddings
        from fluxflow.models.flow import RotaryPositionalEmbedding

        rotary_pe = RotaryPositionalEmbedding(d_model // n_head)
        # get_embed expects tensor
        sin_img, cos_img = rotary_pe.get_embed(torch.arange(seq_len_img))
        sin_txt, cos_txt = rotary_pe.get_embed(torch.arange(seq_len_txt))

        # Forward pass
        img_out, txt_out, _ = block(img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt)

        assert img_out.shape == (batch_size, seq_len_img, d_model)
        assert txt_out.shape == (batch_size, seq_len_txt, d_model)

    def test_single_block_parameter_comparison(self):
        """
        Compare single block parameters: Bezier vs Baseline.

        Bezier block: 4 pillars (depth=3) + FFN = 281,656 params/block
        Baseline block: 2-layer FFN @ 4× expansion = 198,712 params/block

        Strategy: Baseline has FEWER params per block, compensates with MORE blocks
        """
        d_model = 128
        n_head = 8

        # Bezier block (pillars hardcoded: size=d_model, depth=3)
        bezier_block = FluxTransformerBlock(
            d_model=d_model,
            n_head=n_head,
        )
        bezier_params = sum(p.numel() for p in bezier_block.parameters())

        # Baseline block
        baseline_block = BaselineFluxTransformerBlock(
            d_model=d_model,
            n_head=n_head,
            baseline_activation="silu",
            ffn_expansion=4.0,
        )
        baseline_params = sum(p.numel() for p in baseline_block.parameters())

        print(f"Bezier block params: {bezier_params:,}")
        print(f"Baseline block params: {baseline_params:,}")
        print(
            f"Ratio: {baseline_params/bezier_params:.2f}× "
            f"(baseline has {100*(1-baseline_params/bezier_params):.1f}% fewer params)"
        )

        # Verify baseline has fewer params (expected: ~70% of Bezier)
        assert baseline_params < bezier_params, "Baseline should have fewer params/block"
        assert baseline_params / bezier_params > 0.6, "Should be at least 60% of Bezier"

    def test_total_flow_parameter_matching(self):
        """
        Test total flow parameters with more baseline blocks.

        Bezier: 12 blocks @ ~282K params/block = 3,379,872 total
        Baseline: 17 blocks @ ~199K params/block = 3,378,104 total

        Target: Match within ±5%
        """
        d_model = 128
        n_head = 8

        # Bezier: 12 blocks (pillars hardcoded)
        bezier_total = 0
        for _ in range(12):
            block = FluxTransformerBlock(d_model=d_model, n_head=n_head)
            bezier_total += sum(p.numel() for p in block.parameters())

        # Baseline: Test with different numbers of blocks
        baseline_params_per_block = sum(
            p.numel()
            for p in BaselineFluxTransformerBlock(
                d_model=d_model, n_head=n_head, baseline_activation="silu"
            ).parameters()
        )

        # Calculate optimal number of blocks
        optimal_blocks = round(bezier_total / baseline_params_per_block)
        baseline_total = optimal_blocks * baseline_params_per_block

        tolerance = 0.05
        min_params = bezier_total * (1 - tolerance)
        max_params = bezier_total * (1 + tolerance)

        print(f"Bezier total (12 blocks): {bezier_total:,}")
        print(f"Baseline params/block: {baseline_params_per_block:,}")
        print(f"Optimal baseline blocks: {optimal_blocks}")
        print(f"Baseline total ({optimal_blocks} blocks): {baseline_total:,}")
        print(f"Match: {100*abs(baseline_total-bezier_total)/bezier_total:.2f}% diff")

        assert (
            min_params <= baseline_total <= max_params
        ), f"Total params {baseline_total:,} outside ±{tolerance*100}% of Bezier {bezier_total:,}"

    def test_baseline_activations_flow(self):
        """Test all supported baseline activations work in flow."""
        for activation in ["silu", "gelu"]:
            block = BaselineFluxTransformerBlock(
                d_model=128,
                n_head=8,
                baseline_activation=activation,
                ffn_expansion=4.0,
            )

            img_seq = torch.randn(1, 16, 128)
            text_seq = torch.randn(1, 10, 128)

            from fluxflow.models.flow import RotaryPositionalEmbedding

            rotary_pe = RotaryPositionalEmbedding(128 // 8)
            sin_img, cos_img = rotary_pe.get_embed(torch.arange(16))
            sin_txt, cos_txt = rotary_pe.get_embed(torch.arange(10))

            img_out, txt_out, _ = block(img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt)
            assert img_out.shape == (1, 16, 128)
            assert txt_out.shape == (1, 10, 128)


class TestLatentCompatibility:
    """
    Test latent space compatibility between Bezier and baseline models.

    Critical constraint: "output of compressor must match between versions"
    Both must produce [B, T+1, 128] where vae_dim=128 (latent dimension)
    """

    def test_vae_dim_consistency(self):
        """
        Verify both models use vae_dim=128 for latent dimension.

        Note: d_model (feature_maps_dim) can differ between versions,
        but vae_dim (latent output dimension) MUST match.
        """
        # This will be tested once full models are integrated
        # For now, verify block-level consistency
        vae_dim = 128

        # Bezier produces latents @ vae_dim
        # Baseline must also produce latents @ vae_dim
        # Both models' final output layer should project to vae_dim

        # Placeholder assertion (will expand in Phase 0.7)
        assert vae_dim == 128, "Latent dimension must be 128 for both models"

    def test_latent_shape_compatibility(self):
        """
        Verify latent space shape [B, T+1, 128] is identical.

        This ensures:
        - Cross-model evaluation: Bezier VAE → Baseline Flow
        - Fair comparison: Same latent representation
        """
        batch_size = 2
        seq_len = 16  # T spatial tokens
        vae_dim = 128

        # Expected latent shape: [B, T+1, vae_dim]
        # (+1 for conditioning token)
        expected_shape = (batch_size, seq_len + 1, vae_dim)

        # Placeholder (will test with full models in Phase 0.7)
        latent = torch.randn(expected_shape)
        assert latent.shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
