"""Unit tests for VAE model shapes (src/models/vae.py)."""

import torch

from fluxflow.models.vae import (
    Clamp,
    ProgressiveUpscaler,
    ResidualUpsampleBlock,
)


class TestClamp:
    """Tests for Clamp layer."""

    def test_initialization(self):
        """Test Clamp layer initializes with correct min/max."""
        clamp = Clamp(min=-1, max=1)
        assert clamp.min == -1
        assert clamp.max == 1

    def test_clamps_values(self):
        """Should clamp values to specified range."""
        clamp = Clamp(min=-1, max=1)
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        output = clamp(x)

        assert output[0] == -1.0  # Clamped from -2.0
        assert output[1] == -0.5  # Unchanged
        assert output[2] == 0.0  # Unchanged
        assert output[3] == 0.5  # Unchanged
        assert output[4] == 1.0  # Clamped from 2.0

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        clamp = Clamp()
        x = torch.randn(4, 3, 64, 64)
        output = clamp(x)
        assert output.shape == x.shape

    def test_different_ranges(self):
        """Should work with different min/max ranges."""
        clamp = Clamp(min=0, max=255)
        x = torch.tensor([-10.0, 100.0, 300.0])
        output = clamp(x)

        assert output[0] == 0.0
        assert output[1] == 100.0
        assert output[2] == 255.0

    def test_gradient_flow(self):
        """Gradients should flow through clamp (except at boundaries)."""
        clamp = Clamp(min=-1, max=1)
        x = torch.tensor([0.5], requires_grad=True)
        output = clamp(x)
        output.backward()

        # Gradient should flow for non-clamped values
        assert x.grad is not None


class TestResidualUpsampleBlock:
    """Tests for ResidualUpsampleBlock."""

    def test_initialization_no_conditioning(self):
        """Test initialization without conditioning."""
        block = ResidualUpsampleBlock(channels=64)

        # Should have SPADE by default
        assert block.use_spade
        assert hasattr(block, "spade")

    def test_initialization_with_spade(self):
        """Test initialization with SPADE conditioning."""
        block = ResidualUpsampleBlock(channels=64, context_size=256, use_spade=True)

        assert block.use_spade
        assert hasattr(block, "spade")

    def test_initialization_without_spade(self):
        """Test initialization without SPADE."""
        block = ResidualUpsampleBlock(channels=64, use_spade=False)

        assert not block.use_spade

    def test_output_shape_doubles_resolution(self):
        """Output should have 2x resolution."""
        block = ResidualUpsampleBlock(channels=64, use_spade=False)

        x = torch.randn(2, 64, 16, 16)
        output = block(x)

        # Should double spatial dimensions
        assert output.shape == (2, 64, 32, 32)

    def test_output_shape_without_spade(self):
        """Should work without SPADE conditioning."""
        block = ResidualUpsampleBlock(channels=64, use_spade=False)

        x = torch.randn(2, 64, 16, 16)
        output = block(x)

        assert output.shape == (2, 64, 32, 32)

    def test_output_shape_with_spade(self):
        """Should work with SPADE conditioning."""
        block = ResidualUpsampleBlock(channels=64, context_size=128, use_spade=True)

        x = torch.randn(2, 64, 16, 16)
        context = torch.randn(2, 128, 16, 16)
        output = block(x, context=context)

        assert output.shape == (2, 64, 32, 32)

    def test_output_shape_with_context(self):
        """Should work with context (SPADE conditioning)."""
        block = ResidualUpsampleBlock(channels=64, context_size=128, use_spade=True)

        x = torch.randn(2, 64, 8, 8)
        context = torch.randn(2, 128, 8, 8)

        output = block(x, context=context)

        assert output.shape == (2, 64, 16, 16)

    def test_different_channel_counts(self):
        """Should work with different channel counts."""
        for channels in [32, 64, 128, 256]:
            block = ResidualUpsampleBlock(channels=channels, use_spade=False)
            x = torch.randn(1, channels, 8, 8)
            output = block(x)

            assert output.shape == (1, channels, 16, 16)

    def test_different_spatial_sizes(self):
        """Should work with different input sizes."""
        block = ResidualUpsampleBlock(channels=64, use_spade=False)

        for size in [4, 8, 16, 32]:
            x = torch.randn(1, 64, size, size)
            output = block(x)

            assert output.shape == (1, 64, size * 2, size * 2)

    def test_context_upsample(self):
        """SPADE should upsample context if needed."""
        block = ResidualUpsampleBlock(channels=64, context_size=128, use_spade=True)

        x = torch.randn(2, 64, 16, 16)
        context = torch.randn(2, 128, 8, 8)  # Half resolution

        # Should not crash (SPADE upsamples context internally)
        output = block(x, context=context)
        assert output.shape == (2, 64, 32, 32)


class TestProgressiveUpscaler:
    """Tests for ProgressiveUpscaler."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        upscaler = ProgressiveUpscaler(channels=64, steps=2)

        # Should have 2 layers
        assert len(upscaler.layers) == 2

    def test_output_shape_2_steps(self):
        """2 steps should give 4x upsampling (2^2)."""
        upscaler = ProgressiveUpscaler(channels=64, steps=2, use_spade=False)

        x = torch.randn(2, 64, 8, 8)
        context = None  # Not used when use_spade=False

        output = upscaler(x, context)

        # 2 steps: 8 -> 16 -> 32
        assert output.shape == (2, 64, 32, 32)

    def test_output_shape_3_steps(self):
        """3 steps should give 8x upsampling (2^3)."""
        upscaler = ProgressiveUpscaler(channels=64, steps=3, use_spade=False)

        x = torch.randn(2, 64, 4, 4)
        output = upscaler(x, context=None)

        # 3 steps: 4 -> 8 -> 16 -> 32
        assert output.shape == (2, 64, 32, 32)

    def test_output_shape_4_steps(self):
        """4 steps should give 16x upsampling (2^4)."""
        upscaler = ProgressiveUpscaler(channels=64, steps=4, use_spade=False)

        x = torch.randn(2, 64, 4, 4)
        output = upscaler(x, context=None)

        # 4 steps: 4 -> 8 -> 16 -> 32 -> 64
        assert output.shape == (2, 64, 64, 64)

    def test_with_spade_conditioning(self):
        """Should work with SPADE conditioning."""
        upscaler = ProgressiveUpscaler(channels=64, steps=2, context_size=128, use_spade=True)

        x = torch.randn(2, 64, 8, 8)
        context = torch.randn(2, 128, 8, 8)

        output = upscaler(x, context)

        assert output.shape == (2, 64, 32, 32)

    def test_without_spade_conditioning(self):
        """Should work without SPADE conditioning."""
        upscaler = ProgressiveUpscaler(channels=64, steps=2, use_spade=False)

        x = torch.randn(2, 64, 8, 8)

        output = upscaler(x, context=None)

        assert output.shape == (2, 64, 32, 32)

    def test_different_channel_counts(self):
        """Should work with different channel counts."""
        for channels in [32, 64, 128]:
            upscaler = ProgressiveUpscaler(channels=channels, steps=2, use_spade=False)
            x = torch.randn(1, channels, 8, 8)
            output = upscaler(x, context=None)

            assert output.shape == (1, channels, 32, 32)

    def test_single_step(self):
        """Should work with single step."""
        upscaler = ProgressiveUpscaler(channels=64, steps=1, use_spade=False)

        x = torch.randn(2, 64, 16, 16)
        output = upscaler(x, context=None)

        # 1 step: 16 -> 32
        assert output.shape == (2, 64, 32, 32)

    def test_rectangular_input(self):
        """Should work with non-square inputs."""
        upscaler = ProgressiveUpscaler(channels=64, steps=2, use_spade=False)

        x = torch.randn(2, 64, 8, 16)
        output = upscaler(x, context=None)

        # Should double both dimensions
        assert output.shape == (2, 64, 32, 64)


class TestVAEShapesIntegration:
    """Integration tests for VAE component shapes."""

    def test_encoder_decoder_symmetry(self):
        """Upscaler steps should reverse downsampling."""
        # Simulate 4 downscales: 256 -> 128 -> 64 -> 32 -> 16
        latent_size = 16
        channels = 64

        # Upscaler with 4 steps should reverse this
        upscaler = ProgressiveUpscaler(channels=channels, steps=4, use_spade=False)

        x = torch.randn(1, channels, latent_size, latent_size)
        output = upscaler(x, context=None)

        # Should return to original size
        assert output.shape == (1, channels, 256, 256)

    def test_multi_scale_upsampling(self):
        """Test progressive upsampling at each stage."""
        upscaler = ProgressiveUpscaler(channels=64, steps=3, use_spade=False)

        # Start from small latent
        x = torch.randn(2, 64, 4, 4)

        # Manually upsample through each layer
        for i, layer in enumerate(upscaler.layers):
            x = layer(x)
            expected_size = 4 * (2 ** (i + 1))
            assert x.shape[2] == expected_size
            assert x.shape[3] == expected_size

    def test_conditioning_combination(self):
        """Test with SPADE conditioning."""
        upscaler = ProgressiveUpscaler(
            channels=64,
            steps=2,
            context_size=128,
            use_spade=True,
        )

        x = torch.randn(2, 64, 8, 8)
        context = torch.randn(2, 128, 8, 8)

        output = upscaler(x, context)

        assert output.shape == (2, 64, 32, 32)

    def test_batch_size_independence(self):
        """Output shape should scale with batch size."""
        upscaler = ProgressiveUpscaler(channels=64, steps=2, use_spade=False)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 64, 8, 8)
            output = upscaler(x, context=None)

            assert output.shape == (batch_size, 64, 32, 32)

    def test_gradient_flow_through_upscaler(self):
        """Gradients should flow through full upscaler."""
        upscaler = ProgressiveUpscaler(channels=64, steps=2, use_spade=False)

        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        output = upscaler(x, context=None)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_typical_vae_workflow(self):
        """Test typical VAE encoder-decoder workflow shapes."""
        # Typical VAE dimensions
        batch_size = 4
        img_size = 256
        latent_channels = 128
        downscales = 4

        # After encoding: 256 -> 128 -> 64 -> 32 -> 16
        latent_size = img_size // (2**downscales)

        # Simulated latent
        latent = torch.randn(batch_size, latent_channels, latent_size, latent_size)

        # Decode with upscaler
        upscaler = ProgressiveUpscaler(
            channels=latent_channels,
            steps=downscales,
            use_spade=False,
        )

        decoded = upscaler(latent, context=None)

        # Should match original image size
        assert decoded.shape == (batch_size, latent_channels, img_size, img_size)
