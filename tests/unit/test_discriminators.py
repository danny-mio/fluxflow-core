"""Unit tests for discriminator models (src/models/discriminators.py)."""

import torch
import torch.nn as nn

from fluxflow.models.discriminators import DBlock, PatchDiscriminator, snconv, snlinear


class TestSpectralNormHelpers:
    """Tests for spectral normalization helper functions."""

    def test_snconv_creates_conv2d(self):
        """snconv should create Conv2d with spectral normalization."""
        conv = snconv(in_ch=64, out_ch=128, k=3)

        # snconv returns a Sequential with Conv2d layers inside
        assert isinstance(conv, nn.Sequential)
        # Check first Conv2d in the Sequential
        first_conv = conv[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 64
        assert first_conv.out_channels == 128
        assert first_conv.kernel_size == (3, 3)

    def test_snconv_applies_spectral_norm(self):
        """snconv should apply spectral normalization to weights."""
        conv = snconv(in_ch=32, out_ch=64, k=3)

        # Check for spectral norm parametrization in Conv2d layers
        first_conv = conv[0]
        assert hasattr(first_conv, "weight_u")
        assert hasattr(first_conv, "weight_v")

    def test_snconv_output_shape(self):
        """snconv should produce correct output shape."""
        conv = snconv(in_ch=3, out_ch=64, k=4, s=2, p=1)
        x = torch.randn(2, 3, 64, 64)

        output = conv(x)

        # snconv has TWO convs with stride=2, so dimensions halve TWICE: 64 -> 32 -> 16
        assert output.shape == (2, 64, 16, 16)

    def test_snconv_gradient_flow(self):
        """Gradients should flow through snconv."""
        conv = snconv(in_ch=16, out_ch=32, k=3, p=1)
        x = torch.randn(1, 16, 8, 8, requires_grad=True)

        output = conv(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_snlinear_creates_linear(self):
        """snlinear should create Linear with spectral normalization."""
        linear = snlinear(in_f=256, out_f=1)

        # snlinear returns a Sequential with Linear layers inside
        assert isinstance(linear, nn.Sequential)
        # Check first Linear in the Sequential
        first_linear = linear[0]
        assert isinstance(first_linear, nn.Linear)
        assert first_linear.in_features == 256
        assert first_linear.out_features == 1

    def test_snlinear_applies_spectral_norm(self):
        """snlinear should apply spectral normalization to weights."""
        linear = snlinear(in_f=128, out_f=64)

        # Check for spectral norm parametrization in Linear layers
        first_linear = linear[0]
        assert hasattr(first_linear, "weight_u")
        assert hasattr(first_linear, "weight_v")

    def test_snlinear_output_shape(self):
        """snlinear should produce correct output shape."""
        linear = snlinear(in_f=512, out_f=1)
        x = torch.randn(4, 512)

        output = linear(x)

        assert output.shape == (4, 1)

    def test_snlinear_gradient_flow(self):
        """Gradients should flow through snlinear."""
        linear = snlinear(in_f=64, out_f=32)
        x = torch.randn(2, 64, requires_grad=True)

        output = linear(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestDBlock:
    """Tests for DBlock downsampling block."""

    def test_initialization_default(self):
        """Should initialize with default parameters."""
        block = DBlock(in_ch=64, out_ch=128)

        assert isinstance(block, nn.Module)

    def test_initialization_with_downsample(self):
        """Should initialize with downsampling enabled."""
        block = DBlock(in_ch=64, out_ch=128, down=True)

        # Should have pooling layer
        assert hasattr(block, "pool")

    def test_initialization_without_downsample(self):
        """Should initialize without downsampling."""
        block = DBlock(in_ch=64, out_ch=128, down=False)

        # down is False
        assert block.down is False

    def test_output_shape_no_downsample(self):
        """Without downsampling, spatial dimensions should be preserved."""
        block = DBlock(in_ch=64, out_ch=128, down=False)
        x = torch.randn(2, 64, 32, 32)

        output = block(x)

        # Channels should change, spatial dims preserved
        assert output.shape == (2, 128, 32, 32)

    def test_output_shape_with_downsample(self):
        """With downsampling, spatial dimensions should be halved."""
        block = DBlock(in_ch=64, out_ch=128, down=True)
        x = torch.randn(2, 64, 32, 32)

        output = block(x)

        # Channels change, spatial dims halved
        assert output.shape == (2, 128, 16, 16)

    def test_different_input_sizes(self):
        """Should handle different spatial dimensions."""
        block = DBlock(in_ch=32, out_ch=64, down=True)

        for size in [16, 32, 64, 128]:
            x = torch.randn(1, 32, size, size)
            output = block(x)
            assert output.shape == (1, 64, size // 2, size // 2)

    def test_batch_size_independence(self):
        """Should work with different batch sizes."""
        block = DBlock(in_ch=64, out_ch=128)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 64, 16, 16)
            output = block(x)
            assert output.shape[0] == batch_size

    def test_gradient_flow(self):
        """Gradients should flow through DBlock."""
        block = DBlock(in_ch=64, out_ch=128, down=True)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)

        output = block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_spectral_normalization_applied(self):
        """DBlock should use spectral normalization on convolutions."""
        block = DBlock(in_ch=32, out_ch=64)

        # Check for spectral norm in at least one conv layer
        has_spectral_norm = False
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight_u"):
                    has_spectral_norm = True
                    break

        assert has_spectral_norm, "DBlock should use spectral normalization"

    def test_rectangular_input(self):
        """Should handle non-square inputs."""
        block = DBlock(in_ch=64, out_ch=128, down=True)
        x = torch.randn(2, 64, 32, 48)

        output = block(x)

        # Both dimensions should be halved
        assert output.shape == (2, 128, 16, 24)


class TestPatchDiscriminator:
    """Tests for PatchDiscriminator model."""

    def test_initialization_default(self):
        """Should initialize with default depth."""
        disc = PatchDiscriminator(in_channels=3)

        assert isinstance(disc, nn.Module)

    def test_initialization_custom_depth(self):
        """Should initialize with custom depth."""
        for depth in [3, 4, 5, 6]:
            disc = PatchDiscriminator(in_channels=3, depth=depth)
            assert isinstance(disc, nn.Module)

    def test_initialization_with_conditioning(self):
        """Should initialize with projection conditioning."""
        disc = PatchDiscriminator(in_channels=3, ctx_dim=512)

        # Should have projection layer for conditioning
        assert hasattr(disc, "ctx_proj")

    def test_output_is_patch_logits(self):
        """Output should be patch logits (not single scalar)."""
        disc = PatchDiscriminator(in_channels=3, depth=4)
        x = torch.randn(2, 3, 256, 256)

        output = disc(x)

        # Should be 4D: (batch, 1, H, W) where H, W are patch dimensions
        assert output.dim() == 4
        assert output.shape[0] == 2
        assert output.shape[1] == 1  # Single channel logits

    def test_output_shape_depth_3(self):
        """Depth 3 discriminator output shape."""
        disc = PatchDiscriminator(in_channels=3, depth=3)
        x = torch.randn(4, 3, 128, 128)

        output = disc(x)

        # 3 downsamples: 128 -> 64 -> 32 -> 16
        assert output.shape[0] == 4
        assert output.shape[1] == 1
        assert output.shape[2:] == (16, 16) or output.shape[2:] == (8, 8)

    def test_output_shape_depth_5(self):
        """Depth 5 discriminator output shape."""
        disc = PatchDiscriminator(in_channels=3, depth=5)
        x = torch.randn(2, 3, 256, 256)

        output = disc(x)

        # 5 downsamples: 256 -> ... -> smaller patches
        assert output.shape[0] == 2
        assert output.shape[1] == 1
        # Verify spatial dimensions are reduced
        assert output.shape[2] < 256
        assert output.shape[3] < 256

    def test_different_input_sizes(self):
        """Should handle different input resolutions."""
        disc = PatchDiscriminator(in_channels=3, depth=4)

        for size in [64, 128, 256, 512]:
            x = torch.randn(1, 3, size, size)
            output = disc(x)

            # Should produce valid output
            assert output.dim() == 4
            assert output.shape[0] == 1
            assert output.shape[1] == 1

    def test_batch_size_independence(self):
        """Should work with different batch sizes."""
        disc = PatchDiscriminator(in_channels=3, depth=4)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 128, 128)
            output = disc(x)
            assert output.shape[0] == batch_size

    def test_gradient_flow(self):
        """Gradients should flow through discriminator."""
        disc = PatchDiscriminator(in_channels=3, depth=4)
        x = torch.randn(2, 3, 128, 128, requires_grad=True)

        output = disc(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_with_conditioning_vector(self):
        """Should handle conditioning vector for projection."""
        disc = PatchDiscriminator(in_channels=3, depth=4, ctx_dim=512)
        x = torch.randn(2, 3, 256, 256)
        ctx_vec = torch.randn(2, 512)

        # Should accept context vector
        output = disc(x, ctx_vec=ctx_vec)

        assert output.shape[0] == 2
        assert output.shape[1] == 1

    def test_without_conditioning_no_ctx_vec(self):
        """Should work without ctx_vec when not using conditioning."""
        disc = PatchDiscriminator(in_channels=3, depth=4, ctx_dim=0)
        x = torch.randn(2, 3, 128, 128)

        # Should work without ctx_vec argument
        output = disc(x)

        assert output.shape[0] == 2

    def test_return_feats_parameter(self):
        """Should return features when return_feats=True."""
        disc = PatchDiscriminator(in_channels=3, depth=4)
        x = torch.randn(2, 3, 128, 128)

        feats, patch_logits = disc(x, return_feats=True)

        # Features should be pooled [B, C]
        assert feats.dim() == 2
        assert feats.shape[0] == 2
        # Patch logits should be [B, 1, H, W]
        assert patch_logits.dim() == 4
        assert patch_logits.shape[0] == 2
        assert patch_logits.shape[1] == 1

    def test_rectangular_input(self):
        """Should handle non-square inputs."""
        disc = PatchDiscriminator(in_channels=3, depth=4)
        x = torch.randn(2, 3, 128, 256)

        output = disc(x)

        assert output.dim() == 4
        assert output.shape[0] == 2
        assert output.shape[1] == 1

    def test_single_channel_input(self):
        """Should work with single-channel images."""
        disc = PatchDiscriminator(in_channels=1, depth=3)
        x = torch.randn(2, 1, 128, 128)

        output = disc(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 1

    def test_multi_channel_input(self):
        """Should work with multi-channel inputs."""
        disc = PatchDiscriminator(in_channels=4, depth=3)  # e.g., RGBA
        x = torch.randn(2, 4, 128, 128)

        output = disc(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 1

    def test_spectral_normalization_applied(self):
        """PatchDiscriminator should use spectral normalization."""
        disc = PatchDiscriminator(in_channels=3, depth=4)

        # Check for spectral norm in conv layers
        has_spectral_norm = False
        for module in disc.modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "weight_u"):
                    has_spectral_norm = True
                    break

        assert has_spectral_norm, "PatchDiscriminator should use spectral normalization"


class TestDiscriminatorIntegration:
    """Integration tests for discriminator usage."""

    def test_gan_training_step(self):
        """Test typical GAN discriminator training step."""
        disc = PatchDiscriminator(in_channels=3, depth=4)

        # Real images
        real_imgs = torch.randn(4, 3, 128, 128)
        real_logits = disc(real_imgs)

        # Fake images
        fake_imgs = torch.randn(4, 3, 128, 128)
        fake_logits = disc(fake_imgs)

        # Both should produce valid logits
        assert real_logits.shape == fake_logits.shape
        assert real_logits.dim() == 4

        # Can compute hinge loss (simplified)
        d_loss_real = torch.relu(1.0 - real_logits).mean()
        d_loss_fake = torch.relu(1.0 + fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake

        # Should be able to backprop
        d_loss.backward()

    def test_progressive_downsampling(self):
        """Test that deeper discriminators downsample more."""
        disc_shallow = PatchDiscriminator(in_channels=3, depth=3)
        disc_deep = PatchDiscriminator(in_channels=3, depth=5)

        x = torch.randn(1, 3, 256, 256)

        out_shallow = disc_shallow(x)
        out_deep = disc_deep(x)

        # Deeper discriminator should have smaller spatial dimensions
        assert out_deep.shape[2] <= out_shallow.shape[2]
        assert out_deep.shape[3] <= out_shallow.shape[3]

    def test_discriminator_with_conditional_gan(self):
        """Test conditional GAN setup with projection."""
        disc = PatchDiscriminator(in_channels=3, depth=4, ctx_dim=256)

        # Images and text embeddings
        imgs = torch.randn(4, 3, 128, 128)
        text_emb = torch.randn(4, 256)

        # Discriminate with conditioning
        logits = disc(imgs, ctx_vec=text_emb)

        assert logits.shape[0] == 4
        assert logits.shape[1] == 1

        # Should be able to compute loss
        loss = logits.mean()
        loss.backward()

    def test_multi_scale_discrimination(self):
        """Test discriminating at multiple resolutions."""
        disc = PatchDiscriminator(in_channels=3, depth=4)

        # Different resolutions
        sizes = [64, 128, 256]
        outputs = []

        for size in sizes:
            x = torch.randn(2, 3, size, size)
            output = disc(x)
            outputs.append(output)

        # All should produce valid outputs
        for output in outputs:
            assert output.dim() == 4
            assert output.shape[0] == 2
            assert output.shape[1] == 1

    def test_dblock_stacking(self):
        """Test stacking multiple DBlocks."""
        blocks = nn.Sequential(
            DBlock(in_ch=3, out_ch=64, down=True),
            DBlock(in_ch=64, out_ch=128, down=True),
            DBlock(in_ch=128, out_ch=256, down=True),
        )

        x = torch.randn(2, 3, 128, 128)
        output = blocks(x)

        # 3 downsamples: 128 -> 64 -> 32 -> 16
        assert output.shape == (2, 256, 16, 16)

    def test_discriminator_eval_mode(self):
        """Test discriminator in eval mode (no dropout, stable BN)."""
        disc = PatchDiscriminator(in_channels=3, depth=4)
        disc.eval()

        x = torch.randn(2, 3, 128, 128)

        # Multiple forward passes should give same result
        with torch.no_grad():
            out1 = disc(x)
            out2 = disc(x)

        assert torch.allclose(out1, out2)

    def test_device_compatibility(self):
        """Test discriminator works on CPU."""
        disc = PatchDiscriminator(in_channels=3, depth=4)
        disc = disc.to("cpu")

        x = torch.randn(2, 3, 128, 128, device="cpu")
        output = disc(x)

        assert output.device.type == "cpu"
        assert output.shape[0] == 2

    def test_hinge_loss_validation(self):
        """Validate hinge GAN loss implementation with discriminator."""
        disc = PatchDiscriminator(in_channels=3, depth=4)

        # Real images (should have positive logits)
        real_imgs = torch.randn(4, 3, 128, 128)
        real_logits = disc(real_imgs)

        # Fake images (should have negative logits after training)
        fake_imgs = torch.randn(4, 3, 128, 128)
        fake_logits = disc(fake_imgs)

        # Hinge loss for discriminator:
        # L_D = E[max(0, 1 - D(real))] + E[max(0, 1 + D(fake))]
        d_loss_real = torch.relu(1.0 - real_logits).mean()
        d_loss_fake = torch.relu(1.0 + fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake

        # Hinge loss for generator:
        # L_G = -E[D(G(z))]
        g_loss = -fake_logits.mean()

        # Validate loss properties
        assert d_loss.item() >= 0, "Discriminator loss must be non-negative"
        assert not torch.isnan(d_loss), "Discriminator loss must not be NaN"
        assert not torch.isnan(g_loss), "Generator loss must not be NaN"

        # Test gradient flow
        d_loss.backward(retain_graph=True)
        g_loss.backward()

        # Verify gradients exist
        for param in disc.parameters():
            if param.requires_grad:
                assert param.grad is not None, "All discriminator params must have gradients"
                assert not torch.isnan(param.grad).any(), "Gradients must not contain NaN"

    def test_hinge_loss_with_conditioning(self):
        """Validate hinge loss with conditional discriminator."""
        disc = PatchDiscriminator(in_channels=3, depth=4, ctx_dim=256)

        # Images and conditioning vectors
        real_imgs = torch.randn(4, 3, 128, 128)
        fake_imgs = torch.randn(4, 3, 128, 128)
        ctx_vec = torch.randn(4, 256)

        # Get logits with conditioning
        real_logits = disc(real_imgs, ctx_vec=ctx_vec)
        fake_logits = disc(fake_imgs, ctx_vec=ctx_vec)

        # Compute hinge loss
        d_loss_real = torch.relu(1.0 - real_logits).mean()
        d_loss_fake = torch.relu(1.0 + fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake
        g_loss = -fake_logits.mean()

        # Validate
        assert d_loss.item() >= 0
        assert not torch.isnan(d_loss)
        assert not torch.isnan(g_loss)

        # Test backprop
        d_loss.backward(retain_graph=True)
        g_loss.backward()

    def test_hinge_loss_convergence_behavior(self):
        """Test that hinge loss behaves correctly during optimization."""
        disc = PatchDiscriminator(in_channels=3, depth=3)
        optimizer = torch.optim.Adam(disc.parameters(), lr=0.0001)

        # Create clearly separable real/fake data
        real_imgs = torch.ones(4, 3, 64, 64) * 0.5  # Centered at 0.5
        fake_imgs = torch.ones(4, 3, 64, 64) * -0.5  # Centered at -0.5

        initial_loss = 0.0
        final_loss = 0.0
        for step in range(5):
            optimizer.zero_grad()

            real_logits = disc(real_imgs)
            fake_logits = disc(fake_imgs)

            # Discriminator should push real_logits > 1 and fake_logits < -1
            d_loss_real = torch.relu(1.0 - real_logits).mean()
            d_loss_fake = torch.relu(1.0 + fake_logits).mean()
            d_loss = d_loss_real + d_loss_fake

            if step == 0:
                initial_loss = d_loss.item()

            d_loss.backward()
            optimizer.step()
            final_loss = d_loss.item()

        # After optimization, loss should decrease (or stay low if already converged)
        assert final_loss <= initial_loss + 0.1, "Hinge loss should decrease or stabilize"
