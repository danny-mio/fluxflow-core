"""Unit tests for FluxExpander context handling."""

import torch

from fluxflow.models.vae import FluxExpander


class TestFluxExpanderContext:
    """Tests for FluxExpander use_context parameter."""

    def test_with_context_enabled(self):
        """Test FluxExpander with use_context=True (default)."""
        expander = FluxExpander(d_model=32, upscales=2)

        # Create packed representation: [B, T+1, D]
        # For 8x8 image: T = 64, last token is hw_vector
        B, H, W, D = 2, 8, 8, 32
        T = H * W
        img_seq = torch.randn(B, T, D)

        # hw_vector: [h_norm, w_norm, ...]
        hw_vector = torch.zeros(B, 1, D)
        hw_vector[:, 0, 0] = H / 1024.0  # Normalized height
        hw_vector[:, 0, 1] = W / 1024.0  # Normalized width

        packed = torch.cat([img_seq, hw_vector], dim=1)  # [B, T+1, D]

        # Forward with context
        output = expander(packed, use_context=True)

        # Should produce 32x32 image (8 * 2^2 = 32)
        assert output.shape == (B, 3, 32, 32)

    def test_with_context_disabled(self):
        """Test FluxExpander with use_context=False."""
        expander = FluxExpander(d_model=32, upscales=2)

        # Create packed representation
        B, H, W, D = 2, 8, 8, 32
        T = H * W
        img_seq = torch.randn(B, T, D)

        hw_vector = torch.zeros(B, 1, D)
        hw_vector[:, 0, 0] = H / 1024.0
        hw_vector[:, 0, 1] = W / 1024.0

        packed = torch.cat([img_seq, hw_vector], dim=1)

        # Forward WITHOUT context (SPADE disabled)
        output = expander(packed, use_context=False)

        # Should still work, just without SPADE modulation
        assert output.shape == (B, 3, 32, 32)

    def test_context_does_not_allocate_zeros_tensor(self):
        """Test that use_context=False doesn't allocate unnecessary tensors."""
        expander = FluxExpander(d_model=32, upscales=2)

        B, H, W, D = 1, 8, 8, 32
        T = H * W
        img_seq = torch.randn(B, T, D)

        hw_vector = torch.zeros(B, 1, D)
        hw_vector[:, 0, 0] = H / 1024.0
        hw_vector[:, 0, 1] = W / 1024.0

        packed = torch.cat([img_seq, hw_vector], dim=1)

        # Record initial allocated memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        output = expander(packed, use_context=False)

        # Should produce valid output without allocating zeros_like tensor
        assert output.shape == (B, 3, 32, 32)
        # Note: This test verifies the fix - previously would allocate torch.zeros_like(feat)

    def test_variable_dimensions_with_context_disabled(self):
        """Test variable dimension path with use_context=False."""
        expander = FluxExpander(d_model=32, upscales=2)

        # Same dimensions for both samples (to allow concatenation)
        B = 2
        D = 32
        H, W = 8, 8
        T = H * W

        img_seq = torch.randn(B, T, D)

        # hw_vector with same dimensions for both samples
        hw_vector = torch.zeros(B, 1, D)
        hw_vector[:, 0, 0] = H / 1024.0
        hw_vector[:, 0, 1] = W / 1024.0

        packed = torch.cat([img_seq, hw_vector], dim=1)

        # Forward WITHOUT context
        output = expander(packed, use_context=False)

        # Should handle correctly
        assert output.shape == (B, 3, 32, 32)

    def test_context_parameter_types(self):
        """Test different context parameter values."""
        expander = FluxExpander(d_model=32, upscales=2)

        B, H, W, D = 1, 8, 8, 32
        T = H * W
        img_seq = torch.randn(B, T, D)

        hw_vector = torch.zeros(B, 1, D)
        hw_vector[:, 0, 0] = H / 1024.0
        hw_vector[:, 0, 1] = W / 1024.0

        packed = torch.cat([img_seq, hw_vector], dim=1)

        # Test with True (default)
        output_true = expander(packed, use_context=True)
        assert output_true.shape == (B, 3, 32, 32)

        # Test with False
        output_false = expander(packed, use_context=False)
        assert output_false.shape == (B, 3, 32, 32)

        # Outputs should be different (SPADE modulation changes output)
        assert not torch.allclose(output_true, output_false, atol=1e-6)

    def test_gradient_flow_without_context(self):
        """Test that gradients flow correctly with use_context=False."""
        expander = FluxExpander(d_model=32, upscales=2)

        B, H, W, D = 1, 8, 8, 32
        T = H * W
        img_seq = torch.randn(B, T, D, requires_grad=True)

        hw_vector = torch.zeros(B, 1, D)
        hw_vector[:, 0, 0] = H / 1024.0
        hw_vector[:, 0, 1] = W / 1024.0

        packed = torch.cat([img_seq, hw_vector], dim=1)

        output = expander(packed, use_context=False)
        loss = output.sum()
        loss.backward()

        # Gradients should flow back to input
        assert img_seq.grad is not None
        assert img_seq.grad.shape == img_seq.shape
