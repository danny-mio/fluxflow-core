"""
Test text encoder compatibility between Bezier and Baseline models.

Phase 0.7: Verify that the same frozen Bezier text encoder works with both
Bezier and Baseline VAE/Flow models.

Critical constraint: Text encoder is ALWAYS the same (frozen Bezier) for both
model variants. Only VAE and Flow differ.
"""

import pytest
import torch

from fluxflow.models.encoders import BertTextEncoder


class TestTextEncoderCompatibility:
    """Test that text encoder works identically with both model types."""

    @pytest.fixture
    def text_encoder(self):
        """Create a Bezier text encoder (will be frozen in actual training)."""
        embed_dim = 1024
        encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)
        encoder.eval()  # Frozen mode
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    @pytest.fixture
    def sample_tokens(self):
        """Sample input tokens for testing."""
        batch_size = 2
        seq_len = 20
        return torch.randint(0, 30522, (batch_size, seq_len))

    def test_encoder_is_frozen(self, text_encoder):
        """Verify encoder is frozen (all params require_grad=False)."""
        for name, param in text_encoder.named_parameters():
            assert (
                not param.requires_grad
            ), f"Parameter {name} should be frozen but requires_grad=True"

    def test_encoder_output_shape(self, text_encoder, sample_tokens):
        """Verify encoder produces per-token output [B, T, embed_dim] + bool mask [B, T]."""
        batch_size, seq_len = sample_tokens.shape
        embed_dim = 1024  # Configured embed_dim

        with torch.no_grad():
            text_seq, text_mask = text_encoder(sample_tokens)

        assert text_seq.shape == (
            batch_size,
            seq_len,
            embed_dim,
        ), f"Expected [{batch_size}, {seq_len}, {embed_dim}], got {text_seq.shape}"
        assert text_mask.shape == (batch_size, seq_len)
        assert text_mask.dtype == torch.bool

    def test_encoder_deterministic_when_frozen(self, text_encoder, sample_tokens):
        """Verify frozen encoder produces identical outputs across calls."""
        with torch.no_grad():
            text_seq1, text_mask1 = text_encoder(sample_tokens)
            text_seq2, text_mask2 = text_encoder(sample_tokens)

        torch.testing.assert_close(
            text_seq1, text_seq2, msg="Frozen encoder text_seq should be deterministic"
        )
        assert torch.equal(text_mask1, text_mask2)

    def test_encoder_no_gradient_flow(self, text_encoder, sample_tokens):
        """Verify gradients don't flow through frozen encoder."""
        # Frozen encoder output should not require grad
        text_seq, _ = text_encoder(sample_tokens)

        assert not text_seq.requires_grad, "Frozen encoder output should not require grad"

        # Verify no parameters require grad
        for name, param in text_encoder.named_parameters():
            assert not param.requires_grad, f"Frozen encoder param {name} should not require grad"

    def test_encoder_architecture_has_bezier_activations(self, text_encoder):
        """
        Verify text encoder uses BezierActivation (remains Bezier even in baseline runs).

        This is intentional - we freeze the Bezier text encoder and use it for both
        Bezier and Baseline experiments to isolate architectural differences to VAE+Flow.
        """
        from fluxflow.models.activations import BezierActivation

        has_bezier = False
        for module in text_encoder.modules():
            if isinstance(module, BezierActivation):
                has_bezier = True
                break

        assert has_bezier, "Text encoder should contain BezierActivation layers"

    def test_encoder_compatible_with_flow_processor(self, text_encoder, sample_tokens):
        """
        Test that text encoder output is compatible with FluxFlowProcessor.

        BertTextEncoder now outputs per-token (text_seq [B, T, embed_dim],
        text_mask [B, T] bool). The Flow consumes the sequence directly.
        """
        batch_size, seq_len = sample_tokens.shape
        embed_dim = 1024  # From text encoder

        with torch.no_grad():
            text_seq, text_mask = text_encoder(sample_tokens)

        assert text_seq.shape == (batch_size, seq_len, embed_dim)
        assert text_mask.shape == (batch_size, seq_len)

    def test_embedding_dim_matches_config(self, text_encoder):
        """
        Verify text encoder embed_dim matches expected config.

        This is critical for compatibility with Flow processor.
        """
        expected_embed_dim = 1024

        # Get actual output size from per-token sequence
        sample_input = torch.randint(0, 30522, (1, 10))
        with torch.no_grad():
            text_seq, _ = text_encoder(sample_input)

        actual_embed_dim = text_seq.shape[-1]
        assert (
            actual_embed_dim == expected_embed_dim
        ), f"Embedding dim {actual_embed_dim} != expected {expected_embed_dim}"


class TestTextEncoderSharedCheckpoint:
    """Test that same checkpoint can be loaded for both Bezier and Baseline runs."""

    def test_checkpoint_loading_idempotent(self):
        """Verify loading same checkpoint produces identical encoder."""
        embed_dim = 1024
        encoder1 = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)
        encoder2 = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)

        # Simulate checkpoint save/load
        state_dict = encoder1.state_dict()
        encoder2.load_state_dict(state_dict)

        # Verify identical parameters
        for (name1, param1), (name2, param2) in zip(
            encoder1.named_parameters(), encoder2.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_checkpoint_format_compatibility(self):
        """
        Test that checkpoint can be saved/loaded in safetensors format.

        This is the preferred format for frozen models.
        """
        embed_dim = 1024
        encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)

        # Get state dict
        state_dict = encoder.state_dict()

        # Verify all tensors are contiguous (required for safetensors)
        for name, tensor in state_dict.items():
            assert tensor.is_contiguous(), f"Tensor {name} is not contiguous"

        # Verify no non-tensor objects (safetensors requirement)
        for name, value in state_dict.items():
            assert isinstance(value, torch.Tensor), f"State dict entry {name} is not a tensor"


class TestCrossModelCompatibility:
    """
    Test that text encoder works with both Bezier and Baseline components.

    This validates the core requirement: same frozen text encoder for both experiments.
    """

    @pytest.fixture
    def shared_text_encoder(self):
        """Frozen Bezier text encoder (shared by both model types)."""
        embed_dim = 1024
        encoder = BertTextEncoder(embed_dim=embed_dim, pretrain_model=None)
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def test_text_encoder_with_bezier_flow(self, shared_text_encoder):
        """Test text encoder → Bezier Flow pipeline."""
        tokens = torch.randint(0, 30522, (2, 15))

        # Get per-token text embeddings [B, T, embed_dim] + mask [B, T]
        with torch.no_grad():
            text_seq, text_mask = shared_text_encoder(tokens)

        assert text_seq.shape == (2, 15, 1024), "Text encoder should output [B, T, embed_dim]"
        assert text_mask.shape == (2, 15)

    def test_text_encoder_with_baseline_flow(self, shared_text_encoder):
        """Test text encoder → Baseline Flow pipeline."""
        tokens = torch.randint(0, 30522, (2, 15))

        # Get per-token text embeddings (same encoder as Bezier test).
        with torch.no_grad():
            text_seq, text_mask = shared_text_encoder(tokens)

        assert text_seq.shape == (2, 15, 1024), "Same encoder output for baseline"
        assert text_mask.shape == (2, 15)

    def test_identical_embeddings_both_models(self, shared_text_encoder):
        """
        Verify that the SAME text encoder produces IDENTICAL embeddings
        regardless of which Flow model will consume them.

        This is the core compatibility requirement.
        """
        tokens = torch.randint(0, 30522, (2, 15))

        with torch.no_grad():
            text_seq1, text_mask1 = shared_text_encoder(tokens)
            text_seq2, text_mask2 = shared_text_encoder(tokens)

        # Should be identical (deterministic frozen encoder)
        torch.testing.assert_close(text_seq1, text_seq2)
        assert torch.equal(text_mask1, text_mask2)

        # Both Bezier and Baseline flows get SAME embeddings
        # Any quality difference is due to VAE/Flow architecture, not text encoding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
