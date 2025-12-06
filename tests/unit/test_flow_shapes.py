"""Unit tests for flow model components (src/models/flow.py)."""

import torch
import torch.nn as nn

from fluxflow.models.flow import (
    FluxFlowProcessor,
    FluxTransformerBlock,
    ParallelAttention,
    RotaryPositionalEmbedding,
    pillarLayer,
)


class TestPillarLayer:
    """Tests for pillarLayer helper function."""

    def test_creates_sequential_module(self):
        """Should create a sequential module."""
        layer = pillarLayer(in_size=64, out_size=128, depth=2)

        assert isinstance(layer, nn.Sequential)

    def test_depth_parameter(self):
        """Should create modules with specified depth."""
        for depth in [1, 2, 3, 4]:
            layer = pillarLayer(in_size=32, out_size=64, depth=depth)
            # Each "layer" is a Sequential with Linear + activation
            assert len(layer) == depth

    def test_output_shape(self):
        """Should preserve input dimension in output."""
        layer = pillarLayer(in_size=128, out_size=256, depth=3)
        x = torch.randn(2, 128)

        output = layer(x)

        # Output should match in_size (last layer outputs in_size)
        assert output.shape == (2, 128)

    def test_batch_independence(self):
        """Should work with different batch sizes."""
        layer = pillarLayer(in_size=64, out_size=128, depth=2)

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 64)
            output = layer(x)
            assert output.shape == (batch_size, 64)

    def test_gradient_flow(self):
        """Gradients should flow through pillar layer."""
        layer = pillarLayer(in_size=32, out_size=64, depth=3)
        x = torch.randn(2, 32, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_custom_activation(self):
        """Should support custom activation functions."""
        activation = nn.ReLU()
        layer = pillarLayer(in_size=64, out_size=128, depth=2, activation=activation)

        x = torch.randn(2, 64)
        output = layer(x)

        assert output.shape == (2, 64)


class TestRotaryPositionalEmbedding:
    """Tests for RoPE (Rotary Position Embedding)."""

    def test_initialization(self):
        """Should initialize with specified dimension."""
        rope = RotaryPositionalEmbedding(dim=64)

        assert isinstance(rope, nn.Module)
        assert rope.dim == 64

    def test_inv_freq_buffer(self):
        """Should register inv_freq as buffer."""
        rope = RotaryPositionalEmbedding(dim=128)

        assert hasattr(rope, "inv_freq")
        assert rope.inv_freq.shape == (64,)  # dim // 2

    def test_get_embed_shapes(self):
        """get_embed should return sin/cos with correct shapes."""
        rope = RotaryPositionalEmbedding(dim=64)
        pos_ids = torch.arange(100)

        sin, cos = rope.get_embed(pos_ids)

        # Should be [seq_len, dim] after repeat_interleave
        assert sin.shape == (100, 64)
        assert cos.shape == (100, 64)

    def test_get_embed_values(self):
        """get_embed should produce valid sin/cos values."""
        rope = RotaryPositionalEmbedding(dim=32)
        pos_ids = torch.arange(50)

        sin, cos = rope.get_embed(pos_ids)

        # Values should be in [-1, 1]
        assert (sin >= -1.0).all() and (sin <= 1.0).all()
        assert (cos >= -1.0).all() and (cos <= 1.0).all()

    def test_apply_rotary_shape(self):
        """apply_rotary should preserve input shape."""
        rope = RotaryPositionalEmbedding(dim=64)
        x = torch.randn(2, 8, 10, 64)  # [B, H, S, D]
        pos_ids = torch.arange(10)
        sin, cos = rope.get_embed(pos_ids)

        rotated = rope.apply_rotary(x, sin, cos)

        assert rotated.shape == x.shape

    def test_apply_rotary_gradient_flow(self):
        """Gradients should flow through rotary application."""
        rope = RotaryPositionalEmbedding(dim=32)
        x = torch.randn(2, 4, 8, 32, requires_grad=True)
        pos_ids = torch.arange(8)
        sin, cos = rope.get_embed(pos_ids)

        rotated = rope.apply_rotary(x, sin, cos)
        loss = rotated.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_sequence_lengths(self):
        """Should handle different sequence lengths."""
        rope = RotaryPositionalEmbedding(dim=64)

        for seq_len in [16, 32, 64, 128]:
            pos_ids = torch.arange(seq_len)
            sin, cos = rope.get_embed(pos_ids)
            assert sin.shape == (seq_len, 64)
            assert cos.shape == (seq_len, 64)


class TestParallelAttention:
    """Tests for ParallelAttention module."""

    def test_initialization(self):
        """Should initialize with specified parameters."""
        attn = ParallelAttention(d_model=512, n_head=8)

        assert isinstance(attn, nn.Module)
        assert attn.n_head == 8
        assert attn.d_head == 64  # 512 // 8

    def test_output_shape_self_attention(self):
        """Output shape for self-attention (x_q == x_kv)."""
        attn = ParallelAttention(d_model=512, n_head=8)
        x = torch.randn(2, 16, 512)  # [B, S, D]

        def identity_rotary(x):
            return x

        output = attn(x, x, identity_rotary, identity_rotary)

        # Should preserve input shape
        assert output.shape == (2, 16, 512)

    def test_output_shape_cross_attention(self):
        """Output shape for cross-attention (x_q != x_kv)."""
        attn = ParallelAttention(d_model=256, n_head=4)
        x_q = torch.randn(2, 10, 256)  # Query: [B, S_q, D]
        x_kv = torch.randn(2, 20, 256)  # Key/Value: [B, S_kv, D]

        def identity_rotary(x):
            return x

        output = attn(x_q, x_kv, identity_rotary, identity_rotary)

        # Output should match query sequence length
        assert output.shape == (2, 10, 256)

    def test_batch_size_independence(self):
        """Should work with different batch sizes."""
        attn = ParallelAttention(d_model=128, n_head=4)

        def identity_rotary(x):
            return x

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 8, 128)
            output = attn(x, x, identity_rotary, identity_rotary)
            assert output.shape[0] == batch_size

    def test_gradient_flow(self):
        """Gradients should flow through attention."""
        attn = ParallelAttention(d_model=256, n_head=8)
        x_q = torch.randn(2, 10, 256, requires_grad=True)
        x_kv = torch.randn(2, 15, 256, requires_grad=True)

        def identity_rotary(x):
            return x

        output = attn(x_q, x_kv, identity_rotary, identity_rotary)
        loss = output.sum()
        loss.backward()

        assert x_q.grad is not None
        assert x_kv.grad is not None
        assert not torch.isnan(x_q.grad).any()
        assert not torch.isnan(x_kv.grad).any()

    def test_with_rotary_embeddings(self):
        """Should work with actual rotary embeddings."""
        attn = ParallelAttention(d_model=512, n_head=8)
        rope = RotaryPositionalEmbedding(dim=64)  # d_head = 512 // 8

        x_q = torch.randn(2, 16, 512)
        x_kv = torch.randn(2, 16, 512)

        sin_q, cos_q = rope.get_embed(torch.arange(16))
        sin_kv, cos_kv = rope.get_embed(torch.arange(16))

        output = attn(
            x_q,
            x_kv,
            lambda x: rope.apply_rotary(x, sin_q, cos_q),
            lambda x: rope.apply_rotary(x, sin_kv, cos_kv),
        )

        assert output.shape == (2, 16, 512)

    def test_attention_scaling(self):
        """Attention should use proper scaling factor."""
        attn = ParallelAttention(d_model=512, n_head=8)

        # d_head = 512 // 8 = 64, so scaling should be 1/sqrt(64) = 1/8
        assert attn.d_head == 64


class TestFluxTransformerBlock:
    """Tests for FluxTransformerBlock."""

    def test_initialization(self):
        """Should initialize with specified parameters."""
        block = FluxTransformerBlock(d_model=512, n_head=8)

        assert isinstance(block, nn.Module)
        assert hasattr(block, "self_attn")
        assert hasattr(block, "cross_attn")
        assert hasattr(block, "bezier_activation")

    def test_output_shapes(self):
        """Should return img_seq and 4 control point features."""
        block = FluxTransformerBlock(d_model=256, n_head=4)

        img_seq = torch.randn(2, 16, 256)
        text_seq = torch.randn(2, 1, 256)

        sin_img, cos_img = block.rotary_pe.get_embed(torch.arange(16))
        sin_txt, cos_txt = block.rotary_pe.get_embed(torch.arange(1))

        # First pass with None control points
        img_out, p0, p1, p2, p3 = block(
            img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, None, None, None, None
        )

        assert img_out.shape == (2, 16, 256)
        assert p0.shape == (2, 16, 256)
        assert p1.shape == (2, 16, 256)
        assert p2.shape == (2, 16, 256)
        assert p3.shape == (2, 16, 256)

    def test_with_previous_control_points(self):
        """Should accept previous control points as input."""
        block = FluxTransformerBlock(d_model=128, n_head=4)

        img_seq = torch.randn(2, 8, 128)
        text_seq = torch.randn(2, 1, 128)

        sin_img, cos_img = block.rotary_pe.get_embed(torch.arange(8))
        sin_txt, cos_txt = block.rotary_pe.get_embed(torch.arange(1))

        # Provide previous control points
        p0_prev = torch.randn(2, 8, 128)
        p1_prev = torch.randn(2, 8, 128)
        p2_prev = torch.randn(2, 8, 128)
        p3_prev = torch.randn(2, 8, 128)

        img_out, p0, p1, p2, p3 = block(
            img_seq,
            text_seq,
            sin_img,
            cos_img,
            sin_txt,
            cos_txt,
            p0_prev,
            p1_prev,
            p2_prev,
            p3_prev,
        )

        assert img_out.shape == (2, 8, 128)

    def test_gradient_flow(self):
        """Gradients should flow through transformer block."""
        block = FluxTransformerBlock(d_model=256, n_head=8)

        img_seq = torch.randn(2, 16, 256, requires_grad=True)
        text_seq = torch.randn(2, 1, 256, requires_grad=True)

        sin_img, cos_img = block.rotary_pe.get_embed(torch.arange(16))
        sin_txt, cos_txt = block.rotary_pe.get_embed(torch.arange(1))

        img_out, p0, p1, p2, p3 = block(
            img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, None, None, None, None
        )
        loss = img_out.sum()
        loss.backward()

        assert img_seq.grad is not None
        assert text_seq.grad is not None
        assert not torch.isnan(img_seq.grad).any()

    def test_batch_size_independence(self):
        """Should work with different batch sizes."""
        block = FluxTransformerBlock(d_model=128, n_head=4)

        for batch_size in [1, 2, 4]:
            img_seq = torch.randn(batch_size, 10, 128)
            text_seq = torch.randn(batch_size, 1, 128)

            sin_img, cos_img = block.rotary_pe.get_embed(torch.arange(10))
            sin_txt, cos_txt = block.rotary_pe.get_embed(torch.arange(1))

            img_out, p0, p1, p2, p3 = block(
                img_seq,
                text_seq,
                sin_img,
                cos_img,
                sin_txt,
                cos_txt,
                None,
                None,
                None,
                None,
            )
            assert img_out.shape[0] == batch_size


class TestFluxFlowProcessor:
    """Tests for FluxFlowProcessor model."""

    def test_initialization_default(self):
        """Should initialize with default parameters."""
        model = FluxFlowProcessor()

        assert isinstance(model, nn.Module)
        assert model.max_hw == 1024
        assert model.ctx_tokens == 4

    def test_initialization_custom(self):
        """Should initialize with custom parameters."""
        model = FluxFlowProcessor(
            d_model=256,
            vae_dim=64,
            embedding_size=512,
            n_head=4,
            n_layers=6,
            max_hw=512,
            ctx_tokens=8,
        )

        assert model.max_hw == 512
        assert model.ctx_tokens == 8
        assert len(model.transformer_blocks) == 6

    def test_add_coord_channels(self):
        """add_coord_channels should add 2 coordinate channels."""
        model = FluxFlowProcessor()
        x = torch.randn(2, 128, 16, 16)

        output = model.add_coord_channels(x)

        # Should add 2 channels (x, y coordinates)
        assert output.shape == (2, 130, 16, 16)

    def test_forward_output_shape(self):
        """Forward pass should preserve packed tensor shape."""
        model = FluxFlowProcessor(
            d_model=256,
            vae_dim=64,
            embedding_size=512,
            n_head=4,
            n_layers=2,  # Small for testing
        )

        batch_size = 2
        num_tokens = 16 * 16  # 16x16 image
        packed = torch.randn(batch_size, num_tokens + 1, 64)  # +1 for hw_vec
        text_embeddings = torch.randn(batch_size, 512)
        timesteps = torch.randint(0, 1000, (batch_size,))

        output = model(packed, text_embeddings, timesteps)

        # Should preserve packed shape
        assert output.shape == (batch_size, num_tokens + 1, 64)

    def test_forward_gradient_flow(self):
        """Gradients should flow through forward pass."""
        model = FluxFlowProcessor(d_model=128, vae_dim=32, n_head=4, n_layers=2, embedding_size=256)

        batch_size = 1
        num_tokens = 8 * 8
        packed = torch.randn(batch_size, num_tokens + 1, 32, requires_grad=True)
        text_embeddings = torch.randn(batch_size, 256, requires_grad=True)
        timesteps = torch.randint(0, 1000, (batch_size,))

        output = model(packed, text_embeddings, timesteps)
        loss = output.sum()
        loss.backward()

        assert packed.grad is not None
        assert text_embeddings.grad is not None
        assert not torch.isnan(packed.grad).any()

    def test_different_spatial_sizes(self):
        """Should handle different spatial dimensions."""
        model = FluxFlowProcessor(
            d_model=128, vae_dim=32, n_head=4, n_layers=1, embedding_size=256, max_hw=64
        )

        for h, w in [(8, 8), (16, 16), (8, 16)]:
            num_tokens = h * w
            packed = torch.randn(1, num_tokens + 1, 32)
            # Set hw_vec to indicate spatial size
            packed[0, -1, 0] = h / model.max_hw
            packed[0, -1, 1] = w / model.max_hw

            text_embeddings = torch.randn(1, 256)
            timesteps = torch.randint(0, 1000, (1,))

            output = model(packed, text_embeddings, timesteps)
            assert output.shape == (1, num_tokens + 1, 32)

    def test_batch_size_independence(self):
        """Should work with different batch sizes."""
        model = FluxFlowProcessor(d_model=128, vae_dim=32, n_head=4, n_layers=1, embedding_size=256)

        num_tokens = 4 * 4

        for batch_size in [1, 2]:
            packed = torch.randn(batch_size, num_tokens + 1, 32)
            text_embeddings = torch.randn(batch_size, 256)
            timesteps = torch.randint(0, 1000, (batch_size,))

            output = model(packed, text_embeddings, timesteps)
            assert output.shape[0] == batch_size

    def test_context_injection(self):
        """Should have context injection mechanism."""
        model = FluxFlowProcessor(d_model=256, n_head=4, n_layers=2)

        assert hasattr(model, "context_injection")
        assert hasattr(model, "ctx_mixer")

    def test_time_embedding(self):
        """Should embed timesteps correctly."""
        model = FluxFlowProcessor(embedding_size=512)

        timesteps = torch.tensor([0, 100, 500, 999])
        embedded = model.time_embed(timesteps)

        assert embedded.shape == (4, 512)

    def test_ctx_tokens_extraction(self):
        """Should extract context from first K tokens."""
        model = FluxFlowProcessor(ctx_tokens=4, d_model=128, vae_dim=32, n_head=4, n_layers=1)

        # Create packed with more tokens than ctx_tokens
        num_tokens = 64
        packed = torch.randn(1, num_tokens + 1, 32)
        text_embeddings = torch.randn(1, 1024)
        timesteps = torch.randint(0, 1000, (1,))

        output = model(packed, text_embeddings, timesteps)

        # Should still work and preserve shape
        assert output.shape == (1, num_tokens + 1, 32)


class TestFlowIntegration:
    """Integration tests for flow model components."""

    def test_transformer_block_stacking(self):
        """Test stacking multiple transformer blocks."""
        blocks = nn.ModuleList([FluxTransformerBlock(d_model=256, n_head=8) for _ in range(3)])

        img_seq = torch.randn(2, 16, 256)
        text_seq = torch.randn(2, 1, 256)

        sin_img, cos_img = blocks[0].rotary_pe.get_embed(torch.arange(16))
        sin_txt, cos_txt = blocks[0].rotary_pe.get_embed(torch.arange(1))

        p0 = p1 = p2 = p3 = None
        for block in blocks:
            img_seq, p0, p1, p2, p3 = block(
                img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, p0, p1, p2, p3
            )

        assert img_seq.shape == (2, 16, 256)

    def test_full_flow_prediction_pipeline(self):
        """Test complete flow prediction pipeline."""
        model = FluxFlowProcessor(d_model=128, vae_dim=32, embedding_size=256, n_head=4, n_layers=2)

        # Simulate a batch of latent representations
        batch_size = 2
        h, w = 8, 8
        num_tokens = h * w
        packed = torch.randn(batch_size, num_tokens + 1, 32)

        # Set hw_vec to indicate 8x8 spatial size
        for i in range(batch_size):
            packed[i, -1, 0] = h / model.max_hw
            packed[i, -1, 1] = w / model.max_hw

        text_embeddings = torch.randn(batch_size, 256)
        timesteps = torch.randint(0, 1000, (batch_size,))

        # Forward pass
        output = model(packed, text_embeddings, timesteps)

        # Output should have same shape as input
        assert output.shape == packed.shape

        # Should be able to compute loss and backprop
        loss = output.sum()
        loss.backward()

    def test_eval_mode_deterministic(self):
        """Model in eval mode should be deterministic."""
        model = FluxFlowProcessor(d_model=128, vae_dim=32, n_head=4, n_layers=1)
        model.eval()

        packed = torch.randn(1, 17, 32)  # 16 tokens + 1 hw_vec
        text_embeddings = torch.randn(1, 1024)
        timesteps = torch.tensor([500])

        with torch.no_grad():
            out1 = model(packed, text_embeddings, timesteps)
            out2 = model(packed, text_embeddings, timesteps)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_different_timesteps(self):
        """Should handle different timesteps in batch."""
        model = FluxFlowProcessor(d_model=128, vae_dim=32, n_head=4, n_layers=1)

        packed = torch.randn(3, 17, 32)
        text_embeddings = torch.randn(3, 1024)
        timesteps = torch.tensor([0, 500, 999])  # Different timesteps

        output = model(packed, text_embeddings, timesteps)

        assert output.shape == (3, 17, 32)

    def test_device_compatibility(self):
        """Should work on CPU."""
        model = FluxFlowProcessor(d_model=128, vae_dim=32, n_head=4, n_layers=1)
        model = model.to("cpu")

        packed = torch.randn(1, 17, 32, device="cpu")
        text_embeddings = torch.randn(1, 1024, device="cpu")
        timesteps = torch.tensor([500], device="cpu")

        output = model(packed, text_embeddings, timesteps)

        assert output.device.type == "cpu"
