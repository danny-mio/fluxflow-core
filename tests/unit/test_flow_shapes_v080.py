"""Unit tests for FluxFlow v0.8.0 flow model components (pillar-attention)."""

import pytest
import torch
import torch.nn as nn

from fluxflow.models.v070.vae import CONTEXT_DIMS
from fluxflow.models.v080.flow import FluxFlowProcessor_v080, FluxTransformerBlock_v080

D = 32  # small d_model for CPU tests
N_HEAD = 4
T = 16  # sequence length (image tokens)
B = 2  # batch size


@pytest.fixture
def block():
    return FluxTransformerBlock_v080(d_model=D, n_head=N_HEAD)


@pytest.fixture
def processor():
    return FluxFlowProcessor_v080(
        d_model=D,
        vae_dim=D,
        embedding_size=D,
        n_head=N_HEAD,
        n_layers=2,
        max_hw=64,
        ctx_tokens=2,
    )


class TestFluxTransformerBlock_v080:
    """Tests for FluxTransformerBlock_v080."""

    def test_initialization(self):
        """Block initializes with expected sub-modules."""
        blk = FluxTransformerBlock_v080(d_model=D, n_head=N_HEAD)
        assert hasattr(blk, "film_p0")
        assert hasattr(blk, "film_p1")
        assert hasattr(blk, "film_p2")
        assert hasattr(blk, "film_p3")
        assert hasattr(blk, "pillar_cross_attn")
        assert hasattr(blk, "norm_pillar")

    def test_film_linear_shapes(self):
        """FiLM projections should map D → 2*D."""
        blk = FluxTransformerBlock_v080(d_model=D, n_head=N_HEAD)
        assert blk.film_p0.in_features == D
        assert blk.film_p0.out_features == 2 * D

    def test_pillar_cross_attn_heads(self):
        """pillar_cross_attn should use n_head // 4 heads (min 1)."""
        blk = FluxTransformerBlock_v080(d_model=D, n_head=N_HEAD)
        expected = max(1, N_HEAD // 4)
        assert blk.pillar_cross_attn.n_head == expected

    def _make_inputs(self, blk):
        img_seq = torch.randn(B, T, D)
        text_seq = torch.randn(B, 1, D)
        text_cond = torch.randn(B, D)
        sin_img, cos_img = blk.rotary_pe.get_embed(torch.arange(T))
        sin_txt, cos_txt = blk.rotary_pe.get_embed(torch.arange(1))
        return img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, text_cond

    def test_output_shapes_first_block(self, block):
        """First block (p_x=None) returns correct output shapes."""
        img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, text_cond = self._make_inputs(block)

        out, p0, p1, p2, p3 = block(
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
            text_cond,
        )

        assert out.shape == (B, T, D)
        assert p0.shape == (B, T, D)
        assert p1.shape == (B, T, D)
        assert p2.shape == (B, T, D)
        assert p3.shape == (B, T, D)

    def test_output_shapes_subsequent_block(self, block):
        """Subsequent blocks (p_x provided) return correct shapes."""
        img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, text_cond = self._make_inputs(block)
        p_prev = torch.randn(B, T, D)

        out, p0, p1, p2, p3 = block(
            img_seq,
            text_seq,
            sin_img,
            cos_img,
            sin_txt,
            cos_txt,
            p_prev,
            p_prev,
            p_prev,
            p_prev,
            text_cond,
        )

        assert out.shape == (B, T, D)

    def test_film_modulation_changes_output(self, block):
        """Different text_cond should produce different outputs (FiLM is active)."""
        img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, _ = self._make_inputs(block)
        tc1 = torch.zeros(B, D)
        tc2 = torch.ones(B, D)

        out1, *_ = block(
            img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, None, None, None, None, tc1
        )
        out2, *_ = block(
            img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, None, None, None, None, tc2
        )

        assert not torch.allclose(
            out1, out2
        ), "FiLM should make outputs differ with different text_cond"

    def test_gradient_flow(self, block):
        """Gradients should flow through all new paths."""
        img_seq = torch.randn(B, T, D, requires_grad=True)
        text_seq = torch.randn(B, 1, D)
        text_cond = torch.randn(B, D, requires_grad=True)
        sin_img, cos_img = block.rotary_pe.get_embed(torch.arange(T))
        sin_txt, cos_txt = block.rotary_pe.get_embed(torch.arange(1))

        out, p0, p1, p2, p3 = block(
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
            text_cond,
        )
        loss = out.sum() + p0.sum() + p1.sum() + p2.sum() + p3.sum()
        loss.backward()

        assert img_seq.grad is not None
        assert text_cond.grad is not None
        assert not torch.isnan(img_seq.grad).any()
        assert not torch.isnan(text_cond.grad).any()

    def test_batch_independence(self, block):
        """Output for each sample should be independent."""
        img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, text_cond = self._make_inputs(block)

        out_full, *_ = block(
            img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, None, None, None, None, text_cond
        )

        out_0, *_ = block(
            img_seq[:1],
            text_seq[:1],
            sin_img,
            cos_img,
            sin_txt,
            cos_txt,
            None,
            None,
            None,
            None,
            text_cond[:1],
        )

        assert torch.allclose(out_full[:1], out_0, atol=1e-5)


class TestFluxFlowProcessor_v080:
    """Tests for FluxFlowProcessor_v080."""

    def _make_packed(self, vae_dim, T=T, max_hw=64):
        """Build a valid packed tensor with HW metadata token."""
        img_tokens = torch.randn(B, T, vae_dim + CONTEXT_DIMS)
        # HW token: H=W=8 (8*8=64=T), normalized by max_hw=64
        hw = torch.zeros(B, vae_dim + CONTEXT_DIMS)
        hw[:, 0] = 8 / max_hw  # H
        hw[:, 1] = 8 / max_hw  # W
        return torch.cat([img_tokens, hw.unsqueeze(1)], dim=1)

    def test_output_shape(self, processor):
        """Forward pass should preserve packed tensor shape."""
        packed = self._make_packed(D)
        text_emb = torch.randn(B, D)
        t = torch.tensor([0.3, 0.7])

        out = processor(packed, text_emb, t)

        assert out.shape == packed.shape

    def test_hw_token_preserved(self, processor):
        """HW metadata token should be unchanged in output."""
        packed = self._make_packed(D)
        text_emb = torch.randn(B, D)
        t = torch.tensor([0.3, 0.7])

        out = processor(packed, text_emb, t)

        assert torch.allclose(out[:, -1, :], packed[:, -1, :], atol=1e-5)

    def test_text_cond_proj_exists(self, processor):
        """Processor must have text_cond_proj layer."""
        assert hasattr(processor, "text_cond_proj")
        assert isinstance(processor.text_cond_proj, nn.Linear)

    def test_transformer_blocks_are_v080(self, processor):
        """All transformer blocks should be FluxTransformerBlock_v080."""
        for blk in processor.transformer_blocks:
            assert isinstance(blk, FluxTransformerBlock_v080)

    def test_film_modulation_active(self, processor):
        """Different text embeddings should produce different outputs."""
        packed = self._make_packed(D)
        t = torch.tensor([0.5, 0.5])
        text1 = torch.zeros(B, D)
        text2 = torch.ones(B, D)

        with torch.no_grad():
            out1 = processor(packed, text1, t)
            out2 = processor(packed, text2, t)

        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self, processor):
        """Gradients should flow through all components."""
        packed = self._make_packed(D)
        packed.requires_grad_(True)
        text_emb = torch.randn(B, D, requires_grad=True)
        t = torch.tensor([0.3, 0.7])

        out = processor(packed, text_emb, t)
        out.sum().backward()

        assert packed.grad is not None
        assert text_emb.grad is not None

    def test_single_batch(self, processor):
        """Should work with batch size 1 (slow path may trigger)."""
        img_tokens = torch.randn(1, T, D + CONTEXT_DIMS)
        hw = torch.zeros(1, D + CONTEXT_DIMS)
        hw[0, 0] = 8 / 64
        hw[0, 1] = 8 / 64
        packed = torch.cat([img_tokens, hw.unsqueeze(1)], dim=1)
        text_emb = torch.randn(1, D)
        t = torch.tensor([0.5])

        with torch.no_grad():
            out = processor(packed, text_emb, t)

        assert out.shape == packed.shape

    @pytest.mark.gpu
    def test_gpu_larger_model(self):
        """GPU test with larger d_model=512."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = "cuda"
        proc = FluxFlowProcessor_v080(
            d_model=512,
            vae_dim=128,
            embedding_size=1024,
            n_head=8,
            n_layers=3,
            max_hw=256,
        ).to(device)

        T_gpu = 64
        packed = torch.randn(2, T_gpu + 1, 128 + CONTEXT_DIMS, device=device)
        packed[:, -1, :] = 0
        packed[:, -1, 0] = 8 / 256
        packed[:, -1, 1] = 8 / 256
        text_emb = torch.randn(2, 1024, device=device)
        t = torch.tensor([0.3, 0.7], device=device)

        with torch.no_grad():
            out = proc(packed, text_emb, t)

        assert out.shape == packed.shape
