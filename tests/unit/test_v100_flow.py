"""Unit tests for FluxFlow v0.10.0 flow processor.

Run *before* implementation — all tests must fail until implementation exists.
"""

import torch


class TestFluxFlowProcessorV100:
    """Tests for FluxFlowProcessor_v100."""

    def test_forward_preserves_packed_shape(self):
        """Forward pass must return same shape as input packed tensor."""
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        proc = FluxFlowProcessor_v100(d_model=128, vae_dim=32, embedding_size=64, n_layers=2)
        packed = torch.randn(1, 17, 64)  # T=16, +1 HW, 2*32=64 dims
        text = torch.randn(1, 64)
        t = torch.tensor([0.5])
        with torch.no_grad():
            out = proc(packed, text, t)
        assert out.shape == packed.shape

    def test_vae_to_dmodel_width_is_2x_vae_dim(self):
        """vae_to_dmodel.in_features must equal 2 * vae_dim."""
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        proc = FluxFlowProcessor_v100(d_model=64, vae_dim=32, embedding_size=64, n_layers=1)
        assert proc.vae_to_dmodel.in_features == 64  # 2 * 32

    def test_dmodel_to_vae_width_is_2x_vae_dim(self):
        """dmodel_to_vae.out_features must equal 2 * vae_dim."""
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        proc = FluxFlowProcessor_v100(d_model=64, vae_dim=32, embedding_size=64, n_layers=1)
        assert proc.dmodel_to_vae.out_features == 64  # 2 * 32

    def test_context_dims_attribute_defaults_to_vae_dim(self):
        """context_dims attribute must default to vae_dim when not specified."""
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        proc = FluxFlowProcessor_v100(d_model=64, vae_dim=32, embedding_size=64, n_layers=1)
        assert proc.context_dims == 32

    def test_context_dims_attribute_explicit(self):
        """context_dims can be set explicitly and is stored as attribute."""
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        proc = FluxFlowProcessor_v100(
            d_model=64, vae_dim=32, embedding_size=64, n_layers=1, context_dims=16
        )
        assert proc.context_dims == 16

    def test_flow_does_not_separate_context_dims(self):
        """vae_to_dmodel projects the full 2D token without internal split."""
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        proc = FluxFlowProcessor_v100(d_model=64, vae_dim=32, embedding_size=64, n_layers=1)
        # Single linear weight with in_features = 2*vae_dim — no split bias param
        assert proc.vae_to_dmodel.in_features == 64  # 2 * vae_dim

    def test_no_context_dims_constant_import(self):
        """v100/flow.py must not import CONTEXT_DIMS from v070."""
        import fluxflow.models.v100.flow as flow_module

        # If CONTEXT_DIMS was imported, it would be an int attribute on the module
        assert not hasattr(
            flow_module, "CONTEXT_DIMS"
        ), "v100/flow.py must not use module-level CONTEXT_DIMS"

    def test_forward_batch_size_two(self):
        """Forward must work with batch_size > 1."""
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        proc = FluxFlowProcessor_v100(d_model=64, vae_dim=32, embedding_size=64, n_layers=1)
        T = 16
        packed = torch.randn(2, T + 1, 64)
        # Set valid HW tokens
        packed[:, -1, :] = 0
        packed[:, -1, 0] = 16 / 1024.0
        packed[:, -1, 1] = 16 / 1024.0
        text = torch.randn(2, 64)
        t = torch.tensor([0.3, 0.7])
        with torch.no_grad():
            out = proc(packed, text, t)
        assert out.shape == (2, T + 1, 64)


class TestFluxTransformerBlockV100:
    """TDD tests for FluxTransformerBlock_v100 — write BEFORE implementation."""

    D = 32
    N_HEAD = 4
    T = 16
    B = 2

    def _make_block(self):
        from fluxflow.models.v100.flow import FluxTransformerBlock_v100

        return FluxTransformerBlock_v100(d_model=self.D, n_head=self.N_HEAD)

    def _make_inputs(self, blk):
        img_seq = torch.randn(self.B, self.T, self.D)
        text_seq = torch.randn(self.B, 1, self.D)
        text_cond = torch.randn(self.B, self.D)
        sin_img, cos_img = blk.rotary_pe.get_embed(torch.arange(self.T))
        sin_txt, cos_txt = blk.rotary_pe.get_embed(torch.arange(1))
        return img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, text_cond

    def test_class_exists(self):
        from fluxflow.models.v100.flow import FluxTransformerBlock_v100

        assert FluxTransformerBlock_v100 is not None

    def test_forward_shape(self):
        blk = self._make_block()
        img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, text_cond = self._make_inputs(blk)
        with torch.no_grad():
            out, p0, p1, p2, p3 = blk(
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
        assert out.shape == (self.B, self.T, self.D)
        for p in (p0, p1, p2, p3):
            assert p.shape == (self.B, self.T, self.D)

    def test_film_beta_propagates_through_zeroed_pillars(self):
        """
        Correct ordering: Pillar MLPs run first, FiLM modulates the output.
        With zeroed pillar weights (g_p* == 0), FiLM gives: 0*(1+gamma)+beta = beta.
        beta must propagate to the final output.
        If FiLM were first (v080 ordering), zeroed pillar MLPs would zero everything.
        """
        blk = self._make_block()
        # Zero all pillar MLP weights so g_p* == 0 regardless of input
        for attr in ("p0", "p1", "p2", "p3"):
            for m in getattr(blk, attr).modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.zeros_(m.weight)
                    torch.nn.init.zeros_(m.bias)
        # Set film beta to a large nonzero constant (second half of film output)
        for attr in ("film_p0", "film_p1", "film_p2", "film_p3"):
            film = getattr(blk, attr)
            torch.nn.init.zeros_(film.weight)
            bias = torch.zeros(film.out_features)
            bias[self.D :] = 1.0  # beta=1.0, gamma=0.0
            film.bias.data.copy_(bias)

        img_seq, text_seq, sin_img, cos_img, sin_txt, cos_txt, text_cond = self._make_inputs(blk)
        with torch.no_grad():
            _, p0, _, _, _ = blk(
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
        # With zeroed pillars + FiLM-second: p0 = FiLM(0) = beta = 1.0
        # Values flow through pillar_cross_attn (additive residual), so p0 != 0
        assert not torch.allclose(
            p0, torch.zeros_like(p0)
        ), "FiLM beta must be nonzero in pillar output — proves FiLM runs AFTER Pillar MLP"

    def test_film_layer_shapes(self):
        blk = self._make_block()
        for attr in ("film_p0", "film_p1", "film_p2", "film_p3"):
            film = getattr(blk, attr)
            assert film.in_features == self.D
            assert film.out_features == 2 * self.D

    def test_pillar_cross_attn_exists(self):
        blk = self._make_block()
        assert hasattr(blk, "pillar_cross_attn")
        assert hasattr(blk, "norm_pillar")

    def test_no_v080_import_in_v100_flow(self):
        """v100/flow.py must not import any symbol from v080 after this fix."""
        import ast
        import pathlib

        src = pathlib.Path(
            "/Volumes/DanieleExt/ai/ffnew/fluxflow-core/src/fluxflow/models/v100/flow.py"
        ).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert (
                        "v080" not in node.module
                    ), f"v100/flow.py must not import from v080: {node.module}"
