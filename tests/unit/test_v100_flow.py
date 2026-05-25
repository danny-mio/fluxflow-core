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
