"""Tests for zero-init residual seam-smoother in FluxExpander_v100.

Three properties verified:
1. Zero-init no-op: seam_smoother weights == 0 → output bit-identical to a
   ZeroConv stand-in (returns zeros), proving feat + 0 == feat.
2. Learnability: weights become non-zero after one optimizer step.
3. State-dict compat: loading checkpoint without seam_smoother.* keys (strict=False)
   keeps output identical to the unmodified expander.
"""

import torch
import torch.nn as nn


class _ZeroConv(nn.Module):
    """Drop-in replacement that always returns zeros with the same shape as input.

    Used in zero-init tests to simulate a seam_smoother whose residual is zero,
    verifying feat + _ZeroConv(feat) == feat + 0 == feat.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _make_v100_packed(b: int, h_lat: int, w_lat: int, d_model: int = 32) -> torch.Tensor:
    """Build a valid packed tensor for FluxExpander_v100."""
    T = h_lat * w_lat
    packed = torch.randn(b, T + 1, 2 * d_model)
    packed[:, -1, :] = 0.0
    packed[:, -1, 0] = h_lat / 1024.0
    packed[:, -1, 1] = w_lat / 1024.0
    return packed


class TestSeamSmootherV100:
    """Seam-smoother tests for FluxExpander_v100 (v0.10.0)."""

    def test_zero_init_noop(self):
        """At init, seam_smoother weights == 0 → output identical to zero-residual path."""
        from fluxflow.models.v100.vae import FluxExpander_v100

        exp = FluxExpander_v100(d_model=32, upscales=2)
        exp.eval()

        assert torch.all(exp.seam_smoother.weight == 0), "seam_smoother weight must be zero at init"
        assert torch.all(exp.seam_smoother.bias == 0), "seam_smoother bias must be zero at init"
        assert torch.all(
            exp.seam_smoother_ctx.weight == 0
        ), "seam_smoother_ctx weight must be zero at init"
        assert torch.all(
            exp.seam_smoother_ctx.bias == 0
        ), "seam_smoother_ctx bias must be zero at init"

        packed = _make_v100_packed(1, 4, 4, d_model=32)

        # Replace both smoothers with zero-output modules to simulate no-op residual
        orig = exp.seam_smoother
        orig_ctx = exp.seam_smoother_ctx
        exp.seam_smoother = _ZeroConv()
        exp.seam_smoother_ctx = _ZeroConv()
        with torch.no_grad():
            out_zero_ref = exp(packed)
        exp.seam_smoother = orig
        exp.seam_smoother_ctx = orig_ctx

        with torch.no_grad():
            out_zero_init = exp(packed)

        torch.testing.assert_close(out_zero_init, out_zero_ref, atol=0, rtol=0)

    def test_learnability(self):
        """After one optimizer step, seam_smoother weights must be non-zero."""
        from fluxflow.models.v100.vae import FluxExpander_v100

        exp = FluxExpander_v100(d_model=32, upscales=2)
        packed = _make_v100_packed(1, 4, 4, d_model=32)
        target = torch.zeros(1, 3, 16, 16)  # 4x4 tokens * 2^2 upscales = 16x16

        opt = torch.optim.SGD(exp.parameters(), lr=0.1)
        out = exp(packed)
        loss = (out - target).pow(2).mean()
        loss.backward()
        opt.step()

        assert not torch.all(
            exp.seam_smoother.weight == 0
        ), "seam_smoother.weight must be non-zero after step"

    def test_state_dict_compat(self):
        """Loading a checkpoint without seam_smoother.* keys (strict=False) preserves output.

        Simulates loading an old checkpoint: save state_dict, delete seam_smoother.*,
        reload into fresh expander. Output must equal output before any training since
        the freshly initialised seam_smoother weights default to zero (no-op residual).
        """
        from fluxflow.models.v100.vae import FluxExpander_v100

        exp = FluxExpander_v100(d_model=32, upscales=2)
        exp.eval()

        packed = _make_v100_packed(1, 4, 4, d_model=32)
        with torch.no_grad():
            out_before = exp(packed)

        # Build a "legacy" checkpoint without seam_smoother keys
        sd = {k: v for k, v in exp.state_dict().items() if "seam_smoother" not in k}

        fresh = FluxExpander_v100(d_model=32, upscales=2)
        fresh.eval()
        missing, unexpected = fresh.load_state_dict(sd, strict=False)

        # Only seam_smoother keys should be missing
        assert all("seam_smoother" in k for k in missing), f"Unexpected missing keys: {missing}"
        assert unexpected == [], f"Unexpected keys: {unexpected}"

        with torch.no_grad():
            out_after = fresh(packed)

        torch.testing.assert_close(out_after, out_before, atol=0, rtol=0)
