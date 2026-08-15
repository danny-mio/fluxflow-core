"""Regression tests: FluxCompressor_v100/FluxExpander_v100 must not emit
NaN/Inf on real image-range input from a genuinely random (no checkpoint)
initialization.

Root cause: FluxCompressor_v100 and FluxExpander_v100 never called
``self.apply(xavier_init)`` (the convention used by every other model class
in this codebase -- see conditioning.py, encoders.py, and every flow.py).
They relied on PyTorch's raw default Conv2d init, which was never validated
against this architecture's deep unnormalized Conv2d->BezierActivation
stacks. Confirmed via an actual training run using the real from-scratch
config (``vae_dim: 32``, ``img_size: 1024`` --
config-v0.10.0-from-scratch*.yaml): fp32 forward passes stay large-but-
finite, but the *same* unstable activations clip straight to Inf under
fp16's narrow (~65504) dynamic range.

The seeds in ``_KNOWN_BAD_SEEDS`` were found by sweeping seeds 0-59 at the
real production scale (d_model=32, downscales=4, 1024x1024 input) under a
real ``torch.autocast(dtype=torch.float16)`` context on the pre-fix code --
4/60 seeds (~7%) produced non-finite output. Pure fp32 forward passes did
not reproduce the bug at any scale tried (up to d_model=512, downscales=6),
matching the user's independent observation that disabling ``use_fp16``
avoids the crash. The bug is seed-dependent (a NaN-at-init bug can hide for
many draws), so these specific seeds pin down a deterministic regression
check rather than relying on chance across a fresh random range.
"""

import pytest
import torch

from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

# Confirmed to produce NaN/Inf pre-fix under real fp16 autocast at
# production scale (d_model=32, downscales=4, 1024x1024 input); confirmed
# finite post-fix. Mixed with a few arbitrary seeds for general coverage.
_KNOWN_BAD_SEEDS = [6, 15, 27, 42]
_COVERAGE_SEEDS = [0, 1, 2, 3, 4, 5]
_PROD_SEEDS = _KNOWN_BAD_SEEDS + _COVERAGE_SEEDS

# Real from-scratch training config (config-v0.10.0-from-scratch*.yaml):
# model.vae_dim=32, data.img_size=1024.
_PROD_D_MODEL = 32
_PROD_DOWNSCALES = 4
_PROD_IMG_SIZE = 1024

# Smaller config for fast CPU sanity checks (always run, not GPU-gated).
_FAST_D_MODEL = 64
_FAST_DOWNSCALES = 4
_FAST_IMG_SIZE = 64
_FAST_SEEDS = list(range(10))


def _real_image_batch(seed: int, size: int, batch: int = 1) -> torch.Tensor:
    """Real, well-formed image-range input: uniform in [-1, 1]."""
    g = torch.Generator().manual_seed(seed)
    return torch.empty(batch, 3, size, size).uniform_(-1.0, 1.0, generator=g)


def _fresh_compressor(seed: int, d_model: int, downscales: int) -> FluxCompressor_v100:
    torch.manual_seed(seed)
    return FluxCompressor_v100(
        d_model=d_model, downscales=downscales, use_gradient_checkpointing=False
    ).eval()


def _fresh_expander(seed: int, d_model: int, upscales: int) -> FluxExpander_v100:
    torch.manual_seed(seed)
    return FluxExpander_v100(
        d_model=d_model, upscales=upscales, use_gradient_checkpointing=False
    ).eval()


class TestCompressorRandomInitStabilityFp32Fast:
    """Fast CPU sanity check: fresh random-init compressor stays finite
    under fp32 at a modest scale."""

    @pytest.mark.parametrize("seed", _FAST_SEEDS)
    def test_finite_output(self, seed):
        comp = _fresh_compressor(seed, _FAST_D_MODEL, _FAST_DOWNSCALES)
        img = _real_image_batch(seed, _FAST_IMG_SIZE, batch=2)
        with torch.no_grad():
            packed = comp(img)
        assert torch.isfinite(packed).all(), f"seed={seed}: NaN/Inf in compressor output (fp32)"


class TestRoundTripRandomInitStabilityFp32Fast:
    """Fast CPU sanity check: fresh random-init compressor -> expander round
    trip stays finite under fp32."""

    @pytest.mark.parametrize("seed", _FAST_SEEDS)
    def test_finite_roundtrip(self, seed):
        comp = _fresh_compressor(seed, _FAST_D_MODEL, _FAST_DOWNSCALES)
        exp = _fresh_expander(seed + 1000, _FAST_D_MODEL, _FAST_DOWNSCALES)
        img = _real_image_batch(seed, _FAST_IMG_SIZE, batch=2)
        with torch.no_grad():
            packed = comp(img)
            out = exp(packed)
        assert torch.isfinite(packed).all(), f"seed={seed}: NaN/Inf in packed latent (fp32)"
        assert torch.isfinite(out).all(), f"seed={seed}: NaN/Inf in decoded image (fp32)"


@pytest.mark.gpu
class TestCompressorRandomInitStabilityFp16Autocast:
    """Fresh random-init compressor at real production scale (vae_dim=32,
    img_size=1024) must not produce NaN/Inf under a REAL fp16 autocast
    context -- this is precisely the condition (and scale) under which the
    bug was confirmed via an actual training run."""

    @pytest.mark.parametrize("seed", _PROD_SEEDS)
    def test_finite_output_under_real_fp16_autocast(self, seed):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = "cuda"
        comp = _fresh_compressor(seed, _PROD_D_MODEL, _PROD_DOWNSCALES).to(device)
        img = _real_image_batch(seed, _PROD_IMG_SIZE).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            packed = comp(img)
        assert torch.isfinite(packed).all(), f"seed={seed}: NaN/Inf in compressor output (fp16)"


@pytest.mark.gpu
class TestRoundTripRandomInitStabilityFp16Autocast:
    """Fresh random-init compressor -> expander round trip at real
    production scale under a REAL fp16 autocast context."""

    @pytest.mark.parametrize("seed", _PROD_SEEDS)
    def test_finite_roundtrip_under_real_fp16_autocast(self, seed):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = "cuda"
        comp = _fresh_compressor(seed, _PROD_D_MODEL, _PROD_DOWNSCALES).to(device)
        exp = _fresh_expander(seed + 1000, _PROD_D_MODEL, _PROD_DOWNSCALES).to(device)
        img = _real_image_batch(seed, _PROD_IMG_SIZE).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            packed = comp(img)
            out = exp(packed)
        assert torch.isfinite(packed).all(), f"seed={seed}: NaN/Inf in packed latent (fp16)"
        assert torch.isfinite(out).all(), f"seed={seed}: NaN/Inf in decoded image (fp16)"
