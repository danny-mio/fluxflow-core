"""Smooth mu/logvar bound with gradient floor + from-scratch training regression."""

import pytest
import torch
import torch.nn.functional as F

from fluxflow.models.v100.vae import (
    LOGVAR_RANGE,
    MU_RANGE,
    FluxCompressor_v100,
    _bound,
)


def _kl(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
    return (0.5 * (mu.pow(2) + lv.exp() - 1 - lv)).mean()


def test_identity_in_normal_range():
    mu = torch.linspace(-5, 5, 101)
    lv = torch.linspace(-16, 0, 101)
    assert torch.allclose(_bound(mu, *MU_RANGE), mu, atol=1e-6)
    assert torch.allclose(_bound(lv, *LOGVAR_RANGE), lv, atol=1e-6)


@pytest.mark.parametrize("rng", [MU_RANGE, LOGVAR_RANGE])
def test_output_inside_range_and_finite(rng):
    x = torch.tensor([-1e9, -3000.0, -50.0, 50.0, 3000.0, 1e9, float("inf"), -float("inf")])
    y = _bound(x, *rng)
    assert torch.isfinite(y).all()
    assert (y >= rng[0]).all() and (y <= rng[1]).all()


@pytest.mark.parametrize("rng", [MU_RANGE, LOGVAR_RANGE])
@pytest.mark.parametrize("v", [-3000.0, -50.0, 50.0, 3000.0])
def test_gradient_never_vanishes(rng, v):
    x = torch.tensor([v], requires_grad=True)
    _bound(x, *rng).sum().backward()
    assert x.grad.abs().item() >= 0.05


def test_far_out_of_range_returns_under_kl_only_optimiser():
    mu = torch.full((64,), 1000.0, requires_grad=True)
    lv = torch.full((64,), 50.0, requires_grad=True)
    opt = torch.optim.Adam([mu, lv], lr=20.0)
    for _ in range(300):
        opt.zero_grad()
        _kl(_bound(mu, *MU_RANGE), _bound(lv, *LOGVAR_RANGE)).backward()
        opt.step()
    assert mu.abs().max() < MU_RANGE[1]
    assert lv.max() < LOGVAR_RANGE[1]
    assert torch.isfinite(_kl(_bound(mu, *MU_RANGE), _bound(lv, *LOGVAR_RANGE)))


def _comp(fam: str, d: int = 16) -> FluxCompressor_v100:
    torch.manual_seed(0)
    return FluxCompressor_v100(
        d_model=d, downscales=2, activation_type=fam, use_gradient_checkpointing=False
    )


def test_pade_forward_bounds_extreme_heads():
    c = _comp("pade").eval()
    with torch.no_grad():
        c.mu_activation.a0.fill_(1000.0)
        c.logvar_activation.a0.fill_(50.0)
        _, mu, lv = c(torch.rand(1, 3, 64, 64) * 2 - 1, training=True)
    assert torch.isfinite(mu).all() and torch.isfinite(lv).all()
    assert mu.abs().max() <= MU_RANGE[1] and lv.max() <= LOGVAR_RANGE[1]


def test_pade_extreme_head_receives_gradient():
    c = _comp("pade")
    with torch.no_grad():
        c.logvar_activation.a0.fill_(50.0)
    _, mu, lv = c(torch.rand(1, 3, 64, 64) * 2 - 1, training=True)
    _kl(mu, lv).backward()
    assert c.logvar_activation.a0.grad.abs().max() > 0


def test_bezier_forward_unchanged_by_bound():
    c = _comp("bezier").eval()
    with torch.no_grad():
        _, mu, lv = c(torch.rand(1, 3, 64, 64) * 2 - 1, training=True)
    assert lv.max() <= 4.0 + 1e-4 and torch.isfinite(mu).all()


@pytest.mark.slow
def test_pade_from_scratch_plain_l1_adamw_learns():
    """Regression: init calibration made this explode (grad norm 1e6+, L1 stuck)."""
    from fluxflow.models.v100.vae import FluxExpander_v100

    torch.manual_seed(0)
    c = FluxCompressor_v100(
        d_model=16, downscales=4, activation_type="pade", use_gradient_checkpointing=False
    )
    e = FluxExpander_v100(
        d_model=16, upscales=4, activation_type="pade", use_gradient_checkpointing=False
    )
    params = list(c.parameters()) + list(e.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    img = torch.rand(4, 3, 64, 64) * 2 - 1
    losses, norms = [], []
    for _ in range(60):
        opt.zero_grad()
        loss = F.l1_loss(e(c(img, training=True)[0]), img)
        loss.backward()
        norms.append(torch.nn.utils.clip_grad_norm_(params, float("inf")).item())
        opt.step()
        losses.append(loss.item())
    assert all(map(float.__ne__, losses, [float("nan")] * len(losses)))
    assert max(norms) < 1e3
    assert losses[-1] < 0.8 * losses[0]
