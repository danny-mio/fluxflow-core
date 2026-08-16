"""Tests for the redesigned FluxTransformerBlock_v100."""

import inspect

import torch

from fluxflow.models.v100.flow import FluxTransformerBlock_v100


def _block(d=64, nh=4):
    return FluxTransformerBlock_v100(d_model=d, n_head=nh)


def test_block_forward_signature_has_text_mask_and_time_cond():
    sig = inspect.signature(FluxTransformerBlock_v100.forward)
    assert "text_mask" in sig.parameters
    assert "time_cond" in sig.parameters


def test_block_no_pillar_cross_attn():
    b = _block()
    assert not hasattr(b, "pillar_cross_attn")
    assert not hasattr(b, "norm_pillar")


def test_block_uses_widened_pillar():
    """First Linear in any pillar maps D → 2D (widened)."""
    b = _block(d=64)
    first_linear = next(m for m in b.p0.modules() if isinstance(m, torch.nn.Linear))
    assert first_linear.out_features == 128  # 2 * 64


def test_block_has_dual_film_per_pillar():
    b = _block()
    for i in range(4):
        assert hasattr(b, f"film_p{i}_text")
        assert hasattr(b, f"film_p{i}_time")


def test_block_has_split_norm2():
    b = _block()
    assert hasattr(b, "norm2_q")
    assert hasattr(b, "norm2_kv")
    assert not hasattr(b, "norm2")


def test_block_forward_runs():
    """Forward pass produces correct shape with new signature."""
    torch.manual_seed(0)
    d, nh = 64, 4
    b = _block(d=d, nh=nh)
    B, T_img, T_txt = 1, 16, 5
    img_seq = torch.randn(B, T_img, d)
    text_seq = torch.randn(B, T_txt, d)
    text_mask = torch.ones(B, T_txt, dtype=torch.bool)
    head_dim = d // nh
    half = head_dim // 2
    # Axial buffers for img (each half_dim)
    sin_w_img = torch.zeros(T_img, half)
    cos_w_img = torch.ones(T_img, half)
    sin_h_img = torch.zeros(T_img, half)
    cos_h_img = torch.ones(T_img, half)
    # Text uses 1D RoPE (full head_dim per the rotary helper)
    sin_txt = torch.zeros(T_txt, head_dim)
    cos_txt = torch.ones(T_txt, head_dim)
    text_cond = torch.randn(B, d)
    time_cond = torch.randn(B, d)
    out, p0, p1, p2, p3 = b(
        img_seq,
        text_seq,
        text_mask,
        sin_w_img,
        cos_w_img,
        sin_h_img,
        cos_h_img,
        sin_txt,
        cos_txt,
        None,
        None,
        None,
        None,
        text_cond,
        time_cond,
    )
    assert out.shape == img_seq.shape
    for p in (p0, p1, p2, p3):
        assert p.shape == img_seq.shape


def _forward_inputs(d=64, nh=4, B=1, T_img=16, T_txt=5, seed=0):
    torch.manual_seed(seed)
    img_seq = torch.randn(B, T_img, d)
    text_seq = torch.randn(B, T_txt, d)
    text_mask = torch.ones(B, T_txt, dtype=torch.bool)
    head_dim = d // nh
    half = head_dim // 2
    sin_w_img = torch.zeros(T_img, half)
    cos_w_img = torch.ones(T_img, half)
    sin_h_img = torch.zeros(T_img, half)
    cos_h_img = torch.ones(T_img, half)
    sin_txt = torch.zeros(T_txt, head_dim)
    cos_txt = torch.ones(T_txt, head_dim)
    text_cond = torch.randn(B, d)
    time_cond = torch.randn(B, d)
    return (
        img_seq,
        text_seq,
        text_mask,
        sin_w_img,
        cos_w_img,
        sin_h_img,
        cos_h_img,
        sin_txt,
        cos_txt,
        text_cond,
        time_cond,
    )


def _call_forward(b, args, time_cond=None):
    (
        img_seq,
        text_seq,
        text_mask,
        sin_w_img,
        cos_w_img,
        sin_h_img,
        cos_h_img,
        sin_txt,
        cos_txt,
        text_cond,
        default_time_cond,
    ) = args
    return b(
        img_seq,
        text_seq,
        text_mask,
        sin_w_img,
        cos_w_img,
        sin_h_img,
        cos_h_img,
        sin_txt,
        cos_txt,
        None,
        None,
        None,
        None,
        text_cond,
        time_cond if time_cond is not None else default_time_cond,
    )


def test_film_dual_identity_at_init():
    """At init, film_text_scale/film_time_scale == 0 so _film_dual is a no-op."""
    d, nh = 64, 4
    b = _block(d=d, nh=nh)
    assert torch.equal(b.film_text_scale, torch.zeros(1))
    assert torch.equal(b.film_time_scale, torch.zeros(1))

    B = 2
    gate = torch.randn(B, 16, d)
    text_cond = torch.randn(B, d)
    time_cond = torch.randn(B, d)
    out = b._film_dual(gate, b.film_p0_text, b.film_p0_time, text_cond, time_cond)
    assert torch.equal(out, gate)

    # Verify through the full forward() call too: capture pre-FiLM gates.
    args = _forward_inputs(d=d, nh=nh)
    captured_gates = []
    real_film_dual = b._film_dual

    def _spy(gate, film_text, film_time, text_cond, time_cond):
        captured_gates.append(gate)
        return real_film_dual(gate, film_text, film_time, text_cond, time_cond)

    b._film_dual = _spy
    _, p0, p1, p2, p3 = _call_forward(b, args)
    assert len(captured_gates) == 4
    for p, g in zip((p0, p1, p2, p3), captured_gates):
        assert torch.equal(p, g)


def test_film_dual_learnable():
    """film_text_scale/film_time_scale receive nonzero gradient and move off 0."""
    d, nh = 64, 4
    b = _block(d=d, nh=nh)
    args = _forward_inputs(d=d, nh=nh)
    _, p0, _, _, _ = _call_forward(b, args)
    loss = p0.sum()
    loss.backward()

    assert b.film_text_scale.grad is not None
    assert b.film_time_scale.grad is not None
    assert torch.any(b.film_text_scale.grad != 0)
    assert torch.any(b.film_time_scale.grad != 0)

    optimizer = torch.optim.SGD(b.parameters(), lr=0.1)
    optimizer.step()

    assert not torch.equal(b.film_text_scale, torch.zeros(1))
    assert not torch.equal(b.film_time_scale, torch.zeros(1))


def test_film_dual_strict_false_load_from_pre_fix_checkpoint():
    """Pre-fix checkpoints (missing the two new scale keys) load via
    strict=False, defaulting scales to 0: timestep conditioning is fully
    zeroed post-reload (identical outputs across different time_cond), and
    this stops being true after a few fine-tuning steps."""
    d, nh = 64, 4
    src = _block(d=d, nh=nh)
    stripped = {
        k: v for k, v in src.state_dict().items() if k not in ("film_text_scale", "film_time_scale")
    }

    b = _block(d=d, nh=nh)
    missing, unexpected = b.load_state_dict(stripped, strict=False)
    assert "film_text_scale" in missing
    assert "film_time_scale" in missing
    assert unexpected == []
    assert torch.equal(b.film_text_scale, torch.zeros(1))
    assert torch.equal(b.film_time_scale, torch.zeros(1))

    args = _forward_inputs(d=d, nh=nh, seed=1)
    time_cond_a = args[-1]
    time_cond_b = time_cond_a + torch.randn_like(time_cond_a)

    out_a, *_ = _call_forward(b, args, time_cond=time_cond_a)
    out_b, *_ = _call_forward(b, args, time_cond=time_cond_b)
    assert torch.equal(out_a, out_b)

    optimizer = torch.optim.SGD(b.parameters(), lr=0.5)
    for _ in range(5):
        optimizer.zero_grad()
        out, *_ = _call_forward(b, args, time_cond=time_cond_a)
        out.sum().backward()
        optimizer.step()

    out_a2, *_ = _call_forward(b, args, time_cond=time_cond_a)
    out_b2, *_ = _call_forward(b, args, time_cond=time_cond_b)
    assert not torch.equal(out_a2, out_b2)


def test_film_dual_scale_bounded_via_tanh():
    """film_text_scale/film_time_scale are unclamped; simulate aggressive
    drift and verify the *effective* scales consumed at _film_dual's use-sites
    stay within (-1, 1), matching the tanh-wrapped formula exactly.
    """
    d, nh = 64, 4
    b = _block(d=d, nh=nh)
    b.film_text_scale.data.fill_(8.0)  # fp32 tanh saturates to 1.0 above ~9
    b.film_time_scale.data.fill_(-8.0)  # fp32 tanh saturates to -1.0 below ~-9

    gate = torch.randn(2, 16, d)
    text_cond = torch.randn(2, d)
    time_cond = torch.randn(2, d)
    with torch.no_grad():
        out = b._film_dual(gate, b.film_p0_text, b.film_p0_time, text_cond, time_cond)
        gt, bt = b.film_p0_text(text_cond).chunk(2, dim=-1)
        gtau, btau = b.film_p0_time(time_cond).chunk(2, dim=-1)
        text_scale = torch.tanh(b.film_text_scale)
        time_scale = torch.tanh(b.film_time_scale)
        expected = (
            gate * (1.0 + text_scale * gt[:, None, :] + time_scale * gtau[:, None, :])
            + text_scale * bt[:, None, :]
            + time_scale * btau[:, None, :]
        )

    assert torch.isfinite(out).all()
    assert text_scale.abs().item() < 1.0
    assert time_scale.abs().item() < 1.0
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)
