"""Unit tests for Padé activation functions (src/fluxflow/models/pade_activation.py)."""

import pytest
import torch
import torch.nn as nn

from fluxflow.models.pade_activation import (
    PadeActivation,
    PadeActivationModule,
    TrainablePade,
    WideTrainablePade,
)


class TestPadeActivationModule:
    """Tests for the core low-order (m=2, n=1) Padé computation."""

    def test_matches_manual_computation(self):
        module = PadeActivationModule()
        x = torch.tensor([[2.0]])
        a0 = torch.tensor([[1.0]])
        a1 = torch.tensor([[0.5]])
        a2 = torch.tensor([[-0.25]])
        b1 = torch.tensor([[3.0]])

        output = module(x, a0, a1, a2, b1)

        numerator = 1.0 + 0.5 * 2.0 + (-0.25) * 2.0**2  # 1 + 1 - 1 = 1
        denominator = 1.0 + abs(3.0 * 2.0)  # 1 + 6 = 7
        expected = numerator / denominator
        assert torch.allclose(output, torch.tensor([[expected]]), atol=1e-5)

    def test_denominator_never_below_one(self):
        """Q(x) = 1 + |b1*x| must be >= 1 for any real x, b1 -- the no-pole property."""
        module = PadeActivationModule()
        x = torch.randn(1000) * 1e6
        b1 = torch.randn(1000) * 1e6
        a0 = torch.zeros(1000)
        a1 = torch.zeros(1000)
        a2 = torch.zeros(1000)

        denominator = 1.0 + (b1 * x).abs()
        output = module(x, a0, a1, a2, b1)

        assert (denominator >= 1.0).all()
        assert torch.isfinite(output).all()


class TestPadeActivation:
    """Tests for PadeActivation (data-driven, mirrors BezierActivation)."""

    def test_2d_input_shape(self):
        activation = PadeActivation()
        x = torch.randn(4, 50)  # 50 = 10 * 5
        output = activation(x)
        assert output.shape == (4, 10)

    def test_3d_input_shape(self):
        activation = PadeActivation()
        x = torch.randn(4, 16, 25)  # 25 = 5 * 5
        output = activation(x)
        assert output.shape == (4, 16, 5)

    def test_4d_input_shape(self):
        activation = PadeActivation()
        x = torch.randn(2, 15, 8, 8)  # 15 = 3 * 5
        output = activation(x)
        assert output.shape == (2, 3, 8, 8)

    def test_channel_not_divisible_by_5_raises(self):
        activation = PadeActivation()
        x = torch.randn(4, 13)
        with pytest.raises(AssertionError):
            activation(x)

    def test_gradient_flow(self):
        activation = PadeActivation()
        x = torch.randn(4, 25, requires_grad=True)
        output = activation(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_finite_on_large_inputs(self):
        """No pole means large inputs must never produce inf/nan."""
        activation = PadeActivation()
        x = torch.randn(4, 25) * 1e4
        output = activation(x)
        assert torch.isfinite(output).all()


class TestTrainablePade:
    """Tests for TrainablePade module."""

    def test_initialization_is_identity_at_default(self):
        """Default coefficients (a0=0, a1=1, rest 0) must compute P(x)/Q(x) == x."""
        module = TrainablePade((4,), b1=0.0)
        x = torch.randn(2, 4)
        output = module(x)
        assert torch.allclose(output, x, atol=1e-5)

    def test_default_is_near_identity(self):
        """Default b1 is small nonzero (E1) so the init stays close to identity."""
        module = TrainablePade((4,))
        x = torch.randn(2, 4)
        assert torch.allclose(module(x), x, atol=2e-2)

    def test_parameters_are_learnable(self):
        module = TrainablePade((3, 4, 4))
        param_count = sum(1 for _ in module.parameters())
        assert param_count == 10  # a0..a5, b1..b4
        for param in module.parameters():
            assert param.requires_grad

    def test_forward_output_shape(self):
        shape = (3, 8, 8)
        module = TrainablePade(shape)
        x = torch.randn(2, 3, 8, 8)
        output = module(x)
        assert output.shape == (2, 3, 8, 8)

    def test_gradient_updates_parameters(self):
        module = TrainablePade((2, 4, 4))
        x = torch.randn(1, 2, 4, 4)
        output = module(x)
        loss = output.sum()
        loss.backward()
        assert module.a0.grad is not None
        assert module.b1.grad is not None

    def test_denominator_never_below_one_under_training_drift(self):
        """After large synthetic parameter updates, Q(x) must stay >= 1 (no pole)."""
        module = TrainablePade((4,))
        with torch.no_grad():
            for p in module.parameters():
                p.copy_(torch.randn_like(p) * 100)
        x = torch.randn(8, 4) * 100
        output = module(x)
        assert torch.isfinite(output).all()

    def test_channel_only_broadcasts_over_spatial_dims(self):
        module = TrainablePade((3,), channel_only=True)
        x = torch.randn(2, 3, 8, 8)
        output = module(x)
        assert output.shape == (2, 3, 8, 8)


class TestWideTrainablePade:
    """Tests for WideTrainablePade module."""

    def test_initialization_is_shifted_identity_at_default(self):
        """Default coefficients (a0=-8, a1=1, rest 0) must compute P(x)/Q(x) == x - 8."""
        module = WideTrainablePade((4,), b1=0.0)
        x = torch.randn(2, 4)
        output = module(x)
        assert torch.allclose(output, x - 8.0, atol=1e-5)

    def test_is_subclass_of_trainable_pade(self):
        module = WideTrainablePade((4,))
        assert isinstance(module, TrainablePade)


class TestTrainablePadeInputClamp:
    """Tests for the numerator input clamp (overflow fix).

    TrainablePade's numerator (degree 5, raw x) has no input range limiting,
    unlike the safe-by-construction denominator. A finite but large x (a
    normal early-training outlier) can overflow x^5 even with small learned
    coefficients. Fix: clamp x to +-65504**0.2 (~9.1887) before the Horner
    evaluation, sized off fp16's max representable value (65504) for a
    degree-5 numerator with worst-case unit-magnitude coefficients.
    """

    def test_clamp_constant_matches_fp16_derivation(self):
        from fluxflow.models.pade_activation import _INPUT_CLAMP

        assert _INPUT_CLAMP == pytest.approx(65504.0**0.2, rel=1e-9)

    def test_regression_within_clamp_range_matches_unclamped(self):
        """Effective coefficients (ctor args) reproduce the plain rational math in range."""
        x = torch.linspace(-5.0, 5.0, steps=20).view(5, 4)
        a = dict(a0=0.1, a1=1.0, a2=0.02, a3=-0.01, a4=1e-4, a5=1e-4)
        module = TrainablePade((4,), b1=0.0, b2=0.0, b3=0.0, b4=0.0, **a)
        expected = sum(a[f"a{k}"] * x**k for k in range(6))
        assert torch.allclose(module(x), expected, atol=1e-4, rtol=1e-5)

    def test_denominator_matches_abs_within_smooth_eps(self):
        x = torch.linspace(-4.0, 4.0, steps=20).view(5, 4)
        b = dict(b1=0.1, b2=-0.02, b3=0.003, b4=-1e-4)
        module = TrainablePade((4,), **b)
        q = sum(b[f"b{k}"] * x**k for k in range(1, 5))
        expected = x / (1.0 + q.abs())
        assert torch.allclose(module(x), expected, atol=5e-3)

    def test_extreme_input_produces_no_inf_or_nan(self):
        """Coefficients at observed crash scale (a1~=1, tiny a4/a5, b*=0) + extreme x."""
        module = TrainablePade((4,))
        with torch.no_grad():
            module.a1.copy_(torch.full_like(module.a1, 1.0))
            module.a4.copy_(torch.full_like(module.a4, 1e-3))
            module.a5.copy_(torch.full_like(module.a5, 1e-3))

        x = torch.full((2, 4), 1e8)
        output = module(x)
        assert torch.isfinite(output).all()

    def test_wide_trainable_pade_extreme_input_produces_no_inf_or_nan(self):
        """WideTrainablePade inherits forward unchanged -- same fix must cover it."""
        module = WideTrainablePade((4,))
        with torch.no_grad():
            module.a4.copy_(torch.full_like(module.a4, 1e-3))
            module.a5.copy_(torch.full_like(module.a5, 1e-3))

        x = torch.full((2, 4), 1e8)
        output = module(x)
        assert torch.isfinite(output).all()

    def test_soft_clamp_bounded_monotone_with_gradient(self):
        """E5: bounded by K, monotone, nonzero gradient beyond K, identity well inside."""
        from fluxflow.models.pade_activation import _INPUT_CLAMP, _soft_clamp

        x = torch.linspace(-50.0, 50.0, 2001, requires_grad=True)
        y = _soft_clamp(x, _INPUT_CLAMP)
        assert (y.abs() <= _INPUT_CLAMP).all()
        assert (y[1:] >= y[:-1]).all()
        y.sum().backward()
        assert (x.grad > 0).all()
        inside = torch.linspace(-5.0, 5.0, 11)
        assert torch.equal(_soft_clamp(inside, _INPUT_CLAMP), inside)

    def test_gradient_flows_beyond_old_hard_clamp(self):
        from fluxflow.models.pade_activation import _INPUT_CLAMP

        module = TrainablePade((1,))
        x = torch.tensor([[_INPUT_CLAMP + 3.0]], requires_grad=True)
        module(x).sum().backward()
        assert x.grad.abs().item() > 0

    def test_infinite_input_stays_finite(self):
        module = TrainablePade((1,))
        out = module(torch.tensor([[float("inf")], [-float("inf")]]))
        assert torch.isfinite(out).all()


class TestTrainablePadeParameterization:
    """E1/E2: gradient reaches the denominator at init; params live in a normalized basis."""

    def test_default_b_gradients_nonzero(self):
        for cls in (TrainablePade, WideTrainablePade):
            module = cls((4,))
            module(torch.randn(8, 4) * 2).pow(2).sum().backward()
            assert module.b1.grad.abs().sum() > 0

    def test_all_b_gradients_nonzero_with_nonzero_b1(self):
        module = TrainablePade((4,), a2=0.1, a3=0.05)
        module(torch.randn(8, 4) * 2).pow(2).sum().backward()
        for name in ("b1", "b2", "b3", "b4"):
            assert getattr(module, name).grad.abs().sum() > 0

    def test_raw_param_grads_balanced(self):
        """E2: high-order raw grads are not amplified by |x|^k (K^5 ~ 65k) vs a1."""
        torch.manual_seed(0)
        module = TrainablePade((16,))
        x = torch.randn(256, 16) * 5
        module(x).pow(2).mean().backward()
        g1, g5 = module.a1.grad.abs().mean(), module.a5.grad.abs().mean()
        assert g5 < 10 * g1

    def test_effective_coefficients_roundtrip(self):
        module = TrainablePade((2,), a0=0.3, a3=-0.2, a5=1e-3, b2=0.05)
        eff = module.effective_coefficients()
        assert eff["a0"].allclose(torch.full((2,), 0.3))
        assert eff["a3"].allclose(torch.full((2,), -0.2), rtol=1e-5)
        assert eff["a5"].allclose(torch.full((2,), 1e-3), rtol=1e-5)
        assert eff["b2"].allclose(torch.full((2,), 0.05), rtol=1e-5)

    def test_legacy_state_dict_loads_with_identical_forward(self):
        """Old checkpoints stored effective a*/b* under same names, version 1."""
        from collections import OrderedDict

        coeffs = dict(a0=0.2, a1=0.9, a2=0.03, a3=-0.004, a4=1e-4, a5=-2e-5)
        coeffs.update(b1=0.0, b2=0.0, b3=0.0, b4=0.0)
        legacy = OrderedDict((k, torch.full((3,), v)) for k, v in coeffs.items())
        legacy._metadata = {"": {"version": 1}}  # type: ignore[attr-defined]

        module = TrainablePade((3,))
        module.load_state_dict(legacy)
        x = torch.linspace(-3, 3, 12).view(4, 3)
        expected = sum(coeffs[f"a{k}"] * x**k for k in range(6))
        assert torch.allclose(module(x), expected, atol=1e-4)

    def test_legacy_state_dict_without_metadata_treated_as_legacy(self):
        module = TrainablePade((2,))
        legacy = {k: torch.full((2,), 0.0) for k in ("a0", "a2", "a3", "a4", "a5")}
        legacy.update({"a1": torch.ones(2), **{f"b{i}": torch.zeros(2) for i in range(1, 5)}})
        module.load_state_dict(legacy)
        x = torch.randn(3, 2)
        assert torch.allclose(module(x), x, atol=1e-5)

    def test_new_state_dict_roundtrip(self):
        src = TrainablePade((2,), a2=0.07, a5=3e-4, b1=0.02)
        dst = TrainablePade((2,))
        dst.load_state_dict(src.state_dict())
        x = torch.randn(4, 2)
        assert torch.equal(src(x), dst(x))
        assert torch.equal(src.a5, dst.a5)

    def test_safetensors_roundtrip_twice_identical(self, tmp_path):
        from safetensors.torch import load_file, save_file

        src = TrainablePade((3,), a2=0.07, a5=3e-4, b1=0.02)
        x = torch.randn(4, 3)
        ref = src(x)
        sd = src.state_dict()
        for i in range(2):
            save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / f"{i}.st"))
            sd = load_file(str(tmp_path / f"{i}.st"))
            dst = TrainablePade((3,))
            dst.load_state_dict(sd)
            assert torch.equal(dst(x), ref)
            sd = dst.state_dict()

    def test_plain_dict_copy_not_reconverted(self):
        src = TrainablePade((2,), a3=0.05)
        dst = TrainablePade((2,))
        dst.load_state_dict(dict(src.state_dict()))
        assert torch.equal(dst.a3, src.a3)

    def test_legacy_strict_load_converted_once_and_marker_added(self):
        legacy = {
            k: v * 0 + (1.0 if k == "a1" else 0.5)
            for k, v in TrainablePade((2,)).state_dict().items()
            if k != "pade_basis_version"
        }
        m = TrainablePade((2,))
        m.load_state_dict(legacy, strict=True)
        assert torch.allclose(m.effective_coefficients()["a3"], torch.full((2,), 0.5), rtol=1e-5)
        assert "pade_basis_version" in m.state_dict()

    def test_nested_prefix_and_strict_false(self):
        net = nn.Sequential(TrainablePade((2,)), WideTrainablePade((2,)))
        legacy = {k: v for k, v in net.state_dict().items() if "pade_basis_version" not in k}
        missing, unexpected = net.load_state_dict(legacy, strict=False)
        assert not missing and not unexpected

    def test_param_names(self):
        names = {n for n, _ in TrainablePade((1,)).named_parameters()}
        assert names == {f"a{i}" for i in range(6)} | {f"b{i}" for i in range(1, 5)}


class TestPadeActivationModuleInputClamp:
    """Tests for the fixed-mode numerator input clamp (overflow fix).

    Unlike TrainablePade, here all five of x, a0, a1, a2, b1 are unbound
    values split directly from an upstream activation tensor -- none carry a
    smallness guarantee. The dominant overflow term is a2*x^2, where both
    factors are unbounded: with everything clamped to +-K, the worst case is
    K * K^2 = K^3. Horner worst case is K + K^2 + K^3 <= 65504 (fp16 max), K = 30.
    """

    def test_clamp_constant_matches_fp16_derivation(self):
        from fluxflow.models.pade_activation import _FIXED_MODE_INPUT_CLAMP

        k = _FIXED_MODE_INPUT_CLAMP
        assert k + k**2 + k**3 <= 65504.0  # Horner worst case: a0 + x*(a1 + x*a2)
        assert k == 30.0

    def test_fp16_extreme_inputs_finite(self):
        """E4: math runs in fp32; fp16 output must stay finite at the worst case."""
        module = PadeActivationModule()
        big = torch.full((8,), 60000.0, dtype=torch.float16)
        out = module(big, big, big, big, torch.zeros(8, dtype=torch.float16))
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()

    def test_gradient_beyond_clamp_nonzero(self):
        """E5: soft clamp keeps gradient alive past K."""
        module = PadeActivationModule()
        x = torch.tensor([45.0], requires_grad=True)
        one = torch.tensor([0.5])
        module(x, one, one, one, one * 0.0).sum().backward()
        assert x.grad.abs().item() > 0

    def test_regression_within_clamp_range_matches_unclamped(self):
        """For inputs well within the clamp bound, output must match the unclamped math."""
        module = PadeActivationModule()
        torch.manual_seed(0)
        x = torch.empty(100).uniform_(-5.0, 5.0)
        a0 = torch.empty(100).uniform_(-5.0, 5.0)
        a1 = torch.empty(100).uniform_(-5.0, 5.0)
        a2 = torch.empty(100).uniform_(-5.0, 5.0)
        b1 = torch.empty(100).uniform_(-5.0, 5.0)

        expected_numerator = a0 + a1 * x + a2 * x**2
        expected_denominator = 1.0 + (b1 * x).abs()
        expected = expected_numerator / expected_denominator

        output = module(x, a0, a1, a2, b1)
        assert torch.allclose(output, expected, atol=1e-5)

    def test_extreme_input_overflows_pre_fix_but_not_post_fix(self):
        """a2*x^2 with both factors at 1e13 must overflow to Inf (float32) before the clamp."""
        x = torch.tensor([1e13])
        a0 = torch.tensor([1e13])
        a1 = torch.tensor([1e13])
        a2 = torch.tensor([1e13])
        b1 = torch.tensor([1e13])

        unclamped_numerator = a0 + a1 * x + a2 * x**2
        assert torch.isinf(unclamped_numerator).all()

        module = PadeActivationModule()
        output = module(x, a0, a1, a2, b1)
        assert torch.isfinite(output).all()

    def test_extreme_input_via_public_pade_activation_2d(self):
        """PadeActivation (2D dispatch) must not produce inf/nan on extreme inputs."""
        activation = PadeActivation()
        x = torch.full((2, 25), 1e13)  # 25 = 5 * 5
        output = activation(x)
        assert torch.isfinite(output).all()


class TestCompilePadeActivations:
    """compile_pade_activations patches/restores the class forwards."""

    def test_patch_is_idempotent_and_reversible(self, monkeypatch):
        import fluxflow.models.pade_activation as pa

        compiled = []

        def fake_compile(fn, **kwargs):
            assert kwargs == {"dynamic": True}

            def wrapper(*a, **k):
                compiled.append(fn)
                return fn(*a, **k)

            return wrapper

        monkeypatch.setattr(torch, "compile", fake_compile)
        import torch._dynamo.config as dynamo_config

        monkeypatch.setattr(dynamo_config, "recompile_limit", 8)
        eager_module = pa.PadeActivationModule.forward
        eager_trainable = pa.TrainablePade.forward
        try:
            pa.compile_pade_activations()
            patched = pa.PadeActivationModule.forward
            pa.compile_pade_activations()
            assert pa.PadeActivationModule.forward is patched

            pa.PadeActivation()(torch.randn(2, 10, 4, 4))
            pa.WideTrainablePade((3,), channel_only=True)(torch.randn(2, 3, 4, 4))
            assert len(compiled) == 2
            assert dynamo_config.recompile_limit == pa._RECOMPILE_LIMIT
        finally:
            pa.compile_pade_activations(enabled=False)

        assert pa.PadeActivationModule.forward is eager_module
        assert pa.TrainablePade.forward is eager_trainable
