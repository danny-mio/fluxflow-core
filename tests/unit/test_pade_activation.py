"""Unit tests for Padé activation functions (src/fluxflow/models/pade_activation.py)."""

import pytest
import torch

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
        module = TrainablePade((4,))
        x = torch.randn(2, 4)
        output = module(x)
        assert torch.allclose(output, x, atol=1e-5)

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
        module = WideTrainablePade((4,))
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
        """For |x| well within the clamp bound, output must match the unclamped math."""
        torch.manual_seed(0)
        module = TrainablePade((4,))
        with torch.no_grad():
            module.a1.copy_(torch.full_like(module.a1, 1.0))
            module.a4.copy_(torch.full_like(module.a4, 1e-4))
            module.a5.copy_(torch.full_like(module.a5, 1e-4))

        x = torch.linspace(-5.0, 5.0, steps=20).view(5, 4)

        a0, a1, a2, a3, a4, a5 = (
            module.a0.expand_as(x),
            module.a1.expand_as(x),
            module.a2.expand_as(x),
            module.a3.expand_as(x),
            module.a4.expand_as(x),
            module.a5.expand_as(x),
        )
        b1, b2, b3, b4 = (
            module.b1.expand_as(x),
            module.b2.expand_as(x),
            module.b3.expand_as(x),
            module.b4.expand_as(x),
        )
        expected_numerator = a0 + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4 + a5 * x**5
        expected_denominator = 1.0 + (b1 * x + b2 * x**2 + b3 * x**3 + b4 * x**4).abs()
        expected = expected_numerator / expected_denominator

        output = module(x)
        assert torch.allclose(output, expected, atol=1e-5)

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

    def test_clamp_boundary(self):
        """Output at exactly the clamp boundary must equal output just past it."""
        from fluxflow.models.pade_activation import _INPUT_CLAMP

        module = TrainablePade((1,))
        at_boundary = module(torch.tensor([[_INPUT_CLAMP]]))
        past_boundary = module(torch.tensor([[_INPUT_CLAMP + 100.0]]))
        assert torch.allclose(at_boundary, past_boundary, atol=1e-5)
        assert torch.isfinite(at_boundary).all()
        assert torch.isfinite(past_boundary).all()
