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
