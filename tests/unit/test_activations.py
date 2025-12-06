"""Unit tests for activation functions (src/models/activations.py)."""

import pytest
import torch
import torch.nn as nn

from fluxflow.models.activations import (
    BezierActivation,
    Flip,
    Rot90,
    TrainableBezier,
    xavier_init,
)


class TestBezierActivation:
    """Tests for BezierActivation module."""

    def test_2d_input_shape(self):
        """Test with 2D input [B, D] where D % 5 == 0."""
        activation = BezierActivation()
        x = torch.randn(4, 50)  # 50 = 10 * 5
        output = activation(x)
        assert output.shape == (4, 10)  # 50 / 5 = 10 features

    def test_3d_input_shape(self):
        """Test with 3D input [B, S, D] where D % 5 == 0."""
        activation = BezierActivation()
        x = torch.randn(4, 16, 25)  # 25 = 5 * 5
        output = activation(x)
        assert output.shape == (4, 16, 5)  # 25 / 5 = 5 features

    def test_4d_input_shape(self):
        """Test with 4D input [B, C, H, W] where C % 5 == 0."""
        activation = BezierActivation()
        x = torch.randn(2, 15, 8, 8)  # 15 = 3 * 5
        output = activation(x)
        assert output.shape == (2, 3, 8, 8)  # 15 / 5 = 3 features

    def test_channel_not_divisible_by_5_raises(self):
        """Should raise assertion error if channels not divisible by 5."""
        activation = BezierActivation()
        x = torch.randn(4, 13)  # 13 % 5 != 0
        with pytest.raises(AssertionError):
            activation(x)

    def test_bezier_curve_computation(self):
        """Verify Bezier curve formula is applied correctly."""
        activation = BezierActivation()

        # Create controlled input: [t, p0, p1, p2, p3]
        # With t=0.5 and specific control points
        t = torch.tensor([[0.5]])
        p0 = torch.tensor([[0.0]])
        p1 = torch.tensor([[1.0]])
        p2 = torch.tensor([[2.0]])
        p3 = torch.tensor([[3.0]])

        x = torch.cat([t, p0, p1, p2, p3], dim=1)  # [1, 5]
        output = activation(x)

        # Manual Bezier computation at t=0.5
        expected = (
            (1 - 0.5) ** 3 * 0.0
            + 3 * (1 - 0.5) ** 2 * 0.5 * 1.0
            + 3 * (1 - 0.5) * 0.5**2 * 2.0
            + 0.5**3 * 3.0
        )

        # Apply sigmoid to t (default t_pre_activation)
        t_activated = torch.sigmoid(t).item()
        expected = (
            (1 - t_activated) ** 3 * 0.0
            + 3 * (1 - t_activated) ** 2 * t_activated * 1.0
            + 3 * (1 - t_activated) * t_activated**2 * 2.0
            + t_activated**3 * 3.0
        )

        assert torch.allclose(output, torch.tensor([[expected]]), atol=1e-5)

    def test_t_pre_activation_sigmoid(self):
        """Test t pre-activation with sigmoid."""
        activation = BezierActivation(t_pre_activation="sigmoid")
        x = torch.randn(2, 10)
        output = activation(x)
        assert output.shape == (2, 2)
        assert torch.isfinite(output).all()

    def test_t_pre_activation_tanh(self):
        """Test t pre-activation with tanh."""
        activation = BezierActivation(t_pre_activation="tanh")
        x = torch.randn(2, 10)
        output = activation(x)
        assert output.shape == (2, 2)
        assert torch.isfinite(output).all()

    def test_t_pre_activation_silu(self):
        """Test t pre-activation with silu."""
        activation = BezierActivation(t_pre_activation="silu")
        x = torch.randn(2, 10)
        output = activation(x)
        assert output.shape == (2, 2)
        assert torch.isfinite(output).all()

    def test_p_preactivation_sigmoid(self):
        """Test p pre-activation with sigmoid."""
        activation = BezierActivation(p_preactivation="sigmoid")
        x = torch.randn(2, 10)
        output = activation(x)
        assert output.shape == (2, 2)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Verify gradients can flow through activation."""
        activation = BezierActivation()
        x = torch.randn(4, 25, requires_grad=True)
        output = activation(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestTrainableBezier:
    """Tests for TrainableBezier module."""

    def test_initialization(self):
        """Test that parameters are initialized correctly."""
        shape = (3, 8, 8)
        module = TrainableBezier(shape, p0=0.0, p1=0.25, p2=0.75, p3=1.0)

        assert module.p0.shape == shape
        assert module.p1.shape == shape
        assert module.p2.shape == shape
        assert module.p3.shape == shape

        # Check initialization values
        assert torch.allclose(module.p0, torch.zeros(shape))
        assert torch.allclose(module.p1, torch.ones(shape) * 0.25)
        assert torch.allclose(module.p2, torch.ones(shape) * 0.75)
        assert torch.allclose(module.p3, torch.ones(shape))

    def test_parameters_are_learnable(self):
        """Test that control points are learnable parameters."""
        module = TrainableBezier((3, 4, 4))

        param_count = sum(1 for _ in module.parameters())
        assert param_count == 4  # p0, p1, p2, p3

        for param in module.parameters():
            assert param.requires_grad

    def test_forward_output_shape(self):
        """Test forward pass produces correct output shape."""
        shape = (3, 8, 8)
        module = TrainableBezier(shape)

        # Input is just t, control points are learned
        x = torch.randn(2, 3, 8, 8)
        output = module(x)

        # Output should have same spatial dims, but 5x channels (added p0-p3)
        # Then BezierActivation reduces back to original
        assert output.shape == (2, 3, 8, 8)

    def test_gradient_updates_parameters(self):
        """Test that backward pass updates control point parameters."""
        module = TrainableBezier((2, 4, 4))
        x = torch.randn(1, 2, 4, 4)

        # Forward and backward
        output = module(x)
        loss = output.sum()
        loss.backward()

        # Parameters should have gradients
        assert module.p0.grad is not None
        assert module.p1.grad is not None
        assert module.p2.grad is not None
        assert module.p3.grad is not None

    def test_different_initialization_values(self):
        """Test with different initialization values."""
        module = TrainableBezier((2, 2, 2), p0=-1.0, p1=0.0, p2=0.5, p3=2.0)

        assert torch.allclose(module.p0, torch.ones(2, 2, 2) * -1.0)
        assert torch.allclose(module.p1, torch.zeros(2, 2, 2))
        assert torch.allclose(module.p2, torch.ones(2, 2, 2) * 0.5)
        assert torch.allclose(module.p3, torch.ones(2, 2, 2) * 2.0)


class TestFlip:
    """Tests for Flip module."""

    def test_default_flip_2d_spatial(self):
        """Test default flip along dimensions (2, 3) for 4D tensor."""
        flip = Flip()
        x = torch.randn(2, 3, 4, 5)
        output = flip(x)

        assert output.shape == x.shape
        # Verify flipping: first element should match last after flip
        assert torch.equal(output[0, 0, 0, 0], x[0, 0, -1, -1])

    def test_custom_flip_dims(self):
        """Test flip with custom dimensions."""
        flip = Flip(dims=(1, 2))
        x = torch.randn(2, 3, 4, 5)
        output = flip(x)

        assert output.shape == x.shape
        # Verify flipping along dims 1 and 2
        assert torch.equal(output[0, 0, 0, 0], x[0, -1, -1, 0])

    def test_single_dim_flip(self):
        """Test flip along single dimension."""
        flip = Flip(dims=(2,))
        x = torch.randn(2, 3, 4, 5)
        output = flip(x)

        assert output.shape == x.shape
        # Verify flipping along dim 2 only
        assert torch.equal(output[0, 0, 0, :], x[0, 0, -1, :])

    def test_2d_tensor(self):
        """Test flip on 2D tensor."""
        flip = Flip(dims=(1,))
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        output = flip(x)

        expected = torch.tensor([[3, 2, 1], [6, 5, 4]])
        assert torch.equal(output, expected)

    def test_gradient_flow(self):
        """Verify gradients flow through flip."""
        flip = Flip()
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        output = flip(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestRot90:
    """Tests for Rot90 module."""

    def test_default_rotation(self):
        """Test default 90-degree rotation (k=1)."""
        rot = Rot90()
        x = torch.randn(2, 3, 4, 5)
        output = rot(x)

        # After 90-degree rotation, H and W swap
        assert output.shape == (2, 3, 5, 4)

    def test_180_rotation(self):
        """Test 180-degree rotation (k=2)."""
        rot = Rot90(k=2)
        x = torch.randn(2, 3, 4, 5)
        output = rot(x)

        # After 180-degree rotation, shape stays same but content flips
        assert output.shape == x.shape

    def test_270_rotation(self):
        """Test 270-degree rotation (k=3)."""
        rot = Rot90(k=3)
        x = torch.randn(2, 3, 4, 5)
        output = rot(x)

        # After 270-degree rotation, H and W swap
        assert output.shape == (2, 3, 5, 4)

    def test_360_rotation(self):
        """Test 360-degree rotation (k=4) returns to original."""
        rot = Rot90(k=4)
        x = torch.randn(2, 3, 4, 5)
        output = rot(x)

        # After 360-degree rotation, should be identical
        assert output.shape == x.shape
        assert torch.equal(output, x)

    def test_rotation_correctness(self):
        """Verify rotation is applied correctly."""
        rot = Rot90(k=1)
        x = torch.tensor([[[[1, 2], [3, 4]]]])  # [1, 1, 2, 2]
        output = rot(x)

        # torch.rot90 with k=1 rotates 90-degree counter-clockwise
        # Input:  [[1, 2],    Expected:  [[2, 4],
        #          [3, 4]]                 [1, 3]]
        expected = torch.tensor([[[[2, 4], [1, 3]]]])  # [1, 1, 2, 2]
        assert torch.equal(output, expected)

    def test_gradient_flow(self):
        """Verify gradients flow through rotation."""
        rot = Rot90(k=1)
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        output = rot(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Gradient shape matches input
        assert x.grad.shape == x.shape


class TestXavierInit:
    """Tests for xavier_init function."""

    def test_conv2d_initialization(self):
        """Test Xavier init on Conv2d layer."""
        layer = nn.Conv2d(3, 16, 3)
        xavier_init(layer)

        # Check that weights are initialized (not default)
        # Xavier should have roughly uniform distribution with specific variance
        weight_std = layer.weight.std().item()
        assert 0.05 < weight_std < 0.5  # Reasonable range for Xavier

        # Bias should be zeros
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_linear_initialization(self):
        """Test Xavier init on Linear layer."""
        layer = nn.Linear(128, 64)
        xavier_init(layer)

        weight_std = layer.weight.std().item()
        assert 0.05 < weight_std < 0.5

        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_transposed_conv_initialization(self):
        """Test Xavier init on ConvTranspose2d layer."""
        layer = nn.ConvTranspose2d(16, 8, 4, stride=2)
        xavier_init(layer)

        weight_std = layer.weight.std().item()
        assert 0.05 < weight_std < 0.5
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_batch_norm_skipped(self):
        """Test that BatchNorm is skipped (no error)."""
        layer = nn.BatchNorm2d(16)
        # Should not raise error, just skips
        xavier_init(layer)

    def test_module_recursion(self):
        """Test applying xavier_init to a module recursively."""
        module = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
        )

        module.apply(xavier_init)

        # Check both Conv layers are initialized
        conv1 = module[0]
        conv3 = module[2]

        assert 0.05 < conv1.weight.std().item() < 0.5
        assert 0.05 < conv3.weight.std().item() < 0.5


class TestActivationsIntegration:
    """Integration tests combining multiple activation functions."""

    def test_bezier_in_network(self):
        """Test BezierActivation in a simple network."""
        network = nn.Sequential(
            nn.Linear(20, 50),
            BezierActivation(),
            nn.Linear(10, 5),
        )

        x = torch.randn(4, 20)
        output = network(x)
        assert output.shape == (4, 5)
        assert torch.isfinite(output).all()

    def test_trainable_bezier_training_step(self):
        """Test TrainableBezier in a training step."""
        module = TrainableBezier((3, 4, 4))
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)

        x = torch.randn(2, 3, 4, 4)
        target = torch.randn(2, 3, 4, 4)

        # Store initial parameter values
        p0_before = module.p0.clone()

        # Training step
        optimizer.zero_grad()
        output = module(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # Parameters should have changed
        assert not torch.equal(module.p0, p0_before)

    def test_spatial_transforms_composition(self):
        """Test composing Flip and Rot90."""
        transform = nn.Sequential(
            Flip(dims=(2, 3)),
            Rot90(k=1),
        )

        x = torch.randn(2, 3, 8, 8)
        output = transform(x)

        # After flip and 90-degree rotation
        assert output.shape == (2, 3, 8, 8)
        assert torch.isfinite(output).all()


class TestSlidingBezierActivation:
    pass
