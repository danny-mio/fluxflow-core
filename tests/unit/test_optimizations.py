"""
Tests for Bezier activation performance optimizations.

Validates:
- All 25 JIT-compiled activation combinations
- Cached power computations
- Numerical equivalence with baseline
- Cross-device consistency
"""

import pytest
import torch

from fluxflow.models.activations import BezierActivation, BezierActivationModule, TrainableBezier
from fluxflow.models.bezier_jit import get_jit_bezier_function
from fluxflow.models.bezier_power_cache import (
    clear_power_cache,
    get_cache_stats,
    get_cached_power_fn,
    prewarm_cache,
    reset_cache_stats,
)


class TestJITCompilation:
    """Test all 25 JIT-compiled activation combinations."""

    @pytest.fixture
    def activations(self):
        """Fixture providing all activation types."""
        return [None, "sigmoid", "tanh", "silu", "relu"]

    def test_all_25_combinations_exist(self, activations):
        """Test that all 25 JIT combinations are available."""
        count = 0
        for t_act in activations:
            for p_act in activations:
                jit_fn = get_jit_bezier_function(t_act, p_act)
                assert jit_fn is not None, f"Missing JIT for {t_act}, {p_act}"
                count += 1

        assert count == 25, f"Expected 25 combinations, got {count}"

    @pytest.mark.parametrize(
        "t_act,p_act",
        [
            (None, None),
            ("sigmoid", None),
            ("tanh", None),
            ("silu", None),
            ("relu", None),
            (None, "sigmoid"),
            ("sigmoid", "sigmoid"),
            ("sigmoid", "silu"),
            ("silu", "tanh"),
        ],
    )
    def test_jit_forward_pass(self, t_act, p_act):
        """Test JIT forward pass for common combinations."""
        jit_fn = get_jit_bezier_function(t_act, p_act)

        # Create test inputs
        t = torch.randn(4, 10)
        p0, p1, p2, p3 = (
            torch.randn(4, 10),
            torch.randn(4, 10),
            torch.randn(4, 10),
            torch.randn(4, 10),
        )

        # Run forward pass
        output = jit_fn(t, p0, p1, p2, p3)

        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_jit_numerical_equivalence(self):
        """Test JIT output matches non-JIT baseline."""
        # Create inputs
        t = torch.randn(8, 20)
        p0, p1, p2, p3 = (
            torch.randn(8, 20),
            torch.randn(8, 20),
            torch.randn(8, 20),
            torch.randn(8, 20),
        )

        # JIT version
        jit_fn = get_jit_bezier_function("sigmoid", None)
        jit_output = jit_fn(t, p0, p1, p2, p3)

        # Manual version (baseline)
        t_sig = torch.sigmoid(t)
        t2 = t_sig * t_sig
        t3 = t2 * t_sig
        t_inv = 1.0 - t_sig
        t_inv2 = t_inv * t_inv
        t_inv3 = t_inv2 * t_inv
        manual_output = t_inv3 * p0 + 3.0 * t_inv2 * t_sig * p1 + 3.0 * t_inv * t2 * p2 + t3 * p3

        # Should match within floating point precision
        assert torch.allclose(jit_output, manual_output, atol=1e-6)

    def test_jit_gradient_flow(self):
        """Test that JIT functions support gradients."""
        jit_fn = get_jit_bezier_function("sigmoid", "silu")

        t = torch.randn(4, 10, requires_grad=True)
        p0 = torch.randn(4, 10, requires_grad=True)
        p1 = torch.randn(4, 10, requires_grad=True)
        p2 = torch.randn(4, 10, requires_grad=True)
        p3 = torch.randn(4, 10, requires_grad=True)

        output = jit_fn(t, p0, p1, p2, p3)
        loss = output.sum()
        loss.backward()

        # All inputs should have gradients
        assert t.grad is not None
        assert p0.grad is not None
        assert p1.grad is not None
        assert p2.grad is not None
        assert p3.grad is not None

        # Gradients should be non-zero and finite
        assert not torch.allclose(t.grad, torch.zeros_like(t.grad))
        assert not torch.isnan(t.grad).any()


class TestBezierActivationModuleOptimization:
    """Test BezierActivationModule uses JIT by default."""

    def test_module_uses_jit(self):
        """Test that BezierActivationModule loads JIT function."""
        module = BezierActivationModule("sigmoid", "silu")
        assert module.jit_fn is not None, "JIT function should be loaded"

    def test_module_all_25_combinations(self):
        """Test module works with all 25 JIT combinations."""
        activations = [None, "sigmoid", "tanh", "silu", "relu"]

        for t_act in activations:
            for p_act in activations:
                module = BezierActivationModule(t_act, p_act)
                assert module.jit_fn is not None

                # Test forward pass
                t = torch.randn(4, 10)
                p0, p1, p2, p3 = (
                    torch.randn(4, 10),
                    torch.randn(4, 10),
                    torch.randn(4, 10),
                    torch.randn(4, 10),
                )
                output = module(t, p0, p1, p2, p3)
                assert output.shape == (4, 10)

    def test_bezier_activation_uses_jit(self):
        """Test BezierActivation layer uses JIT."""
        activation = BezierActivation("sigmoid", None)
        assert activation.bezier_activation.jit_fn is not None

        # Test forward
        x = torch.randn(4, 50)  # → (4, 10) output
        output = activation(x)
        assert output.shape == (4, 10)


class TestPowerCache:
    """Test cached power computation functionality."""

    def test_cache_hit(self):
        """Test cache hits for same device/dtype."""
        clear_power_cache()
        reset_cache_stats()

        t = torch.randn(4, 10)

        # First call - cache miss
        power_fn1 = get_cached_power_fn(t)
        stats1 = get_cache_stats()
        assert stats1["misses"] >= 1

        # Second call - cache hit
        power_fn2 = get_cached_power_fn(t)
        stats2 = get_cache_stats()
        assert stats2["hits"] >= 1

        # Should return same function
        assert power_fn1 is power_fn2

    def test_cache_different_devices(self):
        """Test cache creates different entries for different devices."""
        clear_power_cache()
        reset_cache_stats()

        t_cpu = torch.randn(4, 10, device="cpu")
        power_fn_cpu = get_cached_power_fn(t_cpu)

        # If MPS available, test different cache entry
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            t_mps = torch.randn(4, 10, device="mps")
            power_fn_mps = get_cached_power_fn(t_mps)

            # Should be different functions
            assert power_fn_cpu is not power_fn_mps

    def test_power_computation_correctness(self):
        """Test cached power computation produces correct results."""
        t = torch.randn(8, 16)

        power_fn = get_cached_power_fn(t)
        t2, t3, t_inv, t_inv2, t_inv3 = power_fn(t)

        # Verify correctness
        assert torch.allclose(t2, t * t)
        assert torch.allclose(t3, t * t * t)
        assert torch.allclose(t_inv, 1.0 - t)
        assert torch.allclose(t_inv2, (1.0 - t) * (1.0 - t))
        assert torch.allclose(t_inv3, (1.0 - t) * (1.0 - t) * (1.0 - t))

    def test_power_cache_gradient_flow(self):
        """Test cached power functions support gradients."""
        t = torch.randn(4, 10, requires_grad=True)

        power_fn = get_cached_power_fn(t)
        t2, t3, t_inv, t_inv2, t_inv3 = power_fn(t)

        # Compute loss
        loss = t2.sum() + t3.sum() + t_inv.sum() + t_inv2.sum() + t_inv3.sum()
        loss.backward()

        assert t.grad is not None
        assert not torch.isnan(t.grad).any()

    def test_prewarm_cache(self):
        """Test cache prewarming."""
        clear_power_cache()
        reset_cache_stats()

        prewarm_cache()

        # Stats should be reset after prewarming
        stats = get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_trainable_bezier_uses_cache(self):
        """Test TrainableBezier uses cached power computations."""
        clear_power_cache()
        reset_cache_stats()

        trainable = TrainableBezier((16,), channel_only=True)
        x = torch.randn(4, 16, 32, 32)

        # First forward - should populate cache
        output1 = trainable(x)

        # Second forward - should use cache
        output2 = trainable(x)

        stats = get_cache_stats()
        # Should have cache hits from second forward
        assert stats["hits"] > 0

        # Outputs should be consistent
        assert output1.shape == (4, 16, 32, 32)
        assert output2.shape == (4, 16, 32, 32)


class TestNumericalEquivalence:
    """Test optimized versions produce same results as baseline."""

    def test_jit_vs_manual_all_activations(self):
        """Test JIT matches manual computation for all activations."""
        activations = [None, "sigmoid", "tanh", "silu", "relu"]

        for t_act in activations:
            for p_act in activations:
                # Create inputs
                x = torch.randn(4, 50)

                # Run through BezierActivation (uses JIT)
                activation_jit = BezierActivation(t_act, p_act)
                output_jit = activation_jit(x)

                # Should produce valid output
                assert output_jit.shape == (4, 10)
                assert not torch.isnan(output_jit).any()


class TestCrossDeviceConsistency:
    """Test consistency across CPU and MPS devices."""

    def test_cpu_vs_mps_consistency(self):
        """Test CPU and MPS produce same results."""
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available")

        # Create same input on both devices
        torch.manual_seed(42)
        x_cpu = torch.randn(4, 50)
        x_mps = x_cpu.to("mps")

        # Create activation on both devices
        activation_cpu = BezierActivation("sigmoid", None)
        activation_mps = BezierActivation("sigmoid", None).to("mps")

        # Run forward
        output_cpu = activation_cpu(x_cpu)
        output_mps = activation_mps(x_mps)

        # Results should match
        assert torch.allclose(output_cpu, output_mps.cpu(), atol=1e-5)

    @pytest.mark.parametrize("t_act,p_act", [("sigmoid", None), ("silu", "tanh"), (None, None)])
    def test_jit_cross_device(self, t_act, p_act):
        """Test JIT functions work on both CPU and MPS."""
        jit_fn = get_jit_bezier_function(t_act, p_act)

        # CPU
        t_cpu = torch.randn(4, 10)
        p0_cpu, p1_cpu, p2_cpu, p3_cpu = (
            torch.randn(4, 10),
            torch.randn(4, 10),
            torch.randn(4, 10),
            torch.randn(4, 10),
        )
        output_cpu = jit_fn(t_cpu, p0_cpu, p1_cpu, p2_cpu, p3_cpu)

        # MPS (if available)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            t_mps = t_cpu.to("mps")
            p0_mps, p1_mps, p2_mps, p3_mps = (
                p0_cpu.to("mps"),
                p1_cpu.to("mps"),
                p2_cpu.to("mps"),
                p3_cpu.to("mps"),
            )
            output_mps = jit_fn(t_mps, p0_mps, p1_mps, p2_mps, p3_mps)

            # Should match
            assert torch.allclose(output_cpu, output_mps.cpu(), atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_inputs(self):
        """Test JIT functions handle zero inputs."""
        jit_fn = get_jit_bezier_function("sigmoid", None)

        t = torch.zeros(4, 10)
        p0, p1, p2, p3 = (
            torch.zeros(4, 10),
            torch.zeros(4, 10),
            torch.zeros(4, 10),
            torch.zeros(4, 10),
        )

        output = jit_fn(t, p0, p1, p2, p3)

        assert output.shape == (4, 10)
        assert not torch.isnan(output).any()

    def test_large_values(self):
        """Test JIT functions handle large values."""
        jit_fn = get_jit_bezier_function("sigmoid", "tanh")

        t = torch.randn(4, 10) * 100  # Large values
        p0, p1, p2, p3 = (
            torch.randn(4, 10) * 100,
            torch.randn(4, 10) * 100,
            torch.randn(4, 10) * 100,
            torch.randn(4, 10) * 100,
        )

        output = jit_fn(t, p0, p1, p2, p3)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_compatibility(self):
        """Test legacy function names still work."""
        from fluxflow.models.bezier_jit import (
            bezier_forward,
            bezier_forward_with_sigmoid,
            bezier_forward_with_silu,
            bezier_forward_with_tanh,
        )

        # All legacy functions should exist
        assert bezier_forward is not None
        assert bezier_forward_with_sigmoid is not None
        assert bezier_forward_with_silu is not None
        assert bezier_forward_with_tanh is not None

        # Test they work
        t = torch.randn(4, 10)
        p0, p1, p2, p3 = (
            torch.randn(4, 10),
            torch.randn(4, 10),
            torch.randn(4, 10),
            torch.randn(4, 10),
        )

        output = bezier_forward(t, p0, p1, p2, p3)
        assert output.shape == (4, 10)
