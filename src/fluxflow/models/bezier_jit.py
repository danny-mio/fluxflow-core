"""
JIT-compiled Bezier activation functions for cross-platform acceleration.

Provides torch.jit.script compiled versions of core Bezier operations
for 20-30% speedup on CPU, CUDA, and MPS devices.

This module now supports all 25 combinations of t_pre_activation × p_preactivation
using auto-generated JIT variants for maximum performance.
"""

# Import all 25 auto-generated JIT variants
from fluxflow.models.bezier_jit_generated import (  # noqa: F401
    bezier_forward_none_none,
    bezier_forward_none_sigmoid,
    bezier_forward_none_tanh,
    bezier_forward_none_silu,
    bezier_forward_none_relu,
    bezier_forward_sigmoid_none,
    bezier_forward_sigmoid_sigmoid,
    bezier_forward_sigmoid_tanh,
    bezier_forward_sigmoid_silu,
    bezier_forward_sigmoid_relu,
    bezier_forward_tanh_none,
    bezier_forward_tanh_sigmoid,
    bezier_forward_tanh_tanh,
    bezier_forward_tanh_silu,
    bezier_forward_tanh_relu,
    bezier_forward_silu_none,
    bezier_forward_silu_sigmoid,
    bezier_forward_silu_tanh,
    bezier_forward_silu_silu,
    bezier_forward_silu_relu,
    bezier_forward_relu_none,
    bezier_forward_relu_sigmoid,
    bezier_forward_relu_tanh,
    bezier_forward_relu_silu,
    bezier_forward_relu_relu,
    get_jit_bezier_function as _get_jit_bezier_function,
)

# Legacy aliases for backward compatibility
bezier_forward = bezier_forward_none_none
bezier_forward_with_sigmoid = bezier_forward_sigmoid_none
bezier_forward_with_silu = bezier_forward_silu_none
bezier_forward_with_tanh = bezier_forward_tanh_none
bezier_forward_sigmoid_silu = bezier_forward_sigmoid_silu  # Already exists


def get_jit_bezier_function(t_pre_activation=None, p_preactivation=None):
    """
    Get the appropriate JIT-compiled Bezier function for given pre-activations.

    Now supports all 25 combinations of:
        t_pre_activation: "sigmoid", "tanh", "silu", "relu", None
        p_preactivation: "sigmoid", "tanh", "silu", "relu", None

    Args:
        t_pre_activation: Transform for t ("sigmoid", "silu", "tanh", "relu", None)
        p_preactivation: Transform for control points ("sigmoid", "silu", "tanh", "relu", None)

    Returns:
        JIT-compiled Bezier function, or None if combination not supported

    Example:
        >>> bezier_fn = get_jit_bezier_function("sigmoid", "silu")
        >>> output = bezier_fn(t, p0, p1, p2, p3)
    """
    return _get_jit_bezier_function(t_pre_activation, p_preactivation)
