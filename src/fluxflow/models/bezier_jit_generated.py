"""
JIT-compiled Bezier activation functions (auto-generated).

Provides torch.jit.script compiled versions for all 25 combinations
of t_pre_activation × p_preactivation for maximum performance.
"""

import torch
import torch.nn.functional as F


@torch.jit.script
def bezier_forward_none_none(t, p0, p1, p2, p3):
    """
    Bezier with none on t, none on control points.

    Configuration: t_pre_activation="None", p_preactivation="None"
    """

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_none_sigmoid(t, p0, p1, p2, p3):
    """
    Bezier with none on t, sigmoid on control points.

    Configuration: t_pre_activation="None", p_preactivation="sigmoid"
    """
    p0 = torch.sigmoid(p0)
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    p3 = torch.sigmoid(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_none_tanh(t, p0, p1, p2, p3):
    """
    Bezier with none on t, tanh on control points.

    Configuration: t_pre_activation="None", p_preactivation="tanh"
    """
    p0 = torch.tanh(p0)
    p1 = torch.tanh(p1)
    p2 = torch.tanh(p2)
    p3 = torch.tanh(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_none_silu(t, p0, p1, p2, p3):
    """
    Bezier with none on t, silu on control points.

    Configuration: t_pre_activation="None", p_preactivation="silu"
    """
    p0 = F.silu(p0)
    p1 = F.silu(p1)
    p2 = F.silu(p2)
    p3 = F.silu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_none_relu(t, p0, p1, p2, p3):
    """
    Bezier with none on t, relu on control points.

    Configuration: t_pre_activation="None", p_preactivation="relu"
    """
    p0 = F.relu(p0)
    p1 = F.relu(p1)
    p2 = F.relu(p2)
    p3 = F.relu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_sigmoid_none(t, p0, p1, p2, p3):
    """
    Bezier with sigmoid on t, none on control points.

    Configuration: t_pre_activation="sigmoid", p_preactivation="None"
    """
    t = torch.sigmoid(t)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_sigmoid_sigmoid(t, p0, p1, p2, p3):
    """
    Bezier with sigmoid on t, sigmoid on control points.

    Configuration: t_pre_activation="sigmoid", p_preactivation="sigmoid"
    """
    t = torch.sigmoid(t)
    p0 = torch.sigmoid(p0)
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    p3 = torch.sigmoid(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_sigmoid_tanh(t, p0, p1, p2, p3):
    """
    Bezier with sigmoid on t, tanh on control points.

    Configuration: t_pre_activation="sigmoid", p_preactivation="tanh"
    """
    t = torch.sigmoid(t)
    p0 = torch.tanh(p0)
    p1 = torch.tanh(p1)
    p2 = torch.tanh(p2)
    p3 = torch.tanh(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_sigmoid_silu(t, p0, p1, p2, p3):
    """
    Bezier with sigmoid on t, silu on control points.

    Configuration: t_pre_activation="sigmoid", p_preactivation="silu"
    """
    t = torch.sigmoid(t)
    p0 = F.silu(p0)
    p1 = F.silu(p1)
    p2 = F.silu(p2)
    p3 = F.silu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_sigmoid_relu(t, p0, p1, p2, p3):
    """
    Bezier with sigmoid on t, relu on control points.

    Configuration: t_pre_activation="sigmoid", p_preactivation="relu"
    """
    t = torch.sigmoid(t)
    p0 = F.relu(p0)
    p1 = F.relu(p1)
    p2 = F.relu(p2)
    p3 = F.relu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_tanh_none(t, p0, p1, p2, p3):
    """
    Bezier with tanh on t, none on control points.

    Configuration: t_pre_activation="tanh", p_preactivation="None"
    """
    t = torch.tanh(t)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_tanh_sigmoid(t, p0, p1, p2, p3):
    """
    Bezier with tanh on t, sigmoid on control points.

    Configuration: t_pre_activation="tanh", p_preactivation="sigmoid"
    """
    t = torch.tanh(t)
    p0 = torch.sigmoid(p0)
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    p3 = torch.sigmoid(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_tanh_tanh(t, p0, p1, p2, p3):
    """
    Bezier with tanh on t, tanh on control points.

    Configuration: t_pre_activation="tanh", p_preactivation="tanh"
    """
    t = torch.tanh(t)
    p0 = torch.tanh(p0)
    p1 = torch.tanh(p1)
    p2 = torch.tanh(p2)
    p3 = torch.tanh(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_tanh_silu(t, p0, p1, p2, p3):
    """
    Bezier with tanh on t, silu on control points.

    Configuration: t_pre_activation="tanh", p_preactivation="silu"
    """
    t = torch.tanh(t)
    p0 = F.silu(p0)
    p1 = F.silu(p1)
    p2 = F.silu(p2)
    p3 = F.silu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_tanh_relu(t, p0, p1, p2, p3):
    """
    Bezier with tanh on t, relu on control points.

    Configuration: t_pre_activation="tanh", p_preactivation="relu"
    """
    t = torch.tanh(t)
    p0 = F.relu(p0)
    p1 = F.relu(p1)
    p2 = F.relu(p2)
    p3 = F.relu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_silu_none(t, p0, p1, p2, p3):
    """
    Bezier with silu on t, none on control points.

    Configuration: t_pre_activation="silu", p_preactivation="None"
    """
    t = F.silu(t)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_silu_sigmoid(t, p0, p1, p2, p3):
    """
    Bezier with silu on t, sigmoid on control points.

    Configuration: t_pre_activation="silu", p_preactivation="sigmoid"
    """
    t = F.silu(t)
    p0 = torch.sigmoid(p0)
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    p3 = torch.sigmoid(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_silu_tanh(t, p0, p1, p2, p3):
    """
    Bezier with silu on t, tanh on control points.

    Configuration: t_pre_activation="silu", p_preactivation="tanh"
    """
    t = F.silu(t)
    p0 = torch.tanh(p0)
    p1 = torch.tanh(p1)
    p2 = torch.tanh(p2)
    p3 = torch.tanh(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_silu_silu(t, p0, p1, p2, p3):
    """
    Bezier with silu on t, silu on control points.

    Configuration: t_pre_activation="silu", p_preactivation="silu"
    """
    t = F.silu(t)
    p0 = F.silu(p0)
    p1 = F.silu(p1)
    p2 = F.silu(p2)
    p3 = F.silu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_silu_relu(t, p0, p1, p2, p3):
    """
    Bezier with silu on t, relu on control points.

    Configuration: t_pre_activation="silu", p_preactivation="relu"
    """
    t = F.silu(t)
    p0 = F.relu(p0)
    p1 = F.relu(p1)
    p2 = F.relu(p2)
    p3 = F.relu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_relu_none(t, p0, p1, p2, p3):
    """
    Bezier with relu on t, none on control points.

    Configuration: t_pre_activation="relu", p_preactivation="None"
    """
    t = F.relu(t)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_relu_sigmoid(t, p0, p1, p2, p3):
    """
    Bezier with relu on t, sigmoid on control points.

    Configuration: t_pre_activation="relu", p_preactivation="sigmoid"
    """
    t = F.relu(t)
    p0 = torch.sigmoid(p0)
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    p3 = torch.sigmoid(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_relu_tanh(t, p0, p1, p2, p3):
    """
    Bezier with relu on t, tanh on control points.

    Configuration: t_pre_activation="relu", p_preactivation="tanh"
    """
    t = F.relu(t)
    p0 = torch.tanh(p0)
    p1 = torch.tanh(p1)
    p2 = torch.tanh(p2)
    p3 = torch.tanh(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_relu_silu(t, p0, p1, p2, p3):
    """
    Bezier with relu on t, silu on control points.

    Configuration: t_pre_activation="relu", p_preactivation="silu"
    """
    t = F.relu(t)
    p0 = F.silu(p0)
    p1 = F.silu(p1)
    p2 = F.silu(p2)
    p3 = F.silu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


@torch.jit.script
def bezier_forward_relu_relu(t, p0, p1, p2, p3):
    """
    Bezier with relu on t, relu on control points.

    Configuration: t_pre_activation="relu", p_preactivation="relu"
    """
    t = F.relu(t)
    p0 = F.relu(p0)
    p1 = F.relu(p1)
    p2 = F.relu(p2)
    p3 = F.relu(p3)

    # Bezier curve computation
    one_minus_t = 1.0 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cube = one_minus_t_sq * one_minus_t

    t_sq = t * t
    t_cube = t_sq * t

    return (
        one_minus_t_cube * p0
        + 3.0 * one_minus_t_sq * t * p1
        + 3.0 * one_minus_t * t_sq * p2
        + t_cube * p3
    )


def get_jit_bezier_function(t_pre_activation=None, p_preactivation=None):
    """
    Get the appropriate JIT-compiled Bezier function for given pre-activations.

    Args:
        t_pre_activation: Transform for t ("sigmoid", "silu", "tanh", "relu", None)
        p_preactivation: Transform for control points ("sigmoid", "silu", "tanh", "relu", None)

    Returns:
        JIT-compiled Bezier function

    Example:
        >>> bezier_fn = get_jit_bezier_function("sigmoid", "silu")
        >>> output = bezier_fn(t, p0, p1, p2, p3)
    """
    # Lookup table for all 25 combinations
    lookup = {
        (None, None): bezier_forward_none_none,
        (None, "sigmoid"): bezier_forward_none_sigmoid,
        (None, "tanh"): bezier_forward_none_tanh,
        (None, "silu"): bezier_forward_none_silu,
        (None, "relu"): bezier_forward_none_relu,
        ("sigmoid", None): bezier_forward_sigmoid_none,
        ("sigmoid", "sigmoid"): bezier_forward_sigmoid_sigmoid,
        ("sigmoid", "tanh"): bezier_forward_sigmoid_tanh,
        ("sigmoid", "silu"): bezier_forward_sigmoid_silu,
        ("sigmoid", "relu"): bezier_forward_sigmoid_relu,
        ("tanh", None): bezier_forward_tanh_none,
        ("tanh", "sigmoid"): bezier_forward_tanh_sigmoid,
        ("tanh", "tanh"): bezier_forward_tanh_tanh,
        ("tanh", "silu"): bezier_forward_tanh_silu,
        ("tanh", "relu"): bezier_forward_tanh_relu,
        ("silu", None): bezier_forward_silu_none,
        ("silu", "sigmoid"): bezier_forward_silu_sigmoid,
        ("silu", "tanh"): bezier_forward_silu_tanh,
        ("silu", "silu"): bezier_forward_silu_silu,
        ("silu", "relu"): bezier_forward_silu_relu,
        ("relu", None): bezier_forward_relu_none,
        ("relu", "sigmoid"): bezier_forward_relu_sigmoid,
        ("relu", "tanh"): bezier_forward_relu_tanh,
        ("relu", "silu"): bezier_forward_relu_silu,
        ("relu", "relu"): bezier_forward_relu_relu,
    }

    return lookup.get((t_pre_activation, p_preactivation), None)
