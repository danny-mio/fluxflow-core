"""
Code generation utility for JIT-compiled Bezier activation functions.

Generates all 25 combinations of t_pre_activation × p_preactivation to avoid
manual boilerplate while maintaining optimal performance.
"""

from typing import Optional


def generate_activation_code(activation: Optional[str], var_name: str) -> str:
    """
    Generate code for applying an activation function.

    Args:
        activation: One of "sigmoid", "tanh", "silu", "relu", or None
        var_name: Variable name to apply activation to (e.g., "t", "p0")

    Returns:
        Python code string applying the activation

    Example:
        >>> generate_activation_code("sigmoid", "t")
        't = torch.sigmoid(t)'
        >>> generate_activation_code(None, "t")
        ''
    """
    if activation is None:
        return ""
    elif activation == "sigmoid":
        return f"    {var_name} = torch.sigmoid({var_name})"
    elif activation == "tanh":
        return f"    {var_name} = torch.tanh({var_name})"
    elif activation == "silu":
        return f"    {var_name} = F.silu({var_name})"
    elif activation == "relu":
        return f"    {var_name} = F.relu({var_name})"
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def generate_function_name(t_activation: Optional[str], p_activation: Optional[str]) -> str:
    """
    Generate function name from activation pair.

    Args:
        t_activation: Activation for t parameter
        p_activation: Activation for control points

    Returns:
        Function name string

    Example:
        >>> generate_function_name("sigmoid", "silu")
        'bezier_forward_sigmoid_silu'
        >>> generate_function_name(None, None)
        'bezier_forward_none_none'
    """
    t_part = t_activation if t_activation else "none"
    p_part = p_activation if p_activation else "none"
    return f"bezier_forward_{t_part}_{p_part}"


def generate_jit_function(t_activation: Optional[str], p_activation: Optional[str]) -> str:
    """
    Generate complete JIT-compiled Bezier function code.

    Args:
        t_activation: Activation for t parameter ("sigmoid", "tanh", "silu", "relu", None)
        p_activation: Activation for control points ("sigmoid", "tanh", "silu", "relu", None)

    Returns:
        Complete Python function definition as string

    Example:
        >>> code = generate_jit_function("sigmoid", "silu")
        >>> print(code)
        @torch.jit.script
        def bezier_forward_sigmoid_silu(t, p0, p1, p2, p3):
            ...
    """
    func_name = generate_function_name(t_activation, p_activation)
    t_code = generate_activation_code(t_activation, "t")
    p_codes = [generate_activation_code(p_activation, var) for var in ["p0", "p1", "p2", "p3"]]

    # Build docstring
    t_desc = t_activation if t_activation else "none"
    p_desc = p_activation if p_activation else "none"
    docstring = f'    """\n    Bezier with {t_desc} on t, {p_desc} on control points.\n\n'
    docstring += (
        f'    Configuration: t_pre_activation="{t_activation}", p_preactivation="{p_activation}"\n'
    )
    docstring += '    """'

    # Build function body
    lines = [
        "@torch.jit.script",
        f"def {func_name}(t, p0, p1, p2, p3):",
        docstring,
    ]

    # Add t activation
    if t_code:
        lines.append(t_code)

    # Add p activations
    for p_code in p_codes:
        if p_code:
            lines.append(p_code)

    # Add core Bezier computation (always present)
    lines.extend(
        [
            "",
            "    # Bezier curve computation",
            "    one_minus_t = 1.0 - t",
            "    one_minus_t_sq = one_minus_t * one_minus_t",
            "    one_minus_t_cube = one_minus_t_sq * one_minus_t",
            "",
            "    t_sq = t * t",
            "    t_cube = t_sq * t",
            "",
            "    return (",
            "        one_minus_t_cube * p0",
            "        + 3.0 * one_minus_t_sq * t * p1",
            "        + 3.0 * one_minus_t * t_sq * p2",
            "        + t_cube * p3",
            "    )",
        ]
    )

    return "\n".join(lines)


def generate_all_jit_functions() -> str:
    """
    Generate all 25 JIT-compiled Bezier function combinations.

    Returns:
        Complete Python module code with all 25 functions

    Generates functions for all combinations of:
        t_activations = [None, "sigmoid", "tanh", "silu", "relu"]
        p_activations = [None, "sigmoid", "tanh", "silu", "relu"]
    """
    activations = [None, "sigmoid", "tanh", "silu", "relu"]

    # Module header
    header = [
        '"""',
        "JIT-compiled Bezier activation functions (auto-generated).",
        "",
        "Provides torch.jit.script compiled versions for all 25 combinations",
        "of t_pre_activation × p_preactivation for maximum performance.",
        '"""',
        "",
        "import torch",
        "import torch.nn.functional as F",
        "",
        "",
    ]

    # Generate all 25 functions
    functions = []
    for t_act in activations:
        for p_act in activations:
            func_code = generate_jit_function(t_act, p_act)
            functions.append(func_code)

    # Combine all
    all_code = "\n".join(header) + "\n\n".join(functions) + "\n"
    return all_code


def generate_lookup_function() -> str:
    """
    Generate the get_jit_bezier_function() lookup function.

    Returns:
        Python code for function lookup dispatch
    """
    activations = [None, "sigmoid", "tanh", "silu", "relu"]

    lines = [
        "def get_jit_bezier_function(t_pre_activation=None, p_preactivation=None):",
        '    """',
        "    Get the appropriate JIT-compiled Bezier function for given pre-activations.",
        "",
        "    Args:",
        '        t_pre_activation: Transform for t ("sigmoid", "silu", "tanh", "relu", None)',
        '        p_preactivation: Transform for control points ("sigmoid", "silu", "tanh", "relu", None)',
        "",
        "    Returns:",
        "        JIT-compiled Bezier function",
        "",
        "    Example:",
        '        >>> bezier_fn = get_jit_bezier_function("sigmoid", "silu")',
        "        >>> output = bezier_fn(t, p0, p1, p2, p3)",
        '    """',
        "    # Lookup table for all 25 combinations",
        "    lookup = {",
    ]

    # Generate lookup table
    for t_act in activations:
        for p_act in activations:
            func_name = generate_function_name(t_act, p_act)
            t_key = f'"{t_act}"' if t_act else "None"
            p_key = f'"{p_act}"' if p_act else "None"
            lines.append(f"        ({t_key}, {p_key}): {func_name},")

    lines.extend(
        [
            "    }",
            "",
            "    return lookup.get((t_pre_activation, p_preactivation), None)",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    # Generate complete bezier_jit.py module
    print("Generating bezier_jit.py with all 25 JIT variants...")

    all_functions = generate_all_jit_functions()
    lookup_function = generate_lookup_function()

    complete_module = all_functions + "\n\n" + lookup_function + "\n"

    # Write to file
    output_path = "src/fluxflow/models/bezier_jit_generated.py"
    with open(output_path, "w") as f:
        f.write(complete_module)

    print(f"✓ Generated {output_path} with 25 JIT variants")
