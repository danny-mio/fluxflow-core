"""Widened pillar MLP for v0.10.0 flow transformer."""

import torch.nn as nn


def pillarLayerWide(in_size: int, hidden: int | None = None, depth: int = 3) -> nn.Sequential:
    """
    Widened depth-D MLP: in_size → hidden → hidden → ... → in_size with SiLU between.

    Replaces v070's pillarLayer which collapsed to in→in→in (depth=3 with
    in_size==out_size); the new version actually expands then contracts so the
    pillar has real expressive capacity.

    Args:
        in_size: Input / output channel width (= d_model in the transformer block).
        hidden: Intermediate width. Defaults to ``2 * in_size``.
        depth: Number of Linear+SiLU pairs. Must be >= 2.

    Returns:
        nn.Sequential with ``depth`` Linear+SiLU pairs.
    """
    if hidden is None:
        hidden = 2 * in_size
    assert depth >= 2, "pillarLayerWide depth must be >= 2"

    layers: list[nn.Module] = [nn.Sequential(nn.Linear(in_size, hidden), nn.SiLU())]
    for _ in range(depth - 2):
        layers.append(nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU()))
    layers.append(nn.Sequential(nn.Linear(hidden, in_size), nn.SiLU()))
    return nn.Sequential(*layers)
