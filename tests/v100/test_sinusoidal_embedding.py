"""Tests for continuous sinusoidal time embedding."""

import torch

from fluxflow.models.v100.positional import sinusoidal_embedding


def test_sinusoidal_embedding_shape():
    t = torch.linspace(0, 1, steps=4)
    emb = sinusoidal_embedding(t, dim=64)
    assert emb.shape == (4, 64)


def test_sinusoidal_embedding_continuous():
    """Embedding for t=0.5 ≠ t=0.5001 by a small but nonzero amount."""
    t1 = torch.tensor([0.5])
    t2 = torch.tensor([0.5001])
    e1 = sinusoidal_embedding(t1, dim=64)
    e2 = sinusoidal_embedding(t2, dim=64)
    diff = (e2 - e1).norm()
    assert 0 < diff < 1e-2


def test_sinusoidal_embedding_differentiable():
    """Gradient flows through the embedding wrt t."""
    t = torch.tensor([0.5], requires_grad=True)
    emb = sinusoidal_embedding(t, dim=64)
    emb.sum().backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()


def test_sinusoidal_embedding_distinct_timesteps():
    """Distinct t values produce distinct embeddings."""
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    emb = sinusoidal_embedding(t, dim=32)
    for i in range(emb.size(0)):
        for j in range(i + 1, emb.size(0)):
            assert not torch.allclose(emb[i], emb[j])
