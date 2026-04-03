import pytest

mx = pytest.importorskip("mlx.core")


def test_spade_output_shape():
    from fluxflow.mlx.layers.conditioning import SPADE

    spade = SPADE(context_nc=5, num_features=64)
    x = mx.random.normal((1, 64, 8, 8))
    ctx = mx.random.normal((1, 5, 8, 8))
    out = spade(x, ctx)
    mx.eval(out)
    assert out.shape == (1, 64, 8, 8)


def test_spade_none_context_returns_normalized():
    from fluxflow.mlx.layers.conditioning import SPADE

    spade = SPADE(context_nc=5, num_features=32)
    x = mx.random.normal((1, 32, 4, 4))
    out = spade(x, None)
    mx.eval(out)
    assert out.shape == (1, 32, 4, 4)


def test_spade_no_gamma_attribute():
    """SPADE must be beta-only — no mlp_gamma."""
    from fluxflow.mlx.layers.conditioning import SPADE

    spade = SPADE(context_nc=8, num_features=16)
    assert not hasattr(spade, "mlp_gamma"), "SPADE must be beta-only (no mlp_gamma)"


def test_film_output_shape():
    from fluxflow.mlx.layers.conditioning import FiLM

    film = FiLM(d_in=128, d_feat=64)
    cond = mx.random.normal((2, 128))
    x = mx.random.normal((2, 64))
    out = film(x, cond)
    mx.eval(out)
    assert out.shape == (2, 64)
