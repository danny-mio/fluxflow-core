import pytest

mx = pytest.importorskip("mlx.core")
import hashlib  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402


def _make_fake_checkpoint(path: str) -> dict:
    """Create a minimal fake checkpoint with conv + linear weights."""
    state = {
        "expander.to_rgb_conv.0.weight": torch.randn(96, 128, 3, 3),
        "expander.to_rgb_conv.0.bias": torch.randn(96),
        "expander.to_rgb_conv.3.weight": torch.randn(48, 96, 3, 3),
        "flow.transformer_blocks.0.attn.q_proj.weight": torch.randn(128, 128),
        "flow.transformer_blocks.0.attn.q_proj.bias": torch.randn(128),
    }
    save_file(state, path)
    return state


def test_convert_produces_npz():
    from fluxflow.mlx.convert import convert_checkpoint

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.safetensors")
        _make_fake_checkpoint(src)
        out = convert_checkpoint(src)
        assert out.endswith(".npz")
        assert os.path.exists(out)


def test_convert_does_not_modify_original():
    from fluxflow.mlx.convert import convert_checkpoint

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.safetensors")
        _make_fake_checkpoint(src)
        before = hashlib.md5(open(src, "rb").read()).hexdigest()
        convert_checkpoint(src)
        after = hashlib.md5(open(src, "rb").read()).hexdigest()
        assert before == after


def test_convert_preserves_keys():
    from fluxflow.mlx.convert import convert_checkpoint

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.safetensors")
        pt_state = _make_fake_checkpoint(src)
        out = convert_checkpoint(src)
        mlx_arrays = dict(mx.load(out))
        assert set(mlx_arrays.keys()) == set(pt_state.keys())


def test_convert_transposes_conv2d_weights():
    from fluxflow.mlx.convert import convert_checkpoint

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.safetensors")
        _make_fake_checkpoint(src)
        out = convert_checkpoint(src)
        arrays = dict(mx.load(out))
        # PyTorch [96, 128, 3, 3] → MLX [96, 3, 3, 128]
        assert arrays["expander.to_rgb_conv.0.weight"].shape == (96, 3, 3, 128)


def test_convert_leaves_linear_weights_unchanged():
    from fluxflow.mlx.convert import convert_checkpoint

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.safetensors")
        _make_fake_checkpoint(src)
        out = convert_checkpoint(src)
        arrays = dict(mx.load(out))
        # Linear [out, in] — no transposition
        assert arrays["flow.transformer_blocks.0.attn.q_proj.weight"].shape == (128, 128)


def test_convert_custom_dst_path():
    from fluxflow.mlx.convert import convert_checkpoint

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.safetensors")
        dst = os.path.join(d, "model_mlx.npz")
        _make_fake_checkpoint(src)
        out = convert_checkpoint(src, dst_path=dst)
        assert out == dst
        assert os.path.exists(dst)
