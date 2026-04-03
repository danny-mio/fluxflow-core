import pytest

mx = pytest.importorskip("mlx.core")
import os  # noqa: E402
import tempfile  # noqa: E402

from safetensors.torch import save_file  # noqa: E402


def _make_tiny_checkpoint(path: str) -> None:
    """Create a minimal valid checkpoint with expander weights only."""
    from fluxflow.models.v070.vae import FluxExpander

    exp = FluxExpander(d_model=32, upscales=1, max_hw=64)
    state = {f"expander.{k}": v for k, v in exp.state_dict().items()}
    save_file(state, path)


def test_pipeline_from_checkpoint_loads():
    """FluxFlowPipelineMLX.from_checkpoint must not raise."""
    from fluxflow.mlx.pipeline import FluxFlowPipelineMLX

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.safetensors")
        _make_tiny_checkpoint(ckpt)
        pipe = FluxFlowPipelineMLX.from_checkpoint(ckpt, d_model=32, upscales=1, max_hw=64)
        assert pipe is not None


def test_pipeline_decode_output_shape():
    """FluxFlowPipelineMLX.decode must return [B, 3, H, W]."""
    from fluxflow.mlx.pipeline import FluxFlowPipelineMLX

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.safetensors")
        _make_tiny_checkpoint(ckpt)
        pipe = FluxFlowPipelineMLX.from_checkpoint(ckpt, d_model=32, upscales=1, max_hw=64)

        # Build a minimal packed tensor: [1, T+1, D] with h=2, w=2
        h, w = 2, 2
        T = h * w
        D = 32 + 5  # d_model + CONTEXT_DIMS
        img_tokens = mx.zeros((1, T, 32))
        ctx_tokens = mx.zeros((1, T, 5))
        hw_token = mx.array([[[h / 64, w / 64] + [0.0] * (D - 2)]])
        img_seq = mx.concatenate([img_tokens, ctx_tokens], axis=-1)
        packed = mx.concatenate([img_seq, hw_token], axis=1)

        out = pipe.decode(packed)
        mx.eval(out)
        assert out.shape == (1, 3, h * 2, w * 2)  # 1 upscale → 2x each dim


def test_pipeline_npz_cached():
    """from_checkpoint must reuse existing .npz and not recreate it."""
    from fluxflow.mlx.pipeline import FluxFlowPipelineMLX
    from fluxflow.mlx.convert import convert_checkpoint

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.safetensors")
        _make_tiny_checkpoint(ckpt)
        # Pre-create the .npz
        npz = convert_checkpoint(ckpt)
        mtime_before = os.path.getmtime(npz)
        # Load pipeline — should reuse existing .npz
        FluxFlowPipelineMLX.from_checkpoint(ckpt, d_model=32, upscales=1, max_hw=64)
        mtime_after = os.path.getmtime(npz)
        assert mtime_before == mtime_after, "from_checkpoint must not overwrite existing .npz"
