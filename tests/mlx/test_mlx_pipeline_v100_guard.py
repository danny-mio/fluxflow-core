import pytest

mx = pytest.importorskip("mlx.core")
import os  # noqa: E402
import tempfile  # noqa: E402

from fluxflow.exceptions import ModelArchitectureError  # noqa: E402


def test_from_checkpoint_rejects_v100_only_beta_low_key(monkeypatch):
    """from_checkpoint must raise before load_weights when a v0.10.0-only
    SPADE_v100b marker key (beta_low) is present in the loaded array dict."""
    from fluxflow.mlx import pipeline as pipeline_mod

    fake_arrays = {
        "expander.decoder.spade.beta_low.weight": mx.zeros((1,)),
        "expander.rgb_conv1.conv.weight": mx.zeros((1,)),
    }

    # Skip the real safetensors->npz conversion and hand back our fake dict
    # from the spot the real `mx.load(npz_path)` call would normally fill.
    monkeypatch.setattr(pipeline_mod, "convert_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod.mx, "load", lambda path: fake_arrays)

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.safetensors")
        with open(ckpt, "wb") as f:
            f.write(b"fake")

        with pytest.raises(ModelArchitectureError, match="beta_low"):
            pipeline_mod.FluxFlowPipelineMLX.from_checkpoint(
                ckpt, d_model=32, upscales=1, max_hw=64
            )


def test_from_checkpoint_rejects_v100_only_gamma_head_key(monkeypatch):
    """Same guard, triggered by a gamma_head marker instead of beta_low."""
    from fluxflow.mlx import pipeline as pipeline_mod

    fake_arrays = {
        "expander.upscale.layers.0.spade.gamma_head.weight": mx.zeros((1,)),
    }

    monkeypatch.setattr(pipeline_mod, "convert_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod.mx, "load", lambda path: fake_arrays)

    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.safetensors")
        with open(ckpt, "wb") as f:
            f.write(b"fake")

        with pytest.raises(ModelArchitectureError, match="v0.10.0"):
            pipeline_mod.FluxFlowPipelineMLX.from_checkpoint(
                ckpt, d_model=32, upscales=1, max_hw=64
            )


def test_guard_helper_passes_clean_dict():
    """Sanity check: the guard helper itself does not raise on marker-free keys."""
    from fluxflow.mlx.pipeline import _check_not_v100_only_keys

    clean_arrays = {
        "expander.rgb_conv1.conv.weight": mx.zeros((1,)),
        "expander.upscale.layers.0.spade.mlp_beta.weight": mx.zeros((1,)),
    }
    _check_not_v100_only_keys(clean_arrays)  # must not raise
