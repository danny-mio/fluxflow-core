"""Checkpoint round-trip tests for activation_type auto-detection and override."""

from fluxflow.models.pipeline import FluxPipeline
from fluxflow.models.v100.flow import FluxFlowProcessor_v100
from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100
from fluxflow.models.versioning import load_versioned_checkpoint, save_versioned_checkpoint


def _build_pipeline(activation_type: str) -> FluxPipeline:
    compressor = FluxCompressor_v100(d_model=32, downscales=2, activation_type=activation_type)
    flow_processor = FluxFlowProcessor_v100(
        d_model=32, vae_dim=32, embedding_size=32, n_layers=1, activation_type=activation_type
    )
    expander = FluxExpander_v100(d_model=32, upscales=2, activation_type=activation_type)
    return FluxPipeline(compressor, flow_processor, expander)


def test_detect_architecture_captures_activation_type():
    from fluxflow.models.versioning import _detect_architecture

    pipeline = _build_pipeline("pade")
    config = _detect_architecture(pipeline)
    assert config["activation_type"] == "pade"


def test_save_then_load_auto_detects_pade(tmp_path):
    pipeline = _build_pipeline("pade")
    out_dir = tmp_path / "ckpt_pade"
    save_versioned_checkpoint(pipeline, out_dir, model_version="0.10.0")

    loaded = load_versioned_checkpoint(out_dir, device="cpu")
    assert loaded.compressor.activation_type == "pade"
    assert loaded.flow_processor.activation_type == "pade"
    assert loaded.expander.activation_type == "pade"


def test_save_bezier_then_load_with_force_pade_override(tmp_path):
    pipeline = _build_pipeline("bezier")
    out_dir = tmp_path / "ckpt_bezier"
    save_versioned_checkpoint(pipeline, out_dir, model_version="0.10.0")

    loaded = load_versioned_checkpoint(out_dir, device="cpu", force_activation_type="pade")
    assert loaded.compressor.activation_type == "pade"


def test_legacy_checkpoint_without_activation_type_key_defaults_to_bezier(tmp_path):
    """Metadata saved before this feature has no 'activation_type' key -- must default to bezier."""
    pipeline = _build_pipeline("bezier")
    out_dir = tmp_path / "ckpt_legacy"
    save_versioned_checkpoint(pipeline, out_dir, model_version="0.10.0")

    import json

    metadata_path = out_dir / "model_metadata.json"
    data = json.loads(metadata_path.read_text())
    del data["architecture"]["activation_type"]
    metadata_path.write_text(json.dumps(data, indent=2))

    loaded = load_versioned_checkpoint(out_dir, device="cpu")
    assert loaded.compressor.activation_type == "bezier"
