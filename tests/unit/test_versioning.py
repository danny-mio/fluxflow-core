"""Unit tests for model versioning system."""

import json

import pytest
import torch

from fluxflow.models.versioning import (
    ModelMetadata,
    ModelVersionRegistry,
    _detect_architecture,
    _detect_components,
    load_versioned_checkpoint,
    save_versioned_checkpoint,
)


class TestModelMetadata:
    """Tests for ModelMetadata class."""

    def test_create_metadata(self):
        """Should create metadata with required fields."""
        metadata = ModelMetadata(
            model_version="0.3.0",
            library_version="0.3.1",
            architecture={"vae_dim": 128},
            components={"compressor": "FluxCompressor"},
        )

        assert metadata.model_version == "0.3.0"
        assert metadata.architecture["vae_dim"] == 128

    def test_save_and_load_metadata(self, tmp_path):
        """Should save and load metadata correctly."""
        metadata = ModelMetadata(
            model_version="0.3.0",
            library_version="0.3.1",
            architecture={"vae_dim": 128, "flow_dim": 128},
            components={"compressor": "FluxCompressor"},
            training_info={"steps": 1000},
        )

        path = tmp_path / "metadata.json"
        metadata.save(path)

        loaded = ModelMetadata.load(path)
        assert loaded.model_version == "0.3.0"
        assert loaded.architecture["vae_dim"] == 128
        assert loaded.training_info["steps"] == 1000

    def test_metadata_to_dict(self):
        """Should convert to dict correctly."""
        metadata = ModelMetadata(
            model_version="0.3.0",
            library_version="0.3.1",
            architecture={},
            components={},
        )

        data = metadata.to_dict()
        assert "model_version" in data
        assert "architecture" in data
        assert data["model_version"] == "0.3.0"

    def test_metadata_from_dict(self):
        """Should load from dict correctly."""
        data = {
            "model_version": "0.3.0",
            "library_version": "0.3.1",
            "architecture": {"vae_dim": 64},
            "components": {"compressor": "FluxCompressor"},
            "training_info": {"steps": 500},
            "checksum": {"algorithm": "sha256"},
        }

        metadata = ModelMetadata.from_dict(data)
        assert metadata.model_version == "0.3.0"
        assert metadata.architecture["vae_dim"] == 64
        assert metadata.training_info["steps"] == 500


class TestModelVersionRegistry:
    """Tests for ModelVersionRegistry."""

    def test_register_loader(self):
        """Should register version loader."""
        assert "0.3.0" in ModelVersionRegistry.list_versions()

    def test_get_loader_exact_match(self):
        """Should get loader for exact version."""
        loader = ModelVersionRegistry.get_loader("0.3.0")
        assert loader is not None
        assert loader.VERSION == "0.3.0"

    def test_get_loader_compatible_version(self):
        """Should get loader for compatible version."""
        loader = ModelVersionRegistry.get_loader("0.3.1")
        assert loader is not None
        # 0.3.1 should use 0.3.0 loader

    def test_get_loader_unknown_version(self):
        """Should return None for unknown version."""
        loader = ModelVersionRegistry.get_loader("99.99.99")
        assert loader is None

    def test_list_versions(self):
        """Should list all registered versions."""
        versions = ModelVersionRegistry.list_versions()
        assert isinstance(versions, list)
        assert "0.3.0" in versions
        assert "0.2.0" in versions  # Legacy loader

    def test_v080_registered(self):
        """v0.8.0 should appear in list_versions."""
        # Trigger auto-discovery (factory import ensures v080 is loaded)
        import fluxflow.models.factory  # noqa: F401

        versions = ModelVersionRegistry.list_versions()
        assert "0.8.0" in versions

    def test_get_loader_v080(self):
        """Should return ModelLoaderV08 for version 0.8.0."""
        import fluxflow.models.factory  # noqa: F401

        loader = ModelVersionRegistry.get_loader("0.8.0")
        assert loader is not None
        assert loader.VERSION == "0.8.0"

    def test_get_loader_v080_compatible(self):
        """Compatible patch versions (0.8.1) should resolve to ModelLoaderV08."""
        import fluxflow.models.factory  # noqa: F401

        loader = ModelVersionRegistry.get_loader("0.8.1")
        assert loader is not None


class TestDetectionFunctions:
    """Tests for architecture and component detection."""

    def test_detect_architecture(self, simple_pipeline):
        """Should detect architecture from model."""
        arch = _detect_architecture(simple_pipeline)

        assert "vae_dim" in arch
        assert "flow_dim" in arch
        assert "downscales" in arch
        assert arch["vae_dim"] == 32  # From fixture

    def test_detect_components(self, simple_pipeline):
        """Should detect component names."""
        components = _detect_components(simple_pipeline)

        assert "compressor" in components
        assert "flow_processor" in components
        assert "expander" in components
        assert components["compressor"] == "FluxCompressor"


class TestVersionedSaveAndLoad:
    """Integration tests for save/load with versioning."""

    def test_save_and_load_roundtrip(self, tmp_path, simple_pipeline):
        """Should maintain model through save/load cycle."""
        output_dir = tmp_path / "versioned_model"

        # Save
        save_versioned_checkpoint(
            simple_pipeline,
            output_dir,
            model_version="0.3.0",
            training_info={"steps": 5000},
        )

        # Check files exist
        assert (output_dir / "model.safetensors").exists()
        assert (output_dir / "model_metadata.json").exists()

        # Load metadata and verify
        with open(output_dir / "model_metadata.json") as f:
            metadata = json.load(f)
        assert metadata["model_version"] == "0.3.0"
        assert metadata["training_info"]["steps"] == 5000

        # Load model
        loaded = load_versioned_checkpoint(output_dir, device="cpu")

        # Check weights match (sample check)
        orig_param = next(simple_pipeline.parameters())
        loaded_param = next(loaded.parameters())
        assert torch.allclose(orig_param.cpu(), loaded_param.cpu())

    def test_save_with_auto_detection(self, tmp_path, simple_pipeline):
        """Should auto-detect architecture when saving."""
        output_dir = tmp_path / "auto_model"

        save_versioned_checkpoint(
            simple_pipeline,
            output_dir,
            model_version="0.3.0",
        )

        # Load metadata and verify architecture was detected
        with open(output_dir / "model_metadata.json") as f:
            metadata = json.load(f)

        assert "vae_dim" in metadata["architecture"]
        assert "flow_dim" in metadata["architecture"]


class TestDetectArchitectureVersion:
    """Tests for detect_architecture_version() -- v0.10.0 misdetection regression guard."""

    def test_v100_not_misdetected_as_v070(self):
        """v0.10.0 checkpoints must not be classified as v0.7.0 (reused submodule names)."""
        from fluxflow.models.pipeline import detect_architecture_version
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100

        flow = FluxFlowProcessor_v100(d_model=32, vae_dim=16, n_head=4, n_layers=1)
        keys = [f"flow_processor.{k}" for k in flow.state_dict().keys()]
        assert detect_architecture_version(keys) == "0.10.0"

    def test_v070_still_v070(self):
        from fluxflow.models.pipeline import detect_architecture_version
        from fluxflow.models.v070.flow import FluxFlowProcessor

        flow = FluxFlowProcessor(d_model=32, vae_dim=16, n_head=4, n_layers=1)
        keys = [f"flow_processor.{k}" for k in flow.state_dict().keys()]
        assert detect_architecture_version(keys) == "0.7.0"

    def test_no_markers_defaults_v030(self):
        from fluxflow.models.pipeline import detect_architecture_version

        assert detect_architecture_version(["flow_processor.some_other_layer.weight"]) == "0.3.0"


class TestLegacyLoaderRoutesV100:
    """Regression guard: v0.10.0 checkpoints must load via v100 classes, not a v070 shell."""

    def test_legacy_loader_routes_v100_to_v010_loader(self, tmp_path):
        import safetensors.torch

        from fluxflow.models.pipeline import FluxPipeline
        from fluxflow.models.v100.flow import FluxFlowProcessor_v100
        from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

        comp = FluxCompressor_v100(d_model=16, downscales=2)
        flow = FluxFlowProcessor_v100(d_model=32, vae_dim=16, n_head=4, n_layers=1)
        exp = FluxExpander_v100(d_model=16, upscales=2)
        pipeline = FluxPipeline(comp, flow, exp)

        state = {f"diffuser.{k}": v for k, v in pipeline.state_dict().items()}
        ckpt_path = tmp_path / "flxflow_final.safetensors"
        safetensors.torch.save_file(state, str(ckpt_path))

        loaded = load_versioned_checkpoint(ckpt_path, device="cpu")

        # Proves it was reconstructed as a v100 model, not a v070 shell with
        # most weights silently skipped by strict=False.
        assert hasattr(loaded.flow_processor, "ctx_gate_proj")
        assert hasattr(loaded.flow_processor, "time_mlp")

        # Sharper guard: every weight must load into a freshly-built model
        # with the same shapes -- zero missing/unexpected keys.
        fresh = FluxPipeline(
            FluxCompressor_v100(d_model=16, downscales=2),
            FluxFlowProcessor_v100(d_model=32, vae_dim=16, n_head=4, n_layers=1),
            FluxExpander_v100(d_model=16, upscales=2),
        )
        fresh.load_state_dict(loaded.state_dict(), strict=True)


# Fixtures


@pytest.fixture
def simple_pipeline():
    """Create minimal FluxPipeline for testing."""
    from fluxflow.models.flow import FluxFlowProcessor
    from fluxflow.models.pipeline import FluxPipeline
    from fluxflow.models.vae import FluxCompressor, FluxExpander

    compressor = FluxCompressor(d_model=32, downscales=2, max_hw=256, attn_layers=1)
    flow = FluxFlowProcessor(d_model=32, vae_dim=32, n_layers=2, n_head=4, max_hw=256)
    expander = FluxExpander(d_model=32, upscales=2, max_hw=256)

    return FluxPipeline(compressor, flow, expander)
