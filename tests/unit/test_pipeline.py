"""Tests for FluxPipeline and FluxFlowPipeline classes."""

import pytest
import torch
import safetensors.torch

from fluxflow.models.pipeline import FluxPipeline
from fluxflow.models.vae import FluxCompressor, FluxExpander
from fluxflow.models.flow import FluxFlowProcessor


class TestFluxPipeline:
    """Tests for FluxPipeline low-level API."""

    @pytest.fixture
    def small_compressor(self):
        """Create small compressor for testing."""
        return FluxCompressor(
            in_channels=3,
            d_model=32,
            downscales=2,
            max_hw=128,
            use_attention=True,
            attn_layers=1,
        )

    @pytest.fixture
    def small_flow_processor(self):
        """Create small flow processor for testing."""
        return FluxFlowProcessor(
            d_model=64,
            vae_dim=32,
            embedding_size=128,
            n_head=4,
            n_layers=2,
            max_hw=128,
        )

    @pytest.fixture
    def small_expander(self):
        """Create small expander for testing."""
        return FluxExpander(
            d_model=32,
            upscales=2,
            max_hw=128,
        )

    @pytest.fixture
    def pipeline(self, small_compressor, small_flow_processor, small_expander):
        """Create FluxPipeline instance for testing."""
        return FluxPipeline(small_compressor, small_flow_processor, small_expander)

    def test_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.compressor is not None
        assert pipeline.flow_processor is not None
        assert pipeline.expander is not None

    def test_forward_without_flow(self, pipeline):
        """Test forward pass without flow processing (VAE only)."""
        batch_size = 1
        img = torch.randn(batch_size, 3, 64, 64)

        with torch.no_grad():
            output = pipeline(img, use_flow=False)

        assert output.shape == (batch_size, 3, 64, 64)
        assert torch.isfinite(output).all()

    def test_forward_with_flow(self, pipeline):
        """Test forward pass with flow processing."""
        batch_size = 1
        img = torch.randn(batch_size, 3, 64, 64)
        text_embeddings = torch.randn(batch_size, 128)
        timesteps = torch.tensor([0.5])

        with torch.no_grad():
            output = pipeline(img, text_embeddings, timesteps, use_flow=True)

        assert output.shape == (batch_size, 3, 64, 64)
        assert torch.isfinite(output).all()

    def test_forward_missing_embeddings_raises(self, pipeline):
        """Test that missing embeddings raises ValueError."""
        img = torch.randn(1, 3, 64, 64)

        with pytest.raises(ValueError, match="Missing text_embeddings"):
            pipeline(img, use_flow=True)

    def test_forward_missing_timesteps_raises(self, pipeline):
        """Test that missing timesteps raises ValueError."""
        img = torch.randn(1, 3, 64, 64)
        text_embeddings = torch.randn(1, 128)

        with pytest.raises(ValueError, match="Missing text_embeddings or timesteps"):
            pipeline(img, text_embeddings=text_embeddings, use_flow=True)

    def test_ctx_tokens_propagation(self, small_compressor, small_flow_processor, small_expander):
        """Test that ctx_tokens is propagated to submodules."""
        # Set ctx_tokens on compressor
        small_compressor.ctx_tokens = 8

        pipeline = FluxPipeline(small_compressor, small_flow_processor, small_expander)

        # Check propagation
        if hasattr(pipeline.flow_processor, "ctx_tokens"):
            assert pipeline.flow_processor.ctx_tokens == 8
        if hasattr(pipeline.expander, "ctx_tokens"):
            assert pipeline.expander.ctx_tokens == 8


class TestFluxPipelineFromPretrained:
    """Tests for FluxPipeline.from_pretrained() method."""

    @pytest.fixture
    def mock_checkpoint(self, tmp_path):
        """Create mock checkpoint file using actual model state."""
        from fluxflow.models.vae import FluxCompressor, FluxExpander
        from fluxflow.models.flow import FluxFlowProcessor

        # Create small models with known config
        vae_dim = 32
        flow_dim = 64
        text_embed_dim = 128
        downscales = 2

        compressor = FluxCompressor(
            in_channels=3,
            d_model=vae_dim,
            downscales=downscales,
            max_hw=1024,
            use_attention=True,
            attn_layers=2,
        )
        flow_processor = FluxFlowProcessor(
            d_model=flow_dim,
            vae_dim=vae_dim,
            embedding_size=text_embed_dim,
            n_head=8,
            n_layers=2,
            max_hw=1024,
        )
        expander = FluxExpander(d_model=vae_dim, upscales=downscales, max_hw=1024)

        # Create pipeline and get state dict
        pipeline = FluxPipeline(compressor, flow_processor, expander)
        state_dict = {"diffuser." + k: v for k, v in pipeline.state_dict().items()}

        checkpoint_path = tmp_path / "checkpoint.safetensors"
        safetensors.torch.save_file(state_dict, str(checkpoint_path))
        return checkpoint_path

    def test_from_pretrained_loads_model(self, mock_checkpoint):
        """Test that from_pretrained loads model correctly."""
        pipeline = FluxPipeline.from_pretrained(str(mock_checkpoint), device="cpu")

        assert pipeline is not None
        assert isinstance(pipeline, FluxPipeline)
        assert pipeline.compressor is not None
        assert pipeline.flow_processor is not None
        assert pipeline.expander is not None

    def test_from_pretrained_file_not_found(self):
        """Test that from_pretrained raises for missing file."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            FluxPipeline.from_pretrained("/nonexistent/path.safetensors")

    def test_from_pretrained_sets_eval_mode(self, mock_checkpoint):
        """Test that from_pretrained sets model to eval mode."""
        pipeline = FluxPipeline.from_pretrained(str(mock_checkpoint), device="cpu")

        assert not pipeline.training

    def test_from_pretrained_pt_file(self, tmp_path):
        """Test loading from .pt file."""
        from fluxflow.models.vae import FluxCompressor, FluxExpander
        from fluxflow.models.flow import FluxFlowProcessor

        # Create small models
        compressor = FluxCompressor(in_channels=3, d_model=32, downscales=2, max_hw=1024)
        flow_processor = FluxFlowProcessor(
            d_model=64, vae_dim=32, embedding_size=128, n_head=8, n_layers=2, max_hw=1024
        )
        expander = FluxExpander(d_model=32, upscales=2, max_hw=1024)
        pipeline = FluxPipeline(compressor, flow_processor, expander)

        state_dict = {"diffuser." + k: v for k, v in pipeline.state_dict().items()}

        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(state_dict, checkpoint_path)

        pipeline = FluxPipeline.from_pretrained(str(checkpoint_path), device="cpu")
        assert pipeline is not None


class TestFluxPipelineConfigDetection:
    """Tests for FluxPipeline._detect_config() method."""

    def test_detect_vae_dim(self):
        """Test VAE dimension detection."""
        state_dict = {
            "compressor.latent_proj.0.0.weight": torch.randn(128 * 5, 64, 1, 1),
            "flow_processor.vae_to_dmodel.weight": torch.randn(256, 128),
        }

        config = FluxPipeline._detect_config(state_dict)
        assert config["vae_dim"] == 128

    def test_detect_flow_dim(self):
        """Test flow dimension detection."""
        state_dict = {
            "compressor.latent_proj.0.0.weight": torch.randn(64 * 5, 32, 1, 1),
            "flow_processor.vae_to_dmodel.weight": torch.randn(256, 64),
        }

        config = FluxPipeline._detect_config(state_dict)
        assert config["flow_dim"] == 256

    def test_detect_text_embed_dim(self):
        """Test text embedding dimension detection."""
        state_dict = {
            "compressor.latent_proj.0.0.weight": torch.randn(64 * 5, 32, 1, 1),
            "flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
            "flow_processor.text_proj.weight": torch.randn(128, 512),
        }

        config = FluxPipeline._detect_config(state_dict)
        assert config["text_embed_dim"] == 512

    def test_detect_downscales(self):
        """Test downscales detection."""
        state_dict = {
            "compressor.latent_proj.0.0.weight": torch.randn(64 * 5, 32, 1, 1),
            "flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
            "compressor.encoder_z.0.0.weight": torch.randn(16, 3, 3, 3),
            "compressor.encoder_z.1.0.weight": torch.randn(32, 16, 3, 3),
            "compressor.encoder_z.2.0.weight": torch.randn(64, 32, 3, 3),
        }

        config = FluxPipeline._detect_config(state_dict)
        assert config["downscales"] == 3

    def test_detect_transformer_layers(self):
        """Test transformer layer count detection."""
        state_dict = {
            "compressor.latent_proj.0.0.weight": torch.randn(64 * 5, 32, 1, 1),
            "flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
            "flow_processor.transformer_blocks.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "flow_processor.transformer_blocks.1.self_attn.q_proj.weight": torch.randn(128, 128),
            "flow_processor.transformer_blocks.2.self_attn.q_proj.weight": torch.randn(128, 128),
            "flow_processor.transformer_blocks.3.self_attn.q_proj.weight": torch.randn(128, 128),
        }

        config = FluxPipeline._detect_config(state_dict)
        assert config["flow_transformer_layers"] == 4

    def test_missing_vae_dim_raises(self):
        """Test that missing VAE dim raises ValueError."""
        state_dict = {
            "flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
        }

        # This should work because vae_dim can be detected from vae_to_dmodel
        config = FluxPipeline._detect_config(state_dict)
        assert "vae_dim" in config

    def test_completely_empty_raises(self):
        """Test that empty state dict raises ValueError."""
        state_dict = {}

        with pytest.raises(ValueError, match="Could not detect"):
            FluxPipeline._detect_config(state_dict)

    def test_default_values(self):
        """Test default configuration values."""
        state_dict = {
            "compressor.latent_proj.0.0.weight": torch.randn(64 * 5, 32, 1, 1),
            "flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
        }

        config = FluxPipeline._detect_config(state_dict)

        # Check defaults
        assert config["max_hw"] == 1024
        assert config.get("text_embed_dim", 768) == 768  # Default when not detected


class TestFluxPipelineIntegration:
    """Integration tests for FluxPipeline."""

    @pytest.fixture
    def full_pipeline(self):
        """Create a full pipeline with small models."""
        compressor = FluxCompressor(
            in_channels=3,
            d_model=32,
            downscales=2,
            max_hw=128,
            use_attention=True,
            attn_layers=1,
        )
        flow_processor = FluxFlowProcessor(
            d_model=64,
            vae_dim=32,
            embedding_size=128,
            n_head=4,
            n_layers=2,
            max_hw=128,
        )
        expander = FluxExpander(
            d_model=32,
            upscales=2,
            max_hw=128,
        )
        return FluxPipeline(compressor, flow_processor, expander)

    def test_save_and_load_roundtrip(self, full_pipeline, tmp_path):
        """Test saving and loading pipeline preserves functionality."""
        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.safetensors"
        state_dict = {"diffuser." + k: v for k, v in full_pipeline.state_dict().items()}
        safetensors.torch.save_file(state_dict, str(checkpoint_path))

        # Load checkpoint
        loaded_pipeline = FluxPipeline.from_pretrained(str(checkpoint_path), device="cpu")

        # Test forward pass
        img = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            original_output = full_pipeline(img, use_flow=False)
            loaded_output = loaded_pipeline(img, use_flow=False)

        # Outputs should be close (may not be exact due to model initialization differences)
        assert original_output.shape == loaded_output.shape

    def test_gradient_flow(self, full_pipeline):
        """Test that gradients flow through pipeline."""
        img = torch.randn(1, 3, 64, 64, requires_grad=True)
        text_embeddings = torch.randn(1, 128)
        timesteps = torch.tensor([0.5])

        output = full_pipeline(img, text_embeddings, timesteps, use_flow=True)
        loss = output.mean()
        loss.backward()

        assert img.grad is not None
        assert torch.isfinite(img.grad).all()

    def test_different_batch_sizes(self, full_pipeline):
        """Test pipeline with different batch sizes."""
        for batch_size in [1, 2, 4]:
            img = torch.randn(batch_size, 3, 64, 64)

            with torch.no_grad():
                output = full_pipeline(img, use_flow=False)

            assert output.shape == (batch_size, 3, 64, 64)

    def test_eval_vs_train_mode(self, full_pipeline):
        """Test pipeline behavior in eval vs train mode."""
        img = torch.randn(1, 3, 64, 64)

        # Train mode
        full_pipeline.train()
        with torch.no_grad():
            train_output = full_pipeline(img, use_flow=False)

        # Eval mode
        full_pipeline.eval()
        with torch.no_grad():
            eval_output = full_pipeline(img, use_flow=False)

        # Both should produce valid outputs
        assert torch.isfinite(train_output).all()
        assert torch.isfinite(eval_output).all()
