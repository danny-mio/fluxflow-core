"""
Tests for config integration with model factory.

Validates that models can be created from config files with model_type parameter.
"""

import pytest
import torch

from fluxflow.config import ModelConfig, FluxFlowConfig
from fluxflow.models.factory import create_models_from_config


class TestConfigIntegration:
    """Test config integration with factory."""

    def test_bezier_model_from_config(self):
        """Test creating Bezier models from config."""
        config = ModelConfig(
            model_type="bezier",
            vae_dim=128,
            feature_maps_dim=512,
            text_embedding_dim=1024,
        )

        vae_enc, vae_dec, flow, text_enc = create_models_from_config(config)

        assert vae_enc is not None
        assert vae_dec is not None
        assert flow is not None
        assert text_enc is not None

        # Verify it's using Bezier components (check for BezierActivation)
        from fluxflow.models.vae import FluxExpander
        from fluxflow.models.flow import FluxFlowProcessor

        assert isinstance(vae_dec, FluxExpander)
        assert isinstance(flow, FluxFlowProcessor)

    def test_baseline_model_from_config(self):
        """Test creating Baseline models from config."""
        config = ModelConfig(
            model_type="baseline",
            vae_dim=128,
            feature_maps_dim=512,
            text_embedding_dim=1024,
            baseline_activation="silu",
            baseline_vae_width_mult=4.5,
            baseline_vae_depth_mult=1.0,
            baseline_flow_blocks=17,
        )

        vae_enc, vae_dec, flow, text_enc = create_models_from_config(config)

        assert vae_enc is not None
        assert vae_dec is not None
        assert flow is not None
        assert text_enc is not None

        # Verify it's using Baseline components
        from fluxflow.models.vae import BaselineFluxExpander
        from fluxflow.models.flow import BaselineFluxFlowProcessor

        assert isinstance(vae_dec, BaselineFluxExpander)
        assert isinstance(flow, BaselineFluxFlowProcessor)

    def test_baseline_default_params(self):
        """Test that baseline defaults are correct."""
        config = ModelConfig(model_type="baseline")

        assert config.baseline_activation == "silu"
        assert config.baseline_vae_width_mult == 4.5
        assert config.baseline_vae_depth_mult == 1.0
        assert config.baseline_flow_blocks == 17
        assert config.baseline_flow_ffn_expansion == 4.0

    def test_baseline_gelu_activation(self):
        """Test baseline with GELU activation."""
        config = ModelConfig(
            model_type="baseline",
            baseline_activation="gelu",
            vae_dim=128,
            feature_maps_dim=512,
        )

        vae_enc, vae_dec, flow, text_enc = create_models_from_config(config)

        # Should create successfully with GELU
        assert vae_dec is not None
        assert flow is not None

    def test_invalid_model_type_raises(self):
        """Test that invalid model_type raises error."""
        config = ModelConfig()
        config.model_type = "invalid"  # type: ignore

        with pytest.raises(ValueError, match="Unknown model_type"):
            create_models_from_config(config)

    def test_baseline_text_encoder_shared(self):
        """Test that text encoder is the same for both model types."""
        bezier_config = ModelConfig(model_type="bezier", vae_dim=128, text_embedding_dim=1024)
        baseline_config = ModelConfig(model_type="baseline", vae_dim=128, text_embedding_dim=1024)

        _, _, _, bezier_text = create_models_from_config(bezier_config)
        _, _, _, baseline_text = create_models_from_config(baseline_config)

        # Both should use BertTextEncoder
        from fluxflow.models.encoders import BertTextEncoder

        assert isinstance(bezier_text, BertTextEncoder)
        assert isinstance(baseline_text, BertTextEncoder)
        assert isinstance(bezier_text, type(baseline_text))

    def test_baseline_latent_compatibility(self):
        """Test that baseline and bezier use same latent dimensions."""
        bezier_config = ModelConfig(model_type="bezier", vae_dim=128)
        baseline_config = ModelConfig(model_type="baseline", vae_dim=128)

        bezier_enc, _, _, _ = create_models_from_config(bezier_config)
        baseline_enc, _, _, _ = create_models_from_config(baseline_config)

        # Test with dummy input
        x = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            bezier_latent = bezier_enc(x)
            baseline_latent = baseline_enc(x)

        # Both should produce same shape
        # Note: encoder outputs mu + logvar concatenated, so channels = 2*vae_dim + 1
        assert bezier_latent.shape == baseline_latent.shape
        assert bezier_latent.shape[1] == 2 * 128 + 1  # mu + logvar + 1

    def test_full_config_yaml_structure(self):
        """Test that full FluxFlowConfig works with model_type."""
        config = FluxFlowConfig(
            model=ModelConfig(
                model_type="baseline",
                vae_dim=128,
                feature_maps_dim=512,
                baseline_activation="silu",
            )
        )

        # Verify structure
        assert config.model.model_type == "baseline"
        assert config.model.vae_dim == 128
        assert config.model.baseline_activation == "silu"

        # Should be able to create models
        models = create_models_from_config(config.model)
        assert len(models) == 4

    def test_baseline_parameter_validation(self):
        """Test that baseline parameters are validated."""
        # Should accept valid range
        config = ModelConfig(
            model_type="baseline",
            baseline_vae_width_mult=5.0,
            baseline_vae_depth_mult=2.0,
            baseline_flow_blocks=20,
        )
        assert config.baseline_vae_width_mult == 5.0

        # Invalid values should raise during Pydantic validation
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelConfig(
                model_type="baseline",
                baseline_vae_width_mult=0.5,  # Below minimum (ge=1.0)
            )

    def test_bezier_ignores_baseline_params(self):
        """Test that Bezier model ignores baseline parameters."""
        config = ModelConfig(
            model_type="bezier",
            vae_dim=128,
            # These should be ignored for Bezier
            baseline_activation="gelu",
            baseline_vae_width_mult=10.0,
        )

        # Should create Bezier models successfully
        vae_enc, vae_dec, flow, text_enc = create_models_from_config(config)

        from fluxflow.models.vae import FluxExpander

        # Should still use Bezier components
        assert isinstance(vae_dec, FluxExpander)
