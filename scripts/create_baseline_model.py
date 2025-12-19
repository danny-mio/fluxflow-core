#!/usr/bin/env python
"""
Example script: Create a baseline model from config.

This demonstrates how to:
1. Load a config file
2. Create baseline models using the factory
3. Verify the models are correctly instantiated
"""

import sys
from pathlib import Path

import torch

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fluxflow.config import FluxFlowConfig, ModelConfig
from fluxflow.models.factory import create_models_from_config


def main():
    """Create and test baseline model from config."""
    print("=" * 60)
    print("Baseline Model Creation Example")
    print("=" * 60)

    # Option 1: Create config programmatically
    print("\n1. Creating baseline config programmatically...")
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

    print(f"   Model type: {config.model_type}")
    print(f"   VAE dim: {config.vae_dim}")
    print(f"   Flow d_model: {config.feature_maps_dim}")
    print(f"   Activation: {config.baseline_activation}")
    print(f"   Flow blocks: {config.baseline_flow_blocks}")

    # Create models
    print("\n2. Creating models from config...")
    vae_encoder, vae_decoder, flow, text_encoder = create_models_from_config(config)

    print(f"   ✓ VAE Encoder: {type(vae_encoder).__name__}")
    print(f"   ✓ VAE Decoder: {type(vae_decoder).__name__}")
    print(f"   ✓ Flow Processor: {type(flow).__name__}")
    print(f"   ✓ Text Encoder: {type(text_encoder).__name__}")

    # Count parameters
    print("\n3. Model statistics...")

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    vae_enc_params = count_params(vae_encoder)
    vae_dec_params = count_params(vae_decoder)
    flow_params = count_params(flow)
    text_params = count_params(text_encoder)

    print(f"   VAE Encoder params: {vae_enc_params:,}")
    print(f"   VAE Decoder params: {vae_dec_params:,}")
    print(f"   Flow params: {flow_params:,}")
    print(f"   Text Encoder params: {text_params:,}")
    print(f"   Total params: {vae_enc_params + vae_dec_params + flow_params + text_params:,}")

    # Test forward pass
    print("\n4. Testing forward pass...")
    batch_size = 1
    img_size = 256

    with torch.no_grad():
        # VAE encoder
        x = torch.randn(batch_size, 3, img_size, img_size)
        latent = vae_encoder(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Latent shape: {latent.shape}")
        print("   ✓ VAE encoder forward pass successful")

        # Text encoder
        text_ids = torch.randint(0, 1000, (batch_size, 77))
        text_emb = text_encoder(text_ids)
        print(f"   Text IDs shape: {text_ids.shape}")
        print(f"   Text embeddings shape: {text_emb.shape}")
        print("   ✓ Text encoder forward pass successful")

    print("\n   Note: Full pipeline testing (flow + decoder) requires")
    print("   proper packing format. See test_baseline_integration.py for")
    print("   complete integration tests.")

    print("\n" + "=" * 60)
    print("✓ Baseline model created and tested successfully!")
    print("=" * 60)

    # Option 2: Show how to load from YAML
    print("\n5. Loading from YAML config file...")
    print("   To use a YAML config:")
    print("   >>> from fluxflow.config import load_config")
    print("   >>> config = load_config('examples/baseline_comparison.yaml')")
    print("   >>> models = create_models_from_config(config.model)")

    # Comparison with Bezier
    print("\n6. Creating Bezier model for comparison...")
    bezier_config = ModelConfig(
        model_type="bezier",
        vae_dim=128,
        feature_maps_dim=512,
        text_embedding_dim=1024,
    )

    bez_enc, bez_dec, bez_flow, bez_text = create_models_from_config(bezier_config)

    bez_flow_params = count_params(bez_flow)
    baseline_flow_params = count_params(flow)

    print(f"   Bezier flow params: {bez_flow_params:,}")
    print(f"   Baseline flow params: {baseline_flow_params:,}")
    diff_pct = abs(bez_flow_params - baseline_flow_params) / bez_flow_params * 100
    print(f"   Parameter difference: {diff_pct:.2f}%")

    print("\n✓ All done!")


if __name__ == "__main__":
    main()
