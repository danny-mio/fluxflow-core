"""Shared pytest fixtures for FluxFlow tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# ============================================================================
# Device and Environment Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def device():
    """Get test device (CPU for CI, CUDA if available locally)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device for tests that should run on CPU."""
    return torch.device("cpu")


# ============================================================================
# Tensor Fixtures
# ============================================================================


@pytest.fixture
def random_tensor_2d():
    """Create random 2D tensor [B, D]."""

    def _make(batch_size=4, dim=128, device="cpu"):
        return torch.randn(batch_size, dim, device=device)

    return _make


@pytest.fixture
def random_tensor_3d():
    """Create random 3D tensor [B, S, D]."""

    def _make(batch_size=4, seq_len=16, dim=128, device="cpu"):
        return torch.randn(batch_size, seq_len, dim, device=device)

    return _make


@pytest.fixture
def random_tensor_4d():
    """Create random 4D tensor [B, C, H, W]."""

    def _make(batch_size=2, channels=3, height=64, width=64, device="cpu"):
        return torch.randn(batch_size, channels, height, width, device=device)

    return _make


@pytest.fixture
def random_image_tensor():
    """Create normalized random image tensor [-1, 1]."""

    def _make(batch_size=2, channels=3, height=64, width=64, device="cpu"):
        return torch.rand(batch_size, channels, height, width, device=device) * 2 - 1

    return _make


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temporary directory that's cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create temporary output directory for checkpoints."""
    output_dir = temp_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# Mock Dataset Fixtures
# ============================================================================


@pytest.fixture
def mock_image_dataset(temp_dir):
    """Create mock image dataset with captions."""
    data_dir = temp_dir / "images"
    data_dir.mkdir(exist_ok=True)

    # Create 10 mock images
    image_paths = []
    captions = []

    for i in range(10):
        # Create random colored image
        img = Image.new(
            "RGB",
            (128, 128),
            color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
        )
        img_path = data_dir / f"image_{i:03d}.jpg"
        img.save(img_path)
        image_paths.append(img_path.name)
        captions.append(f"TEST IMAGE NUMBER {i}")

    # Create captions file
    captions_file = temp_dir / "captions.txt"
    with open(captions_file, "w") as f:
        for img_name, caption in zip(image_paths, captions):
            f.write(f"{img_name}\t{caption}\n")

    return {
        "data_dir": str(data_dir),
        "captions_file": str(captions_file),
        "num_images": len(image_paths),
        "image_paths": image_paths,
        "captions": captions,
    }


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing without HuggingFace downloads."""
    mock = MagicMock()

    # Mock encode_plus to return proper dict with tensors
    def mock_encode_plus(text, **kwargs):
        max_length = kwargs.get("max_length", 128)
        return {
            "input_ids": torch.randint(0, 30522, (1, max_length)),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }

    # Mock __call__ as well for other uses
    def mock_call(text, **kwargs):
        max_length = kwargs.get("max_length", 128)
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
        return {
            "input_ids": torch.randint(0, 30522, (batch_size, max_length)),
            "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
        }

    mock.encode_plus = mock_encode_plus
    mock.__call__ = mock_call
    mock.model_max_length = 128
    mock.pad_token_id = 0
    mock.pad_token = "[PAD]"
    mock.eos_token = "[EOS]"
    mock.add_special_tokens = MagicMock()
    return mock


@pytest.fixture
def mock_dimension_cache():
    """Create mock dimension cache data."""
    return {
        "dataset_path": "/fake/path",
        "captions_file": "/fake/captions.txt",
        "scan_date": "2025-01-01T00:00:00",
        "total_images": 100,
        "multiple": 32,
        "size_groups": {
            "(512, 512)": {"indices": list(range(0, 50)), "count": 50},
            "(768, 512)": {"indices": list(range(50, 75)), "count": 25},
            "(512, 768)": {"indices": list(range(75, 100)), "count": 25},
        },
        "statistics": {
            "num_groups": 3,
            "min_group_size": 25,
            "max_group_size": 50,
            "avg_group_size": 33,
        },
    }


# ============================================================================
# Mock Model Fixtures
# ============================================================================


@pytest.fixture
def mock_vae_config():
    """Small VAE configuration for testing."""
    return {
        "image_channels": 3,
        "d_model": 64,  # Reduced for testing
        "downscales": 2,  # Reduced for testing
        "vae_attn_layers": 2,
        "max_hw": 512,
    }


@pytest.fixture
def mock_flow_config():
    """Small flow model configuration for testing."""
    return {
        "d_model": 128,  # Reduced for testing
        "vae_dim": 64,
        "text_embed_dim": 256,
        "n_head": 4,
        "depth": 2,  # Reduced for testing
        "max_seq_len": 256,
    }


@pytest.fixture
def mock_discriminator_config():
    """Small discriminator configuration for testing."""
    return {
        "channels": 3,
        "base_channels": 32,  # Reduced for testing
        "num_layers": 3,
    }


# ============================================================================
# Mock Checkpoint Fixtures
# ============================================================================


@pytest.fixture
def mock_checkpoint_state():
    """Create mock model state dict for testing."""
    return {
        "diffuser.compressor.latent_proj.0.0.weight": torch.randn(64 * 5, 32, 1, 1),
        "diffuser.flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
        "diffuser.flow_processor.text_proj.weight": torch.randn(128, 256),
        "diffuser.compressor.encoder_z.0.0.weight": torch.randn(32, 3, 3, 3),
        "diffuser.compressor.encoder_z.1.0.weight": torch.randn(64, 32, 3, 3),
        "diffuser.expander.upscale.layers.0.conv1.0.weight": torch.randn(32, 64, 3, 3),
        "diffuser.expander.upscale.layers.1.conv1.0.weight": torch.randn(16, 32, 3, 3),
        "diffuser.compressor.token_attn.0.attn.in_proj_weight": torch.randn(192, 64),
        "diffuser.compressor.token_attn.1.attn.in_proj_weight": torch.randn(192, 64),
        "diffuser.flow_processor.transformer_blocks.0.self_attn.q_proj.weight": torch.randn(
            128, 128
        ),
        "diffuser.flow_processor.transformer_blocks.1.self_attn.q_proj.weight": torch.randn(
            128, 128
        ),
        "diffuser.flow_processor.transformer_blocks.0.rotary_pe.inv_freq": torch.randn(16),
        "text_encoder.ouput_layer.weight": torch.randn(256, 512),
    }


@pytest.fixture
def mock_training_state():
    """Create mock training state for testing."""
    return {
        "version": "1.0",
        "timestamp": "2025-01-01T00:00:00",
        "epoch": 5,
        "batch_idx": 42,
        "global_step": 1234,
        "samples_trained": 50000,
        "total_samples": 100000,
        "learning_rates": {
            "vae": 1e-4,
            "flow": 5e-5,
            "discriminator": 1e-4,
        },
        "sampler_state": {
            "seed": 12345,
            "position": 42,
            "current_epoch": 5,
            "batch_size": 8,
        },
    }


# ============================================================================
# Test Data Helpers
# ============================================================================


@pytest.fixture
def assert_shape():
    """Helper to assert tensor shapes."""

    def _assert(tensor, expected_shape, msg=""):
        actual = tuple(tensor.shape)
        assert (
            actual == expected_shape
        ), f"{msg}\nExpected shape: {expected_shape}\nActual shape: {actual}"

    return _assert


@pytest.fixture
def assert_close():
    """Helper to assert tensors are close."""

    def _assert(tensor1, tensor2, rtol=1e-5, atol=1e-8, msg=""):
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), (
            f"{msg}\nTensors not close:\n" f"Max diff: {(tensor1 - tensor2).abs().max().item()}"
        )

    return _assert


@pytest.fixture
def assert_finite():
    """Helper to assert tensor is finite (no NaN/Inf)."""

    def _assert(tensor, msg=""):
        assert torch.isfinite(tensor).all(), (
            f"{msg}\nTensor contains NaN or Inf values:\n"
            f"NaN count: {torch.isnan(tensor).sum().item()}\n"
            f"Inf count: {torch.isinf(tensor).sum().item()}"
        )

    return _assert
