#!/usr/bin/env python
"""
Trace VAE dimension flow from compression to expansion.
Verifies that all layer operations and kernel sizes are sound.
"""

import torch
import torch.nn as nn

from fluxflow.models.vae import FluxCompressor, FluxExpander


def trace_dimensions():
    """Trace tensor dimensions through the entire VAE pipeline."""

    print("=" * 80)
    print("VAE Dimension Flow Analysis")
    print("=" * 80)

    # Test configuration
    batch_size = 2
    original_img_size = 256  # 256x256 images
    img_size = original_img_size
    d_model = 128  # latent dimension
    downscales = 4  # 4 downsampling stages

    print(f"Input: {batch_size} × 3 × {img_size} × {img_size}")
    print(f"Latent dimension: {d_model}")
    print(f"Downsampling stages: {downscales}")
    print()

    # Initialize models
    encoder = FluxCompressor(d_model=d_model, downscales=downscales)
    decoder = FluxExpander(d_model=d_model, upscales=downscales)

    # Stage channels calculation
    input_ch = 3 + 2  # RGB + coordinate channels
    stage_channels = [
        int(round(c))
        for c in torch.linspace(input_ch, max(d_model, 8), steps=downscales + 1).tolist()
    ]
    print("Encoder stage channels:", stage_channels)

    # Input with coordinate channels
    x = torch.randn(batch_size, 3, img_size, img_size)
    print("\n1. INPUT PROCESSING")
    print(f"   Original input: {x.shape}")

    # Add coordinate channels
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, img_size, device=x.device),
        torch.linspace(-1, 1, img_size, device=x.device),
        indexing="ij",
    )
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    x_coord = torch.cat([x, coords], dim=1)
    print(f"   With coordinates: {x_coord.shape}")

    print("\n2. ENCODER PROGRESSION")

    # Encoder stages
    current = x_coord
    for i in range(downscales):
        print(f"\n   Stage {i+1}:")

        # First step: feature expansion
        print(f"     Input: {current.shape}")
        conv1 = nn.Conv2d(
            stage_channels[i],
            stage_channels[i + 1] * 5,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        current = conv1(current)
        print(
            f"     After Conv2d({stage_channels[i]}→{stage_channels[i+1]*5}, k=3, s=1, p=1): {current.shape}"
        )
        print(f"     BezierActivation (5→1 reduction)")
        current = current.view(batch_size, stage_channels[i + 1], 5, img_size, img_size).mean(
            dim=2
        )  # Simulate Bezier reduction
        print(f"     After Bezier: {current.shape}")

        # Second step: downsampling
        conv2 = nn.Conv2d(
            stage_channels[i + 1], stage_channels[i + 1] * 5, kernel_size=8, stride=2, padding=3
        )
        current = conv2(current)
        print(
            f"     After Conv2d({stage_channels[i+1]}→{stage_channels[i+1]*5}, k=8, s=2, p=3): {current.shape}"
        )
        print(f"     BezierActivation (5→1 reduction)")
        current = current.view(
            batch_size, stage_channels[i + 1], 5, current.shape[2], current.shape[3]
        ).mean(dim=2)
        print(f"     After Bezier: {current.shape}")

        img_size = current.shape[2]  # Update spatial size

    print(f"\n   Final encoder output: {current.shape}")
    final_ch = stage_channels[-1]
    print(f"   Expected final channels: {final_ch}")

    # Latent projection
    print("\n3. LATENT PROJECTION")
    print(f"   Input: {current.shape}")

    # Two latent projection layers
    for layer_idx in range(2):
        conv_latent = nn.Conv2d(final_ch, d_model * 5, kernel_size=1)
        current = conv_latent(current)
        print(
            f"   After latent_proj[{layer_idx}] Conv2d({final_ch}→{d_model*5}, k=1): {current.shape}"
        )
        current = current.view(batch_size, d_model, 5, current.shape[2], current.shape[3]).mean(
            dim=2
        )
        print(f"   After Bezier reduction: {current.shape}")

    print(f"   Final latent spatial: {current.shape[2]}×{current.shape[3]}")
    latent_hw = current.shape[2] * current.shape[3]
    print(f"   Total latent tokens: {latent_hw}")

    print("\n4. DECODER PROGRESSION")

    # Decoder starts from latent tokens
    current = torch.randn(batch_size, d_model, current.shape[2], current.shape[3])
    print(f"   Starting decoder input: {current.shape}")

    # Progressive upsampling (4 stages)
    for stage in range(downscales):
        print(f"\n   Upsample Stage {stage+1}:")

        # Transposed convolution
        trans_conv = nn.ConvTranspose2d(d_model, d_model * 5, kernel_size=16, stride=2, padding=7)
        current = trans_conv(current)
        print(f"     After ConvTranspose2d({d_model}→{d_model*5}, k=16, s=2, p=7): {current.shape}")
        print(f"     BezierActivation (5→1 reduction)")
        current = current.view(batch_size, d_model, 5, current.shape[2], current.shape[3]).mean(
            dim=2
        )
        print(f"     After Bezier: {current.shape}")

        # Regular convolution with dilation
        conv_dilated = nn.Conv2d(
            d_model, d_model * 5, kernel_size=5, padding=4, stride=1, dilation=2
        )
        current = conv_dilated(current)
        print(f"     After Conv2d({d_model}→{d_model*5}, k=5, p=4, d=2): {current.shape}")
        print(f"     BezierActivation (5→1 reduction)")
        current = current.view(batch_size, d_model, 5, current.shape[2], current.shape[3]).mean(
            dim=2
        )
        print(f"     After Bezier: {current.shape}")

    print(f"\n   After {downscales} upsampling stages: {current.shape}")

    print("\n5. RGB CONVERSION")
    print(f"   Decoder output: {current.shape}")

    # RGB conversion layers
    conv1 = nn.Conv2d(d_model, 96, kernel_size=3, padding=1)
    current = conv1(current)
    print(f"   After Conv2d({d_model}→96, k=3, p=1): {current.shape}")

    conv2 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
    current = conv2(current)
    print(f"   After Conv2d(96→48, k=3, p=1): {current.shape}")

    conv3 = nn.Conv2d(48, 3, kernel_size=1, padding=0)
    current = conv3(current)
    print(f"   After Conv2d(48→3, k=1, p=0): {current.shape}")

    print("\n6. VERIFICATION")
    expected_final = (batch_size, 3, original_img_size, original_img_size)
    print(f"   Expected final shape: {expected_final}")
    print(f"   Actual final shape: {current.shape}")

    if current.shape == expected_final:
        print("   ✅ Dimensions match perfectly!")
    else:
        print("   ❌ Dimension mismatch!")

    # Check if spatial dimensions are correct
    expected_spatial = original_img_size
    actual_spatial = current.shape[2]
    if actual_spatial == expected_spatial:
        print("   ✅ Spatial dimensions correct!")
    else:
        print(f"   ❌ Spatial dimension error: expected {expected_spatial}, got {actual_spatial}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    trace_dimensions()
