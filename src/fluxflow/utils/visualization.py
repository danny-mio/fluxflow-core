"""Visualization and sample generation utilities for FluxFlow."""

import gzip
import io
import os
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.schedulers import DPMSolverMultistepScheduler
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Global cache for safe_vae_sample function
_VAE_SAMPLE_CACHE: Dict[str, Tuple[torch.Tensor, str]] = {}


def img_to_random_packet(
    img: torch.Tensor,
    d_model: int = 128,
    downscales: int = 4,
    max_hw: int = 1024,
    context_dims: int = 0,
) -> torch.Tensor:
    """
    Create random latent packet matching image dimensions.

    Args:
        img: Input image tensor [B, C, H, W] or [C, H, W]
        d_model: VAE latent dimension (will be extended by context_dims)
        downscales: Number of downsampling levels (2^downscales)
        max_hw: Normalization factor for H/W encoding
        context_dims: Number of context dimensions to add (0 for v0.6.0 and earlier)

    Returns:
        Random latent packet [B, T+1, D+context_dims] where T = H_lat * W_lat
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
    B, _, H, W = img.shape
    H_lat = max(H // (2**downscales), 1)
    W_lat = max(W // (2**downscales), 1)
    T = H_lat * W_lat
    dtype = img.dtype if img.is_floating_point() else torch.float32

    # Total dimensions: VAE dims + context dims
    total_dims = d_model + context_dims

    tokens = torch.randn(B, T, total_dims, device=img.device, dtype=dtype)
    hw = torch.zeros(B, 1, total_dims, device=img.device, dtype=dtype)
    hw[:, 0, 0] = H_lat / float(max_hw)
    hw[:, 0, 1] = W_lat / float(max_hw)
    # Last context_dims are zero (random context)
    return torch.cat([tokens, hw], dim=1)


@torch.no_grad()
def safe_vae_sample(
    diffuser: Any,
    image_address: str,
    channels: int,
    output_path: str,
    epoch: int,
    device: torch.device,
    filename_prefix: Optional[str] = None,
) -> None:
    """
    Test VAE reconstruction and save debug outputs.

    Saves:
    - Compressed latent (.ptz.gz)
    - Reconstruction with context
    - Reconstruction without context
    - Noise reconstruction tests

    Args:
        diffuser: FluxPipeline model
        image_address: Path to test image
        channels: Number of image channels (3 for RGB)
        output_path: Directory to save outputs
        epoch: Current training epoch
        device: Device to run on
        filename_prefix: Custom filename prefix (default: "vae_epoch_{epoch:04d}")
    """
    if image_address in _VAE_SAMPLE_CACHE:
        tensor_imgs, image_hash = _VAE_SAMPLE_CACHE[image_address]
    else:
        image_hash = md5(f"{image_address}".encode("utf-8")).hexdigest()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5] * channels, [0.5] * channels),
            ]
        )
        img = Image.open(image_address)
        tensor_imgs = torch.stack([transform(img)], 0).to(device)
        _VAE_SAMPLE_CACHE[image_address] = (tensor_imgs, image_hash)

    # Store original gradient checkpointing settings
    orig_compressor_gc = getattr(diffuser.compressor, "use_gradient_checkpointing", False)
    orig_expander_gc = getattr(diffuser.expander, "use_gradient_checkpointing", False)
    orig_expander_upscale_gc = getattr(
        getattr(diffuser.expander, "upscale", None), "use_gradient_checkpointing", False
    )

    # Disable gradient checkpointing during inference to avoid warnings
    if hasattr(diffuser.compressor, "use_gradient_checkpointing"):
        diffuser.compressor.use_gradient_checkpointing = False
    if hasattr(diffuser.expander, "use_gradient_checkpointing"):
        diffuser.expander.use_gradient_checkpointing = False
    if hasattr(diffuser.expander, "upscale") and hasattr(
        diffuser.expander.upscale, "use_gradient_checkpointing"
    ):
        diffuser.expander.upscale.use_gradient_checkpointing = False

    try:
        # Encode image
        out_latent = diffuser.compressor(tensor_imgs.detach())
        buffer = io.BytesIO()
        torch.save(out_latent, buffer)
        buffer.seek(0)
        with gzip.open(
            os.path.join(output_path, f"{image_hash}.ptz"), mode="wb", compresslevel=9
        ) as ptz:
            ptz.write(buffer.getbuffer())

        # Use custom prefix if provided, otherwise use default pattern
        prefix = filename_prefix if filename_prefix else f"vae_epoch_{epoch:04d}"

        # Decode with context
        out_img = torch.clamp(diffuser.expander(out_latent, True), min=-1, max=1)
        save_image(
            out_img,
            os.path.join(output_path, f"{prefix}-{image_hash}-ctx.webp"),
            normalize=True,
            value_range=(-1, 1),
        )

        # Decode without context
        out_img = torch.clamp(diffuser.expander(out_latent, False), min=-1, max=1)
        save_image(
            out_img,
            os.path.join(output_path, f"{prefix}-{image_hash}-nc.webp"),
            normalize=True,
            value_range=(-1, 1),
        )

        # Noise test
        rnd_imgs = torch.randn_like(tensor_imgs.detach())
        rsave_path = os.path.join(output_path, f"{prefix}-{image_hash}_ns_i.webp")
        save_image(rnd_imgs, rsave_path, normalize=True, value_range=(-1, 1))
        rout_img = torch.clamp(diffuser(rnd_imgs.detach(), use_flow=False), min=-1, max=1)
        save_image(
            rout_img,
            os.path.join(output_path, f"{prefix}-{image_hash}_ns_o.webp"),
            normalize=True,
            value_range=(-1, 1),
        )

        # Random packet test
        context_dims = diffuser.compressor.get_context_dims()
        out_img = torch.clamp(
            diffuser.expander(
                img_to_random_packet(
                    rnd_imgs, d_model=diffuser.compressor.d_model, context_dims=context_dims
                ),
                True,
            ),
            min=-1,
            max=1,
        )
        save_image(
            out_img,
            os.path.join(output_path, f"{prefix}-{image_hash}-nr_o.webp"),
            normalize=True,
            value_range=(-1, 1),
        )
    finally:
        # Restore original gradient checkpointing settings
        if hasattr(diffuser.compressor, "use_gradient_checkpointing"):
            diffuser.compressor.use_gradient_checkpointing = orig_compressor_gc
        if hasattr(diffuser.expander, "use_gradient_checkpointing"):
            diffuser.expander.use_gradient_checkpointing = orig_expander_gc
        if hasattr(diffuser.expander, "upscale") and hasattr(
            diffuser.expander.upscale, "use_gradient_checkpointing"
        ):
            diffuser.expander.upscale.use_gradient_checkpointing = orig_expander_upscale_gc


@torch.no_grad()
def generate_latent_images(
    batch_z: torch.Tensor,
    text_embeddings: torch.Tensor,
    diffuser: Any,
    scheduler_cls: Any = DPMSolverMultistepScheduler,
    steps: int = 20,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """
    Denoise latent using diffusion flow model.

    Args:
        batch_z: Noised latent packet [B, T+1, D]
        text_embeddings: Text conditioning [B, D_text]
        diffuser: FluxPipeline model
        scheduler_cls: Diffusers scheduler class
        steps: Number of denoising steps
        prediction_type: "v_prediction" or "epsilon"

    Returns:
        Denoised latent packet [B, T+1, D]
    """
    device = batch_z.device
    scheduler = scheduler_cls(
        num_train_timesteps=1000,
        algorithm_type="dpmsolver++",
        solver_order=2,
        prediction_type=prediction_type,
        lower_order_final=True,
        timestep_spacing="trailing",
    )
    scheduler.set_timesteps(steps, device=device)

    hw_vec = batch_z[:, -1:, :].clone()
    lat = batch_z[:, :-1, :].clone()

    for t in scheduler.timesteps:
        t_batch = torch.full((lat.size(0),), t.item(), device=device, dtype=torch.long)
        full_input = torch.cat([lat, hw_vec], dim=1)
        model_out = diffuser.flow_processor(full_input, text_embeddings, t_batch)
        model_out_lat = model_out[:, :-1, :]
        lat = scheduler.step(
            model_output=model_out_lat, timestep=int(t.item()), sample=lat
        ).prev_sample

    return torch.cat([lat, hw_vec], dim=1)


@torch.no_grad()
def _generate_with_cfg(
    noised_latent: torch.Tensor,
    text_embeddings: torch.Tensor,
    null_embeddings: torch.Tensor,
    diffuser: Any,
    guidance_scale: float,
    device: torch.device,
    steps: int = 20,
) -> torch.Tensor:
    """
    Generate with classifier-free guidance (CFG).

    Args:
        noised_latent: Initial noised latent [B, T+1, D]
        text_embeddings: Conditional text embeddings [B, D_text]
        null_embeddings: Null/unconditional embeddings [B, D_text]
        diffuser: FluxPipeline model
        guidance_scale: CFG strength (typically 5.0)
        device: Device to run on
        steps: Number of denoising steps

    Returns:
        Denoised latent [B, T+1, D]
    """
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        algorithm_type="dpmsolver++",
        solver_order=2,
        prediction_type="v_prediction",
        lower_order_final=True,
        timestep_spacing="trailing",
    )
    scheduler.set_timesteps(steps, device=device)  # type: ignore[attr-defined]

    hw_vec = noised_latent[:, -1:, :].clone()
    lat = noised_latent[:, :-1, :].clone()

    for t in scheduler.timesteps:  # type: ignore[attr-defined]
        t_batch = torch.full((lat.size(0),), t.item(), device=device, dtype=torch.long)
        full_input = torch.cat([lat, hw_vec], dim=1)

        # Conditional prediction
        v_cond = diffuser.flow_processor(full_input, text_embeddings, t_batch)
        v_cond_lat = v_cond[:, :-1, :]

        # Unconditional prediction
        v_uncond = diffuser.flow_processor(full_input, null_embeddings, t_batch)
        v_uncond_lat = v_uncond[:, :-1, :]

        # Apply CFG: v_guided = v_uncond + w * (v_cond - v_uncond)
        v_guided = v_uncond_lat + guidance_scale * (v_cond_lat - v_uncond_lat)

        lat = scheduler.step(  # type: ignore[attr-defined]
            model_output=v_guided, timestep=int(t.item()), sample=lat
        ).prev_sample

    return torch.cat([lat, hw_vec], dim=1)


@torch.no_grad()
def save_sample_images(
    diffuser: Any,
    text_encoder: Any,
    tokenizer: Any,
    output_path: str,
    epoch: int,
    device: torch.device,
    sample_captions: List[str],
    batch_size: int = 1,
    sample_sizes: Optional[List[Union[int, Tuple[int, int]]]] = None,
    use_cfg: bool = False,
    guidance_scale: float = 5.0,
    filename_prefix: Optional[str] = None,
    *,
    scheduler: Optional[Any] = None,
    num_inference_steps: int = 50,
) -> None:
    """
    Generate and save sample images from text prompts.

    Args:
        diffuser: FluxPipeline model
        text_encoder: BertTextEncoder model
        tokenizer: HuggingFace tokenizer
        output_path: Directory to save samples
        epoch: Current training epoch
        device: Device to run on
        sample_captions: List of text prompts
        batch_size: Batch size for generation
        sample_sizes: List of sizes for sample generation. Can be:
            - int: generates square image (e.g., 512 -> 512x512)
            - tuple (width, height): generates image with specific dimensions
            Defaults to [256, 384, 512, 1024] (square images)
        use_cfg: Enable classifier-free guidance (default: False)
        guidance_scale: CFG strength (default: 5.0, only used if use_cfg=True)
        filename_prefix: Custom filename prefix (default: "samples_epoch_{epoch:04d}")
    """
    diffuser.eval()
    text_encoder.eval()

    sample_texts = [s.upper() for s in sample_captions]

    # Default to square images if not specified
    if sample_sizes is None:
        sample_sizes = [256, 384, 512, 1024]

    encodings = tokenizer.batch_encode_plus(
        sample_texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    full_text_embeddings = text_encoder(input_ids, attention_mask=attention_mask)

    for size_spec in sample_sizes:
        # Parse size specification
        if isinstance(size_spec, (list, tuple)):
            width, height = size_spec
            size_str = f"{width}x{height}"
        else:
            width = height = size_spec
            size_str = f"{size_spec:04d}"

        for i in range(0, len(sample_texts), batch_size):
            text_embeddings = full_text_embeddings[i : i + batch_size]
            B = text_embeddings.size(0)

            # Use flow model to predict clean latents from noisy inputs
            dummy_img = torch.randn(B, 3, height, width, device=device) * 2 - 1
            base_latent = diffuser.compressor(dummy_img)  # [B, T + 1, D]

            # Extract HW vector (preserve original)
            hw_vec = base_latent[:, -1:, :]
            img_tokens = base_latent[:, :-1, :]  # Clean latents from VAE

            # Create noisy version for flow input
            t = torch.tensor(500.0, device=device)
            alpha_t = torch.sqrt(t / 1000.0)
            sigma_t = torch.sqrt(1 - alpha_t**2)

            # Add noise: x_t = alpha_t * x_0 + sigma_t * epsilon
            epsilon = torch.randn_like(img_tokens)
            noisy_img_tokens = alpha_t * img_tokens + sigma_t * epsilon
            noisy_latent = torch.cat([noisy_img_tokens, hw_vec], dim=1)

            # Flow model predicts v (normalized velocity)
            v_pred = diffuser.flow_processor(noisy_latent, text_embeddings, t.expand(B))
            v_pred_img = v_pred[:, :-1, :]

            # Extract clean latents from v prediction: x_0 = (x_t - sigma_t * v) / alpha_t
            # This inverts the v-prediction formula
            pred_clean_img = (noisy_img_tokens - sigma_t * v_pred_img) / alpha_t

            # Use predicted clean latents with original HW vector
            final_latent = torch.cat([pred_clean_img, hw_vec], dim=1)

            # Decode with VAE expander (expects packed format)
            generated_images = diffuser.expander(final_latent)

            # Save generated images
            for b, img in enumerate(generated_images):
                global_idx = i + b
                # Use custom prefix if provided, otherwise use default pattern
                if filename_prefix:
                    filename = f"{filename_prefix}_{global_idx}-{size_str}.webp"
                else:
                    filename = f"samples_epoch_{epoch:04d}_caption_{global_idx}-{size_str}.webp"
                save_path = os.path.join(output_path, filename)
                save_image(img, save_path, normalize=True, value_range=(-1, 1))
