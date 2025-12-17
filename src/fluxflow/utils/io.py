"""Model checkpoint save/load utilities for FluxFlow."""

import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

import safetensors.torch
import torch
import torch.nn as nn


def copy_and_replace(source_path: str, destination_path: str) -> None:
    """
    Copy file and replace destination if it exists.

    Args:
        source_path: Source file path
        destination_path: Destination file path
    """
    if os.path.exists(destination_path):
        os.remove(destination_path)
    shutil.copy2(source_path, destination_path)


def save_model(
    diffuser: nn.Module,
    text_encoder: nn.Module,
    output_path: str,
    save_pretrained: bool = False,
    save_metadata: bool = False,
    model_version: str = "0.3.0",
    training_info: Optional[Dict] = None,
) -> None:
    """
    Save FluxFlow pipeline and text encoder to safetensors.

    Creates backup (.bck) of existing checkpoints before overwriting.

    Args:
        diffuser: FluxPipeline model
        text_encoder: BertTextEncoder model
        output_path: Directory to save checkpoints
        save_pretrained: If True, save full text encoder (including language_model)
        save_metadata: If True, save version metadata (recommended for future compatibility)
        model_version: Model version string (used when save_metadata=True)
        training_info: Optional training metadata dict (used when save_metadata=True)

    Example:
        >>> # Legacy save (backward compatible)
        >>> save_model(diffuser, text_encoder, "outputs/model/")

        >>> # Save with version metadata (recommended)
        >>> save_model(
        ...     diffuser,
        ...     text_encoder,
        ...     "outputs/model/",
        ...     save_metadata=True,
        ...     model_version="0.3.0",
        ...     training_info={"total_steps": 50000, "dataset": "COCO"}
        ... )
    """
    if save_metadata:
        # Use versioned save
        from pathlib import Path

        from fluxflow.models.versioning import save_versioned_checkpoint

        save_versioned_checkpoint(
            diffuser,
            Path(output_path),
            model_version=model_version,
            training_info=training_info,
        )

        # Also save text encoder separately for compatibility
        os.makedirs(output_path, exist_ok=True)
        te_path = os.path.join(output_path, "text_encoder.safetensors")

        if not save_pretrained:
            te_state_dict = {
                "text_encoder." + k: v.cpu()
                for k, v in text_encoder.state_dict().items()
                if not k.startswith("language_model.")
            }
        else:
            te_state_dict = {
                "text_encoder." + k: v.cpu() for k, v in text_encoder.state_dict().items()
            }

        safetensors.torch.save_file(te_state_dict, te_path)
        print(f"\nModel saved with version metadata to {output_path}")
        return

    # Legacy save path (default for backward compatibility)
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "flxflow_final.safetensors")
    model_path_bck = os.path.join(output_path, "flxflow_final.safetensors.bck")
    te_path = os.path.join(output_path, "text_encoder.safetensors")

    if os.path.isfile(model_path):
        copy_and_replace(model_path, model_path_bck)

    # Save diffuser state
    state_dict = {"diffuser." + k: v.cpu() for k, v in diffuser.state_dict().items()}

    # Save text encoder state
    if not save_pretrained:
        te_state_dict = {
            "text_encoder." + k: v.cpu()
            for k, v in text_encoder.state_dict().items()
            if not k.startswith("language_model.")
        }
    else:
        te_state_dict = {"text_encoder." + k: v.cpu() for k, v in text_encoder.state_dict().items()}

    state_dict.update(te_state_dict)
    safetensors.torch.save_file(state_dict, model_path)
    safetensors.torch.save_file(te_state_dict, te_path)
    print(f"\nModel saved to {model_path}")


def save_discriminators(d_img: Optional[nn.Module], output_path: str) -> None:
    """
    Save discriminator checkpoint to safetensors.

    Args:
        d_img: PatchDiscriminator model (can be None)
        output_path: Directory to save checkpoints
    """
    os.makedirs(output_path, exist_ok=True)
    if d_img is not None:
        di_path = os.path.join(output_path, "disc_img.safetensors")
        di_path_bck = os.path.join(output_path, "disc_img.safetensors.bck")
        if os.path.isfile(di_path):
            copy_and_replace(di_path, di_path_bck)
        safetensors.torch.save_file(
            {"disc_img." + k: v.cpu() for k, v in d_img.state_dict().items()}, di_path
        )
    print("Discriminators saved.")


def load_discriminators_if_any(d_img: nn.Module, output_path: str) -> None:
    """
    Load discriminator checkpoint if it exists.

    Args:
        d_img: PatchDiscriminator model to load weights into
        output_path: Directory containing checkpoints
    """
    di_path = os.path.join(output_path, "disc_img.safetensors")
    if os.path.exists(di_path):
        print(f"Loading image discriminator from {di_path}")
        sd = safetensors.torch.load_file(di_path)
        d_img.load_state_dict({k.replace("disc_img.", ""): v for k, v in sd.items()}, strict=False)


def format_duration(seconds: float) -> str:
    """
    Format elapsed seconds as HH:MM:SS.mmm.

    Args:
        seconds: Elapsed time in seconds

    Returns:
        Formatted string: HHHH:MM:SS.mmm
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    total_hours = days * 24 + hours
    return f"{total_hours:04d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def save_training_state(
    output_path: str,
    epoch: int = 0,
    batch_idx: Optional[int] = None,
    global_step: Optional[int] = None,
    samples_trained: Optional[int] = None,
    total_samples: Optional[int] = None,
    learning_rates: Optional[Dict[str, float]] = None,
    sampler_state: Optional[Dict] = None,
    optimizers: Optional[Dict[str, Any]] = None,
    ema_state: Optional[Dict] = None,
    # Simplified API for tests
    model: Optional[nn.Module] = None,
    optimizer: Optional[Any] = None,
    step: Optional[int] = None,
) -> None:
    """
    Save complete training state for mid-epoch resume.

    Supports two APIs:
    1. Production API with detailed tracking
    2. Simplified API for tests (model, optimizer, epoch, step)

    Args:
        output_path: Directory or file path to save training state
        epoch: Current epoch number
        batch_idx: Current batch index within epoch
        global_step: Total training steps
        samples_trained: Total samples processed
        total_samples: Total samples in dataset
        learning_rates: Dict of learning rates for each optimizer
        sampler_state: Sampler state dict (for reproducibility)
        optimizers: Dict of optimizer state dicts (optional, large files)
        ema_state: EMA state dict (optional)
        model: Model to save (simplified API)
        optimizer: Optimizer to save (simplified API)
        step: Step number (simplified API, maps to global_step)
    """
    # Handle simplified API
    if model is not None or optimizer is not None:
        # Use simplified checkpoint format for tests
        checkpoint = {
            "epoch": epoch,
            "step": step if step is not None else 0,
        }
        if model is not None:
            checkpoint["model_state_dict"] = model.state_dict()
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        # Ensure output_path is a file, not directory
        if os.path.isdir(output_path):
            checkpoint_path = os.path.join(output_path, "checkpoint.pt")
        else:
            checkpoint_path = str(output_path)
            # Create parent directory if needed
            parent_dir = os.path.dirname(checkpoint_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        torch.save(checkpoint, checkpoint_path)
        return

    # Original production API
    os.makedirs(output_path, exist_ok=True)

    state = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "batch_idx": batch_idx if batch_idx is not None else 0,
        "global_step": global_step if global_step is not None else 0,
        "samples_trained": samples_trained if samples_trained is not None else 0,
        "total_samples": total_samples if total_samples is not None else 0,
        "learning_rates": learning_rates if learning_rates is not None else {},
    }

    if sampler_state is not None:
        state["sampler_state"] = sampler_state

    if ema_state is not None:
        state["ema_state"] = ema_state

    # Save main state to JSON
    state_path = os.path.join(output_path, "training_state.json")
    state_path_bck = os.path.join(output_path, "training_state.json.bck")

    if os.path.exists(state_path):
        copy_and_replace(state_path, state_path_bck)

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    # Save optimizer states separately (they're large)
    if optimizers is not None:
        opt_path = os.path.join(output_path, "optimizer_states.pt")
        opt_path_bck = os.path.join(output_path, "optimizer_states.pt.bck")

        if os.path.exists(opt_path):
            copy_and_replace(opt_path, opt_path_bck)

        torch.save(optimizers, opt_path)


def load_training_state(
    output_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load training state for resume.

    Supports two APIs:
    1. Production API: Returns state dict only
    2. Simplified API for tests: Loads model and optimizer states if provided

    Args:
        output_path: Directory or file path containing training state
        model: Model to load weights into (simplified API)
        optimizer: Optimizer to load state into (simplified API)

    Returns:
        Training state dict or None if not found
    """
    # Try simplified checkpoint format first (for tests)
    if os.path.isfile(output_path):
        checkpoint_path = output_path
    elif os.path.isdir(output_path) and os.path.exists(os.path.join(output_path, "checkpoint.pt")):
        checkpoint_path = os.path.join(output_path, "checkpoint.pt")
    else:
        checkpoint_path = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Load model and optimizer states if provided
            if model is not None and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            return checkpoint  # type: ignore[no-any-return]
        except Exception:
            # Fall through to try production format
            pass

    # Try production format (training_state.json)
    state_path = os.path.join(output_path, "training_state.json")

    if not os.path.exists(state_path):
        return None

    try:
        with open(state_path, "r") as f:
            state = json.load(f)

        # Load optimizer states if they exist
        opt_path = os.path.join(output_path, "optimizer_states.pt")
        if os.path.exists(opt_path):
            state["optimizer_states"] = torch.load(opt_path, weights_only=False)

        return state  # type: ignore[no-any-return]
    except Exception as e:
        print(f"Warning: Could not load training state: {e}")
        return None
