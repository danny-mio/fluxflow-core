"""
FluxFlowPipeline: High-level text-to-image pipeline inheriting from Diffusers DiffusionPipeline.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import PIL.Image
import safetensors.torch
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import BaseOutput
from transformers import AutoTokenizer

from .encoders import BertTextEncoder
from .flow import FluxFlowProcessor
from .vae import FluxCompressor, FluxExpander

logger = logging.getLogger(__name__)

# Checkpoint format versions
CHECKPOINT_VERSION = "1.0"
SUPPORTED_CHECKPOINT_VERSIONS = ["1.0"]


class FluxFlowPipelineOutput(BaseOutput):
    """
    Output class for FluxFlow pipeline.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`):
            List of denoised PIL images of length `batch_size` or numpy array of shape
            `(batch_size, height, width, num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class FluxFlowPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using FluxFlow models.

    This pipeline inherits from Diffusers' DiffusionPipeline and provides a familiar
    interface for generating images from text prompts.

    Args:
        compressor (`FluxCompressor`):
            VAE encoder model for compressing images to latent space.
        flow_processor (`FluxFlowProcessor`):
            Flow-based transformer model for denoising latents.
        expander (`FluxExpander`):
            VAE decoder model for expanding latents back to images.
        text_encoder (`BertTextEncoder`):
            Text encoder model for encoding prompts.
        tokenizer (`PreTrainedTokenizer`):
            Tokenizer for processing text prompts.
        scheduler (`SchedulerMixin`):
            Scheduler for the diffusion process.
    """

    model_cpu_offload_seq = "text_encoder->compressor->flow_processor->expander"
    _optional_components = ["tokenizer"]

    def __init__(
        self,
        compressor: FluxCompressor,
        flow_processor: FluxFlowProcessor,
        expander: FluxExpander,
        text_encoder: BertTextEncoder,
        tokenizer: Any,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            compressor=compressor,
            flow_processor=flow_processor,
            expander=expander,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        # Keep ctx_tokens consistent across submodules
        K = getattr(self.compressor, "ctx_tokens", None) or getattr(
            self.compressor, "downscales", 4
        )
        if hasattr(self.flow_processor, "ctx_tokens"):
            self.flow_processor.ctx_tokens = K
        if hasattr(self.expander, "ctx_tokens"):
            self.expander.ctx_tokens = K

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        use_versioning: bool = False,
        **kwargs,
    ):
        """
        Load a FluxFlowPipeline from a checkpoint file or directory.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                - A path to a `.safetensors` or `.pt` checkpoint file
                - A path to a directory containing pipeline components
            use_versioning: Use versioned loading system (default: False for backward compatibility)
            **kwargs:
                Additional arguments passed to the pipeline constructor.

        Returns:
            `FluxFlowPipeline`: The loaded pipeline ready for inference.

        Example:
            ```python
            from fluxflow.models import FluxFlowPipeline

            # Load from checkpoint (legacy method)
            pipeline = FluxFlowPipeline.from_pretrained("path/to/checkpoint.safetensors")

            # Load with versioning for better compatibility
            pipeline = FluxFlowPipeline.from_pretrained(
                "path/to/checkpoint/",
                use_versioning=True
            )

            # Generate image
            image = pipeline(
                prompt="a beautiful sunset over mountains",
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

            image.save("output.png")
            ```
        """
        # Note: Versioned loading currently supports FluxPipeline only
        # FluxFlowPipeline requires additional components (text_encoder, tokenizer, scheduler)
        # For now, versioning redirects to standard loading
        if use_versioning:
            logger.info(
                "Versioned loading for FluxFlowPipeline currently uses standard loading path. "
                "Full versioning support for FluxFlowPipeline will be added in a future release."
            )

        # Check if it's a checkpoint file or directory
        if pretrained_model_name_or_path:
            path = str(pretrained_model_name_or_path)
            if path.endswith(".safetensors") or path.endswith(".pt"):
                return cls._from_checkpoint(pretrained_model_name_or_path, **kwargs)
            elif os.path.isdir(path):
                # Check for FluxFlow checkpoint directory structure
                model_file = os.path.join(path, "flxflow_final.safetensors")
                if not os.path.exists(model_file):
                    model_file = os.path.join(path, "model.safetensors")
                if os.path.exists(model_file):
                    return cls._from_checkpoint_dir(path, **kwargs)

        # Otherwise use standard DiffusionPipeline loading
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def _from_checkpoint(
        cls,
        checkpoint_path: Union[str, os.PathLike],
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        scheduler: Optional[KarrasDiffusionSchedulers] = None,
        scheduler_config: Optional[dict] = None,
        **kwargs,
    ) -> "FluxFlowPipeline":
        """
        Load a FluxFlowPipeline from a single checkpoint file.

        Args:
            checkpoint_path: Path to .safetensors or .pt checkpoint
            device: Device to load model on ('cuda', 'cpu', 'mps', or None for auto)
            torch_dtype: Data type for model weights
            tokenizer_name: HuggingFace tokenizer name
            scheduler: Custom scheduler instance to use (overrides scheduler_config)
            scheduler_config: Configuration dict for DPMSolverMultistepScheduler.
                Supported keys: num_train_timesteps, beta_start, beta_end, beta_schedule,
                algorithm_type, solver_order, prediction_type, use_karras_sigmas
            **kwargs: Override model configuration

        Returns:
            FluxFlowPipeline: Loaded pipeline

        Example:
            ```python
            from fluxflow.models import FluxFlowPipeline
            from diffusers import EulerDiscreteScheduler

            # With custom scheduler
            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=1000,
                beta_schedule="scaled_linear",
            )
            pipeline = FluxFlowPipeline.from_pretrained(
                "checkpoint.safetensors",
                scheduler=scheduler
            )

            # With scheduler config
            pipeline = FluxFlowPipeline.from_pretrained(
                "checkpoint.safetensors",
                scheduler_config={"use_karras_sigmas": True, "solver_order": 3}
            )
            ```
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Load checkpoint
        if str(checkpoint_path).endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Validate checkpoint version
        cls._validate_checkpoint_version(state_dict)

        # Detect configuration from checkpoint
        config = cls._detect_config(state_dict)
        config.update(kwargs)

        # Initialize models
        compressor = FluxCompressor(
            in_channels=config.get("in_channels", 3),
            d_model=config["vae_dim"],
            downscales=config["downscales"],
            max_hw=config.get("max_hw", 1024),
            use_attention=True,
            attn_layers=config.get("vae_attn_layers", 2),
        )

        flow_processor = FluxFlowProcessor(
            d_model=config["flow_dim"],
            vae_dim=config["vae_dim"],
            embedding_size=config.get("text_embed_dim", 768),
            n_head=config.get("flow_attn_heads", 8),
            n_layers=config.get("flow_transformer_layers", 10),
            max_hw=config.get("max_hw", 1024),
        )

        expander = FluxExpander(
            d_model=config["vae_dim"],
            upscales=config.get("upscales", config["downscales"]),
            max_hw=config.get("max_hw", 1024),
        )

        text_encoder = BertTextEncoder(embed_dim=config.get("text_embed_dim", 768))

        # Load weights
        # Try loading with 'diffuser.' prefix first
        diffuser_state = {
            k.replace("diffuser.", ""): v
            for k, v in state_dict.items()
            if k.startswith("diffuser.")
        }

        if not diffuser_state:
            # Try loading directly (no prefix)
            diffuser_state = {
                k: v for k, v in state_dict.items() if not k.startswith("text_encoder.")
            }

        # Load compressor weights
        compressor_state = {
            k.replace("compressor.", ""): v
            for k, v in diffuser_state.items()
            if k.startswith("compressor.")
        }
        if compressor_state:
            compressor.load_state_dict(compressor_state, strict=False)

        # Load flow_processor weights
        flow_state = {
            k.replace("flow_processor.", ""): v
            for k, v in diffuser_state.items()
            if k.startswith("flow_processor.")
        }
        if flow_state:
            flow_processor.load_state_dict(flow_state, strict=False)

        # Load expander weights
        expander_state = {
            k.replace("expander.", ""): v
            for k, v in diffuser_state.items()
            if k.startswith("expander.")
        }
        if expander_state:
            expander.load_state_dict(expander_state, strict=False)

        # Load text encoder weights
        text_encoder_state = {
            k.replace("text_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("text_encoder.")
        }
        if text_encoder_state:
            text_encoder.load_state_dict(text_encoder_state, strict=False)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize scheduler
        if scheduler is None:
            # Default scheduler configuration
            default_scheduler_config = {
                "num_train_timesteps": 1000,
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "algorithm_type": "dpmsolver++",
                "solver_order": 2,
                "prediction_type": "v_prediction",
                "use_karras_sigmas": False,
            }

            # Override with user-provided config
            if scheduler_config is not None:
                default_scheduler_config.update(scheduler_config)

            scheduler = DPMSolverMultistepScheduler(**default_scheduler_config)

        # Create pipeline
        pipeline = cls(
            compressor=compressor,
            flow_processor=flow_processor,
            expander=expander,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        # Move to device and set dtype
        pipeline.to(device)
        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        return pipeline

    @classmethod
    def _from_checkpoint_dir(
        cls,
        checkpoint_dir: Union[str, os.PathLike],
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        **kwargs,
    ) -> "FluxFlowPipeline":
        """
        Load a FluxFlowPipeline from a directory with separate checkpoint files.

        This handles the standard FluxFlow training output format:
        - flxflow_final.safetensors (or model.safetensors) - diffuser weights
        - text_encoder.safetensors - text encoder weights

        Args:
            checkpoint_dir: Path to directory containing checkpoint files
            device: Device to load model on
            torch_dtype: Data type for model weights
            tokenizer_name: HuggingFace tokenizer name
            **kwargs: Override model configuration

        Returns:
            FluxFlowPipeline: Loaded pipeline
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Find model file
        model_path = checkpoint_dir / "flxflow_final.safetensors"
        if not model_path.exists():
            model_path = checkpoint_dir / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model checkpoint found in {checkpoint_dir}. "
                "Expected 'flxflow_final.safetensors' or 'model.safetensors'"
            )

        # Find text encoder file
        text_encoder_path = checkpoint_dir / "text_encoder.safetensors"

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Load model checkpoint
        model_state = safetensors.torch.load_file(str(model_path))

        # Load text encoder if available
        text_encoder_state = {}
        if text_encoder_path.exists():
            text_encoder_state = safetensors.torch.load_file(str(text_encoder_path))

        # Merge state dicts for config detection
        combined_state = {**model_state, **text_encoder_state}

        # Detect configuration
        config = cls._detect_config(combined_state)
        config.update(kwargs)

        # Initialize models
        compressor = FluxCompressor(
            in_channels=config.get("in_channels", 3),
            d_model=config["vae_dim"],
            downscales=config["downscales"],
            max_hw=config.get("max_hw", 1024),
            use_attention=True,
            attn_layers=config.get("vae_attn_layers", 2),
        )

        flow_processor = FluxFlowProcessor(
            d_model=config["flow_dim"],
            vae_dim=config["vae_dim"],
            embedding_size=config.get("text_embed_dim", 768),
            n_head=config.get("flow_attn_heads", 8),
            n_layers=config.get("flow_transformer_layers", 10),
            max_hw=config.get("max_hw", 1024),
        )

        expander = FluxExpander(
            d_model=config["vae_dim"],
            upscales=config.get("upscales", config["downscales"]),
            max_hw=config.get("max_hw", 1024),
        )

        text_encoder = BertTextEncoder(embed_dim=config.get("text_embed_dim", 768))

        # Load diffuser weights (with 'diffuser.' prefix)
        diffuser_state = {
            k.replace("diffuser.", ""): v
            for k, v in model_state.items()
            if k.startswith("diffuser.")
        }

        # If no prefix, use directly
        if not diffuser_state:
            diffuser_state = model_state

        # Load component weights
        compressor_state = {
            k.replace("compressor.", ""): v
            for k, v in diffuser_state.items()
            if k.startswith("compressor.")
        }
        if compressor_state:
            compressor.load_state_dict(compressor_state, strict=False)

        flow_state = {
            k.replace("flow_processor.", ""): v
            for k, v in diffuser_state.items()
            if k.startswith("flow_processor.")
        }
        if flow_state:
            flow_processor.load_state_dict(flow_state, strict=False)

        expander_state = {
            k.replace("expander.", ""): v
            for k, v in diffuser_state.items()
            if k.startswith("expander.")
        }
        if expander_state:
            expander.load_state_dict(expander_state, strict=False)

        # Load text encoder weights
        if text_encoder_state:
            te_state = {
                k.replace("text_encoder.", ""): v
                for k, v in text_encoder_state.items()
                if k.startswith("text_encoder.")
            }
            if not te_state:
                te_state = text_encoder_state
            text_encoder.load_state_dict(te_state, strict=False)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize scheduler
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            algorithm_type="dpmsolver++",
            solver_order=2,
            prediction_type="v_prediction",
            use_karras_sigmas=False,
        )

        # Create pipeline
        pipeline = cls(
            compressor=compressor,
            flow_processor=flow_processor,
            expander=expander,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        # Move to device and set dtype
        pipeline.to(device)
        if torch_dtype is not None:
            pipeline.to(torch_dtype)

        return pipeline

    @staticmethod
    def _detect_config(state_dict: dict) -> dict:
        """Detect model configuration from checkpoint state dict."""
        config = {}
        keys = list(state_dict.keys())

        # Detect VAE dimension from compressor.latent_proj
        for key in keys:
            if "compressor.latent_proj" in key and ".0.0.weight" in key:
                shape = state_dict[key].shape
                if len(shape) == 4:
                    config["vae_dim"] = shape[0] // 5
                    break

        # Detect flow dimension from vae_to_dmodel
        for key in keys:
            if "flow_processor.vae_to_dmodel.weight" in key:
                shape = state_dict[key].shape
                config["flow_dim"] = shape[0]
                if "vae_dim" not in config:
                    config["vae_dim"] = shape[1]
                break

        # Detect text embedding dimension
        for key in keys:
            if "flow_processor.text_proj.weight" in key:
                shape = state_dict[key].shape
                config["text_embed_dim"] = shape[1]
                break

        if "text_embed_dim" not in config:
            config["text_embed_dim"] = 768

        # Detect downscales
        encoder_stages = [k for k in keys if "compressor.encoder_z." in k and ".0.weight" in k]
        config["downscales"] = len(encoder_stages) if encoder_stages else 4

        # Detect upscales
        upscale_layers = [
            k for k in keys if "expander.upscale.layers." in k and ".conv1.0.weight" in k
        ]
        config["upscales"] = len(upscale_layers) if upscale_layers else config["downscales"]

        # Detect attention layers
        attn_layers = [
            k for k in keys if "compressor.token_attn." in k and ".attn.in_proj_weight" in k
        ]
        config["vae_attn_layers"] = len(attn_layers) if attn_layers else 2

        # Detect transformer blocks
        transformer_blocks = [
            k
            for k in keys
            if "flow_processor.transformer_blocks." in k and ".self_attn.q_proj.weight" in k
        ]
        config["flow_transformer_layers"] = len(transformer_blocks) if transformer_blocks else 10

        config["max_hw"] = 1024

        if "vae_dim" not in config:
            raise ValueError("Could not detect VAE dimension from checkpoint")
        if "flow_dim" not in config:
            raise ValueError("Could not detect flow dimension from checkpoint")

        return config

    @classmethod
    def _validate_checkpoint_version(cls, state_dict: dict) -> None:
        """
        Validate checkpoint version compatibility.

        Args:
            state_dict: Loaded checkpoint state dictionary

        Raises:
            ValueError: If checkpoint version is incompatible

        Note:
            Logs warning for unknown versions but allows loading.
            Future versions may be incompatible and will raise an error.
        """
        version = state_dict.get("__version__") or state_dict.get("_version")

        if version is None:
            # Legacy checkpoint without version - assume compatible
            logger.debug("No version info in checkpoint, assuming compatible format")
            return

        if isinstance(version, torch.Tensor):
            # Handle case where version might be stored as tensor
            version = str(version.item()) if version.numel() == 1 else str(version.tolist())

        version = str(version)

        if version in SUPPORTED_CHECKPOINT_VERSIONS:
            logger.debug(f"Checkpoint version {version} is supported")
            return

        # Check if it's a newer version (might be forward-incompatible)
        try:
            current_major = int(CHECKPOINT_VERSION.split(".")[0])
            checkpoint_major = int(version.split(".")[0])

            if checkpoint_major > current_major:
                raise ValueError(
                    f"Checkpoint version {version} is newer than supported version "
                    f"{CHECKPOINT_VERSION}. Please update fluxflow to load this checkpoint."
                )
        except (ValueError, IndexError):
            pass

        # Unknown version - warn but try to load
        logger.warning(
            f"Unknown checkpoint version {version}. "
            f"Supported versions: {SUPPORTED_CHECKPOINT_VERSIONS}. "
            f"Loading may fail or produce unexpected results."
        )

    @staticmethod
    def _validate_latent_format(latents: torch.Tensor, batch_size: int, vae_dim: int) -> None:
        """
        Validate latent tensor format.

        FluxFlow uses a packed latent format: [B, T+1, D] where:
        - B = batch size
        - T = number of image tokens (H*W / 256)
        - D = VAE dimension
        - The +1 is for the HW dimension token

        Args:
            latents: Latent tensor to validate
            batch_size: Expected batch size
            vae_dim: Expected VAE dimension (D)

        Raises:
            ValueError: If latent format is invalid
        """
        if latents.ndim != 3:
            raise ValueError(
                f"Latents must be 3D tensor [B, T+1, D], got shape {latents.shape} "
                f"with {latents.ndim} dimensions"
            )

        b, seq_len, d = latents.shape

        if b != batch_size:
            raise ValueError(f"Latent batch size mismatch: expected {batch_size}, got {b}")

        if d != vae_dim:
            raise ValueError(
                f"Latent dimension mismatch: expected {vae_dim}, got {d}. "
                f"Ensure latents were generated with the same model configuration."
            )

        # Check minimum sequence length (at least 1 token + 1 HW token)
        if seq_len < 2:
            raise ValueError(
                f"Latent sequence length too short: {seq_len}. "
                f"Expected at least 2 (1 image token + 1 HW token)."
            )

        # Validate sequence length corresponds to valid image size
        num_tokens = seq_len - 1  # Subtract HW token
        # Each token represents 16x16 pixels (after 4 downscales of 2x each)
        implied_pixels = num_tokens * 256  # 16 * 16 = 256

        # Check if it corresponds to a reasonable image size
        min_size = 64  # Minimum 64x64
        max_size = 2048  # Maximum 2048x2048

        if implied_pixels < (min_size * min_size) // 256 * 256:
            logger.warning(
                f"Latent sequence length {seq_len} implies very small image. "
                f"This may cause issues."
            )

        if implied_pixels > max_size * max_size:
            logger.warning(
                f"Latent sequence length {seq_len} implies very large image "
                f"(~{int(implied_pixels**0.5)}x{int(implied_pixels**0.5)} pixels). "
                f"This may cause memory issues."
            )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt: Optional[Union[str, List[str]]] = None,
    ) -> torch.Tensor:
        """
        Encode text prompt into embeddings.

        Args:
            prompt: Text prompt(s) to encode
            device: Device to use
            num_images_per_prompt: Number of images per prompt
            do_classifier_free_guidance: Whether to use CFG
            negative_prompt: Negative prompt for CFG

        Returns:
            Text embeddings tensor
        """
        device = device or self._execution_device

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)

        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(device)

        # Encode
        text_embeddings = self.text_encoder(input_ids, attention_mask=attention_mask)

        # Duplicate for num_images_per_prompt
        if num_images_per_prompt > 1:
            text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        # Classifier-free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            uncond_inputs = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )

            uncond_ids = uncond_inputs.input_ids.to(device)
            uncond_mask = (uncond_ids != self.tokenizer.pad_token_id).long().to(device)
            uncond_embeddings = self.text_encoder(uncond_ids, attention_mask=uncond_mask)

            if num_images_per_prompt > 1:
                uncond_embeddings = uncond_embeddings.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[FluxFlowPipelineOutput, tuple]:
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt(s) for image generation.
            height: Height of generated image in pixels.
            width: Width of generated image in pixels.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale. Higher values = stronger adherence to prompt.
            negative_prompt: Negative prompt(s) for CFG.
            num_images_per_prompt: Number of images to generate per prompt.
            generator: Random number generator for reproducibility.
            latents: Pre-generated latents (optional).
            output_type: Output format - "pil", "np", or "latent".
            return_dict: Whether to return a dict or tuple.
            callback: Callback function for progress updates.
            callback_steps: Frequency of callback calls.

        Returns:
            `FluxFlowPipelineOutput` or `tuple`:
                If `return_dict` is `True`, `FluxFlowPipelineOutput` is returned,
                otherwise a `tuple` is returned with the generated images.

        Example:
            ```python
            from fluxflow.models import FluxFlowPipeline

            pipeline = FluxFlowPipeline.from_pretrained("checkpoint.safetensors")

            image = pipeline(
                prompt="a beautiful sunset over mountains",
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

            image.save("output.png")
            ```
        """
        # 1. Check inputs
        if prompt is None:
            raise ValueError("prompt cannot be None")

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode prompt
        text_embeddings = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

        # 3. Prepare latents
        total_batch = batch_size * num_images_per_prompt

        if latents is None:
            # Create random initial image and compress to latent
            shape = (total_batch, 3, height, width)
            if generator is not None:
                if isinstance(generator, list):
                    latents_list = [
                        torch.randn(
                            (1, 3, height, width),
                            generator=g,
                            device=device,
                            dtype=text_embeddings.dtype,
                        )
                        for g in generator
                    ]
                    latents = torch.cat(latents_list, dim=0)
                else:
                    latents = torch.randn(
                        shape,
                        generator=generator,
                        device=device,
                        dtype=text_embeddings.dtype,
                    )
            else:
                latents = torch.randn(shape, device=device, dtype=text_embeddings.dtype)

            # Normalize to [-1, 1]
            latents = latents * 2 - 1

            # Compress to latent space
            with torch.no_grad():
                latents = self.compressor(latents)
        else:
            # Validate provided latents format
            vae_dim = getattr(self.compressor, "d_model", latents.shape[-1])
            self._validate_latent_format(latents, total_batch, vae_dim)

        # Split into image sequence and hw vector
        hw_vec = latents[:, -1:, :].clone()
        lat = latents[:, :-1, :].clone()

        # 4. Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for CFG
                if do_classifier_free_guidance:
                    lat_input = torch.cat([lat] * 2)
                    hw_input = torch.cat([hw_vec] * 2)
                else:
                    lat_input = lat
                    hw_input = hw_vec

                # Prepare full input
                full_input = torch.cat([lat_input, hw_input], dim=1)

                # Predict noise
                t_batch = torch.full(
                    (full_input.size(0),), t.item(), device=device, dtype=torch.long
                )
                model_output = self.flow_processor(full_input, text_embeddings, t_batch)
                model_output = model_output[:, :-1, :]  # Remove hw vector

                # Classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                    model_output = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # Compute previous sample
                lat = self.scheduler.step(
                    model_output=model_output,
                    timestep=int(t.item()),
                    sample=lat,
                ).prev_sample

                # Callback
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, lat)

        # 6. Decode latents to image
        latents = torch.cat([lat, hw_vec], dim=1)

        if output_type == "latent":
            image = latents
        else:
            with torch.no_grad():
                image = self.expander(latents)

            # Post-process
            image = (image / 2 + 0.5).clamp(0, 1)

            if output_type == "pil":
                image = self.numpy_to_pil(image.cpu().permute(0, 2, 3, 1).float().numpy())
            elif output_type == "np":
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxFlowPipelineOutput(images=image)
