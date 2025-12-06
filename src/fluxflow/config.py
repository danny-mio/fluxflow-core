"""Configuration management for FluxFlow.

This module provides Pydantic-based configuration with YAML support,
replacing shell-based configuration with type-safe, validated configs.
"""

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    vae_dim: int = Field(
        default=128,
        ge=8,
        le=512,
        description="VAE latent dimension (8-512, recommended: 32/128/256)",
    )
    feature_maps_dim: int = Field(
        default=128, ge=8, le=1024, description="Flow processor feature dimension"
    )
    feature_maps_dim_disc: int = Field(
        default=8, ge=4, le=128, description="Discriminator feature dimension"
    )
    text_embedding_dim: int = Field(default=1024, description="Text embedding dimension")
    pretrained_bert_model: Optional[str] = Field(
        default=None, description="Path to pretrained BERT checkpoint"
    )


class DataConfig(BaseModel):
    """Dataset configuration."""

    data_path: Optional[str] = Field(default=None, description="Path to training images directory")
    captions_file: Optional[str] = Field(
        default=None, description="Tab-separated file: image_name<tab>caption"
    )
    use_tt2m: bool = Field(
        default=False, description="Use TTI-2M streaming dataset from HuggingFace"
    )
    tt2m_token: Optional[str] = Field(
        default=None, description="HuggingFace token for TTI-2M dataset access"
    )
    img_size: int = Field(default=1024, ge=64, le=2048, description="Maximum image size")
    channels: int = Field(
        default=3, ge=1, le=4, description="Number of image channels (1=grayscale, 3=RGB, 4=RGBA)"
    )
    tokenizer_name: str = Field(
        default="distilbert-base-uncased", description="HuggingFace tokenizer identifier"
    )

    @field_validator("data_path", "captions_file")
    @classmethod
    def validate_paths(cls, v):
        """Expand user paths like ~/"""
        if v is not None:
            return str(Path(v).expanduser())
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    n_epochs: int = Field(default=1, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=2, ge=1, description="Batch size per GPU")
    workers: int = Field(default=1, ge=0, description="Number of data loading workers")
    lr: float = Field(default=5e-7, gt=0, description="Learning rate for flow model")
    lr_min: float = Field(
        default=1e-1, gt=0, lt=1, description="Minimum LR multiplier for scheduler"
    )
    training_steps: int = Field(default=1, ge=1, description="Inner training steps per batch")
    use_fp16: bool = Field(default=False, description="Use mixed precision training")
    initial_clipping_norm: float = Field(default=1.0, gt=0, description="Gradient clipping norm")
    preserve_lr: bool = Field(
        default=False, description="Load saved learning rates from checkpoint"
    )

    # Training modes
    train_vae: bool = Field(default=False, description="Train VAE (compressor+expander)")
    train_no_gan: bool = Field(default=False, description="Disable GAN training for VAE")
    train_spade: bool = Field(default=True, description="Use SPADE conditioning")
    train_diff: bool = Field(default=False, description="Train flow model")
    train_diff_full: bool = Field(default=False, description="Train flow with full schedule")

    # KL divergence settings
    kl_beta: float = Field(default=0.0001, ge=0, description="Final KL divergence weight")
    kl_warmup_steps: int = Field(default=5000, ge=0, description="KL beta warmup steps")
    kl_free_bits: float = Field(default=0.0, ge=0, description="Free bits (nats) for KL divergence")

    # GAN settings
    lambda_adv: float = Field(default=0.5, ge=0, description="GAN adversarial loss weight")


class OptimizationConfig(BaseModel):
    """Optimizer and scheduler configuration."""

    optim_sched_config: Optional[str] = Field(
        default=None, description="Path to JSON file with detailed optimizer/scheduler settings"
    )


class OutputConfig(BaseModel):
    """Output and logging configuration."""

    output_path: str = Field(default="outputs", description="Directory for checkpoints and samples")
    log_interval: int = Field(default=10, ge=1, description="Log every N batches")
    sample_interval: int = Field(default=50, ge=1, description="Generate samples every N batches")
    no_samples: bool = Field(default=False, description="Disable sample generation during training")
    test_image_address: List[str] = Field(
        default_factory=list, description="Test images for VAE reconstruction"
    )
    sample_captions: List[str] = Field(
        default_factory=lambda: ["A sample caption"], description="Captions for sample generation"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )
    log_file: Optional[str] = Field(default=None, description="Optional log file path")


class FluxFlowConfig(BaseModel):
    """Complete FluxFlow configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    model_checkpoint: Optional[str] = Field(
        default=None, description="Path to checkpoint to resume from"
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FluxFlowConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            FluxFlowConfig instance

        Example:
            >>> config = FluxFlowConfig.from_yaml("config.yaml")
        """
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration

        Example:
            >>> config.to_yaml("config.yaml")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(exclude_none=True), f, default_flow_style=False, sort_keys=False
            )

    def validate_training_mode(self) -> None:
        """Validate that at least one training mode is enabled."""
        if not (
            self.training.train_vae or self.training.train_diff or self.training.train_diff_full
        ):
            raise ValueError(
                "No training mode enabled. Set at least one of: "
                "train_vae, train_diff, or train_diff_full"
            )

    def validate_dataset(self) -> None:
        """Validate dataset configuration."""
        if self.data.use_tt2m:
            if not self.data.tt2m_token:
                raise ValueError("tt2m_token required when use_tt2m=True")
        else:
            if not self.data.data_path or not self.data.captions_file:
                raise ValueError("data_path and captions_file required when use_tt2m=False")

    def validate_all(self) -> None:
        """Run all validation checks."""
        self.validate_training_mode()
        self.validate_dataset()


def load_config(path: str | Path) -> FluxFlowConfig:
    """
    Load and validate configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Validated FluxFlowConfig instance

    Example:
        >>> config = load_config("config.yaml")
        >>> config.validate_all()
    """
    config = FluxFlowConfig.from_yaml(path)
    config.validate_all()
    return config


def create_default_config(save_path: Optional[str | Path] = None) -> FluxFlowConfig:
    """
    Create a default configuration.

    Args:
        save_path: Optional path to save the configuration as YAML

    Returns:
        FluxFlowConfig with default values

    Example:
        >>> config = create_default_config("config.yaml")
    """
    config = FluxFlowConfig()
    if save_path:
        config.to_yaml(save_path)
    return config
