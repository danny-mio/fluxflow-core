"""Type definitions for FluxFlow.

This module provides type aliases and protocols for better type safety.
"""

from typing import Protocol, TypedDict


class DimensionCacheData(TypedDict):
    """Structure of dimension cache data."""

    dataset_path: str
    captions_file: str | None
    scan_date: str
    total_images: int
    multiple: int
    size_groups: dict[str, dict[str, list[int] | int]]
    statistics: dict[str, int]


class SamplerState(TypedDict):
    """State dictionary for ResumableDimensionSampler."""

    seed: int
    position: int
    current_epoch: int
    batch_size: int


class TrainingState(TypedDict, total=False):
    """Training state for checkpointing."""

    version: str
    timestamp: str
    epoch: int
    batch_idx: int
    global_step: int
    kl_warmup: dict[str, float]
    learning_rates: dict[str, float]
    sampler_state: SamplerState


class TensorLike(Protocol):
    """Protocol for tensor-like objects."""

    def __getitem__(self, key): ...
    def size(self, dim: int | None = None): ...
    def to(self, device): ...
