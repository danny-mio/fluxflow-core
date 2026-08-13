"""Device auto-detection and ROCm/CUDA/MPS/CPU diagnostics.

Consolidates device-selection logic that was previously duplicated across
fluxflow-training, fluxflow-ui, and fluxflow-comfyui. ROCm-build PyTorch
reports torch.cuda.is_available() == True and behaves like CUDA for all
compute purposes (tensors, kernels, Accelerator) -- no separate code path
is needed for ROCm compute. This module only adds a *diagnostic* layer so
callers can tell "real NVIDIA CUDA" apart from "ROCm pretending to be CUDA"
for logging, warnings, and UX messaging.

ROCm/gfx1151 (Strix Halo) support is experimental and not yet empirically
validated on real hardware -- see docs/ROCM.md.
"""

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["get_device", "is_rocm", "get_device_info", "parse_device", "DeviceInfo"]


@dataclass(frozen=True)
class DeviceInfo:
    """Diagnostic snapshot of the active torch device.

    Attributes:
        device: Resolved torch.device.
        backend: One of "cuda", "rocm", "mps", "cpu". "rocm" is reported
            instead of "cuda" whenever torch.version.hip is set, even
            though device.type is still "cuda" for a ROCm build.
        device_name: Human-readable GPU/CPU name (best-effort; empty string
            if unavailable).
        rocm_version: torch.version.hip string, or None on non-ROCm builds.
    """

    device: torch.device
    backend: str
    device_name: str
    rocm_version: Optional[str]

    def __str__(self) -> str:
        if self.rocm_version:
            return f"{self.backend} ({self.device_name}, ROCm {self.rocm_version})"
        if self.device_name:
            return f"{self.backend} ({self.device_name})"
        return self.backend


def is_rocm() -> bool:
    """Return True if the active torch build is a ROCm/HIP build.

    Distinguishes "real NVIDIA CUDA" from "ROCm reporting as CUDA" via the
    torch.version.hip attribute, which is only set (non-None) on ROCm
    wheels. Safe to call regardless of whether a GPU is actually present.
    """
    return getattr(torch.version, "hip", None) is not None


def get_device() -> torch.device:
    """Auto-detect the best available device: CUDA/ROCm > MPS > CPU.

    ROCm-build PyTorch satisfies torch.cuda.is_available() and is returned
    as torch.device("cuda") here -- identical behavior to real CUDA. Use
    is_rocm() or get_device_info() separately if you need to distinguish
    them for logging.

    Returns:
        torch.device instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(device: Optional[torch.device] = None) -> DeviceInfo:
    """Build a diagnostic DeviceInfo for the given (or auto-detected) device.

    Args:
        device: Device to describe. If None, calls get_device().

    Returns:
        DeviceInfo with backend classification and best-effort device name.
    """
    device = device or get_device()
    device_name = ""
    rocm_version = None

    if device.type == "cuda":
        rocm_version = torch.version.hip if is_rocm() else None
        backend = "rocm" if rocm_version else "cuda"
        try:
            device_name = torch.cuda.get_device_name(device)
        except Exception:
            device_name = ""
    elif device.type == "mps":
        backend = "mps"
        device_name = "Apple Silicon"
    else:
        backend = "cpu"
        device_name = ""

    return DeviceInfo(
        device=device, backend=backend, device_name=device_name, rocm_version=rocm_version
    )


def parse_device(device_str: str) -> torch.device:
    """Parse a device string into a torch.device, resolving "auto".

    Args:
        device_str: "auto", "cuda", "cpu", "mps", "cuda:0", etc. Any string
            torch.device() itself accepts is passed through unchanged.

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        return get_device()
    return torch.device(device_str)
