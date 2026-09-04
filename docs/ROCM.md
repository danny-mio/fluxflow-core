# AMD ROCm / gfx1151 Support

**Status: experimental, unvalidated.** ROCm/gfx1151 (AMD Strix Halo APU, e.g.
Minisforum MS-S1 Max) support has not yet been empirically tested on real
hardware. Treat all guidance and numeric defaults on this page as targets
pending validation, per this repo's convention of hedging unvalidated
performance claims (see `AGENTS.md`).

## Installation

ROCm and a matching PyTorch build must be installed separately — this
project does not manage or pin that install. Example, using AMD's nightly
wheel index (targets gfx1151):

```bash
pip install -U --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "rocm[libraries,device-gfx1151]"
pip install -U --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    "torch[device-gfx1151]" "torchvision[device-gfx1151]" torchaudio
```

Install FluxFlow normally afterward (`pip install -e .` or from PyPI) — no
FluxFlow-specific installation step is required for ROCm.

## How device detection works

ROCm-build PyTorch reports `torch.cuda.is_available() == True` and behaves
like CUDA for all compute purposes (tensors, kernels, `Accelerator`). FluxFlow
auto-detects it via the existing `cuda > mps > cpu` priority chain in
`fluxflow.utils.device.get_device()` — no separate code path is needed for
ROCm compute.

To tell "real NVIDIA CUDA" apart from "ROCm reporting as CUDA" for logging or
diagnostics, use:

```python
from fluxflow.utils.device import is_rocm, get_device_info

print(is_rocm())          # True only on a ROCm/HIP build
print(get_device_info())  # DeviceInfo(backend="rocm"/"cuda"/"mps"/"cpu", ...)
```

This distinction is internally based on `torch.version.hip` — set only on
ROCm wheels.

## Attention backend: SDPA (default) vs. einsum (fallback)

FluxFlow's flow-transformer attention defaults to `"sdpa"`
(`attn_backend="sdpa"` / `attention_backend: sdpa`,
`torch.nn.functional.scaled_dot_product_attention`) — benchmarking found it
the fastest backend on ROCm, CUDA, and MPS. The original hand-rolled
`"einsum"` implementation remains available as a fallback for
numerical-comparison or debugging purposes:

- YAML: `model.attention_backend: einsum`
- CLI: `--attention_backend einsum` (fluxflow-training)

**Coverage:** only `v0.7.0`, `v0.8.0`, and `v0.10.0` models support this flag
today (`fluxflow.models.v070/v080/v100`). `v0.3.0`/`v0.6.0` and baseline
models have their own independent attention implementation and ignore the
flag entirely.

SDPA is numerically close to the `einsum` path but **not bit-identical** (a
fused kernel may use a different reduction order) — this has been verified to
within float tolerance on CPU (`tests/unit/test_parallel_attention_sdpa.py`).
If you need bit-identical behavior with older checkpoints/traces for
debugging, switch back to `einsum`.

## TunableOp (GEMM autotuning)

PyTorch's `TunableOp` can autotune GEMM kernels (rocBLAS + hipBLASLt
candidates) at runtime. This is env-var driven — no FluxFlow code changes are
needed:

```bash
PYTORCH_TUNABLEOP_ENABLED=1 \
PYTORCH_TUNABLEOP_TUNING=1 \
PYTORCH_TUNABLEOP_FILENAME=tunableop_results.csv \
python -m fluxflow_training.scripts.train --config examples/config-rocm.yaml ...
```

The first run with `PYTORCH_TUNABLEOP_TUNING=1` is slower (it's searching for
the best kernel per shape). Subsequent runs should set
`PYTORCH_TUNABLEOP_TUNING=0` and reuse the CSV. Untested/experimental —
report your results if you try it.

## Unified memory (manual, not automated)

Strix Halo's system memory is shared between a small BIOS VRAM carve-out and
a much larger GTT-mapped pool controlled by Linux kernel boot parameters.
This project does **not** configure this automatically — it's a host/kernel
setting, not something FluxFlow can or should manage. Add to your kernel
command line (e.g. `GRUB_CMDLINE_LINUX` in `/etc/default/grub`, or a
bootloader entry):

```
amdgpu.gttsize=<N>       # GTT (system-RAM-backed graphics) size in MB
ttm.pages_limit=<N>      # TTM page pool limit (must be >= gttsize in pages)
amd_iommu=off            # if IOMMU is causing GTT allocation issues
```

Recommended values are hardware- and workload-dependent (they scale with your
total system RAM and how much you're willing to dedicate to the GPU) —
consult AMD's ROCm unified-memory documentation for your specific
configuration rather than copying a fixed number from here.

## Known caveats

- **VRAM warning message**: `fluxflow-training`'s high-memory-usage warning
  during training reflects the current VRAM/GTT carve-out on ROCm, not total
  system RAM. If it fires spuriously, revisit the kernel parameters above.
- **SDPA coverage gap**: `v0.3.0`/`v0.6.0`/baseline models don't support the
  `attn_backend`/`attention_backend` flag (see above) — they always use their
  own hand-rolled attention.
- **Untested end-to-end**: no ROCm hardware has been used to validate any of
  the above. Perf claims are targets/theoretical only until someone runs
  this on real gfx1151 hardware and reports back.

## See also

- `fluxflow-training/examples/config-rocm.yaml` — starting-point training
  config for Strix Halo (conservative batch size, gradient checkpointing on).
