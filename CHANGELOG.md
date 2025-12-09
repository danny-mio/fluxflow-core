# Changelog

All notable changes to FluxFlow Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.2.0] - 2024-12-09

### Fixed
- **SPADE context handling**: Use `None` instead of `torch.zeros_like(feat)` when SPADE disabled
  - More efficient - avoids unnecessary tensor allocation
  - Semantically correct - `None` explicitly means 'no context'
  - Already supported by `ResidualUpsampleBlock` (checks `context is not None`)
  - Changes: `FluxExpander` now passes `ctx = None` when `use_context=False`
  - Added 6 unit tests in `tests/unit/test_expander_context.py` to verify behavior
  - All tests passing, linters clean

### Changed
- **FluxExpander docstrings**: Document that `context=None` disables SPADE conditioning

## [0.1.1] - 2024-11-XX

### Added
- **TrainableBezier activation** for per-channel learnable transformations
  - Optimized implementation with `torch.addcmul` (1.41× faster)
  - Inline computation with cached intermediate values
  - Used in VAE latent bottleneck (mu/logvar) and RGB output
  - Total 1,036 learnable parameters: 1,024 (latent) + 12 (RGB)
- Input channel dimension validation in `FluxCompressor.forward()` to catch shape mismatches early
- **TrainableBezier in VAE decoder RGB layer** for per-channel color correction (12 params)
- **TrainableBezier in VAE encoder latent bottleneck** for per-channel mu/logvar learning (1024 params)

### Changed
- Increased VAE attention layers from 2 to 4 for improved feature learning and global context modeling
- **VAE decoder `to_rgb` architecture**: wider channels (128→96→48→3) with GroupNorm+SiLU, no squashing
- **Optimized BezierActivation**: fused multiply-add operations, reduced module call overhead
- **Optimized TrainableBezier**: inlined computation, cached t², t³, t_inv², t_inv³

### Removed
- **SlidingBezierActivation** (25× slower than SiLU, 5× memory overhead from `unfold()`)
  - Replaced with standard BezierActivation where needed
  - Benchmark: 90sec/step → 8sec/step after removal
- Removed `--use_sliding_bezier` CLI argument
- Removed SlidingBezier exports from `__init__.py`
- Removed SlidingBezier tests

### Fixed
- **VAE color tinting issue**: Fixed to_rgb layer architecture to prevent color range squashing
- **Performance regression**: Removed SlidingBezier bottleneck from decoder path

### Technical Notes
- Spectral normalization was evaluated but removed due to numerical instability with Bezier activations on random weight initialization
- The existing Bezier activation pre-normalization (sigmoid/tanh/silu) provides sufficient gradient stability
- TrainableBezier uses sigmoid normalization for t and unbounded control points for maximum flexibility
