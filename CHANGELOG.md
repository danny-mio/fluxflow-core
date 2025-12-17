# Changelog

All notable changes to FluxFlow Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.4.0] - 2025-12-17

### Fixed
- **CRITICAL: Image Color and Contrast Bug**
  - Fixed incorrect image saving in `visualization.py` that caused severe contrast issues
  - Expander outputs images in `[-1, 1]` range, but `save_image()` was treating them as `[0, 1]`
  - This caused negative pixel values to be clamped to black, crushing 50% of the dynamic range
  - Added `normalize=True` and `value_range=(-1, 1)` to all `save_image()` calls
  - **Affected functions**: `safe_vae_sample()`, `save_sample_images()`
  - **Impact**: All training sample images and generated images now have correct colors and contrast
  - **Files**: `src/fluxflow/utils/visualization.py`
  - **Note**: fluxflow-ui and fluxflow-comfyui were not affected (they handle conversion manually)

### Added
- **Model Versioning System**
  - Explicit model version metadata stored alongside checkpoints
  - Automatic version detection and routing to appropriate loaders
  - Backward compatibility with legacy checkpoints (auto-detection)
  - Forward compatibility detection (clear errors for newer models)
  - Semantic versioning support (MAJOR.MINOR.PATCH)
  - Architecture metadata eliminates config inference
  - Checksum validation for integrity verification
  - **Files**: `src/fluxflow/models/versioning.py`, `docs/VERSIONING.md`, `docs/MIGRATION.md`
  - **Migration tool**: `scripts/migrate_checkpoints.py` for upgrading legacy checkpoints
  - **API Changes**:
    - `FluxPipeline.from_pretrained()` gains `use_versioning` parameter (opt-in, default: False)
    - `FluxFlowPipeline.from_pretrained()` gains `use_versioning` parameter (opt-in, default: False)
    - `save_model()` gains `save_metadata`, `model_version`, and `training_info` parameters
    - New functions: `load_versioned_checkpoint()`, `save_versioned_checkpoint()`
  - **Testing**: Comprehensive unit tests in `tests/unit/test_versioning.py`

- **CFG Support in Sample Generation**
  - Added `use_cfg` and `guidance_scale` parameters to `save_sample_images()` function
  - New `_generate_with_cfg()` helper function for CFG-guided sample generation
  - Enables classifier-free guidance during training sample generation
  - Default guidance scale: 5.0 (balanced quality/creativity)
  - Compatible with models trained with `cfg_dropout_prob > 0`
  - **Files**: `src/fluxflow/utils/visualization.py`

## [0.3.1] - 2025-12-13

### Note
- Re-release of 0.3.0 with corrected version number
- v0.3.0 does not exist on PyPI due to release process issues
- All features from 0.3.0 are included in this release

## [0.3.0] - 2025-12-12

### Added
- **Classifier-Free Guidance (CFG)** support for enhanced inference control
  - New `use_cfg` parameter in generation pipeline for toggling CFG
  - `guidance_scale` parameter (range 1.0-15.0) to control conditioning strength
  - Negative prompts for better control over unwanted features
  - Dual-pass sampling implementation for CFG inference
  - Compatible with models trained with `cfg_dropout_prob > 0`
- Infrastructure for CFG-aware model loading and validation
- Enhanced configuration validation for CFG parameters

### Changed
- Updated version to 0.3.0 across all modules
- Improved inference pipeline to support both standard and CFG sampling modes
- Enhanced documentation with CFG usage examples

## [0.2.1] - 2024-12-09

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
