# Changelog

All notable changes to FluxFlow Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

Work in progress toward the v0.10.0 release: a coordinated VAE + Flow +
text-path redesign organized around five locked decisions: per-token text,
conditional `ctx = f(img, z)` coupling, full Flow modernization (2D axial
RoPE, sinusoidal time, dual FiLM, widened pillars, gated `ctx_agg`),
multi-scale SPADE, and a clean Gaussian z. See
[`docs/MIGRATION-v0.10.0-redesign.md`](docs/MIGRATION-v0.10.0-redesign.md)
for the upgrade path and salvage instructions. Not yet released.

### Added
- **`WideTrainableBezier`** activation in `models/activations.py`: learnable
  Bezier with wide-range default control points (`p0=-8, p1=-2, p2=2, p3=4`)
  for variational logvar.
- **`SPADE_v100b`** multi-scale conditioning block with three additive heads
  (`beta_low` 1×1, `beta_mid` 3×3, `beta_hi` 3×3 dilated) and a bounded
  multiplicative gamma (`1 + softplus(scale·raw) - softplus(0)`); identity at
  init via zero-initialized `beta_scale` / `gamma_scale` parameters.
- **2D axial RoPE** helper (`build_axial_rope_2d`) with per-axis (H, W)
  buffers for the flow self-attention path.
- **Sinusoidal time embedding** (continuous; replaces the legacy
  `Embedding(1000)` bucketed path).
- **Widened `pillarLayerWide`** (`D→2D→2D→D`, depth 3) for the flow pillar
  MLPs.
- **Conditional ctx coupling** in the VAE compressor: ctx now encodes
  `f(img, z)` via SPADE-style injection of `z` into the ctx conv stack.
- **`build_cfg_null_pair`** helper in `utils/visualization.py` to produce
  `(null_text_seq, null_text_mask)` from the encoded empty prompt for CFG.
- **`IncompatibleCheckpointError`** raised when loading legacy SPADE keys
  (single-scale `mlp_shared` / `mlp_beta` / `mlp_gamma`) into v0.10.0.
- **`FluxTransformerBlock_v100`** and **`FluxFlowProcessor_v100`** flow
  modules with per-token text cross-attention, axial 2D RoPE on image tokens,
  dual independent FiLM (text + time), gated `ctx_agg` (GRU-style), and a
  widened pillarLayer.
- **Salvage script** `scripts/migrate_v0_10_0_to_redesign.py` covering seven
  conversion categories: direct-copy of unchanged tensors, logvar rescale
  from `[-1, 1]` to `[-8, 4]`, SPADE partial-fill (`mlp_beta → beta_mid`,
  zero-init new heads), pillar padding (`D→D` → upper-left of widened
  `D→2D→2D→D`), FiLM duplication (text + time), `norm2` duplication
  (q / kv), and explicit drop of legacy keys (`time_embed`,
  `pillar_cross_attn`, `norm_pillar`). Emits a one-page report. ~80% of
  params from `flxflow_final.safetensors` carry through.
- Mask-aware path on `ParallelAttention`: optional `attn_mask` keyword that
  applies `-inf` to invalid positions before softmax (`None` reproduces prior
  behavior).
- `@pytest.mark.compat_break` marker for tests that assert v0.10.0 refuses
  to silently load legacy v0.10.0-pre checkpoints.
- `fluxflow.models.detect_architecture_version(keys)` — the consolidated,
  version-aware architecture detector, replacing the duplicated
  v0.7.0-only heuristic across `pipeline.py`/`versioning.py`.
- `BertTextEncoder.load_with_override(checkpoint_path, override_path=None)` —
  3-tier text-encoder loading precedence (explicit override path > sibling
  `text_encoder.safetensors` > bundled `text_encoder.*` keys inside the main
  checkpoint), used by `fluxflow-training`, `fluxflow-ui`, and
  `fluxflow-comfyui`.
- **`SPADE_v100b.scale_drift()`**: returns the mean absolute deviation of
  `gamma_scale`/`beta_scale` from their (zero) init values, snapshotted as
  buffers at construction. Gives a direct, ground-truth signal that gradient
  is reaching SPADE's conditioning scales, independent of the "Ctx" console
  metric, which a training-diagnostics audit found was disconnected from
  SPADE's actual weights.
- **`fluxflow.DEFAULT_MAX_TEXT_LENGTH`** (`text_length.py`, re-exported from
  `fluxflow` and `fluxflow.utils`): single source of truth for the
  text-encoder token budget, currently `32`. Consumed by
  `diffusion_pipeline.py` and `utils/visualization.py`; `fluxflow-training`,
  `fluxflow-ui`, and `fluxflow-comfyui` import it with a `32` fallback until
  this version is installed everywhere.

### Changed
- **`BertTextEncoder.forward`** now returns `(text_seq, text_mask)` per
  token instead of a pooled `[B, E]` vector. Mean-pooling removed.
- **`FluxFlowProcessor_v100.forward`** signature is now
  `forward(packed, text_seq, text_mask, timesteps)` (per-token text +
  boolean mask).
- **`FluxPipeline.forward`** plumbs `(text_seq, text_mask)` end-to-end.
- **`SPADEWithLearnableScale`** is now a deprecated alias for `SPADE_v100b`
  (kept for one release; the original beta-only `SPADE` in
  `models/conditioning.py` is unchanged and remains the v060/v070 path).
- **CFG null** is now the encoded empty prompt (precomputed
  `(null_text_seq, null_text_mask)`) rather than `torch.zeros_like(...)`,
  closing the train/test CFG gap.
- **Visualization** (`utils/visualization.py`) updated to consume the
  per-token text tuple.
- **`fluxflow.models.pipeline._flow_processor_takes_pertoken_text`** added
  as a polymorphic dispatcher: inspects the loaded flow processor and routes
  v060/v070 checkpoints through the pooled-text path automatically.

### Removed
- `pillar_cross_attn` and the shared `norm_pillar` LayerNorm from the
  transformer block (the cross-attention was length-1 degenerate over pooled
  text).
- `Embedding(1000, d_model)` time-bucket path (replaced by continuous
  sinusoidal embedding).
- `tanh` squash after LayerNorm on both `z` and `ctx` tokens (was clipping
  the high-frequency tail of the latent distribution).
- Deterministic `+ pe_content` leak around the VAE bottleneck (the
  bottleneck is now genuinely variational so KL pressure pulls toward
  N(0, I)).
- The shared single `norm2` in the v0.10.0 cross-attention block is split
  into separate `norm2_q` and `norm2_kv` LayerNorms.

### Fixed
- **v0.10.0 checkpoints were misdetected as v0.7.0** when loaded without a
  `model_metadata.json` sidecar (the common case, since training never wrote
  one). `FluxFlowProcessor_v100` reuses `ctx_mixer`/`context_injection`/
  `context_final` submodule names inherited from v0.7.0, and the old
  heuristic had no v0.10.0 branch, so every v0.10.0 checkpoint reaching the
  fallback path was force-loaded using v0.7.0 classes with `strict=False` —
  silently dropping most of the trained weights. `detect_architecture_version()`
  now checks v0.10.0-exclusive markers (`ctx_gate_proj`/`ctx_delta_proj`/
  `time_mlp`) first, and `ModelLoaderLegacy` now routes detected v0.10.0
  checkpoints to the already-existing (but previously unreachable)
  `ModelLoaderV010`.
- **Train/inference text-length mismatch**: training tokenized captions with
  `max_length=32` while generation/inference hardcoded `max_length=512` at
  four call sites across four repos, silently truncating (or padding to a
  length never seen in training) with no logging either way. Both sides now
  import the shared `DEFAULT_MAX_TEXT_LENGTH` constant.

### Migration

Old v0.10.0-pre, v0.7.x and v0.8.x checkpoints are not weight-compatible
with v0.10.0; the salvage script provides a warm-start covering the
direct-copy paths and the partial-fill cases listed above. Full details
and an API surface delta table live in
[`docs/MIGRATION-v0.10.0-redesign.md`](docs/MIGRATION-v0.10.0-redesign.md).

CLI one-liner:

```bash
python scripts/migrate_v0_10_0_to_redesign.py \
    --src OLD.safetensors \
    --dst WARM.safetensors
```

After running the script, fine-tune with the standard v0.10.0 pipeline
config to recover quality.

## [0.8.1] - 2026-04-03

### Added
- **`fluxflow[mlx]` optional package**: Native Apple Silicon inference via MLX.
  `pip install fluxflow[mlx]` then `from fluxflow.mlx import FluxFlowPipelineMLX`.
  Includes weight converter (`.safetensors` → `.npz`, original never modified),
  MLX re-implementations of all core layers (Bezier activations, SPADE, FiLM, VAE decoder),
  and an experimental `mlx_forward` bridge for gradient-free hybrid MPS+MLX routing.
- **`fluxflow.utils.mps.mps_safe_pool2d`**: MPS-safe adaptive pooling utility replacing
  ad-hoc try/except patterns in training code.
- **`scripts/profile_mps.py`** (fluxflow-training): Profiles CPU fallbacks during a VAE
  forward+backward pass on MPS. Run on Apple Silicon to identify remaining bottlenecks.
- **`examples/config-mps.yaml`** (fluxflow-training): Recommended training config for
  Apple Silicon (batch_size=2, grad_accum=8, fp16=off, gradient_checkpointing=on).

### Fixed
- **Bezier JIT disabled on MPS**: `torch.jit.script` compiled functions were silently
  falling back to CPU on MPS. JIT is now bypassed for MPS tensors; the pure-PyTorch
  fallback path is used instead.
- **MPS cache clearing**: `torch.mps.empty_cache()` added alongside `torch.cuda.empty_cache()`
  in VAETrainer and FlowTrainer.

### Changed
- **SPADE: beta-only additive conditioning** (v060 & v070): Removed the multiplicative `gamma` path (`out = (1+gamma)*GroupNorm(x) + beta` → `out = GroupNorm(x) + beta`). Under MSE/L1 loss the model was learning near-zero or negative gamma values to damp down spatially uncertain sharp features, producing blurry reconstructions while SPADE was active. The additive-only design lets context inject spatial bias without suppressing sharp convolution features. Checkpoints with `mlp_gamma.*` keys load cleanly via existing `strict=False` loaders (extra keys are silently ignored).

### Removed
- **Dead `context_mixer` module** (v060 & v070 `FluxExpander` / `FluxExpanderBaseline`): `ContextAttentionMixer` was instantiated in `__init__` but never called in `forward()`. Removed the dead submodule and its unused import to eliminate wasted parameters and clarify the architecture. Existing checkpoints are unaffected (no state dict keys change).

## [0.8.0] - 2026-02-21

### Added
- **Pillar-attention architecture** (`v080/flow.py`)
  - `FluxTransformerBlock_v080`: adds FiLM conditioning and shared pillar cross-attention per transformer block
  - FiLM modulation per pillar: `gamma, beta = film_pi(text_cond).chunk(2)` → `gate * (1 + gamma) + beta`
  - Shared `pillar_cross_attn` (Q=stacked pillar outputs, KV=text_seq), `n_pillar_heads = _valid_pillar_heads(d_model, max(1, n_head // 4))` — guarantees divisibility
  - `norm_pillar` LayerNorm shared across all 4 pillars
  - `FluxFlowProcessor_v080`: adds `text_cond_proj` linear, passes pooled text conditioning to every block
- **`ModelLoaderV08`** in `versioning.py`
  - Loads/saves v0.8.0 checkpoints via `load_versioned_checkpoint()` / `save_versioned_checkpoint()`
  - Routes by `model_version` field in `model_metadata.json`; characteristic v0.8.0 state dict keys include `transformer_blocks.N.pillar_cross_attn` and `transformer_blocks.N.film_p0`
  - Compatible version range: `["0.8.1", "0.8.2"]`
- **`v080/__init__.py`** — version registry entry; imports `FluxCompressor` and `FluxExpander` from `v070/vae.py` (VAE unchanged)
- **Tests** (`tests/unit/test_flow_shapes_v080.py`): shape correctness, FiLM modulation effect, pillar cross-attn KV shape, gradient flow, GPU variant

### Changed
- VAE (`FluxCompressor`, `FluxExpander`) unchanged from v0.7.0
- External `FluxFlowProcessor` forward signature unchanged: `forward(packed, text_embeddings, timesteps)`
- `text_cond` extraction is internal to `FluxFlowProcessor_v080.forward()`; no training loop changes required
- **`text_embed_dim` default corrected to 1024** across all loaders and pipelines (`versioning.py`, `pipeline.py`, `diffusion_pipeline.py`) — previous default of 768 was a legacy error from DistilBERT's internal hidden size; FluxFlow's projection output has always been 1024

### Fixed
- Pillar cross-attention head count now uses `_valid_pillar_heads()` to guarantee `d_model % n_pillar_heads == 0`; previously `n_head // 4` could produce non-divisors causing runtime shape errors
- Versioned loader now passes `vae_latent_dim` (raw latent dim) instead of `flow_vae_dim` when instantiating `FluxFlowProcessor_v080`; `FluxFlowProcessor_v080` adds `CONTEXT_DIMS` internally so passing the pre-added value caused `vae_to_dmodel` to be sized `vae_dim + 2×CONTEXT_DIMS`, always mismatching real checkpoints

## [0.7.0] - 2026-02-11

### Added
- Versioned model registry with auto-discovery and v0.6.0/v0.7.0 model implementations
- `model_version` in configuration with validation and default v0.6.0
- v0.7.x checkpoint loader support with legacy auto-detection routing

### Changed
- Sampling and inference utilities normalize timesteps for v-prediction
- Sample generation follows iterative denoising with scheduler-based noise
- Legacy import paths now map to v0.6.0 model modules

## [0.6.0] - internal

> v0.6.0 is an **architecture version only**, not a standalone PyPI release. The v0.6.0 model module (`src/fluxflow/models/v060/`) was bundled with the v0.7.0 library release and remains the default stable architecture (`model_version: "0.6.0"` in config). It is not loadable via `load_versioned_checkpoint()` — use `FluxPipeline.from_pretrained()` for legacy v0.6.0 checkpoints.

### Added
- v0.6.0 model architecture (baseline context-enhanced flow transformer)

## [0.5.0] - 2025-12-23

### Added
- **Bezier Activation Performance Optimizations**
  - Expanded JIT compilation from 6 to 25 pre-activation combinations (417% increase)
  - LRU-cached power computations for 5-15% speedup on repeated forward passes
  - Comprehensive benchmark suite for performance validation
  - TorchScript export support for production deployment
  - 29 new optimization tests, all passing
  - 100% backward compatibility maintained

- **Baseline Model Architecture**
  - Added baseline models using standard activations (ReLU, GELU, SiLU) for comparative evaluation
  - Model factory system for creating Bezier vs Baseline models
  - Parameter-matched baseline variants for fair comparison
  - Integration tests for baseline training pipeline

- **Enhanced Documentation**
  - Complete FluxFlowPipeline API documentation
  - Comprehensive system requirements (CUDA, CPU, MPS)
  - Consolidated duplicate content into single source of truth
  - All code examples verified against source
  - Version references standardized to 0.5.0

### Fixed
- Mathematical accuracy in BEZIER_ACTIVATIONS.md (C∞ → C² smoothness)
- Module exports for optimization functions
- Cross-platform compatibility (CPU, CUDA, MPS)

### Changed
- Version numbering from 0.4.0 to 0.5.0

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

## [0.1.1] - 2024-11-01

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
