# FluxFlow Model Zoo

## 🚧 Training In Progress

**Status**: FluxFlow models are currently in systematic validation training.

**Expected Completion**: Late February 2026

**Current Phase**: Phase 1 - VAE Training (Weeks 1-4)

**Progress**:
- 🔄 Bezier VAE training in progress
- ⏳ ReLU baseline VAE pending
- ⏳ Flow training pending VAE completion
- ⏳ Empirical benchmarks pending training completion

---

## Planned Model Releases

Upon completion of training and validation, this page will contain:

### VAE Models

- **fluxflow-vae-bezier-v1.0** (128-dim latent)
  - Architecture: Bezier activations throughout
  - Target metrics: Will be published upon validation completion
  - Checkpoint size: ~427 MB (112M parameters)
  - Use case: Image compression/reconstruction

- **fluxflow-vae-relu-baseline-v1.0** (256-dim latent)
  - Architecture: Standard ReLU activations
  - Purpose: Baseline comparison for Bezier VAE
  - Checkpoint size: TBD (estimated ~500M parameters)
  - Use case: Empirical validation reference

### Flow (Diffusion) Models

- **fluxflow-flow-v1.0** (text-to-image)
  - Requires: Trained Bezier VAE
  - Features: Text conditioning, CFG support
  - Target quality: FID ≤ 20 on COCO validation
  - Use case: Production text-to-image generation

### Full Pipelines

- **fluxflow-complete-v1.0**
  - Components: VAE + Flow + Text Encoder
  - All-in-one checkpoint for inference
  - Optimized for deployment

---

## Model Specifications

### Bezier VAE Architecture

| Component | Parameters | Activation | Purpose |
|-----------|-----------|-----------|---------|
| FluxCompressor (Encoder) | 12.6M | BezierActivation | Image → latent encoding |
| FluxExpander (Decoder) | 94.0M | BezierActivation | Latent → image decoding |
| VAE Total | 106.6M | Mixed | Image compression/reconstruction |
| FluxFlowProcessor | 5.4M | BezierActivation | Diffusion transformer (Phase 2) |
| BertTextEncoder | 71.0M | Bezier projection | Text embedding (Phase 2) |
| **Complete Pipeline** | **183.0M** | Mixed | Full text-to-image |

### Training Configuration

Training details will be published alongside model releases, including:
- Dataset composition (COCO 2017, TTI-2M, etc.)
- Training hyperparameters (learning rates, batch sizes, optimizers)
- Hardware specifications (GPU type, VRAM, training time)
- Loss curves and convergence metrics
- Ablation study results

---

## Performance Metrics (Planned)

> **Note**: All metrics below are targets based on architecture analysis. Empirical measurements will be published upon training completion.

### VAE Quality Metrics

| Metric | ReLU Baseline (Target) | Bezier VAE (Target) | Expected Improvement |
|--------|----------------------|---------------------|---------------------|
| PSNR (dB) | 28.5±0.5 | ≥28.0 | Equivalent quality |
| LPIPS | 0.12±0.02 | ≤0.12 | Equivalent quality |
| FID (COCO val) | 15.2±0.3 | ≤15.0 | Equivalent quality |
| Parameters | ~500M | 183M | 2.7× smaller |
| Inference time (A100) | 1.82s | 1.12s | 38% faster |
| Training memory (batch=2) | 10.2GB | 4.1GB | 60% reduction |

### Flow Model Metrics

| Metric | Target |
|--------|--------|
| FID (COCO val, 50 steps) | ≤20.0 |
| CLIP Score | ≥0.30 |
| Inference time (512², 50 steps, A100) | ≤2.5s |
| CFG scale range | 1.0-15.0 (recommended: 3.0-7.0) |

---

## Download Links (Coming Soon)

### Hugging Face Hub

Trained models will be hosted on Hugging Face:
- `danny-mio/fluxflow-vae-bezier-v1.0`
- `danny-mio/fluxflow-vae-relu-baseline-v1.0`
- `danny-mio/fluxflow-flow-v1.0`
- `danny-mio/fluxflow-complete-v1.0`

### Direct Downloads

Alternative direct download links will be provided for:
- Individual component checkpoints
- Complete pipeline bundles
- Baseline comparison models

---

## Usage Examples (Coming Soon)

### Loading VAE Checkpoint

```python
from fluxflow.models import FluxCompressor, FluxExpander
from safetensors.torch import load_file

# Load VAE checkpoint
state_dict = load_file("fluxflow-vae-bezier-v1.0.safetensors")

encoder = FluxCompressor(d_model=128, in_channels=3)
decoder = FluxExpander(d_model=128)

# Load weights
encoder.load_state_dict({k.replace('compressor.', ''): v for k, v in state_dict.items() if 'compressor' in k})
decoder.load_state_dict({k.replace('expander.', ''): v for k, v in state_dict.items() if 'expander' in k})
```

### Loading Full Pipeline

```python
from fluxflow.models import FluxFlowPipeline

# Load complete pipeline (VAE + Flow + Text Encoder)
pipeline = FluxFlowPipeline.from_pretrained("danny-mio/fluxflow-complete-v1.0")

# Generate image
image = pipeline(
    prompt="a beautiful sunset over mountains",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
).images[0]

image.save("output.png")
```

---

## Benchmark Comparisons (Planned)

Upon release, this section will include:
- Head-to-head comparison: Bezier VAE vs ReLU baseline
- Comparison with Stable Diffusion VAE
- Ablation studies (Bezier config variations)
- Scaling analysis (different model sizes)
- Hardware performance across GPUs (A100, 4090, 3090, etc.)

---

## Changelog

### Upcoming v1.0 Release

- Initial model release with empirical validation
- Bezier VAE and ReLU baseline
- Flow model with CFG support
- Complete pipeline checkpoint

---

## Citation

Models released from this project should be cited as:

```bibtex
@software{fluxflow_models2026,
  title={FluxFlow Model Zoo: Efficient Text-to-Image Models with Bezier Activations},
  author={FluxFlow Contributors},
  year={2026},
  note={Trained models from the FluxFlow project},
  url={https://github.com/danny-mio/fluxflow-core}
}
```

---

## License

All models released will be under the MIT License, consistent with the FluxFlow codebase.

---

## Support

For questions about models:
- **Issues**: [GitHub Issues](https://github.com/danny-mio/fluxflow-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/danny-mio/fluxflow-core/discussions)
- **Training Status**: Check repository discussions for latest progress updates

---

**Last Updated**: December 14, 2025
**Next Update**: Expected February 2026 (upon training completion)
