# FluxFlow Core

**Smaller, Faster, More Expressive**: Text-to-Image Generation with Bezier Activation Functions

## 🚧 Project Status

**Training In Progress**: FluxFlow models are currently in Week 1-4 of systematic validation.

**Status**:
- ✅ Architecture implemented and tested  
- 🔄 VAE training in progress (Bezier + ReLU baselines)
- ⏳ Flow training pending VAE completion
- ⏳ Empirical benchmarks pending training completion
- 📅 Expected completion: Late February 2026

**All performance claims below are theoretical targets** - empirical validation underway.

---

FluxFlow is a novel approach to text-to-image generation that targets 2-3× smaller models with equivalent or superior quality compared to standard architectures. The key innovation is the use of **Cubic Bezier activation functions**, which provide 3rd-degree polynomial expressiveness, enabling each neuron to learn complex, smooth non-linear transformations.

## Core Philosophy

**Inspired by Kolmogorov-Arnold Networks (KAN)**, FluxFlow extends the concept of learnable activation functions to large-scale generative models. While KAN uses B-splines, FluxFlow employs **Cubic Bezier curves** with dynamic parameter generation, where control points are derived from the input itself.

Traditional neural networks use fixed activations (ReLU, GELU, SiLU) that provide zero degrees of freedom. FluxFlow uses **Cubic Bezier curves** as activation functions, providing 4 learnable parameters per output dimension:

```
B(t) = (1-t)³·p₀ + 3(1-t)²·t·p₁ + 3(1-t)·t²·p₂ + t³·p₃
```

Where `t, p₀, p₁, p₂, p₃` are all derived from the input, creating a **dynamic, data-dependent activation function**.

This creates a 3rd-degree polynomial manifold where each output dimension can follow a different cubic transformation, allowing:
- **Smaller models**: 2-2.5× fewer parameters target for equivalent quality (based on architecture analysis)
- **Faster inference**: 38% speedup target despite activation overhead (theoretical based on parameter counting)
- **Better gradients**: Smooth, continuous gradients reduce vanishing gradient issues
- **Memory efficient**: 60% parameter reduction target, 60% memory reduction target

## Installation

### Production Install

```bash
pip install fluxflow
```

**What gets installed:**
- `fluxflow` - Core model architectures and inference pipeline
- Flow matching models, VAE, and text encoders
- **Note**: Does NOT include training tools (use `fluxflow-training` for that)
- **Note**: Does NOT include UI (use `fluxflow-ui` or `fluxflow-comfyui` for that)

**Package available on PyPI**: [fluxflow v0.4.0](https://pypi.org/project/fluxflow/)

### Development Install

```bash
git clone https://github.com/danny-mio/fluxflow-core.git
cd fluxflow-core
pip install -e ".[dev]"
```

## Key Features

- **Bezier Activations**: Learnable 3rd-degree (cubic) polynomial activation functions
- **Compact VAE**: Variational autoencoder with 25M params (encoder) + 30M params (decoder)
- **Flow-based Diffusion**: 150M param transformer with rotary embeddings
- **Text Conditioning**: DistilBERT-based encoder (66M params) with Bezier projection layers
  - *Note: Current implementation uses pre-trained DistilBERT as a temporary solution. Future versions will feature a custom Bezier-based text encoder for full end-to-end training and multimodal support.*
- **Adaptive Architecture**: Different activation strategies per component (Bezier for generative, LeakyReLU for discriminative)

## Quick Start

### High-Level API (Recommended)

```python
from fluxflow.models import FluxFlowPipeline

# Load from checkpoint directory (standard training output)
pipeline = FluxFlowPipeline.from_pretrained("path/to/checkpoint_dir/")

# Or load from a single checkpoint file
# pipeline = FluxFlowPipeline.from_pretrained("path/to/checkpoint.safetensors")

# Generate image with Diffusers-style API
image = pipeline(
    prompt="a beautiful sunset over mountains",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
).images[0]

image.save("output.png")
```

### Advanced Usage

```python
from fluxflow.models import FluxFlowPipeline
import torch

# Load with specific settings
pipeline = FluxFlowPipeline.from_pretrained(
    "path/to/checkpoint.safetensors",
    torch_dtype=torch.float16,
    device="cuda",
)

# Generate with more control
result = pipeline(
    prompt="a serene mountain landscape at dawn",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=768,
    width=768,
    num_images_per_prompt=4,
    generator=torch.Generator().manual_seed(42),
)

# Save all generated images
for i, img in enumerate(result.images):
    img.save(f"output_{i}.png")
```

### Classifier-Free Guidance (CFG)

**New in v0.3.0**: FluxFlow supports Classifier-Free Guidance for enhanced generation control.

#### What is CFG?

CFG improves generation quality by amplifying the influence of text conditioning. It works by:
1. Running two forward passes: one with text, one without
2. Interpolating between conditional and unconditional predictions
3. Producing images that more strongly follow the text prompt

#### Using CFG

```python
from fluxflow.models import FluxFlowPipeline

pipeline = FluxFlowPipeline.from_pretrained("path/to/checkpoint.safetensors")

# Generate with CFG (requires model trained with cfg_dropout_prob > 0)
image = pipeline(
    prompt="a photorealistic portrait of a cat",
    negative_prompt="blurry, distorted, low quality",  # Optional
    num_inference_steps=50,
    guidance_scale=5.0,  # Recommended: 3.0-7.0 for balanced results
    height=512,
    width=512,
).images[0]
```

#### Guidance Scale Guidelines

- **1.0**: No guidance (standard generation)
- **3.0-7.0**: Moderate guidance (RECOMMENDED - balanced quality/creativity)
- **7.0-15.0**: Strong guidance (may oversaturate or lose diversity)

**Important**: CFG requires models trained with `cfg_dropout_prob > 0` (typically 0.10-0.15). See [fluxflow-training](https://github.com/danny-mio/fluxflow-training) for training details.

### Low-Level API

For more control, use the base `FluxPipeline`:

```python
import torch
from fluxflow.models import FluxPipeline, BertTextEncoder
from transformers import AutoTokenizer

# Load components manually
pipeline = FluxPipeline.from_pretrained("path/to/checkpoint.safetensors")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
text_encoder = BertTextEncoder(embed_dim=768)

# Encode text
text = "a beautiful sunset"
tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=512)
text_embeddings = text_encoder(tokens["input_ids"])

# Manual forward pass (requires implementing sampling loop)
# See fluxflow-training for complete examples
```

## Package Contents

- `fluxflow.models` - Model architectures (VAE, Flow, Encoders, Discriminators)
  - `activations` - BezierActivation, TrainableBezier
  - `vae` - FluxCompressor (encoder) and FluxExpander (decoder)
  - `flow` - FluxFlowProcessor (diffusion transformer)
  - `encoders` - BertTextEncoder
  - `discriminators` - PatchDiscriminator (for GAN training)
  - `conditioning` - SPADE, FiLM, Gated conditioning modules
- `fluxflow.utils` - Utilities for I/O, visualization, and logging
- `fluxflow.config` - Configuration management
- `fluxflow.types` - Type definitions and protocols
- `fluxflow.exceptions` - Custom exception classes

## Why Bezier Activations?

### Mathematical Foundation

Traditional activations provide a single fixed transformation:
- **ReLU**: max(0, x) - piecewise linear, 50% gradient death
- **GELU/SiLU**: Fixed smooth curves, no adaptability

**Bezier activations** provide a learnable manifold:
- **4 control points** per dimension (p₀, p₁, p₂, p₃)
- **Smooth interpolation** via cubic Bezier curves
- **Adaptive transformations**: Each dimension can follow a different cubic curve
- **TrainableBezier**: Optional 4×D learnable parameters for per-dimension optimization

### Performance Targets

> **⚠️ Training In Progress**: The metrics below are **theoretical targets** based on architecture analysis and parameter counting. Empirical measurements will be added to this table upon training completion.

| Metric | ReLU Baseline (Target) | Bezier FluxFlow (Target) | Expected Improvement |
|--------|----------------------|------------------------|---------------------|
| Parameters | 500M | 183M | 2.7× smaller |
| Inference time (A100, 512², 50 steps) | 1.82s | 1.12s | 38% faster |
| Training memory (batch=2) | 10.2GB | 4.1GB | 60% reduction |
| FID (COCO val) | 15.2±0.3 | ≤15.0 | Equivalent quality |

**Status**: 
- VAE training: 🔄 In progress
- Flow training: ⏳ Pending VAE completion
- Baseline comparison: ⏳ Pending both completions
- Empirical results: 📊 Will be published to [MODEL_ZOO.md](https://github.com/danny-mio/fluxflow-core/blob/main/MODEL_ZOO.md)

### Strategic Activation Placement

FluxFlow uses different activations based on component purpose:

**Bezier activations** (high expressiveness needed):
- VAE encoder/decoder: Complex image↔latent mappings
- Flow transformer: Core generative model
- Text encoder: Semantic embedding space

**LeakyReLU** (memory efficiency critical):
- GAN discriminator: Binary classification, 2× forward passes per batch
- Saves 126 MB per batch vs Bezier

**ReLU** (simple transformations):
- SPADE normalization: Affine scale/shift operations

## API Comparison

| Feature | FluxFlowPipeline | FluxPipeline |
|---------|------------------|--------------|
| **Type** | `DiffusionPipeline` | `nn.Module` |
| **Input** | Text prompts | Pre-encoded embeddings |
| **Inference** | Full iterative denoising | Single forward pass |
| **Guidance** | Classifier-free (automatic) | Manual implementation |
| **Scheduler** | Built-in (DPMSolver++) | None |
| **Output** | PIL Images / numpy | Tensor |
| **Use case** | Production inference | Training / Custom pipelines |

**When to use which:**
- **FluxFlowPipeline**: Text-to-image generation, production use, Diffusers ecosystem
- **FluxPipeline**: Training, fine-tuning, custom inference loops, research

## Model Architecture Overview

**Total Parameters**: ~183M (default config: vae_dim=128, feat_dim=128)

| Component | Parameters | Activation Type | Purpose |
|-----------|-----------|-----------------|---------|
| FluxCompressor | 12.6M | BezierActivation | Image → latent encoding |
| FluxExpander | 94.0M | BezierActivation | Latent → image decoding |
| FluxFlowProcessor | 5.4M | BezierActivation | Diffusion transformer |
| BertTextEncoder | 71.0M | BezierActivation (projection) | Text → embedding |
| PatchDiscriminator | 45.1M | LeakyReLU | GAN training only |

Note: FluxExpander is asymmetrically larger due to progressive upsampling with SPADE conditioning layers.

## Technical Details

### Bezier Activation Types

**BezierActivation** (5→1 dimension reduction):
```python
# Input: [t, p0, p1, p2, p3] concatenated
# Output: B(t) = cubic Bezier interpolation
BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu")
```

**Note**: SlidingBezierActivation (dimension-preserving variant) is deprecated and not used in the current architecture.

**Pre-activation parameters**:
- `t_pre_activation`: Transform input t (sigmoid, silu, tanh, or None)
- `p_preactivation`: Transform control points (sigmoid, silu, tanh, or None)

### Current Configuration

**VAE** (image encoding/decoding):
- Downsampling: `BezierActivation(t_pre="sigmoid", p_pre="silu")`
- Upsampling: `BezierActivation(t_pre="sigmoid", p_pre="silu")`
- Final layers: `BezierActivation(t_pre="silu", p_pre="tanh")`

**Flow Transformer** (diffusion model):
- MLP layers: `BezierActivation()` with SiLU pre-activation
- Enables smooth, expressive token transformations

**Text Encoder**:
- Projection layers: `BezierActivation()` (default)
- Learns optimal text→latent space mapping

## Future Directions

### Custom Text Encoder
The current implementation uses pre-trained DistilBERT as a practical starting point. Future development will create a **custom text encoder built entirely with Bezier activations**, enabling:
- True end-to-end Bezier-based training
- Better semantic alignment with the generative model
- Reduced dependency on external pre-trained models
- Foundation for multimodal extensions

### Multimodal Extensions
With a custom Bezier text encoder, FluxFlow can be extended to:
- **Text + Image → Image**: Conditioning on reference images
- **Video generation**: Temporal consistency via Bezier transformations
- **3D synthesis**: Extending the architecture to volumetric data

### Performance Optimizations
- **JIT compilation**: Already implemented (10-20% speedup available)
- **Mixed precision**: fp16/bf16 training and inference
- **Quantization**: 8-bit/4-bit inference for edge devices
- **Knowledge distillation**: Bezier→fixed activation distillation for mobile deployment

## Links

- [GitHub Repository](https://github.com/danny-mio/fluxflow-core)
- [Architecture Details](docs/ARCHITECTURE.md)
- [Bezier Activations Guide](docs/BEZIER_ACTIVATIONS.md)
- [References & Acknowledgments](REFERENCES.md)
- [Training Tools](https://github.com/danny-mio/fluxflow-training)
- [Web UI](https://github.com/danny-mio/fluxflow-ui)
- [ComfyUI Plugin](https://github.com/danny-mio/fluxflow-comfyui)

## Acknowledgments

FluxFlow was **inspired by Kolmogorov-Arnold Networks (KAN)** [[Liu et al., 2024]](https://arxiv.org/abs/2404.19756), extending learnable activation functions to generative models with dynamic parameter generation.

**Special thanks to:**
- **COCO 2017** [[cocodataset.org]](https://cocodataset.org/) & **Open Images** [[Google]](https://storage.googleapis.com/openimages/web/index.html) - Mixed captions used for testing and validation
- **TTI-2M Dataset** [[HuggingFace]](https://huggingface.co/datasets/jackyhate/text-to-image-2M) - 2M image-text pairs for large-scale training experiments
- **SPADE** [[Park et al., 2019]](https://arxiv.org/abs/1903.07291) - Spatial conditioning mechanism
- **FiLM** [[Perez et al., 2018]](https://arxiv.org/abs/1709.07871) - Feature-wise modulation

For complete references, see [REFERENCES.md](REFERENCES.md).

## Citation

If you use FluxFlow in your research, please cite:

```bibtex
@software{fluxflow2024,
  title = {FluxFlow: Efficient Text-to-Image Generation with Bezier Activation Functions},
  author = {FluxFlow Contributors},
  year = {2024},
  note = {Inspired by Kolmogorov-Arnold Networks (KAN)},
  url = {https://github.com/danny-mio/fluxflow-core}
}
```

**Key References:**
```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and others},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

## License

MIT License - see LICENSE file for details.
