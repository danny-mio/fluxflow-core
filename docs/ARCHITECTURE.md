# FluxFlow Architecture

## Core Innovation: Bezier Activation Functions

**Inspired by Kolmogorov-Arnold Networks (KAN)** [[Liu et al., 2024]](https://arxiv.org/abs/2404.19756), FluxFlow extends the concept of learnable activation functions to large-scale text-to-image generation. While KAN uses B-splines, FluxFlow employs **Cubic Bezier curves with dynamic parameter generation**, where all control points are derived from the input itself.

FluxFlow targets **2-2.5× smaller models** with equivalent perceptual quality through the strategic use of **Cubic Bezier activation functions**. Unlike traditional fixed activations (ReLU, GELU), Bezier activations provide 3rd-degree polynomial transformations where each output dimension can follow a different cubic curve based on its inputs.

### Mathematical Foundation

**Cubic Bezier Formula**:
```
B(t) = (1-t)³·p₀ + 3(1-t)²·t·p₁ + 3(1-t)·t²·p₂ + t³·p₃
```

**Representational Capacity**:
- Standard activation: **Fixed function** applied identically to all dimensions
- Bezier activation: **Cubic polynomial** with input-derived control points per dimension
- TrainableBezier: **4×D learnable parameters** for optimal per-dimension transformations

**Key Benefits** (theoretical targets based on architecture analysis):
1. **Smaller models**: 50% fewer channels target for equivalent quality (FID ≤15 target)
2. **Better gradients**: Smooth, continuous (C²) with reduced vanishing gradient issues
3. **Adaptive**: Each dimension can follow different cubic transformation
4. **Efficient**: 60% fewer parameters target, 38% faster inference target

### Strategic Activation Placement

| Component | Activation | Bezier Approach | Parameters | Rationale |
|-----------|-----------|----------------|------------|-----------|
| VAE Encoder/Decoder | BezierActivation | Input-Based | 0 activation (5× in Conv) | Complex image↔latent mapping, spatial dims |
| VAE Latent Bottleneck | TrainableBezier | Trainable | 4×D (1024 for D=256) | Per-channel mu/logvar learning |
| VAE RGB Output | TrainableBezier | Trainable | 12 (4×3 channels) | Per-channel color correction |
| Flow Transformer | BezierActivation | Pillar-Based | 4×depth×D² (~198K/block) | Context-dependent per-token activations |
| Text Encoder | BezierActivation | Input-Based | 0 activation (5× in Linear) | Semantic embedding space |
| Discriminator | LeakyReLU | N/A | 0 | Memory efficiency (2× calls/batch) |
| SPADE | ReLU | N/A | 0 | Simple affine transformation |

## Overview

FluxFlow combines a Variational Autoencoder (VAE) with a flow-based diffusion model for text-to-image generation. The architecture consists of five main components:

```
Text → BertTextEncoder → embeddings
                            ↓
Image → FluxCompressor → latent → FluxFlowProcessor → denoised → FluxExpander → output
           (VAE enc)       ↑            (flow)                      (VAE dec)
                           │
                    PatchDiscriminator (training only)
```

### Detailed Architecture Diagram

> **Note:** The diagram below depicts the pre-v0.10.0 architecture (mean-pooled
> text, single context vector, pooled text injection). v0.10.0 replaces the
> mean-pool / `B x D_text` text path with per-token `(text_seq, text_mask)`,
> swaps the pooled context vector for `ctx = f(img, z)` carried in the packed
> latent, and adds dual FiLM (text + time), 2D axial RoPE, and multi-scale
> SPADE in the decoder. See the "Components" sub-sections below for the
> v0.10.0 contract; the v0.6.x–v0.8.x behaviour summarised in this diagram is
> retained via the `_flow_processor_takes_pertoken_text` dispatcher.

```mermaid
graph TB
    subgraph "Input Layer"
        IMG[Input Image<br/>B x 3 x H x W]
        TXT[Text Prompt<br/>String]
        NOISE[Random Noise<br/>z1 from N 0,I]
    end

    subgraph "Text Encoding"
        TXT --> BERT[BertTextEncoder<br/>DistilBERT 6L 768H<br/>71.0M params]
        BERT --> |Mean Pool| PROJ[MLP Projection<br/>768-512-D_text]
        PROJ --> |Bezier Act| TEMB[Text Embeddings<br/>B x D_text]
    end

    subgraph "VAE Encoder - FluxCompressor"
        IMG --> |add xy coords| COORD[Coordinate Channels<br/>3 to 5]
        COORD --> DS1[Downscale 1<br/>ConvBezier 2x]
        DS1 --> DS2[Downscale 2<br/>ConvBezier 2x]
        DS2 --> DS3[Downscale 3<br/>ConvBezier 2x]
        DS3 --> DS4[Downscale 4<br/>ConvBezier 2x<br/>16x total]
        DS4 --> |5 to D| CEXP[Channel Expand<br/>vae_dim]
        CEXP --> REPARAM[Reparameterization<br/>mu, sigma to z]
        REPARAM --> FLAT[Flatten to Tokens<br/>B x T x D]
        FLAT --> POSENC[Hybrid PE<br/>Fixed + Content]
        POSENC --> SATTN[Self-Attention<br/>4 layers, 8 heads]
        SATTN --> APPHW[Append HW Vector<br/>B x T+1 x D]
        APPHW --> LATENT[Latent Packet<br/>B x T+1 x D]

        REPARAM -.->|KL Loss| KL[KL Divergence<br/>beta 0.001-0.01]
    end

    subgraph "Diffusion Transformer - FluxFlowProcessor"
        LATENT --> |Training| ADDNOISE[Add Noise<br/>zt = alpha*z0 + sigma*eps]
        NOISE --> |Generation| ZINIT[Initial Latent z1]
        ADDNOISE --> ZT[Noised Latent zt]
        ZINIT --> ZT

        ZT --> TEMBED[Timestep Embedding<br/>Sin-Cos + MLP]
        TEMB --> CROSSATTN[Text Injection<br/>Cross-Attention]

        TEMBED --> FLOW1[Transformer Block 1<br/>RoPE + Parallel Attn]
        CROSSATTN --> FLOW1
        FLOW1 --> FLOW2[Transformer Blocks 2-9<br/>Bezier MLP]
        FLOW2 --> FLOW10[Transformer Block 10]
        FLOW10 --> VPROJ[Output Projection<br/>to velocity v]
        VPROJ --> VPRED[Predicted v<br/>B x T+1 x D]

        VPRED -.->|v-prediction loss| VLOSS[MSE v, v_target<br/>v = alpha*eps - sigma*z0]
    end

    subgraph "VAE Decoder - FluxExpander"
        VPRED --> |Generation| DENOISE[Iterative Denoising<br/>z1 to z0]
        LATENT --> |VAE Training| Z0[Clean Latent z0]
        DENOISE --> Z0

        Z0 --> UNPACK[Unpack<br/>Extract T+1 tokens]
        UNPACK --> CTXPOOL[Context Pooling<br/>First K tokens]
        CTXPOOL --> CTX[Context Vector<br/>B x D]
        UNPACK --> RESHAPE[Reshape 2D<br/>D x H_lat x W_lat]

        RESHAPE --> UP1[Upsample 1<br/>TransConv 2x]
        CTX --> SPADE1[SPADE 1<br/>Spatial Conditioning]
        SPADE1 --> UP1
        UP1 --> UP2[Upsample 2<br/>TransConv 2x]
        CTX --> SPADE2[SPADE 2]
        SPADE2 --> UP2
        UP2 --> UP3[Upsample 3<br/>TransConv 2x]
        CTX --> SPADE3[SPADE 3]
        SPADE3 --> UP3
        UP3 --> UP4[Upsample 4<br/>TransConv 2x<br/>16x total]
        CTX --> SPADE4[SPADE 4]
        SPADE4 --> UP4
        UP4 --> RGBPROJ[RGB Projection<br/>D to 3 channels]
        RGBPROJ --> |Clamp -1 to 1| RECON[Reconstructed Image<br/>B x 3 x H x W]
    end

    subgraph "Discriminator - Training Only"
        RECON --> DISC[PatchDiscriminator<br/>45.1M params]
        IMG --> DISC
        DISC --> |Hinge Loss| GLOSS[GAN Loss<br/>+ R1 Penalty]
    end

    subgraph "Output"
        RECON --> OUT[Generated Image<br/>B x 3 x H x W]
    end

    subgraph "Loss Functions"
        KL --> VAELOSS[VAE Loss]
        RECON -.->|L1 + MSE| RECLOSS[Reconstruction]
        RECLOSS --> VAELOSS
        GLOSS --> VAELOSS

        VLOSS --> FLOWLOSS[Flow Loss]
    end

    classDef encoder fill:#e1f5ff,stroke:#0088cc,stroke-width:2px
    classDef decoder fill:#fff4e1,stroke:#ff8800,stroke-width:2px
    classDef flow fill:#e8f5e8,stroke:#00aa00,stroke-width:2px
    classDef text fill:#f5e8ff,stroke:#8800cc,stroke-width:2px
    classDef disc fill:#ffe8e8,stroke:#cc0000,stroke-width:2px
    classDef loss fill:#f0f0f0,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5

    class BERT,PROJ,TEMB text
    class COORD,DS1,DS2,DS3,DS4,CEXP,REPARAM,FLAT,SATTN,POSENC,APPHW,LATENT encoder
    class TEMBED,CROSSATTN,FLOW1,FLOW2,FLOW10,VPROJ,VPRED,ADDNOISE,ZINIT,ZT,DENOISE flow
    class UNPACK,CTXPOOL,CTX,RESHAPE,UP1,UP2,UP3,UP4,SPADE1,SPADE2,SPADE3,SPADE4,RGBPROJ,RECON,Z0 decoder
    class DISC,GLOSS disc
    class KL,VLOSS,RECLOSS,VAELOSS,FLOWLOSS loss
```

**Color Legend** (parameter counts for default config: vae_dim=128, feat_dim=128):
- **Blue**: VAE Encoder (FluxCompressor) - 12.6M params
- **Orange**: VAE Decoder (FluxExpander) - 94.0M params
- **Green**: Diffusion Transformer (FluxFlowProcessor) - 5.4M params
- **Purple**: Text Encoding (BertTextEncoder) - 71.0M params
- **Red**: GAN Discriminator (Training only) - 45.1M params
- **Gray (dashed)**: Loss functions

## Components

### 1. FluxCompressor (VAE Encoder)

**Purpose**: Encode images to compact latent representations

**Architecture (v0.10.0):**
- **Input**: RGB images `[B, 3, H, W]`
- **Output**: Packed latent `[B, T+1, 2*vae_dim]` carrying `z‖ctx + HW token`,
  where `T = (H//16) * (W//16)`. The z-half samples plain `N(0, I)`; the
  ctx-half is `ctx = f(img, z)` (SPADE-injected with `z` before its
  self-attention stack).

**Stages:**
1. Coordinate channels (add normalized x,y)
2. Progressive downsampling (4 stages, 2x each = 16x total)
3. Channel expansion: 5 → `vae_dim`
4. Reparameterization (μ, σ → z) using wide-range learnable logvar
   (`WideTrainableBezier`)
5. Conditional ctx coupling: `ctx = f(img, z)` via SPADE-style injection at
   the bottleneck (ctx-path conditioned on `z` before its 4 self-attention
   layers)
6. Flatten to tokens `[H_lat × W_lat, 2*vae_dim]` (z‖ctx)
7. Hybrid positional encoding (fixed sinusoidal + content-based from latent;
   the deterministic `+ pe_content` leak is removed)
8. Self-attention (4 layers, 8 heads)
9. Append HW token `[1, 2*vae_dim]` with normalized dimensions

**Key Features:**
- Bezier activations for nonlinearity
- Clean Gaussian `z`: KL pressure pulls toward `N(0, I)`; no `tanh` squash
  after LayerNorm on `z` / `ctx`
- KL divergence with free-bits constraint
- Gradient checkpointing for memory efficiency

### 2. FluxFlowProcessor (Diffusion Transformer)

**Purpose**: Denoise latent representations conditioned on text

**Architecture (v0.10.0, `FluxFlowProcessor_v100`):**
- **Input**:
  - `packed` — noised packed latent `[B, T+1, 2*vae_dim]` (z‖ctx + HW token)
  - `text_seq` — per-token text embeddings `[B, T_txt, embedding_size]`
  - `text_mask` — bool mask `[B, T_txt]` (True for valid tokens)
  - `timesteps` — `[B]` continuous diffusion time
- **Output**: Predicted v (velocity) `[B, T+1, 2*vae_dim]`

**Stages:**
1. Continuous sinusoidal time embedding (`sinusoidal_embedding(timesteps, d_model)`)
   piped through `time_mlp`; bucketed `Embedding(1000)` is removed.
2. Per-token text path: `text_proj` produces `text_seq` keys/values for
   cross-attention; `text_cond_proj` projects the first token (`text_cond`,
   the `[CLS]` position under DistilBERT) for FiLM.
3. Transformer blocks (default: 10 × `FluxTransformerBlock_v100`)
   - 2D axial RoPE on image tokens (built per `H_lat, W_lat`)
   - Self-attention on `img_seq`
   - Per-token cross-attention `img -> text` with `text_mask` (separate
     `norm2_q` / `norm2_kv`)
   - Dual independent FiLM (text + time, additive scales/biases)
   - Widened pillar MLPs (`D → 2D → 2D → D`, depth 3)
   - Bezier activation MLPs on `[img_seq | p0 | p1 | p2 | p3]`
4. GRU-style gated `ctx_agg ← gate · ctx_agg + (1 − gate) · ctx_delta_proj(img_seq.mean(1))`.
5. Output projection back to `2*vae_dim`.

**Key Features:**
- v-prediction (predicts velocity between noise and signal)
- Separate Q and KV projections for efficiency
- Per-token text cross-attention (length-1-degenerate `pillar_cross_attn` removed)
- Continuous time on a dedicated channel; dual FiLM (text + time)
- Backward-compat: pooled-text v060/v070 processors are routed via
  `_flow_processor_takes_pertoken_text` in `fluxflow.models.pipeline`.

### 3. FluxExpander (VAE Decoder)

**Purpose**: Decode latent tokens to RGB images

**Architecture (v0.10.0):**
- **Input**: Packed latent `[B, T+1, 2*vae_dim]` carrying `z‖ctx + HW token`
- **Output**: RGB images `[B, 3, H, W]`

**Stages:**
1. Unpack: split the packed token stream into the `z` half, the `ctx` half
   (`ctx = f(img, z)` produced by the compressor), and the trailing HW token.
2. Reshape both `z` and `ctx` tokens to 2D `[D, H_lat, W_lat]`; `ctx` is the
   spatial conditioning signal — no first-K-token pooling.
3. Progressive upsampling (4 stages, 2× each = 16× total)
   - Multi-scale SPADE conditioning at each stage (`SPADE_v100b`) fed by `ctx`
   - Transposed convolutions for upsampling
4. RGB projection (D → 3 channels)
5. Clamp to [-1, 1]

**Key Features:**
- Multi-scale SPADE (`beta_low` 1×1, `beta_mid` 3×3, `beta_hi` 3×3 dilated
  with effective 7×7 receptive field via `dilation=3`); bounded multiplicative
  gamma `1 + softplus(scale·raw) - softplus(0)`. Identity at init via
  zero-initialised `beta_scale` / `gamma_scale`.
- Bezier activations
- Skip connections via residuals

### 4. BertTextEncoder

**Purpose**: Encode text prompts to per-token embeddings

**Current Implementation (v0.10.0):**
- **Input**: `input_ids` `[B, T_txt]`, `attention_mask` `[B, T_txt]`
- **Output**: `(text_seq, text_mask)` where
  - `text_seq` — per-token embeddings `[B, T_txt, embed_dim]`
  - `text_mask` — bool mask `[B, T_txt]` (True for valid tokens)

**Stages:**
1. DistilBERT backbone (6 layers, 768 hidden) — `last_hidden_state` `[B, T_txt, 768]`
2. Two-stage Bezier projection producing per-token embeddings at `embed_dim`;
   `nn.Linear` and `BezierActivation` broadcast cleanly over the leading
   sequence dim, so the projection is applied independently to every token.
3. Mask passthrough: `text_mask = attention_mask.bool()` (or all-True if no
   mask is supplied).
4. Xavier initialization.

Mean pooling over the sequence was the v0.6–v0.8 path; v0.10.0 removes it
entirely so the flow cross-attention can attend over real tokens via the
`text_mask`.

**Design Note:**
The current implementation uses pre-trained DistilBERT as a practical starting point, allowing the project to focus on the core Bezier activation innovation in the VAE and flow components. This is a **temporary solution** - future development will replace this with a custom text encoder built from scratch using Bezier activations throughout, which will:
- Provide better alignment with the Bezier philosophy
- Enable end-to-end Bezier-based training
- Support planned multimodal extensions (text + image → image)
- Reduce reliance on external pre-trained models

**Key Features:**
- Frozen DistilBERT backbone (optional fine-tuning)
- Bezier activation in projection layers (per-token, broadcast over `T_txt`)
- Placeholder for future custom encoder

### 5. PatchDiscriminator (GAN Training)

**Purpose**: Distinguish real from generated images

**Architecture:**
- **Input**: RGB images [B, 3, H, W], optional context [B, D]
- **Output**: Patch logits [B, 1, H', W']

**Stages:**
1. Progressive downsampling (4 stages, 2x each)
2. Optional spectral normalization (`use_spectral_norm=False` by default)
3. LeakyReLU activations (memory-efficient)
4. Patch-level discrimination (not global)
5. Optional projection conditioning (Miyato-style)

**Key Features:**
- Hinge loss (non-saturating)
- R1 gradient penalty (every 16 steps)
- LeakyReLU for memory efficiency during GAN training

## Data Flow

### Training (VAE Only)

```
Image [B,3,H,W]
  ↓
Compressor
  ↓ (encode)
Latent [B,T,D] + μ,σ
  ↓ (decode)
Expander
  ↓
Reconstruction [B,3,H,W]
  ↓
L1 + MSE + β*KL(μ,σ)
  +
Discriminator(real) vs Discriminator(fake)
```

### Training (Flow Only)

```
Image → Compressor → Latent z₀
                      ↓ (add noise)
                    Noisy zₜ
                      ↓
            FlowProcessor(zₜ, text, t)
                      ↓
                  Predicted v
                      ↓
          MSE(v, v_target)
where v_target = αₜ*noise - σₜ*z₀
```

### Generation

```
Text → BertEncoder → embeddings
                        ↓
Random latent z₁ ────→ FlowProcessor(z₁, text, t₁) → z₀.₉
                        ↓
                    FlowProcessor(z₀.₉, text, t₀.₉) → z₀.₈
                        ...
                        ↓
                    FlowProcessor(z₀.₁, text, t₀.₁) → z₀
                        ↓
                    Expander(z₀)
                        ↓
                    Image
```

## Model Sizes

### Default Configuration (vae_dim=128, feat_dim=128)

| Component | Parameters | Memory (fp32) |
|-----------|-----------|--------------|
| FluxCompressor | 12.6M | ~50 MB |
| FluxFlowProcessor | 5.4M | ~22 MB |
| FluxExpander | 94.0M | ~376 MB |
| BertTextEncoder | 71.0M | ~284 MB |
| **Total (Generative)** | **183.0M** | **~732 MB** |
| PatchDiscriminator | 45.1M | ~180 MB (training only) |

Note: FluxExpander is asymmetrically larger than FluxCompressor due to progressive upsampling with SPADE conditioning at each stage.

### Memory Usage (Training, vae_dim=128, feat_dim=128)

**VAE Training** (with discriminator):
- Model weights: ~0.73 GB (generative) + 0.18 GB (discriminator) = 0.91 GB
- Optimizer states (AdamW, 2× params): ~1.82 GB
- Activations (batch=2, 512×512): ~3-4 GB
- Gradient buffers: ~0.91 GB
- **Total**: ~7-8 GB VRAM

**Flow Training** (frozen VAE):
- Model weights: ~0.73 GB
- Optimizer states (AdamW): ~1.46 GB
- Activations (batch=2, 512×512): ~2-3 GB
- Gradient buffers: ~0.73 GB
- **Total**: ~5-6 GB VRAM

### Scaling Options

**Smaller (for limited VRAM):**
```python
vae_dim=64, feat_dim=64
```
Estimated: ~70M parameters, ~4-5 GB VRAM (training)

**Default (balanced):**
```python
vae_dim=128, feat_dim=128
```
Measured: 183M parameters, ~7-8 GB VRAM (VAE training with GAN)

**Larger (for better quality):**
```python
vae_dim=256, feat_dim=256
```
Estimated: ~500M parameters, ~18-20 GB VRAM (training)

Note: Parameter scaling is approximately O(D²) for attention and linear layers.

## Key Design Decisions

### Why Bezier Activations?

Bezier activations are the **core innovation** of FluxFlow, providing adaptive cubic polynomial transformations through three distinct approaches.

**Mathematical Foundation**:
```
Standard neuron: y = σ(Wx + b)        # σ is fixed (ReLU, GELU, etc.)
Bezier neuron:   y = B(t; p₀,p₁,p₂,p₃) # B uses cubic polynomial
```

All three approaches compute: `B(t) = (1-t)³p₀ + 3(1-t)²t·p₁ + 3(1-t)t²·p₂ + t³·p₃`

**What differs:** How (t, p₀, p₁, p₂, p₃) are obtained.

#### 1. Input-Based BezierActivation

**Control point source:** Split from input channels (5→1 reduction)

```python
# Input: [B, 5C, H, W] → split into [t, p₀, p₁, p₂, p₃]
# Output: [B, C, H, W] after Bezier computation
BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu")
```

**Parameter cost:**
- Activation: 0 learnable parameters
- Previous layer: Must output 5× channels (e.g., Conv2d(C, 5C))
- Net effect: 5× parameters in previous layer

**Memory cost:**
- Peak: 5× intermediate tensors during forward pass
- Gradient: 5× gradient memory during backward pass

**Use case:** VAE encoder/decoder
- **Why here:** Convolutional layers benefit from 0 activation params
- **Trade-off:** Accept 5× Conv weights for 0 activation params

**Configuration:**
- **VAE encoding:** `t_pre="sigmoid", p_pre="silu"` - bounds t∈[0,1], smooth control points
- **VAE decoding (final):** `t_pre="silu", p_pre="tanh"` - preserve gradients, bound output to [-1,1]

#### 2. TrainableBezier

**Control point source:** Learned parameters (4 per dimension)

```python
# Input: [B, D] → t via sigmoid(input)
# Control points: Learned tensors [D] each
TrainableBezier((D,), channel_only=True, p0=-1.0, p3=1.0)
```

**Parameter cost:**
- Activation: 4×D learnable parameters
- Example: D=256 latent → 1024 params (4×256)
- Overhead: Minimal vs Linear layers (D² params)

**Memory cost:**
- Minimal: Dimension-preserving (no channel expansion)

**Use case:** VAE latent bottleneck, RGB output
- **Why here:** Small D makes 4×D negligible; per-channel learning critical
- **mu/logvar (D=256):** 1024 params for optimal latent distribution
- **RGB output (D=3):** 12 params for per-channel color correction

#### 3. Pillar-Based

**Control point source:** Generated by 4 separate depth-3 MLP networks (pillars)

See [BEZIER_ACTIVATIONS.md#pillar-based-bezier](BEZIER_ACTIVATIONS.md#pillar-based-bezier-fluxtransformerblock) for detailed implementation, parameter costs, and memory analysis.

**Summary:**
- **Parameter cost:** ~198K params per transformer block (4 pillars × 49.5K each)
- **Memory cost:** 10× base tensor size during forward pass
- **Use case:** Flow transformer MLP layers only (highest expressiveness needed)
- **Trade-off:** 6× ReLU params for context-dependent control points

#### Why Not Everywhere?

- **Discriminator:** LeakyReLU for memory efficiency
  - Called 2× per batch (real+fake)
  - Binary classification doesn't benefit from Bezier complexity
- **SPADE normalization:** ReLU for simple affine transformations
  - Spatially-adaptive denormalization is already expressive
  - Adding Bezier would complicate without clear benefit

### Why v-Prediction?

v-prediction (velocity) instead of noise prediction:
- More stable training than ε-prediction
- Better sample quality at high noise levels
- Balanced between signal and noise

Formula: `v = α_t * noise - σ_t * signal`

### Why SPADE in Decoder?

SPADE (Spatially-Adaptive Denormalization) provides spatial control:
- Context-dependent normalization
- Better preservation of spatial details
- Improved controllability for text conditioning

### Why Separate Compressor/Expander?

Unlike symmetric autoencoders:
- Compressor: Complex (self-attention, KL)
- Expander: Simpler (upsampling + SPADE)
- Allows independent optimization
- Better for progressive training (VAE → Flow)

## Model Versions

| Version | Module | Key Change | Status |
|---------|--------|-----------|--------|
| `0.10.0` | `v100/` | Bezier-coupled redesign: per-token text, multi-scale SPADE, clean Gaussian z, 2D axial RoPE | **Current** |
| `0.8.0` | `v080/flow.py` | Pillar-attention: FiLM + cross-attn on pillars | Stable |
| `0.7.0` | `v070/flow.py` | Context-enhanced transformer blocks | Stable |
| `0.6.0` | `v060/` | Default stable architecture | Stable |
| `0.3.0` | `v030/` | Legacy architecture | Legacy |

### v0.10.0: Bezier-Coupled Architecture

The five locked decisions of the v0.10.0 redesign — per-token text, conditional
`ctx = f(img, z)`, full Flow modernisation, multi-scale `SPADE_v100b`, and
clean Gaussian `z` — are described in place in the **Components** sub-sections
above (`FluxCompressor`, `FluxFlowProcessor`, `FluxExpander`, `BertTextEncoder`).
This sub-section collects the cross-cutting details that don't fit cleanly into
a single component.

**Compressor (z + ctx).** The compressor's z-path samples plain `N(0, I)`
latents via a wide-range learnable logvar (`WideTrainableBezier`, defaults
`p0=-8, p1=-2, p2=2, p3=4`); the `tanh` squash after LayerNorm on `z` and
`ctx` is removed, and the deterministic `+ pe_content` leak around the
bottleneck is removed so KL pressure pulls toward `N(0, I)`. The ctx-path is
SPADE-injected with `z` before its 4 self-attention layers, so `ctx = f(img, z)`
encodes the residual `z` could not capture. Packed token format
`[B, T+1, 2*vae_dim]` (z‖ctx + HW token) is preserved.

**Flow block reference (`FluxTransformerBlock_v100`).**

```
1. Self-attention on img with 2D axial RoPE
2. Cross-attention img -> text with separate norm2_q / norm2_kv,
   mask-aware via ParallelAttention(attn_mask=text_mask)
3. Gate g = sigmoid(img_seq)
4. Pillar MLPs (widened D->2D->2D->D, depth 3)
5. Dual FiLM applied independently (text + time, additive scales/biases)
6. (REMOVED) pillar_cross_attn — was length-1 degenerate over pooled text
7. FFN + BezierActivation on [img_seq | img_p0 | img_p1 | img_p2 | img_p3]
```

`text_cond` is the first token of `text_seq` (the `[CLS]` position when using
DistilBERT). `ctx_agg` is updated GRU-style:
`ctx_agg ← gate · ctx_agg + (1 − gate) · ctx_delta_proj(img_seq.mean(1))`.

**Backward-compat dispatcher.** Legacy `v060` / `v070` flow processors are
still supported by `fluxflow.models.pipeline._flow_processor_takes_pertoken_text`,
which inspects the loaded flow processor signature and routes pooled-text
checkpoints through the old path automatically. No caller changes are
required when loading older checkpoints.

**Checkpoint compatibility.** Old v0.10.0-pre, v0.7.x and v0.8.x checkpoints
are not weight-compatible with v0.10.0. Loading legacy single-scale SPADE
keys raises `IncompatibleCheckpointError`. The salvage script
`scripts/migrate_v0_10_0_to_redesign.py` warm-starts ~80% of params via
direct-copy, logvar rescale (`[-1, 1] → [-8, 4]`), SPADE partial-fill
(`mlp_beta → beta_mid`, zero-init new heads), pillar padding
(`D→D` → upper-left of widened `D→2D→2D→D`), FiLM duplication (text + time),
`norm2` duplication (q / kv), and explicit drop of legacy keys
(`time_embed`, `pillar_cross_attn`, `norm_pillar`). See
[`docs/MIGRATION-v0.10.0-redesign.md`](MIGRATION-v0.10.0-redesign.md).

### v0.8.0: Pillar-Attention Architecture

**Key idea**: Add direct text conditioning to each Bezier pillar, making pillar activation shapes text-aware.

**Problem with v0.7.0**: Text reaches pillars only indirectly — via `sigmoid(img_seq)` compression after cross-attention. This is lossy and prevents pillar activations from adapting directly to the prompt.

**Solution**: Two new sub-modules per `FluxTransformerBlock_v080`:

| Module | Type | Purpose |
|--------|------|---------|
| `film_p0 … film_p3` | `nn.Linear(d_model, 2*d_model)` × 4 | FiLM modulation per pillar |
| `pillar_cross_attn` | `ParallelAttention(d_model, n_pillar_heads)` shared | Text cross-attn on raw pillar outputs |
| `norm_pillar` | `nn.LayerNorm(d_model)` shared | LayerNorm before cross-attn |

`n_pillar_heads = max(1, n_head // 4)` — auto-computed, no new config required.

**Forward pass (new steps highlighted):**
```
1. Self-attention on img_seq                           (unchanged)
2. Cross-attention img_seq × text_seq                  (unchanged)
3. Form gate: g = sigmoid(img_seq)

4. [NEW] FiLM per pillar (text_cond = pooled text):
   gamma_i, beta_i = film_pi(text_cond).chunk(2, dim=-1)
   g_pi = g * (1 + gamma_i[:, None, :]) + beta_i[:, None, :]

5. Pillar MLP on FiLM-modulated gate:
   raw_pi = pillar_i(g_pi * p{i}_x if p{i}_x is not None else g_pi)

6. [NEW] Shared pillar cross-attention (Q=raw_pi, KV=text_seq):
   img_pi = raw_pi + pillar_cross_attn(norm_pillar(raw_pi), text_seq)

7. FFN + BezierActivation on [img_seq, p0…p3]          (unchanged)
8. Return img_seq, img_p0, img_p1, img_p2, img_p3
```

`text_cond` is the `[B, D]` tensor: `text_cond_proj(text_embeddings + time_embed(timesteps))`, extracted inside `FluxFlowProcessor_v080.forward()`. The external forward signature is **unchanged**.

**VAE is unchanged** — `v080/__init__.py` imports `FluxCompressor` and `FluxExpander` directly from `v070/vae.py`.

**Parameter overhead per block** (d=128, h=8):
- FiLM layers (`film_p0..p3`): 4 × (128 × 256 + 256) = 132,096
- Pillar cross-attn: ~66,048
- `norm_pillar`: 256
- **Total per block: ~+198,400 vs v0.7.0**

## Training Insights

### Two-Stage Training

**Stage 1 (VAE):**
- Learn good latent space
- Minimize reconstruction error
- Regularize with KL divergence
- Optional GAN for perceptual quality

**Stage 2 (Flow):**
- Learn text-to-latent mapping
- Denoise latent representations
- Use frozen VAE for stability

**Why separate?**
- Flow requires stable latents
- VAE trains faster alone
- Easier to debug issues
- Can reuse VAE for different flows

### Loss Functions

**VAE:**
```python
L_vae = L1(rec, real) + 0.1*MSE(rec, real) + β*KL(μ, σ)
L_gan_d = hinge(D(real), D(rec))
L_gan_g = -D(rec)
```

**Flow:**
```python
L_flow = MSE(pred_v, target_v)
where target_v = α_t * noise - σ_t * z₀
```

### Schedulers

**Learning rate**: Cosine annealing
- Starts at `LR`
- Decays to `LR * lr_min`
- Over total training steps

**KL weight**: Cosine warmup
- Starts at 0
- Increases to `kl_beta`
- Over `kl_warmup_steps`

**Noise schedule**: DPMSolver++
- Continuous-time formulation
- Order 2 solver
- Trailing timesteps

## Extensions

Active near-term work: empirical FID validation of the v0.10.0 redesign
against the ReLU baseline, a custom Bezier-only text encoder to replace
DistilBERT (see `BertTextEncoder` design note), and MPS/CUDA throughput
tuning of the widened pillar MLPs.
