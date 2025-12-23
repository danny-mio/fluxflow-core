# FluxFlow API Reference

Complete API reference for FluxFlow's versioning and checkpoint management system.

## Table of Contents

- [Loading Checkpoints](#loading-checkpoints)
  - [load_versioned_checkpoint()](#load_versioned_checkpoint)
  - [FluxPipeline.from_pretrained()](#fluxpipelinefrom_pretrained)
- [Saving Checkpoints](#saving-checkpoints)
  - [save_versioned_checkpoint()](#save_versioned_checkpoint)
- [Metadata Management](#metadata-management)
  - [ModelMetadata](#modelmetadata)
- [Version Registry](#version-registry)
  - [ModelVersionRegistry](#modelversionregistry)

---

## Loading Checkpoints

### `load_versioned_checkpoint()`

Load FluxFlow checkpoint with automatic version detection and routing.

**Location:** `fluxflow.models.versioning`

**Signature:**
```python
def load_versioned_checkpoint(
    checkpoint_path: Path,
    device: Optional[str] = None,
    **kwargs
) -> Any
```

**Parameters:**
- `checkpoint_path` (Path): Path to checkpoint file or directory
- `device` (Optional[str]): Target device ('cuda', 'cpu', 'mps', or None for auto-detection)
- `**kwargs`: Additional arguments passed to version-specific loader

**Returns:**
- Loaded model (FluxPipeline or FluxFlowPipeline)

**Raises:**
- `CheckpointError`: If checkpoint is incompatible or corrupted

**Example:**
```python
from fluxflow.models.versioning import load_versioned_checkpoint

# Load from directory (recommended)
pipeline = load_versioned_checkpoint("checkpoints/model_v0.5.0/")

# Load from specific file
pipeline = load_versioned_checkpoint("checkpoints/model.safetensors")

# Specify device
pipeline = load_versioned_checkpoint("checkpoints/model_v0.5.0/", device="cuda")
```

**Behavior:**
- Auto-detects model version from `model_metadata.json`
- Routes to appropriate version-specific loader
- Falls back to legacy loading if no metadata found
- Auto-detects device if not specified
- Supports both directory and file paths

**See Also:**
- [VERSIONING.md](VERSIONING.md) - Versioning system overview
- [MIGRATION.md](MIGRATION.md) - Migration guides between versions

---

### `FluxPipeline.from_pretrained()`

Load FluxPipeline from pretrained checkpoint (class method).

**Location:** `fluxflow.models.FluxPipeline`

**Signature:**
```python
@classmethod
def from_pretrained(
    cls,
    checkpoint_path: str,
    device: Optional[str] = None,
    use_versioning: bool = False,
    **kwargs
) -> FluxPipeline
```

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file or directory
- `device` (Optional[str]): Device to load on ('cuda', 'cpu', 'mps', or None)
- `use_versioning` (bool): Enable versioned loading system (default: False for backward compatibility)
- `**kwargs`: Additional arguments for model initialization

**Returns:**
- `FluxPipeline`: Loaded model ready for inference

**Example:**
```python
from fluxflow.models import FluxPipeline

# Legacy loading (default, for backward compatibility)
pipeline = FluxPipeline.from_pretrained("path/to/checkpoint.safetensors")

# Versioned loading (recommended for new code)
pipeline = FluxPipeline.from_pretrained(
    "path/to/checkpoint/",
    use_versioning=True
)

# Specify device
pipeline = FluxPipeline.from_pretrained(
    "path/to/checkpoint/",
    use_versioning=True,
    device="cuda"
)
```

**Behavior:**
- **Default (use_versioning=False)**: Legacy loading for backward compatibility
- **Versioned (use_versioning=True)**: Delegates to `load_versioned_checkpoint()` for robust version handling
- Auto-detects device if not specified

**Migration Note:**
New projects should use `use_versioning=True` or call `load_versioned_checkpoint()` directly. Legacy mode exists for backward compatibility with existing codebases.

---

## Saving Checkpoints

### `save_versioned_checkpoint()`

Save FluxFlow checkpoint with version metadata.

**Location:** `fluxflow.models.versioning`

**Signature:**
```python
def save_versioned_checkpoint(
    model: Any,
    output_path: Path,
    model_version: str = "0.5.0",
    architecture: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None
```

**Parameters:**
- `model` (Any): Model to save (FluxPipeline or FluxFlowPipeline)
- `output_path` (Path): Directory to save checkpoint
- `model_version` (str): Model version string (semantic versioning, default: "0.5.0")
- `architecture` (Optional[Dict]): Architecture config (auto-detected if None)
- `training_info` (Optional[Dict]): Optional training metadata (steps, dataset, etc.)
- `**kwargs`: Additional arguments

**Returns:**
- None (saves to disk)

**Saves:**
- `model.safetensors` - Model weights in SafeTensors format
- `model_metadata.json` - Version and architecture metadata

**Example:**
```python
from fluxflow.models.versioning import save_versioned_checkpoint

# Basic save
save_versioned_checkpoint(
    pipeline,
    "outputs/model_v0.5.0/",
    model_version="0.5.0"
)

# With training info
save_versioned_checkpoint(
    pipeline,
    "outputs/model_v0.5.0/",
    model_version="0.5.0",
    training_info={
        "total_steps": 50000,
        "dataset": "COCO",
        "batch_size": 4,
        "learning_rate": 5e-5
    }
)

# Custom architecture (rarely needed, auto-detected by default)
save_versioned_checkpoint(
    pipeline,
    "outputs/model_v0.5.0/",
    model_version="0.5.0",
    architecture={
        "vae_dim": 128,
        "feature_maps_dim": 128,
        "text_embed_dim": 768
    }
)
```

**Behavior:**
- Auto-detects architecture from model if not provided
- Creates output directory if it doesn't exist
- Saves both weights and metadata
- Uses SafeTensors format for cross-framework compatibility

**See Also:**
- [VERSIONING.md](VERSIONING.md#saving-versioned-checkpoints) - Detailed saving guide

---

## Metadata Management

### `ModelMetadata`

Model metadata container for versioning system.

**Location:** `fluxflow.models.versioning`

**Attributes:**
```python
@dataclass
class ModelMetadata:
    model_version: str              # Semantic version (e.g., "0.5.0")
    library_version: str            # FluxFlow library version
    architecture: Dict[str, Any]    # Architecture config
    components: Dict[str, str]      # Component names/types
    training_info: Optional[Dict[str, Any]] = None  # Training metadata
    metadata_version: str = "1.0"   # Metadata schema version
    created_at: Optional[str] = None  # ISO timestamp
```

**Methods:**

#### `load(path: Path) -> ModelMetadata`
Load metadata from JSON file.

```python
from fluxflow.models.versioning import ModelMetadata

metadata = ModelMetadata.load("checkpoints/model_metadata.json")
print(f"Model version: {metadata.model_version}")
print(f"VAE dim: {metadata.architecture['vae_dim']}")
```

#### `save(path: Path) -> None`
Save metadata to JSON file.

```python
from fluxflow.models.versioning import ModelMetadata

metadata = ModelMetadata(
    model_version="0.5.0",
    library_version="0.5.0",
    architecture={"vae_dim": 128, "text_embed_dim": 768},
    components={"compressor": "FluxCompressor", "expander": "FluxExpander"}
)
metadata.save("outputs/model_metadata.json")
```

#### `to_dict() -> Dict[str, Any]`
Convert metadata to dictionary.

```python
metadata_dict = metadata.to_dict()
```

#### `from_dict(data: Dict[str, Any]) -> ModelMetadata`
Create metadata from dictionary.

```python
from fluxflow.models.versioning import ModelMetadata

data = {
    "model_version": "0.5.0",
    "library_version": "0.5.0",
    "architecture": {"vae_dim": 128},
    "components": {}
}
metadata = ModelMetadata.from_dict(data)
```

**See Also:**
- [VERSIONING.md#metadata-format](VERSIONING.md#metadata-format) - Metadata schema details

---

## Version Registry

### `ModelVersionRegistry`

Central registry for version-specific loaders and compatibility checks.

**Location:** `fluxflow.models.versioning`

**Methods:**

#### `get_loader(version: str) -> Optional[Type[BaseModelLoader]]`
Get loader class for specific version.

```python
from fluxflow.models.versioning import ModelVersionRegistry

# Get loader for v0.5.0
loader_class = ModelVersionRegistry.get_loader("0.5.0")
if loader_class:
    loader = loader_class()
    # Use loader...
```

#### `register_loader(version: str, loader_class: Type[BaseModelLoader]) -> None`
Register custom loader for version (advanced usage).

```python
from fluxflow.models.versioning import ModelVersionRegistry, BaseModelLoader

class CustomLoader(BaseModelLoader):
    def load(self, checkpoint_path, device, **kwargs):
        # Custom loading logic
        pass
    
    def save(self, model, output_path, **kwargs):
        # Custom saving logic
        pass

ModelVersionRegistry.register_loader("0.5.0", CustomLoader)
```

**Registered Versions:**
- `0.4.0` → `FluxModelLoader_v0_4_0`
- `0.3.0` → `FluxModelLoader_v0_3_0`
- `0.2.x` → `LegacyLoader`

**See Also:**
- [VERSIONING.md#version-routing](VERSIONING.md#version-routing) - Version routing details
- [CONTRIBUTING.md](https://github.com/danny-mio/fluxflow-core/blob/develop/CONTRIBUTING.md) - Adding new version loaders

---

## Quick Reference

### Common Workflows

**Load any checkpoint:**
```python
from fluxflow.models.versioning import load_versioned_checkpoint
pipeline = load_versioned_checkpoint("path/to/checkpoint/")
```

**Save with metadata:**
```python
from fluxflow.models.versioning import save_versioned_checkpoint
save_versioned_checkpoint(pipeline, "outputs/model/", model_version="0.5.0")
```

**Check metadata:**
```python
from fluxflow.models.versioning import ModelMetadata
metadata = ModelMetadata.load("path/to/model_metadata.json")
print(f"Version: {metadata.model_version}, VAE dim: {metadata.architecture['vae_dim']}")
```

**Migrate from legacy:**
```python
# Load legacy checkpoint
from fluxflow.models import FluxPipeline
pipeline = FluxPipeline.from_pretrained("old_checkpoint.safetensors")

# Save with versioning
from fluxflow.models.versioning import save_versioned_checkpoint
save_versioned_checkpoint(pipeline, "new_checkpoint/", model_version="0.5.0")
```

---

## FluxFlowPipeline

High-level text-to-image generation pipeline inheriting from Diffusers' `DiffusionPipeline`.

### Class Definition

```python
class FluxFlowPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using FluxFlow models.
    
    Inherits from diffusers.DiffusionPipeline and provides a familiar API.
    """
```

### Loading from Checkpoints

#### from_pretrained()

```python
FluxFlowPipeline.from_pretrained(
    pretrained_model_name_or_path: str | PathLike,
    use_versioning: bool = False,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
    tokenizer_name: str = "distilbert-base-uncased",
    scheduler: Optional[SchedulerMixin] = None,
    scheduler_config: Optional[dict] = None,
    **kwargs
) -> FluxFlowPipeline
```

**Parameters:**
- `pretrained_model_name_or_path` (str | PathLike): Path to checkpoint file (`.safetensors` or `.pt`) or directory
- `use_versioning` (bool, default=False): Enable versioned loading (future feature)
- `device` (str, default="cuda"): Device to load model on ("cuda", "cpu", "mps")
- `torch_dtype` (torch.dtype, default=torch.float32): Data type for model weights
- `tokenizer_name` (str, default="distilbert-base-uncased"): HuggingFace tokenizer model name
- `scheduler` (SchedulerMixin, optional): Custom scheduler instance
- `scheduler_config` (dict, optional): Scheduler configuration (if not providing scheduler)
- `**kwargs`: Additional arguments passed to parent class

**Returns:** FluxFlowPipeline instance ready for inference

**Example:**
```python
from fluxflow.models import FluxFlowPipeline
import torch

# Load from checkpoint file
pipeline = FluxFlowPipeline.from_pretrained(
    "checkpoints/fluxflow_v0.5.0.safetensors",
    device="cuda",
    torch_dtype=torch.float16
)

# Load from checkpoint directory
pipeline = FluxFlowPipeline.from_pretrained(
    "outputs/experiment_001/",
    device="cuda"
)
```

### Generating Images

#### __call__()

```python
pipeline(
    prompt: str | List[str],
    negative_prompt: Optional[str | List[str]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: str = "pil",
    return_dict: bool = True,
    callback: Optional[Callable] = None,
    callback_steps: int = 1,
) -> FluxFlowPipelineOutput | Tuple
```

**Parameters:**
- `prompt` (str | List[str]): Text prompt(s) to guide generation
- `negative_prompt` (str | List[str], optional): Text prompt(s) to avoid in generation
- `num_inference_steps` (int, default=50): Number of denoising steps
- `guidance_scale` (float, default=7.5): Classifier-free guidance scale (1.0 = no guidance)
- `height` (int, default=512): Output image height in pixels
- `width` (int, default=512): Output image width in pixels
- `num_images_per_prompt` (int, default=1): Number of images to generate per prompt
- `eta` (float, default=0.0): DDIM eta parameter
- `generator` (torch.Generator, optional): Random number generator for reproducibility
- `latents` (torch.Tensor, optional): Pre-generated latents to start from
- `output_type` (str, default="pil"): Output format ("pil" or "np")
- `return_dict` (bool, default=True): Return FluxFlowPipelineOutput object
- `callback` (Callable, optional): Function called after each denoising step
- `callback_steps` (int, default=1): Number of steps between callback calls

**Returns:** 
- If `return_dict=True`: FluxFlowPipelineOutput with `images` attribute
- If `return_dict=False`: Tuple of (images,)

**Example:**
```python
# Single prompt
result = pipeline(
    prompt="a serene mountain landscape at dawn",
    num_inference_steps=50,
    guidance_scale=7.5,
    height=768,
    width=768
)
result.images[0].save("mountain.png")

# Batch generation with negative prompts
result = pipeline(
    prompt=["a cat", "a dog", "a bird"],
    negative_prompt="blurry, low quality",
    num_inference_steps=30,
    num_images_per_prompt=2,  # 2 images per prompt = 6 total
    generator=torch.Generator().manual_seed(42)
)

for i, img in enumerate(result.images):
    img.save(f"animal_{i}.png")
```

### Classifier-Free Guidance

CFG requires models trained with `cfg_dropout_prob > 0` (typically 0.10-0.15).

**Recommended guidance_scale values:**
- `1.0`: No guidance (standard generation)
- `3.0-7.0`: Moderate guidance (recommended)
- `7.0-15.0`: Strong guidance (may oversaturate)

**Example:**
```python
# Enable CFG
image = pipeline(
    prompt="photorealistic portrait of a cat",
    negative_prompt="blurry, distorted",
    guidance_scale=5.0,
    num_inference_steps=50
).images[0]
```

### FluxFlowPipelineOutput

**Attributes:**
- `images` (List[PIL.Image.Image] | np.ndarray): Generated images

---

## Error Handling

All versioning APIs may raise:
- `CheckpointError`: Checkpoint incompatible, corrupted, or missing
- `ValueError`: Invalid parameters (e.g., unknown version)
- `FileNotFoundError`: Checkpoint file/directory not found

**Example:**
```python
from fluxflow.models.versioning import load_versioned_checkpoint, CheckpointError

try:
    pipeline = load_versioned_checkpoint("path/to/checkpoint/")
except CheckpointError as e:
    print(f"Failed to load checkpoint: {e}")
    # Handle error (e.g., fallback to different checkpoint)
except FileNotFoundError:
    print("Checkpoint not found")
```

---

## Additional Resources

- **[VERSIONING.md](VERSIONING.md)** - Versioning system architecture and design
- **[MIGRATION.md](MIGRATION.md)** - Migration guides between model versions
- **[fluxflow-training README](https://github.com/danny-mio/fluxflow-training)** - Training pipeline usage

---

## Support

- **GitHub Issues**: [fluxflow-core/issues](https://github.com/danny-mio/fluxflow-core/issues)
- **Discussions**: [fluxflow-core/discussions](https://github.com/danny-mio/fluxflow-core/discussions)
