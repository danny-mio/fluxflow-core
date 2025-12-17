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
pipeline = load_versioned_checkpoint("checkpoints/model_v0.3.0/")

# Load from specific file
pipeline = load_versioned_checkpoint("checkpoints/model.safetensors")

# Specify device
pipeline = load_versioned_checkpoint("checkpoints/model_v0.3.0/", device="cuda")
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
    model_version: str = "0.3.0",
    architecture: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None
```

**Parameters:**
- `model` (Any): Model to save (FluxPipeline or FluxFlowPipeline)
- `output_path` (Path): Directory to save checkpoint
- `model_version` (str): Model version string (semantic versioning, default: "0.3.0")
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
    "outputs/model_v0.3.0/",
    model_version="0.3.0"
)

# With training info
save_versioned_checkpoint(
    pipeline,
    "outputs/model_v0.3.0/",
    model_version="0.3.0",
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
    "outputs/model_v0.3.0/",
    model_version="0.3.0",
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
    model_version: str              # Semantic version (e.g., "0.3.0")
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
    model_version="0.3.0",
    library_version="0.3.0",
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
    "model_version": "0.3.0",
    "library_version": "0.3.0",
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

# Get loader for v0.3.0
loader_class = ModelVersionRegistry.get_loader("0.3.0")
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

ModelVersionRegistry.register_loader("0.4.0", CustomLoader)
```

**Registered Versions:**
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
save_versioned_checkpoint(pipeline, "outputs/model/", model_version="0.3.0")
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
save_versioned_checkpoint(pipeline, "new_checkpoint/", model_version="0.3.0")
```

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
- **[Examples](https://github.com/danny-mio/fluxflow-core/tree/develop/examples)** - Code examples

---

## Support

- **GitHub Issues**: [fluxflow-core/issues](https://github.com/danny-mio/fluxflow-core/issues)
- **Discussions**: [fluxflow-core/discussions](https://github.com/danny-mio/fluxflow-core/discussions)
