# FluxFlow Model Versioning

FluxFlow includes a robust model versioning system to ensure smooth model evolution and backward compatibility.

## Overview

Starting from version 0.3.1, FluxFlow supports explicit model versioning through metadata files. This enables:

- **Backward compatibility**: Load models created with older FluxFlow versions
- **Forward compatibility detection**: Clear errors when attempting to load newer models
- **Architecture transparency**: Explicit model configuration instead of inference
- **Integrity verification**: Checksums to detect corrupted checkpoints

## Quick Start

### Saving Models with Versioning

```python
from fluxflow.models.versioning import save_versioned_checkpoint

# Save with version metadata
save_versioned_checkpoint(
    pipeline,
    "outputs/model_v0.3.0/",
    model_version="0.3.0",
    training_info={
        "total_steps": 50000,
        "dataset": "COCO2017",
        "final_loss": 0.042
    }
)
```

This creates:
- `outputs/model_v0.3.0/model.safetensors` - Model weights
- `outputs/model_v0.3.0/model_metadata.json` - Version and architecture info

### Loading Models with Versioning

```python
from fluxflow.models import FluxPipeline

# Option 1: Direct versioned loading
from fluxflow.models.versioning import load_versioned_checkpoint
pipeline = load_versioned_checkpoint("outputs/model_v0.3.0/")

# Option 2: Use from_pretrained with versioning flag
pipeline = FluxPipeline.from_pretrained(
    "outputs/model_v0.3.0/",
    use_versioning=True
)
```

### Training Integration

```python
from fluxflow.utils.io import save_model

# Save with metadata during training
save_model(
    diffuser=pipeline,
    text_encoder=text_encoder,
    output_path="checkpoints/epoch_10/",
    save_metadata=True,
    model_version="0.3.0",
    training_info={
        "epoch": 10,
        "total_steps": 50000,
        "learning_rate": 1e-4
    }
)
```

## Metadata Format

`model_metadata.json` contains:

```json
{
  "model_version": "0.3.0",           // REQUIRED: Model version
  "library_version": "0.3.1",         // REQUIRED: FluxFlow version used to save
  "architecture": {                   // REQUIRED: Auto-detected from model
    "vae_dim": 128,                   // REQUIRED
    "flow_dim": 128,                  // REQUIRED
    "text_embed_dim": 768,            // REQUIRED
    "downscales": 4,                  // REQUIRED
    "upscales": 4,                    // REQUIRED
    "vae_attn_layers": 2,             // Auto-detected
    "flow_transformer_layers": 10,    // Auto-detected
    "flow_attn_heads": 8,             // Auto-detected
    "max_hw": 1024,                   // Auto-detected
    "in_channels": 3                  // Auto-detected
  },
  "components": {                     // REQUIRED: Component types
    "compressor": "FluxCompressor",
    "flow_processor": "FluxFlowProcessor",
    "expander": "FluxExpander"
  },
  "training_info": {                  // OPTIONAL: User-provided metadata
    "trained_on": "2025-01-15T10:30:00Z",
    "total_steps": 50000,
    "dataset": "COCO+OpenImages"
  },
  "checksum": {                       // OPTIONAL: For validation
    "algorithm": "sha256",
    "weights_hash": "abc123..."
  }
}
```

**Note**: Fields marked "Auto-detected" are discovered from the model structure. Only the core dimensional parameters (`vae_dim`, `flow_dim`, etc.) are strictly required for loading.

## Semantic Versioning

FluxFlow uses semantic versioning (MAJOR.MINOR.PATCH) for model versions:

- **MAJOR**: Breaking architecture changes (e.g., VAE dimension change)
- **MINOR**: Backward-compatible additions (e.g., new optional layers)
- **PATCH**: Bug fixes, weight updates (fully compatible)

Examples:
- `0.3.0` → `0.3.1`: Patch version, fully compatible
- `0.3.0` → `0.4.0`: Minor version, backward compatible
- `0.3.0` → `1.0.0`: Major version, may require migration

## Backward Compatibility

Legacy checkpoints (without metadata) continue to work:

```python
# Legacy checkpoint without metadata - still works
pipeline = FluxPipeline.from_pretrained("old_model.safetensors")

# Internally uses architecture detection
```

To upgrade legacy checkpoints, see [MIGRATION.md](MIGRATION.md).

## Version Compatibility Matrix

| Model Version | FluxFlow 0.3.0 | FluxFlow 0.3.1+ |
|---------------|----------------|-----------------|
| 0.2.0 (legacy)| ✅ (inferred)  | ✅ (inferred)   |
| 0.3.0         | ✅             | ✅              |
| 0.3.1         | ❌             | ✅              |
| 0.4.0         | ❌             | Future          |

## Best Practices

### For Model Developers

1. **Always use versioned saves** for new models:
   ```python
   save_versioned_checkpoint(model, path, model_version="0.3.0")
   ```

2. **Include training metadata** for reproducibility:
   ```python
   training_info = {
       "total_steps": steps,
       "dataset": "COCO",
       "config_file": "config.yaml"
   }
   ```

3. **Increment version** when changing architecture:
   - Changed layer dimensions? → Bump MAJOR
   - Added optional feature? → Bump MINOR
   - Fixed bug? → Bump PATCH

### For Model Users

1. **Use `use_versioning=True`** for better error messages:
   ```python
   pipeline = FluxPipeline.from_pretrained(path, use_versioning=True)
   ```

2. **Check metadata** before loading:
   ```python
   from fluxflow.models.versioning import ModelMetadata

   metadata = ModelMetadata.load(path / "model_metadata.json")
   print(f"Model version: {metadata.model_version}")
   print(f"Architecture: {metadata.architecture}")
   ```

3. **Migrate legacy models** to versioned format:
   ```bash
   python scripts/migrate_checkpoints.py old_model.pt new_model/
   ```

## Troubleshooting

### Error: "No loader found for checkpoint version X.Y.Z"

Your FluxFlow version doesn't support this model version. Either:
- Upgrade FluxFlow: `pip install --upgrade fluxflow`
- Use an older model version

### Error: "Checkpoint is newer than library version"

The model was created with a newer FluxFlow version. Upgrade:
```bash
pip install --upgrade fluxflow
```

### Warning: "Loading legacy checkpoint without version metadata"

The checkpoint doesn't have metadata. Architecture will be inferred.
To fix, migrate the checkpoint:
```bash
python scripts/migrate_checkpoints.py old.safetensors new/
```

### Checksum Mismatch

The model weights have been modified or corrupted.
Re-download the checkpoint or restore from backup.

## Advanced Usage

### Custom Version Loaders

Create a custom loader for a new model version:

```python
from fluxflow.models.versioning import ModelVersionLoader, ModelVersionRegistry

class ModelLoaderV04(ModelVersionLoader):
    VERSION = "0.4.0"
    COMPATIBLE_VERSIONS = ["0.4.1"]

    def load_checkpoint(self, checkpoint_path, metadata, device, **kwargs):
        # Custom loading logic for v0.4.x
        pass

    def save_checkpoint(self, model, output_path, metadata, **kwargs):
        # Custom saving logic for v0.4.x
        pass

# Register the loader
ModelVersionRegistry.register(ModelLoaderV04)
```

### Programmatic Version Checking

```python
from fluxflow.models.versioning import ModelMetadata, ModelVersionRegistry

metadata = ModelMetadata.load("model/model_metadata.json")
loader = ModelVersionRegistry.get_loader(metadata.model_version)

if loader is None:
    print(f"Unsupported version: {metadata.model_version}")
else:
    print(f"Supported by: {loader.__name__}")
```

## See Also

- [MIGRATION.md](MIGRATION.md) - Migrating legacy checkpoints
