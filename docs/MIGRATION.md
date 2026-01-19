# Checkpoint Migration Guide

This guide explains how to migrate legacy FluxFlow checkpoints to the new versioned format.

## Why Migrate?

Migrating checkpoints to the versioned format provides:

- **Faster loading**: No need to infer architecture from weights
- **Better error messages**: Clear version mismatch errors
- **Integrity checks**: Automatic checksum validation
- **Future-proofing**: Smooth upgrades as models evolve

## Migration Methods

### Method 1: Command-Line Tool (Recommended)

Use the provided migration script:

```bash
python scripts/migrate_checkpoints.py \
    path/to/old_model.safetensors \
    path/to/new_model/ \
    --version 0.3.0
```

**Arguments:**
- First argument: Input checkpoint file (`.safetensors` or `.pt`)
- Second argument: Output directory for versioned checkpoint
- `--version`: Model version to assign (default: `0.3.0`)
- `--force`: Overwrite existing output directory

**Example:**
```bash
# Basic migration
python scripts/migrate_checkpoints.py \
    checkpoints/flxflow_final.safetensors \
    checkpoints/v0.3.0/

# With specific version
python scripts/migrate_checkpoints.py \
    models/checkpoint.pt \
    models/versioned/ \
    --version 0.3.1

# Force overwrite
python scripts/migrate_checkpoints.py \
    checkpoint.safetensors \
    output/ \
    --force
```

### Method 2: Python API

Migrate programmatically:

```python
from pathlib import Path
from fluxflow.models.versioning import (
    load_versioned_checkpoint,
    save_versioned_checkpoint
)

# Load legacy checkpoint (auto-detects architecture)
model = load_versioned_checkpoint(
    Path("checkpoints/old_model.safetensors"),
    device="cpu"
)

# Save with version metadata
save_versioned_checkpoint(
    model,
    Path("checkpoints/versioned_model/"),
    model_version="0.3.0",
    training_info={
        "migrated_from": "old_model.safetensors",
        "original_source": "training_run_42"
    }
)
```

### Method 3: Batch Migration

Migrate multiple checkpoints at once:

```bash
#!/bin/bash
# migrate_all.sh

for checkpoint in checkpoints/*.safetensors; do
    basename=$(basename "$checkpoint" .safetensors)
    python scripts/migrate_checkpoints.py \
        "$checkpoint" \
        "checkpoints_versioned/${basename}/" \
        --version 0.3.0
done
```

## What Gets Created?

After migration, the new directory contains:

```
new_model/
├── model.safetensors      # Model weights (same as original)
└── model_metadata.json    # New: version and architecture info
```

### Example `model_metadata.json`:

```json
{
  "model_version": "0.3.0",
  "library_version": "0.3.1",
  "architecture": {
    "vae_dim": 128,
    "flow_dim": 128,
    "text_embed_dim": 768,          // Standardized order
    "downscales": 4,
    "upscales": 4,
    "vae_attn_layers": 2,
    "flow_transformer_layers": 10,
    "flow_attn_heads": 8,
    "max_hw": 1024,
    "in_channels": 3
  },
  "components": {
    "compressor": "FluxCompressor",
    "flow_processor": "FluxFlowProcessor",
    "expander": "FluxExpander"
  },
  "training_info": {
    "migrated_from": "old_model.safetensors",
    "migration_tool": "migrate_checkpoints.py"
  },
  "checksum": {
    "algorithm": "sha256",
    "weights_hash": "1a2b3c4d5e6f..."
  }
}
```

## Backward Compatibility

**Important:** Legacy checkpoints still work without migration!

```python
# This still works - architecture is inferred automatically
pipeline = FluxPipeline.from_pretrained("old_model.safetensors")
```

However, migration is recommended for:
- Production deployments
- Shared models
- Long-term storage

## Verification

### Verify Migration Success

```python
from pathlib import Path
from fluxflow.models.versioning import ModelMetadata, load_versioned_checkpoint

# Check metadata exists
metadata_path = Path("checkpoints/versioned_model/model_metadata.json")
assert metadata_path.exists(), "Metadata file missing!"

# Load and inspect metadata
metadata = ModelMetadata.load(metadata_path)
print(f"Model version: {metadata.model_version}")
print(f"Architecture: {metadata.architecture}")
print(f"Checksum: {metadata.checksum.get('weights_hash', 'N/A')}")

# Load model to verify it works
model = load_versioned_checkpoint(metadata_path.parent, device="cpu")
print("Model loaded successfully!")
```

### Compare Weights

Ensure weights are identical before and after migration:

```python
import safetensors.torch
import torch

# Load original checkpoint
original = safetensors.torch.load_file("old_model.safetensors")

# Load migrated checkpoint
migrated = safetensors.torch.load_file("new_model/model.safetensors")

# Compare all weights
for key in original:
    if key in migrated:
        assert torch.allclose(original[key], migrated[key]), f"Mismatch in {key}"

print("All weights match!")
```

## Troubleshooting

### Error: "Could not detect VAE dimension from checkpoint"

The checkpoint structure is unexpected. This can happen if:
- The checkpoint is corrupted
- It's from a very old FluxFlow version
- It's not a FluxFlow checkpoint

**Solution:** Manually inspect the checkpoint:
```python
import safetensors.torch

state_dict = safetensors.torch.load_file("checkpoint.safetensors")
print("Keys in checkpoint:")
for key in sorted(state_dict.keys())[:20]:  # First 20 keys
    print(f"  {key}: {state_dict[key].shape}")
```

### Migration Script Fails with Import Errors

Ensure FluxFlow is installed:
```bash
pip install -e .  # If in fluxflow-core directory
# or
pip install fluxflow
```

### Output Directory Already Exists

Use `--force` to overwrite:
```bash
python scripts/migrate_checkpoints.py input.safetensors output/ --force
```

### Checksum Shows as "N/A"

The checksum is computed during save. If missing:
```python
import hashlib
import safetensors.torch

# Manually compute checksum
with open("model/model.safetensors", "rb") as f:
    data = f.read()
    checksum = hashlib.sha256(data).hexdigest()
    print(f"SHA256: {checksum}")
```

## Migration Checklist

Before putting a migrated model into production:

- [ ] Run migration script without errors
- [ ] Verify `model_metadata.json` exists and is valid JSON
- [ ] Check metadata contains correct architecture parameters
- [ ] Load migrated model successfully
- [ ] Compare original vs migrated weights (optional but recommended)
- [ ] Test inference with migrated model
- [ ] Update documentation/README with new model path
- [ ] Archive or backup original checkpoint

## Best Practices

### 1. Test Before Migrating Production Models

```bash
# Test on a copy first
cp checkpoints/production.safetensors /tmp/test.safetensors
python scripts/migrate_checkpoints.py /tmp/test.safetensors /tmp/test_out/

# Verify it works
python -c "from fluxflow.models.versioning import load_versioned_checkpoint; \
           load_versioned_checkpoint('/tmp/test_out/', device='cpu')"
```

### 2. Keep Originals

Don't delete original checkpoints immediately:
```bash
# Create backup
mkdir -p backups/
cp checkpoints/*.safetensors backups/

# Then migrate
for f in checkpoints/*.safetensors; do
    python scripts/migrate_checkpoints.py "$f" "versioned/${f##*/}/"
done
```

### 3. Document Migration

Add migration info to your training logs:
```python
training_info = {
    "migrated_from": "original_checkpoint.safetensors",
    "migration_date": "2025-01-15",
    "migration_version": "0.3.1",
    "original_training_steps": 50000
}
```

### 4. Validate Architecture Detection

If you know the architecture, verify it was detected correctly:
```python
from fluxflow.models.versioning import ModelMetadata

metadata = ModelMetadata.load("versioned_model/model_metadata.json")

assert metadata.architecture["vae_dim"] == 128, "Wrong VAE dimension!"
assert metadata.architecture["flow_dim"] == 128, "Wrong flow dimension!"
assert metadata.architecture["downscales"] == 4, "Wrong downscales!"
```

## Automated Migration in CI/CD

Integrate migration into your deployment pipeline:

```yaml
# .github/workflows/migrate.yml
name: Migrate Checkpoints

on:
  workflow_dispatch:
    inputs:
      checkpoint_path:
        description: 'Path to checkpoint'
        required: true

jobs:
  migrate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install FluxFlow
        run: pip install -e .

      - name: Migrate Checkpoint
        run: |
          python scripts/migrate_checkpoints.py \
            ${{ inputs.checkpoint_path }} \
            checkpoints/versioned/ \
            --version 0.3.0

      - name: Upload Artifact
        uses: actions/upload-artifact@v3
        with:
          name: versioned-checkpoint
          path: checkpoints/versioned/
```

## See Also

- [VERSIONING.md](VERSIONING.md) - Versioning system overview
