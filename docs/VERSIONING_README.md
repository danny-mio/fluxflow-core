# FluxFlow Model Versioning - Quick Reference

## What is it?

A system for storing model version and architecture metadata alongside checkpoints, enabling:
- Backward compatibility (load old models)
- Forward compatibility detection (clear errors for new models)
- Faster loading (no architecture inference needed)
- Model integrity verification (checksums)

## Quick Start

### Save with Versioning

```python
from fluxflow.models.versioning import save_versioned_checkpoint

save_versioned_checkpoint(
    pipeline,
    "outputs/model/",
    model_version="0.3.0",
    training_info={"steps": 50000, "dataset": "COCO"}
)
```text
Creates:
- `outputs/model/model.safetensors` - Weights
- `outputs/model/model_metadata.json` - Version + config

### Load with Versioning

```python
from fluxflow.models import FluxPipeline

# Option 1: Direct
from fluxflow.models.versioning import load_versioned_checkpoint
pipeline = load_versioned_checkpoint("outputs/model/")

# Option 2: from_pretrained
pipeline = FluxPipeline.from_pretrained(
    "outputs/model/",
    use_versioning=True
)
```text
### Migrate Legacy Checkpoints

```bash
python scripts/migrate_checkpoints.py \
    old_model.safetensors \
    versioned_model/ \
    --version 0.3.0
```text
## Documentation

- [VERSIONING.md](VERSIONING.md) - Full versioning guide
- [MIGRATION.md](MIGRATION.md) - Migration instructions
- [CHANGELOG.md](../CHANGELOG.md) - Version history

## Compatibility

| Feature | Status |
|---------|--------|
| Backward compatible with legacy checkpoints | ✅ Yes |
| Breaks existing code | ❌ No (opt-in) |
| Requires migration | ❌ No (optional) |
| Recommended for new models | ✅ Yes |

## API Changes

All changes are **opt-in** (default behavior unchanged):

- `FluxPipeline.from_pretrained(path, use_versioning=False)` 
- `save_model(..., save_metadata=False, model_version="0.3.0")`
- New: `load_versioned_checkpoint(path)`
- New: `save_versioned_checkpoint(model, path, model_version)`

## Files

- `src/fluxflow/models/versioning.py` - Core implementation
- `scripts/migrate_checkpoints.py` - Migration tool
- `tests/unit/test_versioning.py` - Tests
- `docs/VERSIONING.md` - Full guide
- `docs/MIGRATION.md` - Migration guide
