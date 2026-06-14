# Migration to v0.10.0 (Bezier-Coupled Redesign)

## Why this is a breaking change

v0.10.0 rebuilds the VAE decoder, the flow processor, and the text path
together. The five locked decisions:

1. **Per-token text** — `BertTextEncoder.forward` returns
   `(text_seq, text_mask)`, not a pooled vector.
2. **Conditional ctx coupling** — the VAE encoder emits a `ctx` channel that
   modulates SPADE blocks in the decoder.
3. **Full flow modernization** — `FluxFlowProcessor_v100` (2D RoPE +
   per-token cross-attention + dual FiLM).
4. **Multi-scale SPADE** — `SPADE_v100b` with `WideTrainableBezier`.
5. **Clean Gaussian z** — plain N(0, I) latents, `kl_z_weight` cosine warmup,
   `ctx_shrinkage_weight` to keep ctx informative.

The encoder/decoder topology, the flow attention I/O, the CFG null path, and
the ComfyUI conditioning type all change shape. Old checkpoints will not load
directly.

## Salvage script

A warm-start path is provided for v0.7.x / v0.8.x checkpoints:

```bash
python scripts/migrate_v0_10_0_to_redesign.py \
    --src OLD.safetensors \
    --dst WARM.safetensors
```

**Covered:** VAE encoder conv stack, decoder conv backbone, text-encoder
projection, flow MLPs whose shapes are unchanged.

**Lost (re-trained from scratch):** SPADE blocks (single-scale → multi-scale),
flow attention QKV (pooled → per-token), the ctx channel, and any FiLM heads
that depend on per-token text.

After running the script, fine-tune with the standard v0.10.0 pipeline config
to recover quality.

## API surface deltas for downstream consumers

| Old (≤ v0.8.x) | New (v0.10.0) |
| --- | --- |
| `text_pooled = encoder(ids, mask)` | `text_seq, text_mask = encoder(ids, mask)` |
| `FluxFlowProcessor(packed, text_pooled, timesteps)` | `FluxFlowProcessor_v100(packed, text_seq, text_mask, timesteps)` |
| `FluxPipeline.forward(img, text_pooled, timesteps)` | `FluxPipeline.forward(img, text_seq, text_mask, timesteps)` |
| `null = torch.zeros_like(text_pooled)` | `null_seq, null_mask = build_cfg_null_pair(encoder)` |
| ComfyUI `FLUXFLOW_CONDITIONING` | ComfyUI `FLUXFLOW_TEXT` (per-token tuple) |

Legacy v060/v070 callers are still supported via the
`_flow_processor_takes_pertoken_text` dispatcher in
`fluxflow.models.pipeline` — it inspects the flow processor and routes to the
pooled-text path automatically when an old processor is loaded.

## Pointers

- Architecture deep-dive: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) → § "v0.10.0: Bezier-Coupled Architecture"
- Changelog entry: [`CHANGELOG.md`](../CHANGELOG.md) → `[0.10.0]`
- Salvage script: [`scripts/migrate_v0_10_0_to_redesign.py`](../scripts/migrate_v0_10_0_to_redesign.py)
- Per-repo CLAUDE.md sections:
  - `fluxflow-core/CLAUDE.md` → v0.10.0 redesign
  - `fluxflow-training/CLAUDE.md` → new config keys + CFG hook
  - `fluxflow-ui/CLAUDE.md` → `generation_worker` switch
  - `fluxflow-comfyui/CLAUDE.md` → `FLUXFLOW_TEXT` type
