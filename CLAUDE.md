# CLAUDE.md - FluxFlow Core

See `AGENTS.md` for the canonical agent and coordinator conventions used in
this repo. This file holds Claude-specific notes layered on top of that.

## v0.10.0 Bezier-Coupled Redesign (in progress on feature/model-v0.10.0)

The v0.10.0 redesign reworks both VAE and Flow around five locked decisions:

1. **Per-token text** — `BertTextEncoder.forward` returns `(text_seq, text_mask)`
   instead of a pooled vector; cross-attention sees the full sequence.
2. **Conditional ctx coupling** — VAE encoder produces a `ctx` channel that
   modulates the SPADE blocks of the decoder (no global pooling).
3. **Full flow modernization** — `FluxFlowProcessor_v100` adds 2D RoPE on the
   image side, per-token text cross-attention, and dual FiLM (time + text).
4. **Multi-scale SPADE** — `SPADE_v100b` replaces single-scale conditioning in
   the decoder; activation is `WideTrainableBezier`.
5. **Clean Gaussian z** — VAE samples plain N(0, I) latents, KL reweighted by
   `kl_z_weight` with cosine warmup, ctx receives a shrinkage loss
   (`ctx_shrinkage_weight`).

Canonical references:
- Migration guide: `docs/MIGRATION-v0.10.0-redesign.md`
- Architecture overview: `docs/ARCHITECTURE.md` (§ "v0.10.0: Bezier-Coupled Architecture")
- Changelog entry: `CHANGELOG.md` (§ "[0.10.0]")

Salvage script for v0.7.x / v0.8.x → v0.10.0 warm-start:

```bash
python scripts/migrate_v0_10_0_to_redesign.py --src OLD.safetensors --dst WARM.safetensors
```

Legacy `v060`/`v070` callers are dispatched by
`fluxflow.models.pipeline._flow_processor_takes_pertoken_text`, which inspects
the flow processor signature and routes to the old pooled-text path when needed.

Per-repo milestone tags landed for this repo: `m1-foundations`, `m2-vae-redesign`,
`m3-pertoken-text`, `m4-flow-redesign`, `m8-salvage-script`.

## MLX backend status

`src/fluxflow/mlx/` (Apple-Silicon inference path) is **frozen at v0.7.0/v0.8.0
parity** — single-head `SPADE`, no `SPADE_v100b`, no seam-smoother. It has NOT
been ported to the v0.10.0 redesign and is out of scope for this release. Zero
references from `fluxflow-ui`/`fluxflow-comfyui` — not wired into any
production surface. `FluxFlowPipelineMLX.from_checkpoint` guards against
v0.10.0 checkpoints (raises `ModelArchitectureError` on `SPADE_v100b`/
seam-smoother marker keys) but do not assume this backend is otherwise at
parity with `models/v100/vae.py`.
