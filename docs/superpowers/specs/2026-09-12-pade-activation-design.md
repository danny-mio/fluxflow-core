# Padé Activation Units for v0.10.0 — Design Spec

Status: approved, implementing on `feature/pade-activation` (off `feature/model-v0.10.0`).
Part of the unreleased `[0.10.0]` line — no version bump.

## Background

`fluxflow-core` uses learnable Bezier-curve activations (`BezierActivation`,
`TrainableBezier`, `WideTrainableBezier` in `models/activations.py`) as both a
nonlinearity and, in the input-based mode, a 5-channel-to-1 dimensionality
reducer used throughout `v100/vae.py`, `v100/flow.py`, and
`v100/conditioning.py`.

A prior proposal to add a "double pendulum" activation was rejected after
review: the only non-chaotic ("tamed") version reduces to a 2-frequency
sinusoid (a constrained special case of existing Snake/SIREN-style periodic
activations), requires transcendental function calls (cos/sin) at every
element, and is a poor fit for the bounded/monotonic-ish shape every current
Bezier site wants. See conversation history for the full rejection rationale.

This spec replaces that idea with **Padé Activation Units (PAU)** — a
published technique (Molina et al., "Padé Activation Units: End-to-end
Learning of Flexible Activation Functions in Deep Networks," ICLR 2020) that
is a strict generalization of Bezier's polynomial curve (rational functions
⊃ polynomials), stays transcendental-free (Horner's-method polynomial
evaluation + one division), and maps directly onto every existing Bezier use
site without inventing a new site to justify it.

## Goals

1. Increase quality-per-parameter at sites currently served by Bezier
   activations, without a training-run-backed quality claim yet (deferred —
   see "Explicitly out of scope").
2. Keep training/inference speed competitive — no transcendental function
   calls, cost bounded by polynomial degree.
3. Make the activation family a first-class, checkpoint-persisted,
   user-selectable choice for `v100` models: `"bezier"` (default, unchanged
   behavior) or `"pade"`.

## Math

`PAU(x) = P(x) / Q(x)`

- `P(x) = a0 + a1·x + a2·x² + ... + am·x^m` (numerator, degree m)
- `Q(x) = 1 + |b1·x + b2·x² + ... + bn·x^n|` (denominator, degree n)

The `1 + |·|` construction on `Q` is load-bearing: it guarantees `Q(x) ≥ 1`
for every real `x`, so the activation has **no pole, ever** — this is the
"safe PAU" variant from the paper. Both `P` and `Q` are evaluated via
Horner's method: pure fused-multiply-adds, one division, zero transcendental
calls.

### Two variants, mirroring Bezier's two usage patterns

1. **`TrainablePade`** (mirrors `TrainableBezier`) — literature-standard
   order `m=5, n=4` (10 learnable coefficients, per-channel or
   `channel_only`, matching `TrainableBezier`'s constructor shape). Used at
   dimension-preserving sites: VAE mu/logvar, RGB output.
   `WideTrainablePade` mirrors `WideTrainableBezier` for the logvar site
   (needs an effectively-suppressible output range).
2. **`PadeActivation`** (mirrors `BezierActivation`) — data-driven /
   dimensionality-reducing mode. Input split 5 ways into `(x, a0, a1, a2, b1)`
   — same channel-reduction contract as Bezier's `(t, p0, p1, p2, p3)` split,
   at low order (`m=2, n=1`) to keep the per-call cost in the same
   ballpark as Bezier's cubic. Used at VAE conv activations and the flow
   pillar/FFN sites.

Both variants use `nn.Module`-internal Horner evaluation; no JIT codegen in
this pass (Bezier's `bezier_jit_generated.py` was added only after the plain
version proved out — premature here).

Design asymmetry to document explicitly (not silently drop): Bezier needs
`t_pre_activation`/`p_preactivation` to squash its Bernstein-basis input into
`[0,1]`. Padé's rational function is defined on all reals, so it does not
need this squashing. The shared construction helper (below) accepts the same
keyword surface for drop-in convenience but ignores/no-ops these kwargs for
the `"pade"` family, with a docstring note — not a silent behavior change a
future reader would have to discover by reading the source.

## Where it plausibly helps vs. Bezier

A cubic Bezier segment is a degree-(3,0) Padé approximant — i.e. Padé's
function class is a strict superset. Every existing Bezier site (bounded,
roughly-monotonic conv/logvar/RGB/pillar-mixing activations) is therefore a
structurally valid target, unlike the rejected pendulum idea which needed a
new site invented to make sense. This is the honest "quality per size"
argument: same or comparable param budget, strictly larger function class,
at sites Bezier is already proven at.

Cost expectation (to be measured, not assumed):
- `PadeActivation` (m=2, n=1): ~3 FMAs + 1 division vs Bezier's ~5 FMAs + 0
  division. Plausibly competitive.
- `TrainablePade` (m=5, n=4): ~9 FMAs + 1 division vs Bezier's ~5 FMAs.
  Expect measurably slower; report the real number.

## Selectable parameter (train/use) + checkpoint metadata

Scoped to `v100/` only. `v030`/`v060`/`v070`/`v080` are frozen/historical and
stay Bezier-only — passing `activation_type` to their constructors is not
supported (same precedent as `attn_backend`, gated by
`_SDPA_SUPPORTED_VERSIONS` in `factory.py`).

1. `ModelConfig.activation_type: Literal["bezier", "pade"] = "bezier"` in
   `config.py`.
2. `ModelFactory.__init__(..., activation_type: ActivationFamily = "bezier")`
   in `factory.py`, gated by a new `_ACTIVATION_SELECTABLE_VERSIONS = {"0.10.0"}`
   set (mirrors `_SDPA_SUPPORTED_VERSIONS`), passed through to
   `FluxCompressor_v100`, `FluxFlowProcessor_v100`, `FluxExpander_v100`.
3. Each of those `v100` classes accepts `activation_type: Literal["bezier",
   "pade"] = "bezier"`, stores `self.activation_type`, and uses a new
   dispatcher — `make_activation(kind, mode, **kwargs)` added to
   `activations.py` — at every current `BezierActivation`/`TrainableBezier`/
   `WideTrainableBezier` call site, instead of hardcoding the class.
4. `_detect_architecture()` in `versioning.py` adds
   `config["activation_type"] = model.compressor.activation_type` when the
   attribute is present, so **every save automatically records which family
   was used** with no caller changes required.
5. `ModelLoaderV010.load_checkpoint` resolves:
   `activation_type = kwargs.pop("force_activation_type", None) or config.get("activation_type", "bezier")`
   — metadata wins by default; passing `force_activation_type="pade"` (or
   `"bezier"`) to `load_versioned_checkpoint(...)` overrides it. Checkpoints
   saved before this change have no `activation_type` key and default to
   `"bezier"` — no retroactive breakage.

## Deliverables this pass

- `src/fluxflow/models/pade_activation.py`: `PadeActivationModule`,
  `PadeActivation`, `TrainablePade`, `WideTrainablePade`.
- `make_activation(kind, mode, **kwargs)` dispatcher in `activations.py`.
- Wiring in `config.py`, `factory.py`, `v100/vae.py`, `v100/flow.py`,
  `v100/conditioning.py`, `versioning.py` per above.
- `benchmark_pade.py`: same harness shape as `benchmark_bezier.py`, both
  variants, head-to-head vs `BezierActivation`/`TrainableBezier`.
- `docs/PADE-ACTIVATION-ANALYSIS.md`: math, per-site cost/expressiveness
  argument, actual benchmark numbers, and an explicit **"quality claims
  unvalidated — no training run yet"** caveat, plus what a follow-up
  matched-param training ablation (same pattern as
  `BaselineFluxTransformerBlock` in `v070/flow.py`) would need to settle it.
- Unit tests: shape handling (2D/3D/4D, matching `BezierActivation`'s test
  coverage), gradient flow, `activation_type` selection end-to-end through
  `ModelFactory`, checkpoint round-trip (save with one family, load with
  metadata auto-detection, load with `force_activation_type` override), and
  a numerical-stability test asserting `Q(x) ≥ 1` holds under random and
  adversarial (large-magnitude) inputs — this property must not regress
  silently.
- `CHANGELOG.md`: new entries under the existing `[Unreleased]` (0.10.0)
  section.
- `REFERENCES.md`: Molina et al. PAU citation.

## Explicitly out of scope this pass

- Any training run. Quality-per-size is argued structurally (strict
  superset of Bezier's function class) and will carry an explicit
  "unvalidated" caveat in the analysis doc until a matched-param ablation is
  run.
- Wiring `activation_type` into `v030`/`v060`/`v070`/`v080` (frozen/historical).
- `fluxflow-training`/`fluxflow-ui`/`fluxflow-comfyui` changes. **Flagged
  cross-repo impact**: `fluxflow-training`'s YAML/CLI config schema will
  need its own `activation_type` passthrough field before a user can
  actually select `"pade"` from the training CLI — the mechanism this spec
  builds works end-to-end for direct `ModelFactory`/checkpoint-loader
  callers, but the training entry point needs a follow-up change to expose
  it. Not implemented here.
- Any JIT-compiled fast path for the Padé activations (mirrors Bezier's
  `bezier_jit.py`/`bezier_jit_generated.py`, deferred the same way Bezier's
  own JIT layer was added only after the plain version proved out).

## Self-review notes

- Placeholder scan: none found — no TBD/TODO left in this document.
- Consistency: the "no transcendentals" claim in Math is consistent with
  the deferred-JIT decision (there is no need for a JIT layer for
  correctness, only for the speed optimization Bezier's JIT provides on top
  of an already-transcendental-free base).
- Scope: implementation is confined to `fluxflow-core`; the one cross-repo
  dependency (`fluxflow-training` config passthrough) is named as a gap
  rather than silently implied to be handled.
- Ambiguity check: "selectable parameter for training/use" resolved
  concretely as `ModelConfig`/`ModelFactory` constructor field (train-time)
  plus `force_activation_type` loader kwarg (use/inference-time) — both
  paths specified, neither left implicit.
