# Padé Activation Units: Analysis

Rational-function generalization of the Bezier activation family. See
`docs/superpowers/specs/2026-09-12-pade-activation-design.md` for the full
design rationale, including why the earlier "double pendulum" activation
proposal was rejected in favor of this one.

## Math

`PAU(x) = P(x) / Q(x)`, `Q(x) = 1 + |Q_raw(x)|` (guarantees `Q(x) >= 1`,
no pole, for every real x and every real coefficient). Evaluated via
Horner's method: FMAs + one division, zero transcendental calls.

Two variants, mirroring Bezier's two usage patterns:
- `PadeActivation` (data-driven, m=2/n=1): input split 5 ways into
  `(x, a0, a1, a2, b1)`, mirrors `BezierActivation`'s 5-channel reduction.
- `TrainablePade` (learnable, m=5/n=4, literature-standard order): 10
  coefficients per channel, mirrors `TrainableBezier`.

## Where it plausibly helps vs. Bezier

A cubic Bezier segment is a degree-(3,0) Padé approximant -- Padé's
function class is a strict superset. Every existing Bezier site (VAE
conv activations, pillar/FFN mixing, RGB output, mu/logvar) is therefore
a structurally valid target without inventing a new site to justify it.

## Benchmark results (CPU, single run via `benchmark_pade.py`)

No CUDA device was available in the environment this was measured in (the
only GPU present is an unrelated ROCm nightly build environment, not set up
with this project's dependencies) -- these are CPU-only, single-run
numbers. They are directional, not a substitute for a proper multi-run GPU
benchmark before any speed claim goes in front of stakeholders.

Fixed / data-driven mode:

| Shape | BezierActivation (ms) | PadeActivation (ms) | Delta |
|---|---|---|---|
| 2D Small (4,50)→(4,10) | 0.0197 | 0.0097 | Pade ~2.0x faster |
| 2D Medium (32,250)→(32,50) | 0.0256 | 0.0119 | Pade ~2.2x faster |
| 2D Large (64,1000)→(64,200) | 0.0643 | 0.0254 | Pade ~2.5x faster |
| 3D Small (4,64,125)→(4,64,25) | 0.0457 | 0.0220 | Pade ~2.1x faster |
| 3D Medium (8,128,250)→(8,128,50) | 0.2641 | 0.0966 | Pade ~2.7x faster |
| 4D Small (4,15,8,8)→(4,3,8,8) | 0.0311 | 0.0125 | Pade ~2.5x faster |
| 4D Large (16,60,32,32)→(16,12,32,32) | 0.3031 | 0.0827 | Pade ~3.7x faster |

Trainable mode (shape (128,), batch 32):

| | TrainableBezier (ms) | TrainablePade (ms) | Delta |
|---|---|---|---|
| Forward | 0.0342 | 0.0299 | ~parity (Pade marginally faster) |

Forward+backward (shape (8,125)):

| | Bezier (ms) | Pade (ms) | Delta |
|---|---|---|---|
| | 0.3729 | 0.1366 | Pade ~2.7x faster |

Parameter counts (D=128): TrainableBezier=512, TrainablePade=1280.

**Interpretation:** `PadeActivation` (m=2, n=1) consistently beat
`BezierActivation` on CPU across every fixed-mode shape tested. This is
partly an artifact of the comparison, not purely the FMA/division op count:
`BezierActivation()`'s default constructor applies `t_pre_activation="sigmoid"`,
i.e. Bezier's default path includes one transcendental call that
`PadeActivation` never needs -- so this result is consistent with, and
partial evidence for, the "zero transcendental calls" design goal rather
than a mysterious win. `TrainablePade` (m=5, n=4, 1280 params) came out
roughly at parity with `TrainableBezier` (512 params) despite doing
~9 FMAs + 1 division per element versus Bezier's ~5 FMAs + 0 divisions --
at these tensor sizes on CPU, Python/dispatch overhead dominates the actual
arithmetic op-count difference, so this result should not be read as
"Padé's higher-order variant is free at scale." A GPU benchmark at
production tensor sizes is needed before claiming trainable-mode
`TrainablePade` is speed-competitive with `TrainableBezier`; the fixed-mode
result is on firmer ground since it held across every tested shape.

## Quality claims: unvalidated

No training run has been performed. The "quality per parameter" argument
above is structural (strict superset of Bezier's function class, same
param budget for the fixed/data-driven variant -- note `TrainablePade`
uses 2.5x more parameters than `TrainableBezier`, since it matches the
literature-standard order rather than Bezier's own count), not empirical.
A follow-up matched-param training ablation, following the same pattern as
`BaselineFluxTransformerBlock` in `v070/flow.py` (which matches Bezier's
and baseline's total parameter counts to make a fair comparison), is
required before any quality claim can be made in front of stakeholders.
That ablation is explicitly out of scope for this pass -- see the design
spec's "Explicitly out of scope" section.

## Cross-repo follow-up required

`fluxflow-training`'s YAML/CLI config schema needs its own `activation_type`
passthrough field before a user can select `"pade"` from the training CLI.
The mechanism added in this pass (`ModelConfig`, `ModelFactory`,
checkpoint metadata auto-detection/override) works end-to-end for direct
`ModelFactory`/checkpoint-loader callers; the training entry point itself
was not touched.
