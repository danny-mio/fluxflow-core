# Padé Activation Units (v0.10.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Padé (rational-function) activation family as a selectable, checkpoint-persisted alternative to the existing Bezier activation family in FluxFlow's v0.10.0 (`v100/`) architecture.

**Architecture:** New `pade_activation.py` module provides `PadeActivation`/`TrainablePade`/`WideTrainablePade`, mirroring `BezierActivation`/`TrainableBezier`/`WideTrainableBezier`. A `make_activation(kind, mode, ...)` dispatcher in `activations.py` centralizes the choice so every `v100/*.py` call site swaps a hardcoded class for one dispatcher call. `ModelConfig`/`ModelFactory` expose `activation_type` for train-time selection; `versioning.py` auto-detects it from a saved model's own attributes at save time and reads it back at load time, with an explicit override.

**Tech Stack:** PyTorch (`torch.addcmul` Horner-method evaluation, no new dependencies).

**Spec:** `docs/superpowers/specs/2026-09-12-pade-activation-design.md`

## Global Constraints

- Zero transcendental function calls in any Padé forward/backward path (FMA + one division only).
- `Q(x) = 1 + |Q_raw(x)|` construction is mandatory everywhere a denominator is computed — guarantees `Q(x) ≥ 1` for all real inputs and all real coefficients (no pole, ever). This must be covered by a dedicated stability test, not just visual inspection.
- `activation_type` wiring is scoped to `v100/` only. Do not touch `v030/`, `v060/`, `v070/`, `v080/` (frozen/historical, Bezier-only).
- No JIT-compiled fast path for Padé in this pass (mirrors Bezier's own JIT layer being added only after the plain version proved out).
- No training runs. No changes outside `fluxflow-core`.
- Every new/changed public constructor keeps `activation_type: str = "bezier"` as the default, so existing callers and existing checkpoints are unaffected.

---

### Task 1: `PadeActivationModule` + `PadeActivation` (data-driven, low order)

**Files:**
- Create: `src/fluxflow/models/pade_activation.py`
- Test: `tests/unit/test_pade_activation.py`

**Interfaces:**
- Produces: `PadeActivationModule` (`nn.Module`, `forward(x, a0, a1, a2, b1) -> Tensor`), `PadeActivation` (`nn.Module`, `forward(x) -> Tensor`, input channels divisible by 5, reduces 5→1 like `BezierActivation`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_pade_activation.py`:

```python
"""Unit tests for Padé activation functions (src/fluxflow/models/pade_activation.py)."""

import pytest
import torch

from fluxflow.models.pade_activation import PadeActivation, PadeActivationModule


class TestPadeActivationModule:
    """Tests for the core low-order (m=2, n=1) Padé computation."""

    def test_matches_manual_computation(self):
        module = PadeActivationModule()
        x = torch.tensor([[2.0]])
        a0 = torch.tensor([[1.0]])
        a1 = torch.tensor([[0.5]])
        a2 = torch.tensor([[-0.25]])
        b1 = torch.tensor([[3.0]])

        output = module(x, a0, a1, a2, b1)

        numerator = 1.0 + 0.5 * 2.0 + (-0.25) * 2.0**2  # 1 + 1 - 1 = 1
        denominator = 1.0 + abs(3.0 * 2.0)  # 1 + 6 = 7
        expected = numerator / denominator
        assert torch.allclose(output, torch.tensor([[expected]]), atol=1e-5)

    def test_denominator_never_below_one(self):
        """Q(x) = 1 + |b1*x| must be >= 1 for any real x, b1 -- the no-pole property."""
        module = PadeActivationModule()
        x = torch.randn(1000) * 1e6
        b1 = torch.randn(1000) * 1e6
        a0 = torch.zeros(1000)
        a1 = torch.zeros(1000)
        a2 = torch.zeros(1000)

        denominator = 1.0 + (b1 * x).abs()
        output = module(x, a0, a1, a2, b1)

        assert (denominator >= 1.0).all()
        assert torch.isfinite(output).all()


class TestPadeActivation:
    """Tests for PadeActivation (data-driven, mirrors BezierActivation)."""

    def test_2d_input_shape(self):
        activation = PadeActivation()
        x = torch.randn(4, 50)  # 50 = 10 * 5
        output = activation(x)
        assert output.shape == (4, 10)

    def test_3d_input_shape(self):
        activation = PadeActivation()
        x = torch.randn(4, 16, 25)  # 25 = 5 * 5
        output = activation(x)
        assert output.shape == (4, 16, 5)

    def test_4d_input_shape(self):
        activation = PadeActivation()
        x = torch.randn(2, 15, 8, 8)  # 15 = 3 * 5
        output = activation(x)
        assert output.shape == (2, 3, 8, 8)

    def test_channel_not_divisible_by_5_raises(self):
        activation = PadeActivation()
        x = torch.randn(4, 13)
        with pytest.raises(AssertionError):
            activation(x)

    def test_gradient_flow(self):
        activation = PadeActivation()
        x = torch.randn(4, 25, requires_grad=True)
        output = activation(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_finite_on_large_inputs(self):
        """No pole means large inputs must never produce inf/nan."""
        activation = PadeActivation()
        x = torch.randn(4, 25) * 1e4
        output = activation(x)
        assert torch.isfinite(output).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_pade_activation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fluxflow.models.pade_activation'`

- [ ] **Step 3: Implement `PadeActivationModule` and `PadeActivation`**

Create `src/fluxflow/models/pade_activation.py`:

```python
"""
Padé Activation Units (PAU) for FluxFlow models.

Rational-function generalization of the Bezier activation family: a cubic
Bezier segment is a degree-(3,0) Padé approximant, so PAU is a strictly
larger function class at comparable cost. Uses the "safe PAU" denominator
construction (Molina et al., "Padé Activation Units: End-to-end Learning of
Flexible Activation Functions in Deep Networks," ICLR 2020):
Q(x) = 1 + |Q_raw(x)| guarantees Q(x) >= 1 for every real x and every real
coefficient, so the activation has no pole, ever. Evaluated via Horner's
method: fused multiply-adds and one division, zero transcendental calls.

Contains:
- PadeActivationModule: core low-order (m=2, n=1) rational computation
- PadeActivation: data-driven Padé (mirrors BezierActivation) -- input split
  5 ways into (x, a0, a1, a2, b1), reduces 5 channels -> 1
- TrainablePade: learnable high-order (m=5, n=4) Padé (mirrors TrainableBezier)
- WideTrainablePade: TrainablePade with a wide-range default for logvar use
"""

import torch
import torch.nn as nn


class PadeActivationModule(nn.Module):
    """
    Core low-order (m=2, n=1) Padé computation: P(x)/Q(x).

    P(x) = a0 + a1*x + a2*x^2          (numerator, degree 2)
    Q(x) = 1 + |b1*x|                  (denominator, degree 1, "safe" form)

    Q(x) >= 1 for every real x and every real b1 -- no domain restriction is
    required on x, a0, a1, a2, or b1.
    """

    def forward(
        self,
        x: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        b1: torch.Tensor,
    ) -> torch.Tensor:
        inner = torch.addcmul(a1, x, a2)
        numerator = torch.addcmul(a0, x, inner)
        denominator = 1.0 + (b1 * x).abs()
        return numerator / denominator


class PadeActivation(nn.Module):
    """
    Data-driven Padé activation.

    Mirrors BezierActivation: expects input with channels divisible by 5,
    where every 5 channels represent [x, a0, a1, a2, b1] for the Padé
    computation. Reduces channel count by 5x.

    Supports 2D [B, D], 3D [B, S, D], and 4D+ [B, C, H, W, ...] tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pade = PadeActivationModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = x.dim()

        if dims == 2:
            B, D = x.shape
            assert D % 5 == 0, "Channel dimension must be divisible by 5."
            x = x.view(B, D // 5, 5)
            t, a0, a1, a2, b1 = x.unbind(dim=-1)
            return self.pade(t, a0, a1, a2, b1)

        elif dims == 3:
            B, S, D = x.shape
            assert D % 5 == 0, "Channel dimension must be divisible by 5."
            x = x.view(B, S, D // 5, 5)
            t, a0, a1, a2, b1 = x.unbind(dim=-1)
            t = t.transpose(1, 2)
            a0 = a0.transpose(1, 2)
            a1 = a1.transpose(1, 2)
            a2 = a2.transpose(1, 2)
            b1 = b1.transpose(1, 2)
            out = self.pade(t, a0, a1, a2, b1)
            return out.transpose(1, 2)

        elif dims >= 4:
            batch_size, channels, *spatial_dims = x.size()
            assert channels % 5 == 0, "Channel dimension must be divisible by 5."
            num_features = channels // 5
            x = x.view(batch_size, num_features, 5, *spatial_dims)
            t, a0, a1, a2, b1 = x.unbind(dim=2)
            return self.pade(t, a0, a1, a2, b1)

        else:
            raise ValueError(f"Unsupported input dimensions: {dims}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_pade_activation.py -v`
Expected: PASS (all `TestPadeActivationModule`/`TestPadeActivation` tests)

- [ ] **Step 5: Commit**

```bash
git add src/fluxflow/models/pade_activation.py tests/unit/test_pade_activation.py
git commit -m "feat: add PadeActivationModule and PadeActivation"
```

---

### Task 2: `TrainablePade` + `WideTrainablePade` (learnable, high order)

**Files:**
- Modify: `src/fluxflow/models/pade_activation.py`
- Test: `tests/unit/test_pade_activation.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent classes in the same file).
- Produces: `TrainablePade(shape, a0=0.0, a1=1.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0, b1=0.0, b2=0.0, b3=0.0, b4=0.0, channel_only=False)`, `WideTrainablePade(shape, a0=-8.0, a1=1.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0, b1=0.0, b2=0.0, b3=0.0, b4=0.0, channel_only=False)` — both `nn.Module`, `forward(x) -> Tensor` (shape-preserving).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_pade_activation.py`:

```python
import torch.nn as nn

from fluxflow.models.pade_activation import TrainablePade, WideTrainablePade


class TestTrainablePade:
    """Tests for TrainablePade module."""

    def test_initialization_is_identity_at_default(self):
        """Default coefficients (a0=0, a1=1, rest 0) must compute P(x)/Q(x) == x."""
        module = TrainablePade((4,))
        x = torch.randn(2, 4)
        output = module(x)
        assert torch.allclose(output, x, atol=1e-5)

    def test_parameters_are_learnable(self):
        module = TrainablePade((3, 4, 4))
        param_count = sum(1 for _ in module.parameters())
        assert param_count == 10  # a0..a5, b1..b4
        for param in module.parameters():
            assert param.requires_grad

    def test_forward_output_shape(self):
        shape = (3, 8, 8)
        module = TrainablePade(shape)
        x = torch.randn(2, 3, 8, 8)
        output = module(x)
        assert output.shape == (2, 3, 8, 8)

    def test_gradient_updates_parameters(self):
        module = TrainablePade((2, 4, 4))
        x = torch.randn(1, 2, 4, 4)
        output = module(x)
        loss = output.sum()
        loss.backward()
        assert module.a0.grad is not None
        assert module.b1.grad is not None

    def test_denominator_never_below_one_under_training_drift(self):
        """After large synthetic parameter updates, Q(x) must stay >= 1 (no pole)."""
        module = TrainablePade((4,))
        with torch.no_grad():
            for p in module.parameters():
                p.copy_(torch.randn_like(p) * 100)
        x = torch.randn(8, 4) * 100
        output = module(x)
        assert torch.isfinite(output).all()

    def test_channel_only_broadcasts_over_spatial_dims(self):
        module = TrainablePade((3,), channel_only=True)
        x = torch.randn(2, 3, 8, 8)
        output = module(x)
        assert output.shape == (2, 3, 8, 8)


class TestWideTrainablePade:
    """Tests for WideTrainablePade module."""

    def test_initialization_is_shifted_identity_at_default(self):
        """Default coefficients (a0=-8, a1=1, rest 0) must compute P(x)/Q(x) == x - 8."""
        module = WideTrainablePade((4,))
        x = torch.randn(2, 4)
        output = module(x)
        assert torch.allclose(output, x - 8.0, atol=1e-5)

    def test_is_subclass_of_trainable_pade(self):
        module = WideTrainablePade((4,))
        assert isinstance(module, TrainablePade)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_pade_activation.py -v -k "TrainablePade or WideTrainablePade"`
Expected: FAIL with `ImportError: cannot import name 'TrainablePade'`

- [ ] **Step 3: Implement `TrainablePade` and `WideTrainablePade`**

Append to `src/fluxflow/models/pade_activation.py`:

```python
class TrainablePade(nn.Module):
    """
    Padé activation with learnable rational-function coefficients.

    P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5   (degree 5)
    Q(x) = 1 + |b1*x + b2*x^2 + b3*x^3 + b4*x^4|            (degree 4, safe)

    Matches the literature-standard "safe PAU" order (Molina et al., 2019).
    Default coefficients (a0=0, a1=1, rest 0) initialize to the identity
    function: P(x)/Q(x) = x/1 = x.

    Args:
        shape: Shape of the input tensor excluding batch dimension.
        channel_only: If True, learns per-channel only (broadcasts spatially).
    """

    def __init__(
        self,
        shape,
        a0=0.0,
        a1=1.0,
        a2=0.0,
        a3=0.0,
        a4=0.0,
        a5=0.0,
        b1=0.0,
        b2=0.0,
        b3=0.0,
        b4=0.0,
        channel_only=False,
    ):
        super().__init__()
        self.channel_only = channel_only

        def make_param(value):
            return nn.Parameter(torch.ones(shape) * value)

        self.a0 = make_param(a0)
        self.a1 = make_param(a1)
        self.a2 = make_param(a2)
        self.a3 = make_param(a3)
        self.a4 = make_param(a4)
        self.a5 = make_param(a5)
        self.b1 = make_param(b1)
        self.b2 = make_param(b2)
        self.b3 = make_param(b3)
        self.b4 = make_param(b4)

    def _broadcast(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.channel_only and x.dim() == 4:
            return p.view(1, -1, 1, 1)
        return p.expand_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a0 = self._broadcast(self.a0, x)
        a1 = self._broadcast(self.a1, x)
        a2 = self._broadcast(self.a2, x)
        a3 = self._broadcast(self.a3, x)
        a4 = self._broadcast(self.a4, x)
        a5 = self._broadcast(self.a5, x)
        b1 = self._broadcast(self.b1, x)
        b2 = self._broadcast(self.b2, x)
        b3 = self._broadcast(self.b3, x)
        b4 = self._broadcast(self.b4, x)

        # Horner's method, zero transcendental calls.
        numerator = torch.addcmul(a4, x, a5)
        numerator = torch.addcmul(a3, x, numerator)
        numerator = torch.addcmul(a2, x, numerator)
        numerator = torch.addcmul(a1, x, numerator)
        numerator = torch.addcmul(a0, x, numerator)

        denom_raw = torch.addcmul(b3, x, b4)
        denom_raw = torch.addcmul(b2, x, denom_raw)
        denom_raw = torch.addcmul(b1, x, denom_raw)
        denom_raw = x * denom_raw
        denominator = 1.0 + denom_raw.abs()

        return numerator / denominator


class WideTrainablePade(TrainablePade):
    """
    TrainablePade with a wide-range default for unbounded logvar use.

    Default coefficients (a0=-8, a1=1, rest 0) initialize to a shifted
    identity: P(x)/Q(x) = x - 8, mirroring WideTrainableBezier's very
    negative default control point (p0=-8) that lets logvar effectively
    shut off noise on specific channels.

    Args:
        shape: Tensor shape (matches TrainablePade semantics).
        channel_only: Per-channel mode (broadcasts spatially).
    """

    def __init__(
        self,
        shape,
        a0=-8.0,
        a1=1.0,
        a2=0.0,
        a3=0.0,
        a4=0.0,
        a5=0.0,
        b1=0.0,
        b2=0.0,
        b3=0.0,
        b4=0.0,
        channel_only=False,
    ):
        super().__init__(
            shape,
            a0=a0,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            a5=a5,
            b1=b1,
            b2=b2,
            b3=b3,
            b4=b4,
            channel_only=channel_only,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_pade_activation.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/fluxflow/models/pade_activation.py tests/unit/test_pade_activation.py
git commit -m "feat: add TrainablePade and WideTrainablePade"
```

---

### Task 3: `make_activation` dispatcher

**Files:**
- Modify: `src/fluxflow/models/activations.py`
- Test: `tests/unit/test_activations.py`

**Interfaces:**
- Consumes: `PadeActivation`, `TrainablePade`, `WideTrainablePade` from Task 1/2 (`fluxflow.models.pade_activation`); `BezierActivation`, `TrainableBezier`, `WideTrainableBezier` already in `activations.py`.
- Produces: `make_activation(kind: str, mode: str, shape=None, channel_only: bool = False, t_pre_activation="sigmoid", p_preactivation=None, **overrides) -> nn.Module`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_activations.py`:

```python
class TestMakeActivation:
    """Tests for the make_activation dispatcher."""

    def test_bezier_fixed_mode(self):
        from fluxflow.models.activations import BezierActivation, make_activation

        act = make_activation("bezier", "fixed", t_pre_activation="tanh", p_preactivation="silu")
        assert isinstance(act, BezierActivation)

    def test_pade_fixed_mode(self):
        from fluxflow.models.pade_activation import PadeActivation
        from fluxflow.models.activations import make_activation

        act = make_activation("pade", "fixed", t_pre_activation="tanh", p_preactivation="silu")
        assert isinstance(act, PadeActivation)

    def test_bezier_trainable_mode_with_overrides(self):
        from fluxflow.models.activations import make_activation

        act = make_activation("bezier", "trainable", shape=(4,), p0=-0.5, p1=-0.1, p2=0.1, p3=0.5)
        assert torch.allclose(act.p0, torch.ones(4) * -0.5)

    def test_pade_trainable_mode_default(self):
        from fluxflow.models.pade_activation import TrainablePade
        from fluxflow.models.activations import make_activation

        act = make_activation("pade", "trainable", shape=(4,))
        assert isinstance(act, TrainablePade)

    def test_bezier_wide_mode(self):
        from fluxflow.models.activations import WideTrainableBezier, make_activation

        act = make_activation("bezier", "wide", shape=(4,))
        assert isinstance(act, WideTrainableBezier)

    def test_pade_wide_mode(self):
        from fluxflow.models.pade_activation import WideTrainablePade
        from fluxflow.models.activations import make_activation

        act = make_activation("pade", "wide", shape=(4,))
        assert isinstance(act, WideTrainablePade)

    def test_unknown_kind_raises(self):
        from fluxflow.models.activations import make_activation

        with pytest.raises(ValueError):
            make_activation("unknown", "fixed")

    def test_unknown_mode_raises(self):
        from fluxflow.models.activations import make_activation

        with pytest.raises(ValueError):
            make_activation("bezier", "unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_activations.py -v -k TestMakeActivation`
Expected: FAIL with `ImportError: cannot import name 'make_activation'`

- [ ] **Step 3: Implement the dispatcher**

Add to `src/fluxflow/models/activations.py`, after the `WideTrainableBezier` class definition:

```python
def make_activation(
    kind: str,
    mode: str,
    shape=None,
    channel_only: bool = False,
    t_pre_activation: Optional[str] = "sigmoid",
    p_preactivation: Optional[str] = None,
    **overrides,
) -> nn.Module:
    """
    Construct a Bezier or Padé activation module by name.

    Centralizes the activation-family dispatch used throughout v100/*.py so
    the choice between "bezier" (default) and "pade" is made in one place
    per call site instead of hardcoding a class.

    Args:
        kind: "bezier" or "pade".
        mode: "fixed" (data-driven, 5-channel reduction -- BezierActivation /
            PadeActivation), "trainable" (TrainableBezier / TrainablePade),
            or "wide" (WideTrainableBezier / WideTrainablePade).
        shape, channel_only: forwarded to the "trainable"/"wide" constructors.
        t_pre_activation, p_preactivation: Bezier-only. Bezier's Bernstein
            basis needs its `t` input squashed into [0, 1]; Padé's rational
            function is defined on all reals and needs no such squashing, so
            these are silently ignored for kind="pade" -- a deliberate
            asymmetry, not a bug.
        **overrides: control-point / coefficient overrides forwarded to the
            underlying constructor (e.g. p0=... for Bezier, a0=... for
            Padé). Overrides use family-specific names -- callers that need
            custom values for both families must branch on `kind`
            themselves rather than passing one shared kwargs dict.

    Returns:
        The constructed activation nn.Module.
    """
    from fluxflow.models.pade_activation import PadeActivation, TrainablePade, WideTrainablePade

    if kind == "bezier":
        if mode == "fixed":
            return BezierActivation(
                t_pre_activation=t_pre_activation, p_preactivation=p_preactivation
            )
        elif mode == "trainable":
            return TrainableBezier(shape, channel_only=channel_only, **overrides)
        elif mode == "wide":
            return WideTrainableBezier(shape, channel_only=channel_only, **overrides)
        raise ValueError(f"Unknown mode: {mode!r}")

    elif kind == "pade":
        if mode == "fixed":
            return PadeActivation()
        elif mode == "trainable":
            return TrainablePade(shape, channel_only=channel_only, **overrides)
        elif mode == "wide":
            return WideTrainablePade(shape, channel_only=channel_only, **overrides)
        raise ValueError(f"Unknown mode: {mode!r}")

    raise ValueError(f"Unknown activation kind: {kind!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_activations.py -v`
Expected: PASS (all tests in the file, including pre-existing Bezier tests)

- [ ] **Step 5: Commit**

```bash
git add src/fluxflow/models/activations.py tests/unit/test_activations.py
git commit -m "feat: add make_activation dispatcher for Bezier/Pade selection"
```

---

### Task 4: `ModelConfig` + `ModelFactory` `activation_type`

**Files:**
- Modify: `src/fluxflow/config.py`
- Modify: `src/fluxflow/models/factory.py`
- Test: `tests/test_config_integration.py`, `tests/test_model_factory.py`

**Interfaces:**
- Consumes: nothing from prior tasks directly (factory doesn't construct v100 classes with `activation_type` until Tasks 5-7 land; until then this task only threads the parameter through and gates it to v0.10.0).
- Produces: `ModelConfig.activation_type: Literal["bezier", "pade"]`; `ModelFactory(..., activation_type: Literal["bezier", "pade"] = "bezier")`, `ModelFactory.activation_type` attribute; module-level `_ACTIVATION_SELECTABLE_VERSIONS = {"0.10.0"}` in `factory.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_integration.py` (add near other `ModelConfig` tests — inspect the file's existing test class name and match it; if none exists, add a new top-level test function):

```python
def test_model_config_activation_type_default():
    from fluxflow.config import ModelConfig

    config = ModelConfig()
    assert config.activation_type == "bezier"


def test_model_config_activation_type_pade():
    from fluxflow.config import ModelConfig

    config = ModelConfig(activation_type="pade")
    assert config.activation_type == "pade"


def test_model_config_activation_type_rejects_invalid():
    import pytest
    from pydantic import ValidationError

    from fluxflow.config import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(activation_type="chaotic_pendulum")
```

Append to `tests/test_model_factory.py`:

```python
class TestModelFactoryActivationType:
    """Test ModelFactory activation_type selection for v0.10.0."""

    def test_default_activation_type_is_bezier(self):
        factory = ModelFactory(model_type="bezier", model_version="0.10.0")
        assert factory.activation_type == "bezier"

    def test_activation_type_pade(self):
        factory = ModelFactory(model_type="bezier", model_version="0.10.0", activation_type="pade")
        assert factory.activation_type == "pade"

    def test_create_vae_encoder_v0100_passes_activation_type(self):
        factory = ModelFactory(
            model_type="bezier", model_version="0.10.0", activation_type="pade", vae_dim=32
        )
        encoder = factory.create_vae_encoder(downscales=2)
        assert encoder.activation_type == "pade"

    def test_create_vae_encoder_v070_ignores_activation_type(self):
        """v0.7.0 is frozen/historical -- activation_type must not be passed to it."""
        factory = ModelFactory(
            model_type="bezier", model_version="0.7.0", activation_type="pade", vae_dim=32
        )
        # Must not raise TypeError from an unexpected activation_type kwarg.
        encoder = factory.create_vae_encoder(downscales=2)
        assert not hasattr(encoder, "activation_type")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config_integration.py tests/test_model_factory.py -v -k activation_type`
Expected: FAIL — `ModelConfig`/`ModelFactory` reject the unexpected `activation_type` kwarg, or `AttributeError` on `factory.activation_type`.

- [ ] **Step 3: Implement the config field**

In `src/fluxflow/config.py`, add to `ModelConfig` (immediately after the `model_type` field):

```python
    activation_type: Literal["bezier", "pade"] = Field(
        default="bezier",
        description=(
            "Activation family for v0.10.0 models: 'bezier' (default) or 'pade' "
            "(rational-function Padé activation, comparable cost, strictly larger "
            "function class). Ignored for v0.3.0/0.6.0/0.7.0/0.8.0 (Bezier-only, "
            "frozen/historical) and for model_type='baseline'."
        ),
    )
```

- [ ] **Step 4: Implement the factory wiring**

In `src/fluxflow/models/factory.py`, add near `_SDPA_SUPPORTED_VERSIONS`:

```python
# Model versions whose VAE/Flow constructors accept an `activation_type`
# kwarg (Bezier vs Padé selection). Older versions (v0.3.0/0.6.0/0.7.0/0.8.0)
# are frozen/historical and remain Bezier-only -- passing activation_type to
# them raises TypeError, so it is gated the same way as attn_backend.
_ACTIVATION_SELECTABLE_VERSIONS = {"0.10.0"}
```

Add `ActivationFamily = Literal["bezier", "pade"]` next to the existing `ActivationType = Literal["silu", "gelu", "relu"]` line.

In `ModelFactory.__init__`, add a new parameter and store it:

```python
        model_version: str = "0.6.0",
        activation_type: "ActivationFamily" = "bezier",
```
(insert `activation_type` as a new keyword parameter right after `model_version` in the signature)

```python
        self.model_version = model_version
        self.activation_type = activation_type
```
(insert right after `self.model_version = model_version` in the body)

In `create_vae_encoder`, change:

```python
        classes = self._get_versioned_classes()
        FluxCompressor = classes["FluxCompressor"]

        return FluxCompressor(  # type: ignore
            in_channels=in_channels,
```

to:

```python
        classes = self._get_versioned_classes()
        FluxCompressor = classes["FluxCompressor"]
        extra_kwargs = (
            {"activation_type": self.activation_type}
            if self.model_version in _ACTIVATION_SELECTABLE_VERSIONS
            else {}
        )

        return FluxCompressor(  # type: ignore
            in_channels=in_channels,
```

and add `**extra_kwargs,` as the last argument before the closing `)` of that call.

In `create_vae_decoder`'s bezier branch (`if self.model_type == "bezier":`), apply the same pattern: build `extra_kwargs` before the `return FluxExpander(...)` call and add `**extra_kwargs,` to it.

In `create_flow_processor`'s bezier branch, extend the existing `extra_kwargs` dict construction:

```python
            extra_kwargs = (
                {"attn_backend": self.attention_backend}
                if self.model_version in _SDPA_SUPPORTED_VERSIONS
                else {}
            )
```

to:

```python
            extra_kwargs = (
                {"attn_backend": self.attention_backend}
                if self.model_version in _SDPA_SUPPORTED_VERSIONS
                else {}
            )
            if self.model_version in _ACTIVATION_SELECTABLE_VERSIONS:
                extra_kwargs["activation_type"] = self.activation_type
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config_integration.py tests/test_model_factory.py -v`
Expected: FAIL on `test_create_vae_encoder_v0100_passes_activation_type` (`FluxCompressor_v100` doesn't accept `activation_type` yet) — that's expected until Task 6. Confirm `test_model_config_activation_type_*` and the plain `factory.activation_type` tests PASS now.

- [ ] **Step 6: Commit**

```bash
git add src/fluxflow/config.py src/fluxflow/models/factory.py tests/test_config_integration.py tests/test_model_factory.py
git commit -m "feat: add activation_type to ModelConfig and ModelFactory"
```

---

### Task 5: Wire `v100/conditioning.py` (`SPADE_v100b`)

**Files:**
- Modify: `src/fluxflow/models/v100/conditioning.py`
- Test: `tests/v100/test_spade_v100b.py`

**Interfaces:**
- Consumes: `make_activation` from Task 3 (`fluxflow.models.activations`).
- Produces: `SPADE_v100b(context_nc, num_features, activation_type: str = "bezier")`.

- [ ] **Step 1: Write the failing test**

Append to `tests/v100/test_spade_v100b.py`:

```python
def test_spade_v100b_accepts_pade_activation_type():
    from fluxflow.models.v100.conditioning import SPADE_v100b

    spade = SPADE_v100b(context_nc=16, num_features=8, activation_type="pade")
    assert spade.activation_type == "pade"

    import torch

    x = torch.randn(2, 8, 4, 4)
    context = torch.randn(2, 16, 4, 4)
    with torch.no_grad():
        out = spade(x, context)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/v100/test_spade_v100b.py -v -k pade_activation_type`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'activation_type'`

- [ ] **Step 3: Implement**

In `src/fluxflow/models/v100/conditioning.py`, change the import line:

```python
from ..activations import BezierActivation
```
to:
```python
from ..activations import make_activation
```

Change the `__init__` signature and body:

```python
    def __init__(self, context_nc: int, num_features: int) -> None:
        super().__init__(context_nc, num_features)
        # Replace the inherited 1-layer mlp_shared / mlp_beta with the new heads.
        hidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(context_nc, hidden * 5, kernel_size=3, padding=1),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Conv2d(hidden, hidden * 5, kernel_size=3, padding=1),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
        )
```

to:

```python
    def __init__(self, context_nc: int, num_features: int, activation_type: str = "bezier") -> None:
        super().__init__(context_nc, num_features)
        self.activation_type = activation_type
        # Replace the inherited 1-layer mlp_shared / mlp_beta with the new heads.
        hidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(context_nc, hidden * 5, kernel_size=3, padding=1),
            make_activation(activation_type, "fixed", t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Conv2d(hidden, hidden * 5, kernel_size=3, padding=1),
            make_activation(activation_type, "fixed", t_pre_activation="sigmoid", p_preactivation="silu"),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/v100/test_spade_v100b.py -v`
Expected: PASS (all tests in the file, including pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add src/fluxflow/models/v100/conditioning.py tests/v100/test_spade_v100b.py
git commit -m "feat: add activation_type to SPADE_v100b"
```

---

### Task 6: Wire `v100/vae.py` (`FluxCompressor_v100`, `FluxExpander_v100`, and helpers)

**Files:**
- Modify: `src/fluxflow/models/v100/vae.py`
- Test: `tests/unit/test_v100_vae.py`

**Interfaces:**
- Consumes: `make_activation` from Task 3; `SPADE_v100b(..., activation_type=...)` from Task 5.
- Produces: `FluxCompressor_v100(..., activation_type: str = "bezier")`, `FluxExpander_v100(..., activation_type: str = "bezier")`, both storing `self.activation_type`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_v100_vae.py`:

```python
class TestActivationTypeSelection:
    """Tests for activation_type selection in FluxCompressor_v100/FluxExpander_v100."""

    def test_compressor_default_is_bezier(self):
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=32, downscales=2)
        assert comp.activation_type == "bezier"

    def test_compressor_pade_forward_runs(self):
        from fluxflow.models.v100.vae import FluxCompressor_v100

        comp = FluxCompressor_v100(d_model=32, downscales=2, activation_type="pade")
        assert comp.activation_type == "pade"
        img = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            packed = comp(img)
        assert torch.isfinite(packed).all()

    def test_expander_default_is_bezier(self):
        from fluxflow.models.v100.vae import FluxExpander_v100

        exp = FluxExpander_v100(d_model=32, upscales=2)
        assert exp.activation_type == "bezier"

    def test_expander_pade_forward_runs(self):
        from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

        comp = FluxCompressor_v100(d_model=32, downscales=2, activation_type="pade")
        exp = FluxExpander_v100(d_model=32, upscales=2, activation_type="pade")
        assert exp.activation_type == "pade"
        img = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            packed = comp(img)
            z_tokens = packed[:, :-1, : comp.d_model]
            H_lat = W_lat = 32 // 4
            z = z_tokens.transpose(1, 2).reshape(2, comp.d_model, H_lat, W_lat)
            ctx = packed[:, :-1, comp.d_model :].transpose(1, 2).reshape(2, comp.d_model, H_lat, W_lat)
            rgb = exp(z, ctx)
        assert rgb.shape[0] == 2
        assert rgb.shape[1] == 3
        assert torch.isfinite(rgb).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_v100_vae.py -v -k ActivationTypeSelection`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'activation_type'`

If `test_expander_pade_forward_runs`'s manual unpack of `packed`/call to `exp(...)` doesn't match `FluxExpander_v100.forward`'s actual signature, first read `FluxExpander_v100.forward` (below the `__init__` in `v100/vae.py`) and adjust the test's forward-call arguments to match it exactly — the shape-reconstruction logic must mirror whatever `FluxCompressor_v100.forward`'s packed-tensor layout actually is; the goal of the test is only "runs end-to-end with activation_type='pade' and produces finite output," not to re-derive the packing format from scratch.

- [ ] **Step 3: Implement**

In `src/fluxflow/models/v100/vae.py`, change the import line:

```python
from ..activations import BezierActivation, TrainableBezier, WideTrainableBezier, xavier_init
```
to:
```python
from ..activations import make_activation, xavier_init
```

**`_ResidualUpsampleBlock`** — change `__init__`:

```python
    def __init__(self, channels: int, context_size: int = 1024, use_spade: bool = True) -> None:
        super().__init__()
        self.use_spade = use_spade
        if self.use_spade:
            self.spade = SPADE_v100b(context_size, channels)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 5, kernel_size=16, stride=2, padding=7),
            BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
            nn.Conv2d(channels, channels * 5, kernel_size=5, padding=4, stride=1, dilation=2),
            BezierActivation(t_pre_activation="tanh", p_preactivation="silu"),
        )
```

to:

```python
    def __init__(
        self,
        channels: int,
        context_size: int = 1024,
        use_spade: bool = True,
        activation_type: str = "bezier",
    ) -> None:
        super().__init__()
        self.use_spade = use_spade
        if self.use_spade:
            self.spade = SPADE_v100b(context_size, channels, activation_type=activation_type)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 5, kernel_size=16, stride=2, padding=7),
            make_activation(activation_type, "fixed", t_pre_activation="tanh", p_preactivation="silu"),
            nn.Conv2d(channels, channels * 5, kernel_size=5, padding=4, stride=1, dilation=2),
            make_activation(activation_type, "fixed", t_pre_activation="tanh", p_preactivation="silu"),
        )
```

**`_ProgressiveUpscaler`** — change `__init__`:

```python
    def __init__(
        self,
        channels: int = 3,
        steps: int = 2,
        context_size: int = 1024,
        use_spade: bool = True,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                _ResidualUpsampleBlock(channels, context_size, use_spade=use_spade)
                for _ in range(steps)
            ]
        )
```

to:

```python
    def __init__(
        self,
        channels: int = 3,
        steps: int = 2,
        context_size: int = 1024,
        use_spade: bool = True,
        use_gradient_checkpointing: bool = True,
        activation_type: str = "bezier",
    ) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layers = nn.ModuleList(
            [
                _ResidualUpsampleBlock(
                    channels, context_size, use_spade=use_spade, activation_type=activation_type
                )
                for _ in range(steps)
            ]
        )
```

**`_AttnBlock`** — change `__init__`:

```python
    def __init__(self, dim: int, heads: int, drop: float = 0.0, ff_mult: int = 2) -> None:
        super().__init__()
        hidden = dim * ff_mult
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden * 5),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Linear(hidden, dim),
        )
```

to:

```python
    def __init__(
        self,
        dim: int,
        heads: int,
        drop: float = 0.0,
        ff_mult: int = 2,
        activation_type: str = "bezier",
    ) -> None:
        super().__init__()
        hidden = dim * ff_mult
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden * 5),
            make_activation(activation_type, "fixed", t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Linear(hidden, dim),
        )
```

**`FluxCompressor_v100.__init__`** — add `activation_type: str = "bezier"` as the last parameter (before the closing `) -> None:`) and `self.activation_type = activation_type` right after `self.use_gradient_checkpointing = use_gradient_checkpointing`. Then replace every `BezierActivation(t_pre_activation="tanh", p_preactivation="silu")` occurrence in this method with `make_activation(self.activation_type, "fixed", t_pre_activation="tanh", p_preactivation="silu")`, and every `BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu")` occurrence with `make_activation(self.activation_type, "fixed", t_pre_activation="sigmoid", p_preactivation="silu")`. This covers the `encoder_first_step`, `encoder_z`, `latent_proj`, `mu_proj`, `logvar_proj`, `ctx_encoder_first_step`, `ctx_encoder_z`, `ctx_proj`, and `ctx_zinject_proj` sites (12 occurrences total — confirm with `grep -n "BezierActivation(" src/fluxflow/models/v100/vae.py` before and after: it must go from 15 total file-wide matches down to 0).

Replace:

```python
        self.mu_activation = TrainableBezier(
            shape=(d_model,), channel_only=True, p0=-0.5, p1=-0.1, p2=0.1, p3=0.5
        )
        self.logvar_activation = WideTrainableBezier(
            shape=(d_model,),
            channel_only=True,
            p0=-8.0,
            p1=-2.0,
            p2=2.0,
            p3=4.0,
        )
```

with:

```python
        if self.activation_type == "bezier":
            self.mu_activation = make_activation(
                "bezier", "trainable", shape=(d_model,), channel_only=True,
                p0=-0.5, p1=-0.1, p2=0.1, p3=0.5,
            )
            self.logvar_activation = make_activation(
                "bezier", "wide", shape=(d_model,), channel_only=True,
                p0=-8.0, p1=-2.0, p2=2.0, p3=4.0,
            )
        else:
            self.mu_activation = make_activation(
                self.activation_type, "trainable", shape=(d_model,), channel_only=True
            )
            self.logvar_activation = make_activation(
                self.activation_type, "wide", shape=(d_model,), channel_only=True
            )
```

Find the `_AttnBlock(d_model, effective_ctx_heads, attn_dropout, attn_ff_mult)` call (inside `ctx_token_attn`) and change it to:

```python
                _AttnBlock(
                    d_model, effective_ctx_heads, attn_dropout, attn_ff_mult,
                    activation_type=self.activation_type,
                )
```

**`FluxExpander_v100.__init__`** — add `activation_type: str = "bezier"` as the last parameter and `self.activation_type = activation_type` right after `self.d_model = d_model`. Change the `self.upscale = _ProgressiveUpscaler(...)` call to add `activation_type=activation_type,` as its last argument. Replace:

```python
        self.rgb_activation = TrainableBezier(
            shape=(3,),
            p0=-0.5,
            p1=-0.05,
            p2=0.05,
            p3=0.5,
            channel_only=True,
        )
```

with:

```python
        if activation_type == "bezier":
            self.rgb_activation = make_activation(
                "bezier", "trainable", shape=(3,), channel_only=True,
                p0=-0.5, p1=-0.05, p2=0.05, p3=0.5,
            )
        else:
            self.rgb_activation = make_activation(
                activation_type, "trainable", shape=(3,), channel_only=True
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_v100_vae.py tests/v100/ -v`
Expected: PASS (all tests, including every pre-existing v100 VAE/SPADE test — these must still pass unchanged since `activation_type="bezier"` is the default and must reproduce prior behavior exactly)

- [ ] **Step 5: Commit**

```bash
git add src/fluxflow/models/v100/vae.py tests/unit/test_v100_vae.py
git commit -m "feat: add activation_type to FluxCompressor_v100 and FluxExpander_v100"
```

---

### Task 7: Wire `v100/flow.py` (`FluxTransformerBlock_v100`, `FluxFlowProcessor_v100`)

**Files:**
- Modify: `src/fluxflow/models/v100/flow.py`
- Test: `tests/unit/test_v100_flow.py`

**Interfaces:**
- Consumes: `make_activation` from Task 3.
- Produces: `FluxTransformerBlock_v100(..., activation_type: str = "bezier")`, `FluxFlowProcessor_v100(..., activation_type: str = "bezier")`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_v100_flow.py`:

```python
def test_flow_processor_v100_pade_activation_type_forward_runs():
    from fluxflow.models.v100.flow import FluxFlowProcessor_v100

    proc = FluxFlowProcessor_v100(
        d_model=128, vae_dim=32, embedding_size=64, n_layers=1, activation_type="pade"
    )
    assert proc.activation_type == "pade"
    packed = torch.zeros(1, 17, 64)
    packed[:, :-1, :] = torch.randn(1, 16, 64)
    packed[0, -1, 0] = 4 / 1024.0
    packed[0, -1, 1] = 4 / 1024.0
    text_seq = torch.randn(1, 5, 64)
    text_mask = torch.ones(1, 5, dtype=torch.bool)
    t = torch.tensor([0.5])
    with torch.no_grad():
        out = proc(packed, text_seq, text_mask, t)
    assert out.shape == packed.shape
    assert torch.isfinite(out).all()


def test_flow_processor_v100_default_activation_type_is_bezier():
    from fluxflow.models.v100.flow import FluxFlowProcessor_v100

    proc = FluxFlowProcessor_v100(d_model=128, vae_dim=32, embedding_size=64, n_layers=1)
    assert proc.activation_type == "bezier"
    assert proc.transformer_blocks[0].bezier_activation.__class__.__name__ == "BezierActivation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_v100_flow.py -v -k activation_type`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'activation_type'`

- [ ] **Step 3: Implement**

In `src/fluxflow/models/v100/flow.py`, change the import line:

```python
from ..activations import BezierActivation, TrainableBezier, xavier_init
```
to:
```python
from ..activations import make_activation, xavier_init
```

**`FluxTransformerBlock_v100.__init__`** — add `activation_type: str = "bezier"` as a new parameter (alongside `attn_backend`), store `self.activation_type = activation_type`, and change:

```python
        self.bezier_activation = BezierActivation()
```

to:

```python
        self.bezier_activation = make_activation(activation_type, "fixed")
```

(the attribute name `bezier_activation` is kept as-is — renaming it would ripple into every forward-pass reference in this class for no behavioral benefit; it now just holds whichever family was selected)

**`FluxFlowProcessor_v100.__init__`** — add `activation_type: str = "bezier"` as a new parameter, store `self.activation_type = activation_type` near the other stored config attributes. Change:

```python
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 5),
            BezierActivation(t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Linear(d_model, d_model),
        )
```

to:

```python
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 5),
            make_activation(activation_type, "fixed", t_pre_activation="sigmoid", p_preactivation="silu"),
            nn.Linear(d_model, d_model),
        )
```

Change:

```python
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock_v100(d_model, n_head, attn_backend=attn_backend)
                for _ in range(n_layers)
            ]
        )
```

to:

```python
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock_v100(
                    d_model, n_head, attn_backend=attn_backend, activation_type=activation_type
                )
                for _ in range(n_layers)
            ]
        )
```

Change:

```python
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2),
            TrainableBezier((d_model, 1, 1)),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.context_final = nn.Sequential(
            nn.Conv2d(d_model + 2, d_model, kernel_size=7, padding=3),
            TrainableBezier((d_model, 1, 1)),
        )
```

to:

```python
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=5, padding=2),
            make_activation(activation_type, "trainable", shape=(d_model, 1, 1)),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.context_final = nn.Sequential(
            nn.Conv2d(d_model + 2, d_model, kernel_size=7, padding=3),
            make_activation(activation_type, "trainable", shape=(d_model, 1, 1)),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_v100_flow.py tests/v100/ -v`
Expected: PASS (all tests, including every pre-existing v100 flow test)

- [ ] **Step 5: Commit**

```bash
git add src/fluxflow/models/v100/flow.py tests/unit/test_v100_flow.py
git commit -m "feat: add activation_type to FluxTransformerBlock_v100 and FluxFlowProcessor_v100"
```

---

### Task 8: Checkpoint metadata auto-detection and `force_activation_type` override

**Files:**
- Modify: `src/fluxflow/models/versioning.py`
- Test: `tests/v100/test_activation_type_checkpoint_metadata.py` (new)

**Interfaces:**
- Consumes: `FluxCompressor_v100.activation_type`, `FluxFlowProcessor_v100.activation_type`, `FluxExpander_v100.activation_type` from Tasks 6-7; `FluxPipeline` from `fluxflow.models.pipeline`.
- Produces: `_detect_architecture()` now includes `config["activation_type"]`; `ModelLoaderV010.load_checkpoint(..., force_activation_type=None)` resolves and applies the activation family.

- [ ] **Step 1: Write the failing tests**

Create `tests/v100/test_activation_type_checkpoint_metadata.py`:

```python
"""Checkpoint round-trip tests for activation_type auto-detection and override."""

import torch

from fluxflow.models.pipeline import FluxPipeline
from fluxflow.models.v100.flow import FluxFlowProcessor_v100
from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100
from fluxflow.models.versioning import load_versioned_checkpoint, save_versioned_checkpoint


def _build_pipeline(activation_type: str) -> FluxPipeline:
    compressor = FluxCompressor_v100(d_model=32, downscales=2, activation_type=activation_type)
    flow_processor = FluxFlowProcessor_v100(
        d_model=32, vae_dim=32, embedding_size=32, n_layers=1, activation_type=activation_type
    )
    expander = FluxExpander_v100(d_model=32, upscales=2, activation_type=activation_type)
    return FluxPipeline(compressor, flow_processor, expander)


def test_detect_architecture_captures_activation_type(tmp_path):
    from fluxflow.models.versioning import _detect_architecture

    pipeline = _build_pipeline("pade")
    config = _detect_architecture(pipeline)
    assert config["activation_type"] == "pade"


def test_save_then_load_auto_detects_pade(tmp_path):
    pipeline = _build_pipeline("pade")
    out_dir = tmp_path / "ckpt_pade"
    save_versioned_checkpoint(pipeline, out_dir, model_version="0.10.0")

    loaded = load_versioned_checkpoint(out_dir, device="cpu")
    assert loaded.compressor.activation_type == "pade"
    assert loaded.flow_processor.activation_type == "pade"
    assert loaded.expander.activation_type == "pade"


def test_save_bezier_then_load_with_force_pade_override(tmp_path):
    pipeline = _build_pipeline("bezier")
    out_dir = tmp_path / "ckpt_bezier"
    save_versioned_checkpoint(pipeline, out_dir, model_version="0.10.0")

    loaded = load_versioned_checkpoint(out_dir, device="cpu", force_activation_type="pade")
    assert loaded.compressor.activation_type == "pade"


def test_legacy_checkpoint_without_activation_type_key_defaults_to_bezier(tmp_path):
    """Metadata saved before this feature has no 'activation_type' key -- must default to bezier."""
    pipeline = _build_pipeline("bezier")
    out_dir = tmp_path / "ckpt_legacy"
    save_versioned_checkpoint(pipeline, out_dir, model_version="0.10.0")

    import json

    metadata_path = out_dir / "model_metadata.json"
    data = json.loads(metadata_path.read_text())
    del data["architecture"]["activation_type"]
    metadata_path.write_text(json.dumps(data, indent=2))

    loaded = load_versioned_checkpoint(out_dir, device="cpu")
    assert loaded.compressor.activation_type == "bezier"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/v100/test_activation_type_checkpoint_metadata.py -v`
Expected: FAIL — `test_detect_architecture_captures_activation_type` fails with `KeyError: 'activation_type'`; the save/load tests fail once that's fixed because `load_checkpoint` doesn't yet read or apply it.

- [ ] **Step 3: Implement `_detect_architecture` addition**

In `src/fluxflow/models/versioning.py`, inside `_detect_architecture`, immediately after the `if hasattr(model, "compressor"):` block's existing lines (`config["vae_dim"] = ...` etc.), add:

```python
        if hasattr(model.compressor, "activation_type"):
            config["activation_type"] = model.compressor.activation_type
```

- [ ] **Step 4: Implement `ModelLoaderV010.load_checkpoint` resolution**

In `src/fluxflow/models/versioning.py`, in `ModelLoaderV010.load_checkpoint`, immediately after `config = metadata.architecture`, add:

```python
        activation_type = kwargs.pop("force_activation_type", None) or config.get(
            "activation_type", "bezier"
        )
```

Change:

```python
        compressor = FluxCompressor_v100(
            in_channels=config.get("in_channels", 3),
            d_model=vae_latent_dim,
            downscales=config["downscales"],
            max_hw=config.get("max_hw", 1024),
            attn_layers=config.get("vae_attn_layers", 4),
            attn_heads=vae_attn_heads,
        )

        flow_processor = FluxFlowProcessor_v100(
            d_model=actual_d_model,
            vae_dim=vae_latent_dim,
            embedding_size=config.get("text_embed_dim", 1024),
            n_head=flow_attn_heads,
            n_layers=config.get("flow_transformer_layers", 10),
            max_hw=config.get("max_hw", 1024),
        )

        expander = FluxExpander_v100(
            d_model=vae_latent_dim,
            upscales=config.get("upscales", config["downscales"]),
            max_hw=config.get("max_hw", 1024),
        )
```

to:

```python
        compressor = FluxCompressor_v100(
            in_channels=config.get("in_channels", 3),
            d_model=vae_latent_dim,
            downscales=config["downscales"],
            max_hw=config.get("max_hw", 1024),
            attn_layers=config.get("vae_attn_layers", 4),
            attn_heads=vae_attn_heads,
            activation_type=activation_type,
        )

        flow_processor = FluxFlowProcessor_v100(
            d_model=actual_d_model,
            vae_dim=vae_latent_dim,
            embedding_size=config.get("text_embed_dim", 1024),
            n_head=flow_attn_heads,
            n_layers=config.get("flow_transformer_layers", 10),
            max_hw=config.get("max_hw", 1024),
            activation_type=activation_type,
        )

        expander = FluxExpander_v100(
            d_model=vae_latent_dim,
            upscales=config.get("upscales", config["downscales"]),
            max_hw=config.get("max_hw", 1024),
            activation_type=activation_type,
        )
```

The method signature `def load_checkpoint(self, checkpoint_path: Path, metadata: ModelMetadata, device: str, **kwargs) -> Any:` already accepts `**kwargs`, so `load_versioned_checkpoint(path, device="cpu", force_activation_type="pade")` already flows through `load_versioned_checkpoint`'s own `**kwargs` (see its signature in `versioning.py`) into this method without any signature change there.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/v100/test_activation_type_checkpoint_metadata.py tests/v100/ tests/unit/test_v100_vae.py tests/unit/test_v100_flow.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/fluxflow/models/versioning.py tests/v100/test_activation_type_checkpoint_metadata.py
git commit -m "feat: auto-detect activation_type in checkpoint metadata with force override"
```

---

### Task 9: `benchmark_pade.py`

**Files:**
- Create: `benchmark_pade.py` (repo root, alongside `benchmark_bezier.py`)

**Interfaces:**
- Consumes: `BezierActivation`/`TrainableBezier` (`fluxflow.models.activations`), `PadeActivation`/`TrainablePade` (`fluxflow.models.pade_activation`).
- Produces: printed benchmark table + parameter counts, run manually (not part of pytest) to gather numbers for Task 10.

- [ ] **Step 1: Create the benchmark script**

Create `benchmark_pade.py`:

```python
"""Benchmark script comparing BezierActivation/TrainableBezier against
PadeActivation/TrainablePade -- forward+backward speed and parameter count."""

import time

import torch

from fluxflow.models.activations import BezierActivation, TrainableBezier
from fluxflow.models.pade_activation import PadeActivation, TrainablePade


def benchmark_activation(activation, x, name, warmup=10, iterations=100):
    for _ in range(warmup):
        _ = activation(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    start = time.time()
    output = None
    for _ in range(iterations):
        output = activation(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = (elapsed / iterations) * 1000

    assert output is not None
    print(f"{name:30s} | {avg_time:8.4f} ms/iter | {output.shape}")
    return avg_time


def benchmark_forward_backward(activation, x, name):
    x = x.clone().requires_grad_(True)
    start = time.time()
    output = activation(x)
    loss = output.sum()
    loss.backward()
    if x.is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    print(f"{name:30s} | {elapsed:8.4f} ms (fwd+bwd)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Bezier vs Padé benchmarks on: {device}")
    print("=" * 70)

    bezier_fixed = BezierActivation().to(device)
    pade_fixed = PadeActivation().to(device)

    test_cases = [
        ("2D Small", (4, 50)),
        ("2D Medium", (32, 250)),
        ("2D Large", (64, 1000)),
        ("3D Small", (4, 64, 125)),
        ("3D Medium", (8, 128, 250)),
        ("4D Small", (4, 15, 8, 8)),
        ("4D Large", (16, 60, 32, 32)),
    ]

    print("\n--- Fixed / data-driven mode (BezierActivation vs PadeActivation) ---")
    print(f"\n{'Test Case':<30} | {'Time (ms)':<12} | Output Shape")
    print("-" * 70)
    for name, shape in test_cases:
        x = torch.randn(*shape, device=device)
        benchmark_activation(bezier_fixed, x, f"Bezier {name}")
        benchmark_activation(pade_fixed, x, f"Pade {name}")

    print("\n--- Trainable mode (TrainableBezier vs TrainablePade), shape (128,) ---")
    bezier_trainable = TrainableBezier((128,)).to(device)
    pade_trainable = TrainablePade((128,)).to(device)
    x = torch.randn(32, 128, device=device)
    benchmark_activation(bezier_trainable, x, "TrainableBezier")
    benchmark_activation(pade_trainable, x, "TrainablePade")

    print("\n--- Forward + backward pass ---")
    x = torch.randn(8, 125, device=device)
    benchmark_forward_backward(bezier_fixed, x, "Bezier fwd+bwd")
    benchmark_forward_backward(pade_fixed, x, "Pade fwd+bwd")

    print("\n--- Parameter counts ---")
    bezier_params = sum(p.numel() for p in TrainableBezier((128,)).parameters())
    pade_params = sum(p.numel() for p in TrainablePade((128,)).parameters())
    print(f"TrainableBezier params (D=128): {bezier_params}")
    print(f"TrainablePade params (D=128):   {pade_params}")

    print("\n✓ All benchmarks completed successfully!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark and capture output**

Run: `python benchmark_pade.py`
Expected: Completes without error, prints a full comparison table. Save the full stdout — it is the source data for Task 10's analysis doc.

- [ ] **Step 3: Commit**

```bash
git add benchmark_pade.py
git commit -m "feat: add benchmark_pade.py comparing Bezier and Pade activations"
```

---

### Task 10: Analysis doc, CHANGELOG, REFERENCES

**Files:**
- Create: `docs/PADE-ACTIVATION-ANALYSIS.md`
- Modify: `CHANGELOG.md`
- Modify: `REFERENCES.md`

**Interfaces:**
- Consumes: the actual stdout captured in Task 9, Step 2.

- [ ] **Step 1: Write `docs/PADE-ACTIVATION-ANALYSIS.md`**

Create the file with this structure (fill in the `<...>` benchmark-number placeholders with the *actual* numbers captured in Task 9 — this is the one place in this plan where a number is not yet known; every other line is real content):

```markdown
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

## Benchmark results (CPU/GPU: <fill in device from benchmark_pade.py output>)

Fixed / data-driven mode:

| Shape | BezierActivation (ms) | PadeActivation (ms) | Delta |
|---|---|---|---|
| <fill in each row from benchmark_pade.py stdout> | | | |

Trainable mode (shape (128,)):

| | TrainableBezier (ms) | TrainablePade (ms) | Delta |
|---|---|---|---|
| Forward | <...> | <...> | <...> |

Forward+backward:

| | Bezier (ms) | Pade (ms) | Delta |
|---|---|---|---|
| | <...> | <...> | <...> |

Parameter counts (D=128): TrainableBezier=<...>, TrainablePade=<...>.

**Interpretation:** `PadeActivation` (m=2, n=1) does ~3 FMAs + 1 division
vs. Bezier's ~5 FMAs + 0 divisions -- <state whether the measured numbers
came out competitive or not, plainly, based on the actual figures above>.
`TrainablePade` (m=5, n=4) does ~9 FMAs + 1 division vs. Bezier's ~5 FMAs
-- <state the measured delta plainly>. Report what was measured, not what
was hoped for.

## Quality claims: unvalidated

No training run has been performed. The "quality per parameter" argument
above is structural (strict superset of Bezier's function class, same
param budget), not empirical. A follow-up matched-param training ablation,
following the same pattern as `BaselineFluxTransformerBlock` in
`v070/flow.py` (which matches Bezier's and baseline's total parameter
counts to make a fair comparison), is required before any quality claim
can be made in front of stakeholders. That ablation is explicitly out of
scope for this pass -- see the design spec's "Explicitly out of scope"
section.

## Cross-repo follow-up required

`fluxflow-training`'s YAML/CLI config schema needs its own `activation_type`
passthrough field before a user can select `"pade"` from the training CLI.
The mechanism added in this pass (`ModelConfig`, `ModelFactory`,
checkpoint metadata auto-detection/override) works end-to-end for direct
`ModelFactory`/checkpoint-loader callers; the training entry point itself
was not touched.
```

- [ ] **Step 2: Run the benchmark and fill in the numbers**

Run: `python benchmark_pade.py`, copy the printed numbers into the table above, replacing every `<...>` placeholder. Do not commit the doc with any placeholder remaining — grep the file for `<` before committing to confirm none are left:

Run: `grep -n "<" docs/PADE-ACTIVATION-ANALYSIS.md`
Expected: no output (empty grep result)

- [ ] **Step 3: Update `CHANGELOG.md`**

Add a new bullet list under the existing `[Unreleased]` heading (do not create a new version section — this lands in the same unreleased 0.10.0 line as the rest of the in-progress redesign), following the existing `### Added` bullet style:

```markdown
- **Padé Activation Units** (`PadeActivation`, `TrainablePade`,
  `WideTrainablePade` in `models/pade_activation.py`): rational-function
  generalization of the Bezier activation family (a cubic Bezier segment is
  a degree-(3,0) Padé approximant), using the "safe PAU" denominator
  construction (`Q(x) = 1 + |Q_raw(x)|`, guaranteeing no pole) to stay
  transcendental-free. Selectable via `activation_type="pade"` on
  `ModelConfig`/`ModelFactory` and every `v100/` VAE/Flow/conditioning
  constructor (default remains `"bezier"`, unchanged behavior).
- `make_activation(kind, mode, ...)` dispatcher in `models/activations.py`
  centralizing Bezier-vs-Padé construction across all `v100/` call sites.
- Checkpoint metadata now records `activation_type` automatically at save
  time (`_detect_architecture`) and `ModelLoaderV010.load_checkpoint`
  auto-selects it at load time, with an explicit `force_activation_type`
  override kwarg. Checkpoints saved before this change default to
  `"bezier"`.
```

- [ ] **Step 4: Update `REFERENCES.md`**

Add a new subsection after the existing "Kolmogorov-Arnold Networks (KAN)" subsection in `REFERENCES.md`:

```markdown
### Padé Activation Units (PAU)

FluxFlow's Padé activation functions are a direct implementation of PAU,
used as a rational-function generalization of the Bezier activation family
(a cubic Bezier segment is a degree-(3,0) Padé approximant).

**Citation:**
```bibtex
@inproceedings{molina2020pade,
  title={Pad{\'e} Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks},
  author={Molina, Alejandro and Schramowski, Patrick and Kersting, Kristian},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/abs/1907.06732}
}
```

**Resources:**
- Paper: [arXiv:1907.06732](https://arxiv.org/abs/1907.06732)
```

- [ ] **Step 5: Commit**

```bash
git add docs/PADE-ACTIVATION-ANALYSIS.md CHANGELOG.md REFERENCES.md
git commit -m "docs: add Pade activation analysis, changelog, and references"
```

---

## Final full-suite verification

- [ ] Run: `pytest -q`
- [ ] Expected: full suite passes (pre-existing tests unaffected — every new `activation_type` parameter defaults to `"bezier"`, reproducing prior behavior exactly).
