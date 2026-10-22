"""
Padé Activation Units (PAU) for FluxFlow models.

Rational-function generalization of the Bezier activation family: a cubic
Bezier segment is a degree-(3,0) Padé approximant, so PAU is a strictly
larger function class at comparable cost. Uses the "safe PAU" denominator
construction (Molina et al., "Padé Activation Units: End-to-end Learning of
Flexible Activation Functions in Deep Networks," ICLR 2020):
Q(x) >= 1 for every real x and every real coefficient, so the activation has
no pole, ever. Fixed mode uses Q = 1 + |b1*x|; TrainablePade uses the smooth
form Q = 1 + sqrt(q^2 + eps) - sqrt(eps), q = b1*x + ... + b4*x^4, so the
denominator coefficients receive gradient even at q == 0 (|.| has zero
subgradient there). Evaluated via Horner's method: fused multiply-adds and one
division; the only non-FMA ops are the square root of the smooth abs and the
rational soft clamp (no exp/tanh).

Input range limiting uses a rational soft clamp (see ``_soft_clamp``): exact
identity for |x| <= 0.75*K, then a monotone saturation to +-K with nonzero
gradient everywhere, instead of a hard clamp whose gradient is zero beyond K.

TrainablePade stores its parameters in a normalized basis: the polynomials are
evaluated in u = x_clamped / K (|u| <= 1) with stored raw coefficients
r_k = c_k * K^k, so one optimizer step moves every term equally instead of
being amplified by |x|^k (up to K^5 ~ 65k for a5). Constructor arguments and
``effective_coefficients()`` are in the original x basis; state dicts without
the persistent ``pade_basis_version`` buffer (legacy) are converted on load.

Contains:
- PadeActivationModule: core low-order (m=2, n=1) rational computation
- PadeActivation: data-driven Padé (mirrors BezierActivation) -- input split
  5 ways into (x, a0, a1, a2, b1), reduces 5 channels -> 1
- TrainablePade: learnable high-order (m=5, n=4) Padé (mirrors TrainableBezier)
- WideTrainablePade: TrainablePade with a wide-range default for logvar use
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable

# TrainablePade's numerator is degree 5 and, unlike the "safe" denominator,
# has no built-in range limiting. The raw input is soft-clamped to +-K so x^5
# cannot overflow even with a coefficient magnitude approaching 1 (the
# observed near-identity-init scale, e.g. a1~=1.0): 65504 is fp16's max
# representable value, and 65504**(1/5) is the largest |x| for which
# |x|**5 <= 65504 -- i.e. the worst-case unit-coefficient term stays within
# fp16 range. This preserves far more dynamic range than Bezier's [0, 1]
# input squash while eliminating the confirmed overflow.
_INPUT_CLAMP = 65504.0**0.2

# PadeActivationModule: x, a0, a1, a2, b1 all come straight from the input
# tensor's channels, so none carries a smallness guarantee. With every factor
# bounded by K, Horner evaluation of a0 + x*(a1 + x*a2) has worst case
# K + K^2 + K^3 (not K^3 alone: the lower-order terms add). Requiring
# K + K^2 + K^3 <= 65504 (fp16 max) gives K ~= 39.3; K = 30 leaves margin
# (worst case 27930). Math runs in fp32 and the result is cast back, so the
# bound holds for the fp16 output too (Q >= 1 cannot increase |P/Q|).
_FIXED_MODE_INPUT_CLAMP = 30.0

# Soft clamp: identity up to _KNEE_FRACTION * K, rational saturation beyond.
_KNEE_FRACTION = 0.75
# Absolute guard so +-inf cannot reach the saturation arithmetic (inf/inf).
_FINITE_GUARD = 1e4
# Smooth-abs epsilon for the trainable denominator (see TrainablePade).
_SMOOTH_ABS_EPS = 1e-6
# Default b1: small, nonzero so the denominator trains from init (E1).
_DEFAULT_B1 = 1e-3
_BASIS_KEY = "pade_basis_version"
_BASIS_VERSION = 2  # absent: legacy effective coefficients; 2: raw (K^k-scaled)
_COEFF_NAMES = ("a0", "a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4")


def _soft_clamp(x: torch.Tensor, k: float) -> torch.Tensor:
    """Rational soft clamp to (-k, k): identity for |x| <= 0.75k, C1, monotone.

    Beyond the knee ``kn``: sign(x) * (kn + span * u / (1 + u)), u = (|x|-kn)/span,
    span = k - kn. Gradient is 1/(1+u)^2 > 0 everywhere (never exactly zero).
    """
    knee = _KNEE_FRACTION * k
    span = k - knee
    x = x.clamp(-_FINITE_GUARD, _FINITE_GUARD)
    u = (x.abs() - knee).clamp(min=0.0) / span
    return x.clamp(-knee, knee) + x.sign() * span * (u / (1.0 + u))


# Backward recompute is chunked so its transient autograd graph (~10-25x the
# chunk) stays small instead of ~10-25x the full activation.
_RECOMPUTE_CHUNK_ELEMS = 1 << 22


class _RecomputeFn(torch.autograd.Function):
    """Run ``fn`` without a graph; save only its inputs; recompute in backward.

    Elementwise rational math otherwise saves ~10 intermediates per call
    (several x the activation size). Recomputing locally from the inputs costs
    a few cheap kernels and keeps saved memory at ~1x input. ``dims`` gives, per
    tensor, the dim along which it is chunked in backward (None: whole tensor,
    gradient summed over chunks); ``dims[0]`` also chunks the output gradient.
    Autocast state of the forward is replayed in backward. Double backward is
    not supported.
    """

    @staticmethod
    def forward(
        ctx: Any,
        fn: Callable[..., torch.Tensor],
        dims: tuple[int | None, ...],
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        dev = tensors[0].device.type
        ctx.fn, ctx.dims = fn, dims
        ctx.ac = (dev, torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev))
        ctx.save_for_backward(*tensors)
        return fn(*tensors)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        saved, dims, need = ctx.saved_tensors, ctx.dims, ctx.needs_input_grad[2:]
        grads = [torch.zeros_like(t) if n else None for t, n in zip(saved, need)]
        d0, t0 = dims[0], saved[0]
        size = 1 if d0 is None else t0.shape[d0]
        step = size if d0 is None else max(1, _RECOMPUTE_CHUNK_ELEMS // max(1, t0.numel() // size))
        for st in range(0, size, step):
            n = min(step, size - st)
            sl = [t if d is None else t.narrow(d, st, n) for t, d in zip(saved, dims)]
            go = grad_out if d0 is None else grad_out.narrow(d0, st, n)
            part = _recompute_grads(ctx, sl, need, go)
            for i, g in enumerate(part):
                acc, d = grads[i], dims[i]
                if g is None or acc is None:
                    continue
                if d is None:
                    acc += g
                else:
                    acc.narrow(d, st, g.shape[d]).copy_(g)
        return (None, None, *grads)


def _recompute_grads(
    ctx: Any, tensors: list[torch.Tensor], need: tuple[bool, ...], go: torch.Tensor
) -> tuple[torch.Tensor | None, ...]:
    """Recompute ``fn`` on detached ``tensors`` and return grads for those in ``need``."""
    dev, enabled, ac_dtype = ctx.ac
    with torch.enable_grad(), torch.autocast(dev, dtype=ac_dtype, enabled=enabled):
        inputs = [t.detach().requires_grad_(n) for t, n in zip(tensors, need)]
        out = ctx.fn(*inputs)
        wrt = [t for t, n in zip(inputs, need) if n]
        got = iter(torch.autograd.grad(out, wrt, go, allow_unused=True) if wrt else ())
        return tuple(next(got) if n else None for n in need)


def _apply_recompute(
    fn: Callable[..., torch.Tensor], dims: tuple[int | None, ...], *tensors: torch.Tensor
) -> torch.Tensor:
    """Memory-lean ``fn(*tensors)``; plain math under torch.compile (inductor already recomputes)."""
    if torch.compiler.is_compiling():
        return fn(*tensors)
    out: torch.Tensor = _RecomputeFn.apply(fn, dims, *tensors)
    return out


def _pade_fixed_math(
    x: torch.Tensor, a0: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, b1: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    x, a0, a1, a2, b1 = (
        _soft_clamp(t.float(), _FIXED_MODE_INPUT_CLAMP) for t in (x, a0, a1, a2, b1)
    )
    inner = torch.addcmul(a1, x, a2)
    numerator = torch.addcmul(a0, x, inner)
    denominator = 1.0 + (b1 * x).abs()
    return (numerator / denominator).to(dtype)


class PadeActivationModule(nn.Module):
    """
    Core low-order (m=2, n=1) Padé computation: P(x)/Q(x).

    P(x) = a0 + a1*x + a2*x^2          (numerator, degree 2)
    Q(x) = 1 + |b1*x|                  (denominator, degree 1, "safe" form)

    Q(x) >= 1 for every real x and every real b1 -- no domain restriction is
    required on x, a0, a1, a2, or b1. Computed in fp32; result cast back to
    the dtype of ``x``.
    """

    def forward(
        self,
        x: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        b1: torch.Tensor,
    ) -> torch.Tensor:
        ts = (x, a0, a1, a2, b1)
        same = all(t.shape == x.shape for t in ts) and x.dim() >= 2
        return _apply_recompute(_pade_fixed_math, (1 if same else None,) * 5, *ts)


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


class TrainablePade(nn.Module):
    """
    Padé activation with learnable rational-function coefficients.

    P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5   (degree 5)
    Q(x) = 1 + smooth|b1*x + b2*x^2 + b3*x^3 + b4*x^4|      (degree 4, safe)

    smooth|q| = sqrt(q^2 + eps) - sqrt(eps), so Q >= 1 and Q is differentiable
    (and trainable) at q == 0. Matches the literature-standard "safe PAU"
    order (Molina et al., 2019). Defaults (a0=0, a1=1, b1=1e-3, rest 0)
    initialize to a near-identity function.

    Constructor arguments are *effective* coefficients (x basis). Stored
    parameters (same names ``a0..a5``, ``b1..b4``) are raw: ``r_k = c_k * K^k``
    with K = ``_INPUT_CLAMP``, and the polynomials are evaluated in u = x/K.
    Use ``effective_coefficients()`` to read the x-basis values.

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
        b1=_DEFAULT_B1,
        b2=0.0,
        b3=0.0,
        b4=0.0,
        channel_only=False,
    ):
        super().__init__()
        self.channel_only = channel_only
        # Persistent format marker: present => raw (K^k-scaled) coefficients.
        # Lives in the state dict itself (safetensors drops _metadata).
        self.register_buffer(_BASIS_KEY, torch.tensor(_BASIS_VERSION, dtype=torch.long))
        values = dict(zip(_COEFF_NAMES, (a0, a1, a2, a3, a4, a5, b1, b2, b3, b4)))
        for name, value in values.items():
            setattr(self, name, nn.Parameter(torch.ones(shape) * value * _scale(name)))

    def effective_coefficients(self) -> dict[str, torch.Tensor]:
        """Coefficients in the x basis (detached): raw / K^k."""
        return {n: getattr(self, n).detach() / _scale(n) for n in _COEFF_NAMES}

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if prefix + _BASIS_KEY not in state_dict:  # legacy: effective -> raw, once
            for name in _COEFF_NAMES:
                if prefix + name in state_dict:
                    state_dict[prefix + name] = state_dict[prefix + name] * _scale(name)
            state_dict[prefix + _BASIS_KEY] = self._buffers[_BASIS_KEY]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _broadcast(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.channel_only and x.dim() == 4:
            return p.view(1, -1, 1, 1)
        return p.expand_as(x)

    def _math(self, x: torch.Tensor, *coeffs: torch.Tensor) -> torch.Tensor:
        u = _soft_clamp(x, _INPUT_CLAMP) / _INPUT_CLAMP
        a0, a1, a2, a3, a4, a5, b1, b2, b3, b4 = (self._broadcast(c, u) for c in coeffs)

        # Horner's method in u = x/K.
        numerator = torch.addcmul(a4, u, a5)
        numerator = torch.addcmul(a3, u, numerator)
        numerator = torch.addcmul(a2, u, numerator)
        numerator = torch.addcmul(a1, u, numerator)
        numerator = torch.addcmul(a0, u, numerator)

        q = torch.addcmul(b3, u, b4)
        q = torch.addcmul(b2, u, q)
        q = torch.addcmul(b1, u, q)
        q = u * q
        sqrt_eps = _SMOOTH_ABS_EPS**0.5
        denominator = 1.0 + torch.sqrt(q * q + _SMOOTH_ABS_EPS) - sqrt_eps

        out: torch.Tensor = numerator / denominator
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = [getattr(self, n) for n in _COEFF_NAMES]
        chunked = x.dim() == 4 and all(
            c.dim() in (1, 3) and c.shape[0] == x.shape[1] for c in coeffs
        )
        dims = (1 if chunked else None,) + (0 if chunked else None,) * len(coeffs)
        return _apply_recompute(self._math, dims, x, *coeffs)


def _scale(name: str) -> float:
    """K^k for coefficient ``a<k>`` / ``b<k>`` (raw = effective * scale)."""
    return _INPUT_CLAMP ** int(name[1:])


class WideTrainablePade(TrainablePade):
    """
    TrainablePade with a wide-range default for unbounded logvar use.

    Default coefficients (a0=-8, a1=1, b1=1e-3, rest 0) initialize to a shifted
    identity: P(x)/Q(x) = x - 8, (up to the small b1) mirroring WideTrainableBezier's very
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
        b1=_DEFAULT_B1,
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


# Upper bound on compiled variants per Padé forward (see compile_pade_activations).
# Measured: 28 graphs in a latent-cycle VAE run (limit was 64), plus more for
# flow's non-channel_only shapes; 256 leaves ~4x headroom over the old limit.
_RECOMPILE_LIMIT = 256

# Original (eager) forwards, saved by compile_pade_activations() so the patch
# is idempotent and reversible.
_EAGER_FORWARDS: dict[type, Callable[..., torch.Tensor]] = {}


def compile_pade_activations(enabled: bool = True) -> None:
    """Fuse the Padé elementwise math into single kernels via ``torch.compile``.

    Each Padé forward is a chain of ~10 clamp/addcmul/mul/div/abs ops; in eager
    mode every op is a separate kernel with its own read+write of the full
    activation tensor (and matching backward kernels). Compiling only this
    math (not the surrounding model) fuses it into one forward and one
    backward kernel. ``dynamic=True`` keeps one compiled graph across
    resolutions instead of recompiling per image shape.

    Patches the class-level forwards of ``PadeActivationModule`` and
    ``TrainablePade`` (``WideTrainablePade`` inherits it), so it affects all
    existing and future instances in the process. Compiled kernels compute
    intermediates in fp32, so bf16/fp16 outputs differ from eager by rounding.

    Each forward is specialized on a small, bounded set of variants -- input
    rank (2D/3D/4D), autocast on/off, grad mode on/off, parameter shape (static
    per TrainablePade instance) -- which exceeds Dynamo's default recompile
    limit of 8 in a typical VAE training step; the limit is raised to
    ``_RECOMPILE_LIMIT`` so those variants stay compiled instead of silently
    falling back to eager.

    ``requires_grad`` of the parameters is also part of the variant key
    (verified: freezing/unfreezing the expander, as the latent cycle does,
    recompiles once per state; flipping back reuses the cached graph). Dynamo
    guards on ``requires_grad`` of every tensor input, so compiling a pure
    function that takes the coefficient tensors as arguments would not remove
    the guard -- it is unavoidable, hence the generous limit instead.

    Args:
        enabled: True to compile, False to restore the eager forwards.
    """
    if enabled:
        import torch._dynamo.config as dynamo_config

        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, _RECOMPILE_LIMIT)
    for cls in (PadeActivationModule, TrainablePade):
        if enabled and cls not in _EAGER_FORWARDS:
            _EAGER_FORWARDS[cls] = cls.forward
            cls.forward = torch.compile(cls.forward, dynamic=True)  # type: ignore[method-assign]
        elif not enabled and cls in _EAGER_FORWARDS:
            cls.forward = _EAGER_FORWARDS.pop(cls)  # type: ignore[method-assign]
