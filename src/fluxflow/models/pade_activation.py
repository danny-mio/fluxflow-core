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

# TrainablePade's numerator is degree 5 and, unlike the "safe" denominator,
# has no built-in range limiting. Clamp the raw input before evaluation so
# x^5 cannot overflow even with a coefficient magnitude approaching 1 (the
# observed near-identity-init scale, e.g. a1~=1.0): 65504 is fp16's max
# representable value, and 65504**(1/5) is the largest |x| for which
# |x|**5 <= 65504 -- i.e. the worst-case unit-coefficient term stays within
# fp16 range. This preserves far more dynamic range than Bezier's [0, 1]
# input squash while eliminating the confirmed overflow.
_INPUT_CLAMP = 65504.0**0.2

# PadeActivationModule's numerator has the same unbounded-input overflow class
# as TrainablePade's, but here x, a0, a1, a2, and b1 are all unbound directly
# from the input tensor's own channels (not learned, near-identity-init
# parameters), so none of the five factors carry a smallness guarantee. The
# dominant overflow term is a2*x^2, where *both* factors are unbounded: with
# every input clamped to +-K, the worst case is K * K^2 = K^3 (degree-2
# numerator, but cubic rather than TrainablePade's quintic bound since only
# one coefficient multiplies the squared term instead of a chain of five).
# Solving K^3 <= 65504 (fp16 max) gives K = 65504**(1/3).
_FIXED_MODE_INPUT_CLAMP = 65504.0 ** (1 / 3)


class PadeActivationModule(nn.Module):
    """
    Core low-order (m=2, n=1) Padé computation: P(x)/Q(x).

    P(x) = a0 + a1*x + a2*x^2          (numerator, degree 2)
    Q(x) = 1 + |b1*x|                  (denominator, degree 1, "safe" form)

    Q(x) >= 1 for every real x and every real b1 -- no domain restriction is
    required on x, a0, a1, a2, or b1.
    """

    # All five inputs are clamped to +-_FIXED_MODE_INPUT_CLAMP up front (see
    # derivation above) so a2*x^2 cannot overflow fp16 range, mirroring how
    # TrainablePade.forward clamps its input before Horner's method.
    def forward(
        self,
        x: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        b1: torch.Tensor,
    ) -> torch.Tensor:
        x = x.clamp(-_FIXED_MODE_INPUT_CLAMP, _FIXED_MODE_INPUT_CLAMP)
        a0 = a0.clamp(-_FIXED_MODE_INPUT_CLAMP, _FIXED_MODE_INPUT_CLAMP)
        a1 = a1.clamp(-_FIXED_MODE_INPUT_CLAMP, _FIXED_MODE_INPUT_CLAMP)
        a2 = a2.clamp(-_FIXED_MODE_INPUT_CLAMP, _FIXED_MODE_INPUT_CLAMP)
        b1 = b1.clamp(-_FIXED_MODE_INPUT_CLAMP, _FIXED_MODE_INPUT_CLAMP)

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
        x = x.clamp(-_INPUT_CLAMP, _INPUT_CLAMP)
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
