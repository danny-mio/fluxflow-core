"""
Migrate a v0.10.0-pre checkpoint to a v0.10.0-bezier-coupled warm start.

Usage:
    python scripts/migrate_v0_10_0_to_redesign.py --src <old.safetensors> \
        --dst <new.safetensors>

Emits a one-page console report covering: direct-copy keys, rescaled keys
(logvar activation), partial-fill warm starts (SPADE), padded keys (pillar
layers), duplicated keys (FiLM, norm2), and dropped legacy keys (time_embed,
pillar_cross_attn, norm_pillar).
"""

import argparse
import re
from pathlib import Path
from typing import Any

import safetensors.torch as st
import torch

# Keys that survive shape-compatible direct copy.
DIRECT_COPY_PREFIXES = (
    "diffuser.compressor.encoder_first_step.",
    "diffuser.compressor.encoder_z.",
    "diffuser.compressor.latent_proj.",
    "diffuser.compressor.mu_proj.",
    "diffuser.compressor.logvar_proj.",
    "diffuser.compressor.mu_activation.",
    "diffuser.compressor.final_norm.",
    "diffuser.compressor.ctx_encoder_first_step.",
    "diffuser.compressor.ctx_encoder_z.",
    "diffuser.compressor.ctx_proj.",
    "diffuser.compressor.ctx_token_attn.",
    "diffuser.compressor.ctx_final_norm.",
    "diffuser.expander.upscale.layers.",  # conv stack inside ResidualUpsampleBlock
    "diffuser.expander.to_rgb_conv.",
    "diffuser.expander.rgb_activation.",
    "diffuser.expander.seam_smoother.",
    "diffuser.expander.seam_smoother_ctx.",
    "diffuser.flow_processor.vae_to_dmodel.",
    "diffuser.flow_processor.dmodel_to_vae.",
    "diffuser.flow_processor.text_proj.",
    "diffuser.flow_processor.text_cond_proj.",
    "diffuser.flow_processor.ctx_mixer.",
    "diffuser.flow_processor.context_injection.",
    "diffuser.flow_processor.norm_ctx.",
    "diffuser.flow_processor.flow_predictor.",
    "diffuser.flow_processor.context_final.",
)

# Legacy keys that map to nothing in the redesign.
DROP_SUBSTRINGS = (
    "time_embed.",
    "pillar_cross_attn.",
    "norm_pillar.",
)


# Logvar activation rescale: the four control points moved from [-1, 1] to
# WideTrainableBezier's [-8, 4] range.
LOGVAR_KEYS = (
    "diffuser.compressor.logvar_activation.p0",
    "diffuser.compressor.logvar_activation.p1",
    "diffuser.compressor.logvar_activation.p2",
    "diffuser.compressor.logvar_activation.p3",
)

_OLD_LOGVAR_RANGE = (-1.0, 1.0)
_NEW_LOGVAR_RANGE = (-8.0, 4.0)


def _rescale_logvar(t: torch.Tensor) -> torch.Tensor:
    """Linearly map old [-1, 1] tensor values to new [-8, 4]."""
    old_min, old_max = _OLD_LOGVAR_RANGE
    new_min, new_max = _NEW_LOGVAR_RANGE
    return (t - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


# ---------------------------------------------------------------------------
# SPADE partial-fill (M8.3)
# ---------------------------------------------------------------------------

_SPADE_LAYER_RE = re.compile(r"diffuser\.expander\.upscale\.layers\.(\d+)\.spade\.")


def _spade_partial_fill(key: str, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Map an old single-scale SPADE key to its redesign equivalents.

    Returns a dict of {new_key: new_tensor}; an empty dict means this key
    should be dropped (e.g. ``mlp_shared.weight`` is replaced wholesale by a
    new deeper MLP).
    """
    if ".spade.mlp_beta.weight" in key or ".spade.mlp_beta.bias" in key:
        new_key = key.replace(".mlp_beta.", ".beta_mid.")
        return {new_key: value}
    if ".spade.beta_scale" in key:
        return {key: value}  # preserved as-is
    if ".spade.mlp_shared." in key:
        # Old 1-layer shared MLP. New shared MLP has different shape (two
        # Conv3x3+Bezier pairs); the old weights don't fit so we drop them.
        return {}
    # Everything else inside .spade.* (unfamiliar) -> drop.
    return {}


def _maybe_zero_init_new_spade_heads(out: dict[str, torch.Tensor]) -> list[str]:
    """For each layer index found in ``out``, add zero-init entries for the new
    multi-scale heads and gamma so the redesign loads cleanly.

    Returns the list of keys added.
    """
    added: list[str] = []
    layer_indices: set[int] = set()
    for k in list(out.keys()):
        m = _SPADE_LAYER_RE.match(k)
        if m:
            layer_indices.add(int(m.group(1)))
    for i in sorted(layer_indices):
        base = f"diffuser.expander.upscale.layers.{i}.spade"
        # We need concrete shapes — derive them from beta_mid which we just
        # wrote (the salvage source provides it).
        bm = out.get(f"{base}.beta_mid.weight")
        if bm is None:
            continue
        C_out, hidden = bm.shape[0], bm.shape[1]
        defaults = {
            f"{base}.beta_low.weight": torch.zeros(C_out, hidden, 1, 1),
            f"{base}.beta_low.bias": torch.zeros(C_out),
            f"{base}.beta_hi.weight": torch.zeros(C_out, hidden, 3, 3),
            f"{base}.beta_hi.bias": torch.zeros(C_out),
            f"{base}.gamma_head.weight": torch.zeros(C_out, hidden, 3, 3),
            f"{base}.gamma_head.bias": torch.zeros(C_out),
            f"{base}.gamma_scale": torch.zeros(1),
        }
        for nk, nv in defaults.items():
            if nk not in out:
                out[nk] = nv
                added.append(nk)
    return added


# ---------------------------------------------------------------------------
# Pillar padding (M8.4)
# ---------------------------------------------------------------------------

# Matches keys like
#   diffuser.flow_processor.transformer_blocks.<N>.p<I>.<L>.0.weight|bias
# where I in {0..3} is the pillar index and L in {0,1,2} is the layer index
# inside the pillar's nn.Sequential.
_PILLAR_RE = re.compile(
    r"^diffuser\.flow_processor\.transformer_blocks\.\d+\."
    r"p[0-3]\.(?P<layer>[0-2])\.0\.(?P<kind>weight|bias)$"
)


def _pad_pillar_tensor(layer: int, kind: str, value: torch.Tensor) -> torch.Tensor:
    """Embed an old (D, D) / (D,) pillar tensor into the redesign's widened shape.

    Layer 0: weight (D, D) -> (2D, D), upper half = old, lower half = 0.
             bias (D,)    -> (2D,),   first D = old, rest zero.
    Layer 1: weight (D, D) -> (2D, 2D), upper-left D x D = old, rest zero.
             bias (D,)    -> (2D,),   first D = old, rest zero.
    Layer 2: weight (D, D) -> (D, 2D), left half = old, right half = 0.
             bias (D,)    -> (D,)    unchanged.
    """
    if kind == "weight":
        D = value.shape[0]
        if value.dim() != 2:
            return value
        if layer == 0:
            out = torch.zeros(2 * D, D, dtype=value.dtype)
            out[:D] = value
            return out
        if layer == 1:
            out = torch.zeros(2 * D, 2 * D, dtype=value.dtype)
            out[:D, :D] = value
            return out
        if layer == 2:
            out = torch.zeros(D, 2 * D, dtype=value.dtype)
            out[:, :D] = value
            return out
    elif kind == "bias":
        D = value.shape[0]
        if layer in (0, 1):
            out = torch.zeros(2 * D, dtype=value.dtype)
            out[:D] = value
            return out
        if layer == 2:
            return value
    return value


def _maybe_pad_pillar(key: str, value: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    """Return the padded (key, tensor) if ``key`` matches a pillar layer; else None."""
    m = _PILLAR_RE.match(key)
    if not m:
        return None
    layer = int(m.group("layer"))
    kind = m.group("kind")
    return key, _pad_pillar_tensor(layer, kind, value)


# ---------------------------------------------------------------------------
# FiLM and norm2 duplication (M8.5)
# ---------------------------------------------------------------------------

# Matches keys like
#   diffuser.flow_processor.transformer_blocks.<N>.film_p<I>.weight|bias
# Old form: single FiLM per pillar.
_FILM_RE = re.compile(
    r"^(diffuser\.flow_processor\.transformer_blocks\.\d+\.film_p[0-3])" r"\.(weight|bias)$"
)

# Matches the legacy shared norm2 LayerNorm: diffuser.flow_processor.transformer_blocks.<N>.norm2.weight|bias
_NORM2_RE = re.compile(
    r"^(diffuser\.flow_processor\.transformer_blocks\.\d+\.norm2)\.(weight|bias)$"
)


def _maybe_duplicate_film(key: str, value: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Old single FiLM -> {<base>_text: copy, <base>_time: zeros}."""
    m = _FILM_RE.match(key)
    if not m:
        return None
    base, kind = m.group(1), m.group(2)
    return {
        f"{base}_text.{kind}": value,
        f"{base}_time.{kind}": torch.zeros_like(value),
    }


def _maybe_duplicate_norm2(key: str, value: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """Old shared norm2 -> {norm2_q: copy, norm2_kv: copy}."""
    m = _NORM2_RE.match(key)
    if not m:
        return None
    base, kind = m.group(1), m.group(2)
    return {
        f"{base}_q.{kind}": value,
        f"{base}_kv.{kind}": value.clone(),
    }


def _is_dropped(key: str) -> bool:
    return any(sub in key for sub in DROP_SUBSTRINGS)


def _is_direct_copy(key: str) -> bool:
    if any(key.startswith(p) for p in DIRECT_COPY_PREFIXES):
        # Inside expander.upscale.layers.N.* we accept conv1/skip_upsample only;
        # spade.* is partial-fill (handled in the SPADE block below).
        if "upscale.layers." in key and ".spade." in key:
            return False
        return True
    return False


def migrate_checkpoint(src: Path, dst: Path) -> dict[str, Any]:
    """
    Read ``src`` and emit a warm-start checkpoint to ``dst``.

    Returns a report dict with keys: ``direct_copied``, ``rescaled``,
    ``partial_filled``, ``padded``, ``duplicated``, ``dropped``.
    """
    tensors = st.load_file(str(src))
    out: dict[str, torch.Tensor] = {}
    report: dict[str, list[str]] = {
        "direct_copied": [],
        "rescaled": [],
        "partial_filled": [],
        "padded": [],
        "duplicated": [],
        "dropped": [],
    }
    for k, v in tensors.items():
        if _is_dropped(k):
            report["dropped"].append(k)
            continue
        if _is_direct_copy(k):
            out[k] = v
            report["direct_copied"].append(k)
            continue
        if k in LOGVAR_KEYS:
            out[k] = _rescale_logvar(v)
            report["rescaled"].append(k)
            continue
        if ".spade." in k:
            mapped = _spade_partial_fill(k, v)
            if mapped:
                for nk, nv in mapped.items():
                    out[nk] = nv
                    report["partial_filled"].append(nk)
            else:
                report["dropped"].append(k)
            continue
        padded = _maybe_pad_pillar(k, v)
        if padded is not None:
            nk, nv = padded
            out[nk] = nv
            report["padded"].append(nk)
            continue
        film_dup = _maybe_duplicate_film(k, v)
        if film_dup is not None:
            for nk, nv in film_dup.items():
                out[nk] = nv
                report["duplicated"].append(nk)
            continue
        norm2_dup = _maybe_duplicate_norm2(k, v)
        if norm2_dup is not None:
            for nk, nv in norm2_dup.items():
                out[nk] = nv
                report["duplicated"].append(nk)
            continue
        # Remaining cases handled in later sub-tasks; for the skeleton just drop.
        report["dropped"].append(k)

    # Zero-init the new SPADE heads (multi-scale beta + gamma) so the
    # redesigned model loads cleanly with the partial-fill warm start.
    spade_zero_added = _maybe_zero_init_new_spade_heads(out)
    report["partial_filled"].extend(spade_zero_added)

    st.save_file(out, str(dst))
    return report


def _print_report(report: dict[str, list[str]]) -> None:
    print("\n=== v0.10.0-pre -> bezier-coupled migration report ===")
    for category, keys in report.items():
        print(f"  {category}: {len(keys)} tensors")
        for k in keys[:3]:
            print(f"      e.g. {k}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate v0.10.0-pre checkpoint to redesign warm-start")
    p.add_argument("--src", type=Path, required=True, help="Old checkpoint path")
    p.add_argument("--dst", type=Path, required=True, help="Output warm-start path")
    args = p.parse_args()
    report = migrate_checkpoint(args.src, args.dst)
    _print_report(report)


if __name__ == "__main__":
    main()
