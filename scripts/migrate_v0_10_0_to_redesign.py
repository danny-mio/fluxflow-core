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
        # Remaining cases handled in later sub-tasks; for the skeleton just drop.
        report["dropped"].append(k)
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
