"""Verify old v0.10.0-pre checkpoints fail with a clear error."""

import pytest
import torch

from fluxflow.models.v100.vae import FluxExpander_v100
from fluxflow.exceptions import IncompatibleCheckpointError


@pytest.mark.compat_break
def test_old_spade_mlp_beta_key_raises():
    """A state_dict containing the old `mlp_beta.weight` key fails clearly."""
    m = FluxExpander_v100(d_model=32, upscales=4, max_hw=256)
    fake_old = {"upscale.layers.0.spade.mlp_beta.weight": torch.zeros(32, 128, 3, 3)}
    with pytest.raises(IncompatibleCheckpointError) as exc:
        m.load_state_dict(fake_old, strict=False, assign=False)
    assert "v0.10.0-bezier-coupled" in str(exc.value)
    assert "migrate_v0.10.0_to_redesign.py" in str(exc.value)
