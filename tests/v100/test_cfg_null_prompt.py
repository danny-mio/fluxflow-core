"""Tests for encoded-empty-prompt CFG null (replaces zero-vector null)."""

import torch

from fluxflow.models.encoders import BertTextEncoder
from fluxflow.utils.visualization import build_cfg_null_pair


def test_cfg_null_is_encoded_empty_prompt():
    """null_text_seq is the encoded empty string, not a zero tensor."""
    enc = BertTextEncoder(embed_dim=1024)
    null_seq, null_mask = build_cfg_null_pair(enc, max_length=32)
    assert null_seq.shape[1] == 32  # padded to max_length
    assert null_seq.abs().sum() > 0  # NOT all-zeros
    # Mask should have at least [CLS] and [SEP] as True
    assert null_mask[0].sum() >= 2
    assert torch.isfinite(null_seq).all()
