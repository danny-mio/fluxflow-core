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


class _MetaRecordingEncoder(torch.nn.Module):
    """Fake encoder whose parameters live on the ``meta`` device.

    Lets us assert that ``build_cfg_null_pair`` moves tokenizer outputs to the
    encoder's device before calling forward — without needing real MPS/CUDA in
    CI.  Tokenizer always returns CPU tensors; if the move is missing, the
    encoder's ``forward`` sees ``device.type == 'cpu'`` (matching pre-fix
    behaviour).  After the fix, it sees ``device.type == 'meta'``.
    """

    def __init__(self):
        super().__init__()
        # ``meta`` device: shape metadata only, no real storage.
        self.dummy = torch.nn.Parameter(torch.empty(1, device="meta"))
        self.recorded_input_device = None
        self.recorded_mask_device = None

    def forward(self, input_ids, attention_mask=None):
        self.recorded_input_device = input_ids.device
        self.recorded_mask_device = attention_mask.device if attention_mask is not None else None
        b, t = input_ids.shape
        out_device = self.dummy.device
        return (
            torch.zeros(b, t, 4, device=out_device),
            torch.ones(b, t, dtype=torch.bool, device=out_device),
        )


def test_build_cfg_null_pair_moves_inputs_to_encoder_device():
    """Tokenizer outputs must be moved onto the encoder's device.

    Regression test for the MPS-specific "Placeholder storage has not been
    allocated on MPS device" error that surfaced at sample-generation time
    when the encoder lives on MPS but ``enc_in`` stayed on CPU.
    """
    enc = _MetaRecordingEncoder()
    build_cfg_null_pair(enc, max_length=8)
    expected = next(enc.parameters()).device
    assert enc.recorded_input_device is not None, "encoder.forward was never called"
    assert enc.recorded_input_device.type == expected.type, (
        f"input_ids must reach the encoder on its parameter device "
        f"(expected {expected.type}, got {enc.recorded_input_device.type})"
    )
    assert enc.recorded_mask_device is not None, "attention_mask was not passed"
    assert enc.recorded_mask_device.type == expected.type, (
        f"attention_mask must reach the encoder on its parameter device "
        f"(expected {expected.type}, got {enc.recorded_mask_device.type})"
    )
