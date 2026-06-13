"""Tests for per-token BertTextEncoder output."""

import torch

from fluxflow.models.encoders import BertTextEncoder


def _encoder(embed_dim: int = 1024) -> BertTextEncoder:
    return BertTextEncoder(embed_dim=embed_dim, pretrain_model="distilbert-base-uncased")


def test_text_encoder_returns_tuple():
    enc = _encoder()
    input_ids = torch.tensor([[101, 1996, 2829, 3899, 102]])  # [CLS] the brown dog [SEP]
    mask = torch.tensor([[1, 1, 1, 1, 1]])
    out = enc(input_ids, attention_mask=mask)
    assert isinstance(out, tuple) and len(out) == 2


def test_text_encoder_pertoken_shape():
    enc = _encoder()
    input_ids = torch.tensor([[101, 1996, 2829, 3899, 102]])
    mask = torch.tensor([[1, 1, 1, 1, 1]])
    text_seq, text_mask = enc(input_ids, attention_mask=mask)
    assert text_seq.shape == (1, 5, 1024)
    assert text_mask.shape == (1, 5)
    assert text_mask.dtype == torch.bool


def test_text_encoder_mask_reflects_attention_mask():
    enc = _encoder()
    input_ids = torch.tensor([[101, 1996, 2829, 3899, 102, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    _, text_mask = enc(input_ids, attention_mask=mask)
    assert text_mask[0, :5].all()
    assert not text_mask[0, 5:].any()


def test_text_encoder_no_mean_pool_in_source():
    """Source no longer mean-pools last_hidden_state."""
    import inspect

    src = inspect.getsource(BertTextEncoder.forward)
    assert ".mean(dim=1)" not in src
