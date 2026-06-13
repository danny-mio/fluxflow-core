"""Tests that pipeline.forward accepts text_seq + text_mask."""

import inspect

from fluxflow.models.pipeline import FluxPipeline


def test_pipeline_forward_signature_has_text_seq_and_mask():
    sig = inspect.signature(FluxPipeline.forward)
    params = list(sig.parameters)
    assert "text_seq" in params
    assert "text_mask" in params
    assert (
        "text_embeddings" not in params
    ), "text_embeddings should be renamed to text_seq for clarity at v0.10.0-redesign"
