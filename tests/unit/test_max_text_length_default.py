"""Tests for the shared DEFAULT_MAX_TEXT_LENGTH constant (v0.10.0).

Generation previously hardcoded max_length=512 at every prompt-encoding call
site while training tokenized captions to a default of 32 -- a silent
mismatch. Both sides now import the same constant so they cannot diverge
again.
"""

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from fluxflow.models.diffusion_pipeline import FluxFlowPipeline
from fluxflow.text_length import DEFAULT_MAX_TEXT_LENGTH
from fluxflow.utils.visualization import build_cfg_null_pair, save_sample_images


def test_default_is_32():
    assert DEFAULT_MAX_TEXT_LENGTH == 32


def test_exported_from_utils():
    from fluxflow.utils import DEFAULT_MAX_TEXT_LENGTH as via_utils

    assert via_utils is DEFAULT_MAX_TEXT_LENGTH


def test_exported_from_top_level_package():
    import fluxflow

    assert fluxflow.DEFAULT_MAX_TEXT_LENGTH is DEFAULT_MAX_TEXT_LENGTH


def test_build_cfg_null_pair_default_matches_constant():
    sig = inspect.signature(build_cfg_null_pair)
    assert sig.parameters["max_length"].default == DEFAULT_MAX_TEXT_LENGTH


class TestEncodePromptUsesSharedDefault:
    """FluxFlowPipeline.encode_prompt (diffusion_pipeline.py) call sites."""

    def _make_pipeline(self, seq_len: int):
        pipeline = object.__new__(FluxFlowPipeline)
        tok_out = MagicMock()
        tok_out.input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        tokenizer = MagicMock(return_value=tok_out)
        tokenizer.pad_token_id = 0
        pipeline.tokenizer = tokenizer
        pipeline.text_encoder = MagicMock(
            return_value=(
                torch.zeros(1, seq_len, 4),
                torch.ones(1, seq_len, dtype=torch.bool),
            )
        )
        return pipeline, tokenizer

    def test_conditional_branch_uses_shared_default(self):
        pipeline, tokenizer = self._make_pipeline(DEFAULT_MAX_TEXT_LENGTH)
        pipeline.encode_prompt("a prompt", device=torch.device("cpu"))
        _, kwargs = tokenizer.call_args
        assert kwargs["max_length"] == DEFAULT_MAX_TEXT_LENGTH

    def test_uncond_branch_uses_shared_default(self):
        pipeline, tokenizer = self._make_pipeline(DEFAULT_MAX_TEXT_LENGTH)
        pipeline.encode_prompt(
            "a prompt", device=torch.device("cpu"), do_classifier_free_guidance=True
        )
        calls = tokenizer.call_args_list
        assert len(calls) == 2
        for _, kwargs in calls:
            assert kwargs["max_length"] == DEFAULT_MAX_TEXT_LENGTH


class TestSaveSampleImagesUsesSharedDefault:
    def test_batch_encode_plus_uses_shared_default(self):
        tokenizer = MagicMock()
        tokenizer.batch_encode_plus.side_effect = RuntimeError("stop-here")
        with pytest.raises(RuntimeError, match="stop-here"):
            save_sample_images(
                diffuser=MagicMock(),
                text_encoder=MagicMock(),
                tokenizer=tokenizer,
                output_path="/tmp/x",
                epoch=0,
                device=torch.device("cpu"),
                sample_captions=["a cat"],
            )
        _, kwargs = tokenizer.batch_encode_plus.call_args
        assert kwargs["max_length"] == DEFAULT_MAX_TEXT_LENGTH
