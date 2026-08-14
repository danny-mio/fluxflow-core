"""Shared text-token length default for FluxFlow.

Single source of truth for the text-encoder token budget, consumed by both
training (caption tokenization in ``fluxflow_training.data.datasets``) and
generation (prompt encoding in ``diffusion_pipeline.py``, ``visualization.py``,
``fluxflow-ui``'s ``generation_worker.py``, and ``fluxflow-comfyui``'s
``text_encode.py``). Import this constant instead of hardcoding a length so
the two sides cannot silently diverge again -- previously training tokenized
captions to 32 tokens while generation independently hardcoded 512 at every
call site, with truncation silent on both sides.

Raising this value is a real behavior change, not a config tweak: the flow
transformer has never been trained on text sequences longer than whatever
training actually used. Do not raise it without first checking the
truncation-rate / length-distribution logging emitted by the training
dataset classes, and retraining accordingly.
"""

DEFAULT_MAX_TEXT_LENGTH: int = 32
