"""Tests for ROCm SDPA backend probe."""

import pytest
import torch
from fluxflow.utils import rocm_compat


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(rocm_compat, "_RESULTS", {})


@pytest.fixture
def fake_backends(monkeypatch):
    state = {"EFFICIENT_ATTENTION": True, "FLASH_ATTENTION": True}
    backends = tuple(
        (name, lambda n=name: state[n], lambda v, n=name: state.__setitem__(n, v)) for name in state
    )
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(rocm_compat, "_FUSED_SDPA_BACKENDS", backends)
    return state


def test_noop_on_cpu():
    assert rocm_compat.disable_broken_rocm_sdpa(torch.device("cpu")) == []


def test_noop_without_hip(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda")) == []


def test_disables_only_failing_backends(monkeypatch, fake_backends):
    monkeypatch.setattr(
        rocm_compat, "_probe_sdpa_backend", lambda name, device: name == "FLASH_ATTENTION"
    )
    assert rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda")) == ["EFFICIENT_ATTENTION"]
    assert fake_backends == {"EFFICIENT_ATTENTION": False, "FLASH_ATTENTION": True}


def test_skips_already_disabled_backends(monkeypatch):
    backends = (("EFFICIENT_ATTENTION", lambda: False, lambda v: None),)
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(rocm_compat, "_FUSED_SDPA_BACKENDS", backends)
    monkeypatch.setattr(rocm_compat, "_probe_sdpa_backend", lambda *a: False)
    assert rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda")) == []


def test_repeat_call_cached_and_returns_same(monkeypatch, fake_backends):
    calls = []

    def probe(name, device):
        calls.append(name)
        return False

    monkeypatch.setattr(rocm_compat, "_probe_sdpa_backend", probe)
    dev = torch.device("cuda")
    first = rocm_compat.disable_broken_rocm_sdpa(dev)
    second = rocm_compat.disable_broken_rocm_sdpa(dev)
    assert first == second == ["EFFICIENT_ATTENTION", "FLASH_ATTENTION"]
    assert len(calls) == 2


def test_cache_is_per_device(monkeypatch, fake_backends):
    calls = []
    monkeypatch.setattr(
        rocm_compat, "_probe_sdpa_backend", lambda n, d: calls.append((n, d)) or True
    )
    rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda", 0))
    rocm_compat.disable_broken_rocm_sdpa(torch.device("cuda", 1))
    assert len(calls) == 4


def test_result_copy_not_aliased(monkeypatch, fake_backends):
    monkeypatch.setattr(rocm_compat, "_probe_sdpa_backend", lambda *a: False)
    dev = torch.device("cuda")
    rocm_compat.disable_broken_rocm_sdpa(dev).clear()
    assert len(rocm_compat.disable_broken_rocm_sdpa(dev)) == 2
