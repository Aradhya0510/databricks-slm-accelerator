"""Environment helpers: rank identity, precision, attention."""

from __future__ import annotations

import pytest

from src.utils.environment import (
    is_distributed_launch,
    is_rank_zero,
    local_rank,
    resolve_precision,
    world_size,
)


def test_rank_zero_uses_global_rank(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    assert is_rank_zero() is True

    # Second node, first local process: LOCAL_RANK is 0 but RANK is not.
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert is_rank_zero() is False
    assert local_rank() == 0


def test_distributed_launch_detection(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert is_distributed_launch() is False
    assert world_size() == 1

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    assert is_distributed_launch() is True
    assert world_size() == 4


def test_precision_falls_back_to_fp32_without_cuda():
    """bf16 was previously hardcoded on and failed outright on unsupported GPUs."""
    assert resolve_precision("auto") == "fp32"
    assert resolve_precision("bf16") == "fp32"


def test_attention_implementation_never_returns_unavailable_flash():
    """FA2 was forced on whenever CUDA was present, with no fallback."""
    from src.utils.environment import resolve_attn_implementation

    resolved = resolve_attn_implementation("auto")
    assert resolved in {"sdpa", "eager", "flash_attention_2"}
    # No CUDA here, so it must not claim flash attention.
    assert resolved == "sdpa"


def test_explicit_attention_choice_is_respected():
    from src.utils.environment import resolve_attn_implementation

    assert resolve_attn_implementation("eager") == "eager"
