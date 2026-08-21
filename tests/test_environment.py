"""Environment helpers: staging paths, rank identity, precision, attention."""

from __future__ import annotations

import pytest

from src.utils.environment import (
    is_distributed_launch,
    is_rank_zero,
    local_rank,
    resolve_precision,
    stage_data_to_local,
    volumes_staging_path,
    world_size,
)


@pytest.mark.parametrize(
    "volumes_path, expected",
    [
        ("/Volumes/main/slm/data/", "/tmp/staged/main/slm/data"),
        ("/Volumes/users/east/sft", "/tmp/staged/users/east/sft"),
        ("/Volumes/mount/x", "/tmp/staged/mount/x"),
        ("/Volumes/sales/y", "/tmp/staged/sales/y"),
    ],
)
def test_staging_path_preserves_every_segment(volumes_path, expected):
    assert volumes_staging_path(volumes_path, "/tmp/staged") == expected


def test_non_volumes_paths_pass_through():
    assert stage_data_to_local("/dbfs/foo") == "/dbfs/foo"
    assert stage_data_to_local("relative/path") == "relative/path"


def test_stage_copies_a_file(tmp_path, monkeypatch):
    source = tmp_path / "Volumes" / "main" / "train.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text('{"a": 1}')

    monkeypatch.setattr(
        "src.utils.environment._VOLUMES_ROOT", str(tmp_path / "Volumes")
    )
    out = stage_data_to_local("/Volumes/main/train.jsonl", str(tmp_path / "staged"))

    assert out == str(tmp_path / "staged" / "main" / "train.jsonl")
    assert (tmp_path / "staged" / "main" / "train.jsonl").read_text() == '{"a": 1}'


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
