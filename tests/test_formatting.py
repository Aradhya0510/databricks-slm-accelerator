"""Data formatting: whole-text rendering and prompt/completion splitting.

The split path is what enables completion-only loss.  Training on the prompt
as well teaches a small model to reproduce its own template, so these check
that the boundary lands where the assistant starts writing.
"""

from __future__ import annotations

import pytest

from src.config.schema import DataConfig
from src.tasks.instruction_tuning.formatting import (
    build_formatting_fn,
    build_split_fn,
    load_dataset_from_config,
)


def _alpaca_cfg(path, **kw):
    return DataConfig(train_data_path=str(path), data_format="alpaca", **kw)


def _sharegpt_cfg(path, **kw):
    return DataConfig(train_data_path=str(path), data_format="sharegpt", **kw)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_loads_jsonl(alpaca_jsonl):
    ds = load_dataset_from_config(_alpaca_cfg(alpaca_jsonl))
    assert len(ds) == 2
    assert set(ds.column_names) >= {"instruction", "input", "output"}


def test_missing_source_is_rejected_at_config_time():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="No training data configured"):
        DataConfig()


def test_a_unity_catalog_table_satisfies_the_source_requirement():
    cfg = DataConfig(train_table="main.slm.sft_train")
    assert cfg.train_table == "main.slm.sft_train"
    assert cfg.train_data_path is None


# ---------------------------------------------------------------------------
# Whole-text formatting
# ---------------------------------------------------------------------------

def test_alpaca_formatting_renders_the_full_exchange(alpaca_jsonl, chat_tokenizer):
    cfg = _alpaca_cfg(alpaca_jsonl)
    fn = build_formatting_fn(cfg, chat_tokenizer)
    ds = load_dataset_from_config(cfg)

    texts = fn(ds[:])
    assert len(texts) == 2
    # Both the question and the answer are present in the training text.
    assert "primary colour" in texts[1]
    assert "Blue." in texts[1]


def test_alpaca_input_column_is_appended_when_present(alpaca_jsonl, chat_tokenizer):
    cfg = _alpaca_cfg(alpaca_jsonl)
    texts = build_formatting_fn(cfg, chat_tokenizer)(load_dataset_from_config(cfg)[:])
    assert "Databricks is a data and AI platform." in texts[0]


# ---------------------------------------------------------------------------
# Prompt/completion splitting
# ---------------------------------------------------------------------------

def test_alpaca_split_puts_the_answer_only_in_the_completion(alpaca_jsonl, chat_tokenizer):
    cfg = _alpaca_cfg(alpaca_jsonl)
    ds = load_dataset_from_config(cfg)

    out = build_split_fn(cfg, chat_tokenizer)(ds[:])

    assert set(out) == {"prompt", "completion"}
    assert out["completion"][1] == "Blue."
    # The answer must NOT appear in the prompt, or masking buys nothing.
    assert "Blue." not in out["prompt"][1]
    # The question must be in the prompt.
    assert "primary" in out["prompt"][1]


def test_prompt_ends_with_a_generation_marker(alpaca_jsonl, chat_tokenizer):
    """The prompt has to stop where the model should start writing."""
    cfg = _alpaca_cfg(alpaca_jsonl)
    out = build_split_fn(cfg, chat_tokenizer)(load_dataset_from_config(cfg)[:])
    assert out["prompt"][0].rstrip().endswith("<|assistant|>")


def test_system_prompt_lands_in_the_prompt_not_the_completion(alpaca_jsonl, chat_tokenizer):
    cfg = _alpaca_cfg(alpaca_jsonl, system_prompt="You are terse.")
    out = build_split_fn(cfg, chat_tokenizer)(load_dataset_from_config(cfg)[:])

    assert "You are terse." in out["prompt"][0]
    assert "You are terse." not in out["completion"][0]


def test_sharegpt_split_uses_the_final_assistant_turn(sharegpt_jsonl, chat_tokenizer):
    cfg = _sharegpt_cfg(sharegpt_jsonl)
    ds = load_dataset_from_config(cfg)

    out = build_split_fn(cfg, chat_tokenizer)(ds[:])

    assert out["completion"][0] == "Four."
    # Earlier turns are context and belong in the prompt.
    assert "Hi there." in out["prompt"][0]
    assert "2+2" in out["prompt"][0]
    # The final answer must not leak into the prompt.
    assert "Four." not in out["prompt"][0]


def test_split_returns_none_for_formats_without_a_prompt_boundary(alpaca_jsonl):
    cfg = DataConfig(train_data_path=str(alpaca_jsonl), data_format="csv")
    assert build_split_fn(cfg, None) is None


def test_fallback_format_is_used_without_a_chat_template(alpaca_jsonl, chat_tokenizer):
    chat_tokenizer.chat_template = None
    cfg = _alpaca_cfg(alpaca_jsonl)
    out = build_split_fn(cfg, chat_tokenizer)(load_dataset_from_config(cfg)[:])
    assert out["prompt"][0].rstrip().endswith("### Assistant:")
