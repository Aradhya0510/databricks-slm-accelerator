"""Shared fixtures.

Every fixture is offline: models are constructed from configs in code rather
than downloaded, so the suite runs in CI with no network and no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def alpaca_jsonl(tmp_path: Path) -> Path:
    """A tiny Alpaca-format dataset, one record with an input and one without."""
    rows = [
        {
            "instruction": "Summarise the following text.",
            "input": "Databricks is a data and AI platform.",
            "output": "It is a data and AI platform.",
        },
        {
            "instruction": "Name a primary colour.",
            "input": "",
            "output": "Blue.",
        },
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.fixture
def sharegpt_jsonl(tmp_path: Path) -> Path:
    """A ShareGPT-format dataset with a multi-turn conversation."""
    rows = [
        {
            "conversations": [
                {"from": "human", "value": "Hello?"},
                {"from": "gpt", "value": "Hi there."},
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt", "value": "Four."},
            ]
        }
    ]
    path = tmp_path / "sharegpt.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.fixture
def chat_tokenizer():
    """A small tokenizer with a chat template, built offline.

    Uses a GPT-2 style BPE vocabulary constructed in code so no download is
    needed; the chat template is what the formatting code actually exercises.
    """
    import tempfile

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=200,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
    )
    corpus = [
        "Hello there how are you", "Summarise the following text",
        "Databricks is a data and AI platform", "Name a primary colour Blue",
        "What is 2+2 Four Hi",
    ]
    tok.train_from_iterator(corpus, trainer)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tok.save(f.name)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f.name,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
        )

    # A minimal but realistic chat template.
    tokenizer.chat_template = (
        "{% for m in messages %}"
        "<|{{ m['role'] }}|>\n{{ m['content'] }}</s>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    return tokenizer
