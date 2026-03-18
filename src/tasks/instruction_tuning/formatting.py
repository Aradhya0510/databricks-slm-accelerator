"""Data loading and formatting functions for instruction tuning.

Converts structured records (Alpaca, ShareGPT) into prompted text that the
tokenizer can process.  HuggingFace ``datasets`` handles the actual I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from ...config.schema import DataConfig


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_from_config(data_cfg: DataConfig, split: str = "train") -> Dataset:
    """Load a HuggingFace Dataset from the configured path and format."""
    path = data_cfg.train_data_path if split == "train" else data_cfg.val_data_path
    if path is None:
        raise ValueError(f"No data path configured for split '{split}'")

    p = Path(path)

    if p.suffix == ".jsonl" or p.suffix == ".json":
        return load_dataset("json", data_files=str(p), split="train")
    elif p.suffix == ".csv":
        return load_dataset("csv", data_files=str(p), split="train")
    elif p.suffix in (".parquet", ".pq"):
        return load_dataset("parquet", data_files=str(p), split="train")
    elif p.is_dir():
        json_files = list(p.glob("*.jsonl")) + list(p.glob("*.json"))
        csv_files = list(p.glob("*.csv"))
        parquet_files = list(p.glob("*.parquet")) + list(p.glob("*.pq"))

        if json_files:
            return load_dataset("json", data_files=[str(f) for f in json_files], split="train")
        elif csv_files:
            return load_dataset("csv", data_files=[str(f) for f in csv_files], split="train")
        elif parquet_files:
            return load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")

    try:
        return load_dataset(str(p), split="train")
    except Exception:
        pass

    raise ValueError(
        f"Cannot load dataset from '{path}'. "
        "Supported: .jsonl, .json, .csv, .parquet files, or directories containing them."
    )


# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------

def _format_alpaca(
    examples: Dict[str, List],
    data_cfg: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> List[str]:
    """Convert Alpaca-format records into chat-templated strings."""
    instructions = examples[data_cfg.instruction_column]
    inputs = examples.get(data_cfg.input_column, [None] * len(instructions))
    outputs = examples[data_cfg.output_column]

    texts = []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        messages = []
        if data_cfg.system_prompt:
            messages.append({"role": "system", "content": data_cfg.system_prompt})

        user_content = instruction
        if inp:
            user_content = f"{instruction}\n\n{inp}"
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            text = _fallback_chat_format(messages)

        texts.append(text)

    return texts


def _format_sharegpt(
    examples: Dict[str, List],
    data_cfg: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> List[str]:
    """Convert ShareGPT-format conversations into chat-templated strings."""
    conversations_list = examples[data_cfg.conversations_column]
    texts = []

    role_map = {"human": "user", "gpt": "assistant", "system": "system",
                "user": "user", "assistant": "assistant"}

    for conversations in conversations_list:
        if isinstance(conversations, str):
            conversations = json.loads(conversations)

        messages = []
        if data_cfg.system_prompt and not any(
            c.get("from", c.get("role", "")) == "system" for c in conversations
        ):
            messages.append({"role": "system", "content": data_cfg.system_prompt})

        for turn in conversations:
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))
            messages.append({"role": role_map.get(role, role), "content": content})

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            text = _fallback_chat_format(messages)

        texts.append(text)

    return texts


def _fallback_chat_format(messages: List[Dict[str, str]]) -> str:
    """Simple fallback when the tokenizer lacks a chat template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"### System:\n{content}\n")
        elif role == "user":
            parts.append(f"### User:\n{content}\n")
        elif role == "assistant":
            parts.append(f"### Assistant:\n{content}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_formatting_fn(
    data_cfg: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Optional[Callable]:
    """Return a formatting function matching the configured data format.

    Returns ``None`` if no formatting is needed (e.g., pre-formatted text field).
    """
    if data_cfg.data_format == "alpaca":
        def fn(examples):
            return _format_alpaca(examples, data_cfg, tokenizer)
        return fn

    if data_cfg.data_format == "sharegpt":
        def fn(examples):
            return _format_sharegpt(examples, data_cfg, tokenizer)
        return fn

    return None
