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
    """Load a HuggingFace Dataset from the configured source.

    A Unity Catalog table takes precedence over a file path.  On Databricks
    training data most naturally lives in a governed table, and requiring an
    export to JSONL first threw away the lineage that makes the platform
    worth using.
    """
    table = data_cfg.train_table if split == "train" else data_cfg.val_table
    if table:
        return load_dataset_from_table(table)

    path = data_cfg.train_data_path if split == "train" else data_cfg.val_data_path
    if path is None:
        raise ValueError(
            f"No data source configured for split '{split}': set "
            f"{'train' if split == 'train' else 'val'}_data_path or "
            f"{'train' if split == 'train' else 'val'}_table."
        )

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


def load_dataset_from_table(table: str) -> Dataset:
    """Read a Unity Catalog table into a HuggingFace Dataset.

    Requires a Spark session, so this only works on Databricks (or anywhere
    else with Spark configured).
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError(
            f"Reading '{table}' needs PySpark, which is not available here. "
            f"Use a file path instead when running off-cluster."
        ) from exc

    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    pdf = spark.read.table(table).toPandas()

    if pdf.empty:
        raise ValueError(f"Table '{table}' is empty.")

    return Dataset.from_pandas(pdf, preserve_index=False)


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


# ---------------------------------------------------------------------------
# Prompt/completion splitting (for completion-only loss)
# ---------------------------------------------------------------------------

def _split_alpaca(
    examples: Dict[str, List],
    data_cfg: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[str]]:
    """Render Alpaca records as separate prompt and completion strings.

    Keeping the two apart lets TRL mask the prompt so the loss covers only the
    assistant's response.  Training on the prompt as well teaches the model to
    reproduce its own template and wastes capacity that small models do not
    have to spare.
    """
    instructions = examples[data_cfg.instruction_column]
    inputs = examples.get(data_cfg.input_column, [None] * len(instructions))
    outputs = examples[data_cfg.output_column]

    prompts, completions = [], []
    for instruction, inp, output in zip(instructions, inputs, outputs):
        messages = []
        if data_cfg.system_prompt:
            messages.append({"role": "system", "content": data_cfg.system_prompt})

        user_content = instruction
        if inp:
            user_content = f"{instruction}\n\n{inp}"
        messages.append({"role": "user", "content": user_content})

        prompts.append(_render_prompt(messages, tokenizer))
        completions.append(output)

    return {"prompt": prompts, "completion": completions}


def _split_sharegpt(
    examples: Dict[str, List],
    data_cfg: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, List[str]]:
    """Render ShareGPT conversations as prompt (everything up to the final
    assistant turn) and completion (that turn)."""
    conversations_list = examples[data_cfg.conversations_column]

    prompts, completions = [], []
    for conversations in conversations_list:
        if isinstance(conversations, str):
            conversations = json.loads(conversations)

        messages = _to_messages(conversations, data_cfg)

        # Split at the last assistant turn.
        last_assistant = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_assistant = i
                break

        if last_assistant is None:
            # No assistant turn to learn from; emit an empty completion so the
            # row can be filtered rather than silently training on the prompt.
            prompts.append(_render_prompt(messages, tokenizer))
            completions.append("")
            continue

        prompts.append(_render_prompt(messages[:last_assistant], tokenizer))
        completions.append(messages[last_assistant]["content"])

    return {"prompt": prompts, "completion": completions}


def _to_messages(conversations, data_cfg: DataConfig) -> List[Dict[str, str]]:
    """Normalise ShareGPT turns into role/content messages."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system",
                "user": "user", "assistant": "assistant"}

    messages = []
    if data_cfg.system_prompt and not any(
        c.get("from", c.get("role", "")) == "system" for c in conversations
    ):
        messages.append({"role": "system", "content": data_cfg.system_prompt})

    for turn in conversations:
        role = turn.get("from", turn.get("role", "user"))
        content = turn.get("value", turn.get("content", ""))
        messages.append({"role": role_map.get(role, role), "content": content})

    return messages


def _render_prompt(
    messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizer,
) -> str:
    """Render messages as a prompt, ending where the model should start writing."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return _fallback_chat_format(messages) + "\n### Assistant:\n"


def build_split_fn(
    data_cfg: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Optional[Callable]:
    """Return a prompt/completion splitter for the configured data format.

    ``None`` when the format has no notion of a prompt boundary, in which case
    the caller falls back to whole-text training.
    """
    if data_cfg.data_format == "alpaca":
        return lambda examples: _split_alpaca(examples, data_cfg, tokenizer)

    if data_cfg.data_format == "sharegpt":
        return lambda examples: _split_sharegpt(examples, data_cfg, tokenizer)

    return None
