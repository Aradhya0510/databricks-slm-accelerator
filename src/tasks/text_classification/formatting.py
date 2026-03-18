"""Data loading and tokenization for text classification datasets."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from ...config.schema import DataConfig


def load_classification_dataset(data_cfg: DataConfig, split: str = "train") -> Dataset:
    """Load a text classification dataset (CSV, JSON, or directory)."""
    path = data_cfg.train_data_path if split == "train" else data_cfg.val_data_path
    if path is None:
        raise ValueError(f"No data path configured for split '{split}'")

    p = Path(path)

    if p.suffix in (".jsonl", ".json"):
        ds = load_dataset("json", data_files=str(p), split="train")
    elif p.suffix == ".csv":
        ds = load_dataset("csv", data_files=str(p), split="train")
    elif p.suffix in (".parquet", ".pq"):
        ds = load_dataset("parquet", data_files=str(p), split="train")
    elif p.is_dir():
        files = list(p.glob("*.csv")) + list(p.glob("*.jsonl")) + list(p.glob("*.json"))
        if files:
            ext = files[0].suffix
            fmt = "csv" if ext == ".csv" else "json"
            ds = load_dataset(fmt, data_files=[str(f) for f in files], split="train")
        else:
            ds = load_dataset(str(p), split="train")
    else:
        ds = load_dataset(str(p), split="train")

    # Rename columns to standard names if needed
    if data_cfg.text_column != "text" and data_cfg.text_column in ds.column_names:
        ds = ds.rename_column(data_cfg.text_column, "text")
    if data_cfg.label_column != "label" and data_cfg.label_column in ds.column_names:
        ds = ds.rename_column(data_cfg.label_column, "label")

    # Convert string labels to integers if needed
    if ds.features["label"].dtype == "string":
        unique_labels = sorted(set(ds["label"]))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        ds = ds.map(lambda x: {"label": label_map[x["label"]]})

    return ds


def tokenize_classification_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    data_cfg: DataConfig,
) -> Dataset:
    """Tokenize a classification dataset for SequenceClassification models."""

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_cfg.max_seq_length,
            padding=False,
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds
