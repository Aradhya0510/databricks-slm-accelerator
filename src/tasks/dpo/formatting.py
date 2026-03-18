"""Data loading and formatting for DPO preference datasets.

Expected format: each record has prompt, chosen, and rejected fields.
The DPOTrainer expects columns named ``prompt``, ``chosen``, ``rejected``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset

from ...config.schema import DataConfig


def load_preference_dataset(data_cfg: DataConfig, split: str = "train") -> Dataset:
    """Load a preference dataset and rename columns to DPOTrainer's convention."""
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
        json_files = list(p.glob("*.jsonl")) + list(p.glob("*.json"))
        if json_files:
            ds = load_dataset("json", data_files=[str(f) for f in json_files], split="train")
        else:
            ds = load_dataset(str(p), split="train")
    else:
        ds = load_dataset(str(p), split="train")

    column_mapping = {}
    if data_cfg.prompt_column != "prompt" and data_cfg.prompt_column in ds.column_names:
        column_mapping[data_cfg.prompt_column] = "prompt"
    if data_cfg.chosen_column != "chosen" and data_cfg.chosen_column in ds.column_names:
        column_mapping[data_cfg.chosen_column] = "chosen"
    if data_cfg.rejected_column != "rejected" and data_cfg.rejected_column in ds.column_names:
        column_mapping[data_cfg.rejected_column] = "rejected"

    if column_mapping:
        ds = ds.rename_columns(column_mapping)

    required = {"prompt", "chosen", "rejected"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Preference dataset missing columns: {missing}. "
            f"Available: {ds.column_names}. "
            f"Configure prompt_column/chosen_column/rejected_column in data config."
        )

    return ds
