# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Data Preparation
# MAGIC
# MAGIC Prepare and validate fine-tuning data for the SLM accelerator.
# MAGIC Supports Alpaca, ShareGPT, Preference (DPO), and CSV classification formats.

# COMMAND ----------

# MAGIC %pip install pyyaml datasets
# MAGIC %restart_python

# COMMAND ----------

import json
from pathlib import Path
from datasets import load_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Set your Unity Catalog volume paths and data format.

# COMMAND ----------

CATALOG = "<catalog>"
SCHEMA = "<schema>"
VOLUME = "<volume>"

VOLUME_BASE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
DATA_DIR = f"{VOLUME_BASE}/data/sft"
DATA_FORMAT = "alpaca"  # alpaca | sharegpt | preference | csv

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option A: Alpaca Format
# MAGIC Each record has `instruction`, `input` (optional), and `output` fields.

# COMMAND ----------

alpaca_examples = [
    {
        "instruction": "Summarize the following text.",
        "input": "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed.",
        "output": "Machine learning is an AI subset where systems learn from data autonomously."
    },
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris."
    },
    {
        "instruction": "Write a Python function to calculate the factorial of a number.",
        "input": "",
        "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option B: ShareGPT Format
# MAGIC Multi-turn conversations with `from` and `value` fields.

# COMMAND ----------

sharegpt_examples = [
    {
        "conversations": [
            {"from": "human", "value": "What is transfer learning?"},
            {"from": "gpt", "value": "Transfer learning is a technique where a model trained on one task is reused as the starting point for a model on a different task."},
            {"from": "human", "value": "Give me an example."},
            {"from": "gpt", "value": "A language model pre-trained on general text can be fine-tuned for sentiment analysis on product reviews."}
        ]
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Option C: DPO Preference Format
# MAGIC Each record has `prompt`, `chosen`, and `rejected` responses.

# COMMAND ----------

preference_examples = [
    {
        "prompt": "Explain quantum computing in simple terms.",
        "chosen": "Quantum computing uses quantum bits (qubits) that can be in multiple states at once, allowing parallel computation that can solve certain problems much faster than classical computers.",
        "rejected": "Quantum computing is really complicated computer stuff that uses quantum physics."
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Volume
# MAGIC Write the examples to your Unity Catalog volume.

# COMMAND ----------

def save_jsonl(data, path):
    """Save a list of dicts as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} records to {path}")


if DATA_FORMAT == "alpaca":
    save_jsonl(alpaca_examples, f"{DATA_DIR}/train.jsonl")
elif DATA_FORMAT == "sharegpt":
    save_jsonl(sharegpt_examples, f"{DATA_DIR}/train.jsonl")
elif DATA_FORMAT == "preference":
    save_jsonl(preference_examples, f"{DATA_DIR}/train.jsonl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate
# MAGIC Load the dataset back and verify it.

# COMMAND ----------

ds = load_dataset("json", data_files=f"{DATA_DIR}/train.jsonl", split="train")
print(f"Loaded {len(ds)} samples")
print(f"Columns: {ds.column_names}")
print(f"First sample: {ds[0]}")
