# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Model Training
# MAGIC
# MAGIC Fine-tune a small language model using the SLM accelerator.
# MAGIC Supports SFT (instruction tuning), DPO, and text classification.
# MAGIC
# MAGIC Attach this notebook to **AI Runtime** (serverless GPU). An A10 trains the
# MAGIC shipped configs; on an 8xH100 node training runs one process per GPU.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Config

# COMMAND ----------

from src.pipeline import load_pipeline_config

CONFIG_PATH = "configs/sft_phi3_config.yaml"
config = load_pipeline_config(CONFIG_PATH)

print(f"Model: {config.model.model_name}")
print(f"Task: {config.model.task_type}")
print(f"Quantization: {config.model.quantization}")
print(f"LoRA r={config.model.lora_r}, alpha={config.model.lora_alpha}")
print(f"Data format: {config.data.data_format}")
print(f"Max seq length: {config.data.max_seq_length}")
print(f"Batch size: {config.data.batch_size}")
print(f"Epochs: {config.training.max_epochs}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train

# COMMAND ----------

from src.engine import TrainingEngine

result = TrainingEngine(config).train()
RUN_ID = result.run_id

print(f"MLflow run: {RUN_ID}")
for k, v in sorted(result.metrics.items()):
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Generation Test
# MAGIC Loads the model back from the MLflow run, the same way the evaluate and
# MAGIC register stages do.

# COMMAND ----------

from src.evaluation.engine import EvaluationEngine

generation = EvaluationEngine(config).evaluate_generation(
    prompts=["Explain the concept of fine-tuning in machine learning."],
    run_id=RUN_ID,
)
for g in generation["generations"]:
    print(f"Prompt: {g['prompt']}")
    print(f"Response: {g['response']}")
