# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluate
# MAGIC
# MAGIC Job task: measure perplexity, generation quality and throughput of the
# MAGIC model a training run logged, and record the metrics on that run. Runs on
# MAGIC AI Runtime (serverless GPU).
# MAGIC
# MAGIC **Parameters:** `config_path`, `run_id` (from the train task),
# MAGIC `eval_type` (`all`, `perplexity`, `generation` or `benchmark`).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from src import pipeline

dbutils.widgets.text("config_path", "configs/sft_phi3_config.yaml")
dbutils.widgets.text("run_id", "")
dbutils.widgets.dropdown("eval_type", "all", ["all", "perplexity", "generation", "benchmark"])

run_id = dbutils.widgets.get("run_id")
if not run_id:
    raise ValueError("run_id is required: the MLflow run of the model to evaluate.")

evaluation = pipeline.evaluate(
    dbutils.widgets.get("config_path"),
    run_id=run_id,
    eval_type=dbutils.widgets.get("eval_type"),
)

for key, value in sorted(evaluation["metrics"].items()):
    print(f"  {key}: {value:.4f}")
for sample in evaluation["results"].get("generation", {}).get("generations", [])[:3]:
    print(f"\nPrompt: {sample['prompt'][:80]}...")
    print(f"Response: {sample['response'][:200]}...")

dbutils.notebook.exit(json.dumps(evaluation["metrics"]))
