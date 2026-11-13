# Databricks notebook source
# MAGIC %md
# MAGIC # Train
# MAGIC
# MAGIC Job task: fine-tune a small language model and log it to MLflow. Runs on
# MAGIC AI Runtime (serverless GPU); on a multi-GPU accelerator it trains one
# MAGIC process per GPU.
# MAGIC
# MAGIC **Parameters:** `config_path` — pipeline config, relative to the project root.
# MAGIC
# MAGIC **Task values set:** `run_id` — the MLflow run holding the trained model.

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

result = pipeline.train(dbutils.widgets.get("config_path"))

print(f"MLflow run: {result['run_id']}")
for key, value in sorted(result["metrics"].items()):
    print(f"  {key}: {value}")

dbutils.jobs.taskValues.set(key="run_id", value=result["run_id"])
dbutils.notebook.exit(json.dumps({"run_id": result["run_id"]}))
