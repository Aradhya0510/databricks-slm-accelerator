# Databricks notebook source
# MAGIC %md
# MAGIC # Register
# MAGIC
# MAGIC Job task: wrap a training run's model as an MLflow PyFunc and register it
# MAGIC to Unity Catalog. Runs on AI Runtime (serverless GPU): a model trained in
# MAGIC 4-bit stays 4-bit after merging, and loading it needs the GPU kernels.
# MAGIC
# MAGIC **Parameters:** `config_path`, `run_id` (from the train task),
# MAGIC `model_name` (`catalog.schema.model`; defaults to `serving.registered_model_name`).
# MAGIC
# MAGIC **Task values set:** `model_version`.

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
dbutils.widgets.text("model_name", "")

run_id = dbutils.widgets.get("run_id")
if not run_id:
    raise ValueError("run_id is required: the MLflow run of the model to register.")

registration = pipeline.register(
    dbutils.widgets.get("config_path"),
    run_id=run_id,
    model_name=dbutils.widgets.get("model_name") or None,
)

version = str(registration["model_version"])
print(f"Registered {registration['registered_model_name']} version {version}")

dbutils.jobs.taskValues.set(key="model_version", value=version)
dbutils.notebook.exit(json.dumps({"model_version": version}))
