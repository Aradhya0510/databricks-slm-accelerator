# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy
# MAGIC
# MAGIC Job task: point a Model Serving endpoint at a registered model version and
# MAGIC wait until it is ready. Runs on serverless CPU: it only calls the serving
# MAGIC API, and an endpoint can take half an hour to come up.
# MAGIC
# MAGIC **Parameters:** `config_path`, `model_version` (from the register task),
# MAGIC `model_name`, `endpoint_name` (both default to the config's `serving` section).

# COMMAND ----------

# MAGIC %pip install -q pydantic pyyaml databricks-sdk -c ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from src import pipeline

dbutils.widgets.text("config_path", "configs/sft_phi3_config.yaml")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("endpoint_name", "")

model_version = dbutils.widgets.get("model_version")
if not model_version:
    raise ValueError("model_version is required: the registered version to serve.")

state = pipeline.deploy(
    dbutils.widgets.get("config_path"),
    model_version=model_version,
    model_name=dbutils.widgets.get("model_name") or None,
    endpoint_name=dbutils.widgets.get("endpoint_name") or None,
)

print(state)
dbutils.notebook.exit(json.dumps(state))
