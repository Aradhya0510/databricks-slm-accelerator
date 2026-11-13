# Databricks notebook source
# MAGIC %md
# MAGIC # Monitor
# MAGIC
# MAGIC Job task: report on a serving endpoint's health, request metrics and token
# MAGIC usage, and fail on a threshold breach so the job's failure notification
# MAGIC is the alert. Runs on serverless CPU.
# MAGIC
# MAGIC **Parameters:** `config_path` (supplies the thresholds), `endpoint_name`
# MAGIC (defaults to `serving.endpoint_name`), `fail_on_breach` (`true`/`false`).

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
dbutils.widgets.text("endpoint_name", "")
dbutils.widgets.dropdown("fail_on_breach", "false", ["true", "false"])

report = pipeline.monitor(
    dbutils.widgets.get("config_path"),
    endpoint_name=dbutils.widgets.get("endpoint_name") or None,
    fail_on_breach=dbutils.widgets.get("fail_on_breach") == "true",
)

print(json.dumps(report, indent=2, default=str))
dbutils.notebook.exit(json.dumps({"status": report["status"]}))
