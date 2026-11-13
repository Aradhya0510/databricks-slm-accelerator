# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Model Deployment
# MAGIC
# MAGIC Register a fine-tuned SLM to Unity Catalog and deploy to Model Serving.
# MAGIC
# MAGIC Attach this notebook to **AI Runtime** (serverless GPU): registration loads
# MAGIC the model, and a model trained in 4-bit needs the GPU kernels to load.

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
# MAGIC ## Configuration

# COMMAND ----------

RUN_ID = "<your_mlflow_run_id>"

CATALOG = "<catalog>"
SCHEMA = "<schema>"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.phi3_sft"
ENDPOINT_NAME = "phi3-sft-endpoint"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model to Unity Catalog

# COMMAND ----------

from src.serving.registration import register_model

# "latest" cannot be used as an alias: the registry reserves it.
reg_result = register_model(
    run_id=RUN_ID,
    registered_model_name=MODEL_NAME,
    task_type="instruction_tuning",
    aliases=["champion"],
    tags={"model_family": "phi-3", "task": "instruction_tuning"},
    test_prompt="Explain the concept of fine-tuning.",
)

print(f"Model URI: {reg_result['model_uri']}")
print(f"Version: {reg_result['model_version']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Endpoint

# COMMAND ----------

from src.serving.deployment import deploy_endpoint, wait_for_ready, test_endpoint

deploy_endpoint(
    endpoint_name=ENDPOINT_NAME,
    registered_model_name=MODEL_NAME,
    model_version=str(reg_result["model_version"]),
    workload_size="Small",
    workload_type="GPU_SMALL",
    scale_to_zero=True,
)

# COMMAND ----------

# This can take 15-30 minutes for GPU endpoints
wait_for_ready(ENDPOINT_NAME, timeout=3600)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Endpoint

# COMMAND ----------

test_endpoint(
    endpoint_name=ENDPOINT_NAME,
    test_prompt="What are the key benefits of using small language models?",
)
