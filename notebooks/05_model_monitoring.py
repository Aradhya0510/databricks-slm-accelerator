# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Model Monitoring
# MAGIC
# MAGIC Monitor a deployed SLM endpoint: health, request metrics, and token usage.

# COMMAND ----------

# MAGIC %pip install -r ../requirements_runtime.txt
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "src"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

ENDPOINT_NAME = "phi3-sft-endpoint"

# COMMAND ----------

from src.monitoring.endpoint_monitor import EndpointMonitor

monitor = EndpointMonitor(ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Health Check

# COMMAND ----------

health = monitor.get_health()
print(f"Endpoint: {health['endpoint_name']}")
print(f"Ready: {health['ready']}")
print(f"Served models: {health['served_models']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Request Metrics (last 24 hours)

# COMMAND ----------

metrics = monitor.get_request_metrics(hours=24)

if "error" not in metrics:
    print(f"Total requests: {metrics['total_requests']}")
    print(f"Error rate: {metrics.get('error_rate', 0):.2%}")
    print(f"Avg latency: {metrics.get('avg_latency_ms', 0):.0f} ms")
    print(f"P95 latency: {metrics.get('p95_latency_ms', 0):.0f} ms")
else:
    print(f"Could not fetch metrics: {metrics['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Token Usage (SLM-specific)

# COMMAND ----------

token_stats = monitor.get_token_usage(hours=24)

if "error" not in token_stats:
    print(f"Responses sampled: {token_stats['num_responses_sampled']}")
    print(f"Avg output tokens (approx): {token_stats.get('avg_output_tokens_approx', 0):.0f}")
    print(f"Tokens/sec (approx): {token_stats.get('tokens_per_second_approx', 0):.1f}")
else:
    print(f"Could not fetch token stats: {token_stats['error']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full Report

# COMMAND ----------

import json

report = monitor.generate_report()
print(json.dumps(report, indent=2, default=str))
