# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Model Evaluation
# MAGIC
# MAGIC Evaluate a fine-tuned SLM: perplexity, generation quality, and latency benchmarks.
# MAGIC
# MAGIC Attach this notebook to **AI Runtime** (serverless GPU).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# COMMAND ----------

from src.evaluation.engine import EvaluationEngine
from src.pipeline import load_pipeline_config

CONFIG_PATH = "configs/sft_phi3_config.yaml"
config = load_pipeline_config(CONFIG_PATH)

engine = EvaluationEngine(config)

# The MLflow run that holds the trained model (from 02_model_training, or the
# pipeline job's train task).
RUN_ID = "<your_mlflow_run_id>"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perplexity

# COMMAND ----------

ppl_result = engine.evaluate_perplexity(run_id=RUN_ID)
print(f"Perplexity: {ppl_result['perplexity']:.2f}")
print(f"Avg loss: {ppl_result['avg_loss']:.4f}")
print(f"Tokens evaluated: {ppl_result['total_tokens']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generation Quality

# COMMAND ----------

test_prompts = [
    "What is the difference between supervised and unsupervised learning?",
    "Write a Python function to reverse a linked list.",
    "Explain the concept of attention in transformers.",
    "What are the benefits of using Databricks for ML?",
]

gen_result = engine.evaluate_generation(prompts=test_prompts, run_id=RUN_ID)

for g in gen_result["generations"]:
    print(f"\nPrompt: {g['prompt']}")
    print(f"Response: {g['response'][:300]}...")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Latency Benchmark

# COMMAND ----------

bench_result = engine.benchmark(run_id=RUN_ID, num_iterations=20)

print(f"Tokens/sec: {bench_result['tokens_per_second']:.1f}")
print(f"Latency p50: {bench_result['latency_per_generation_ms']['p50']:.1f} ms")
print(f"Latency p95: {bench_result['latency_per_generation_ms']['p95']:.1f} ms")
if "gpu_memory_mb" in bench_result:
    print(f"GPU memory: {bench_result['gpu_memory_mb']:.0f} MB")
