# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Model Evaluation
# MAGIC
# MAGIC Evaluate a fine-tuned SLM: perplexity, generation quality, and latency benchmarks.

# COMMAND ----------

# MAGIC %pip install -r ../requirements_runtime.txt rouge-score
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "src"))

# COMMAND ----------

from src.config.schema import load_config
from src.evaluation.engine import EvaluationEngine

CONFIG_PATH = "../configs/sft_phi3_config.yaml"
config = load_config(CONFIG_PATH)

engine = EvaluationEngine(config)

# Set the model path from your training run
MODEL_PATH = os.path.join(config.training.checkpoint_dir, "final_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perplexity

# COMMAND ----------

ppl_result = engine.evaluate_perplexity(model_path=MODEL_PATH)
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

gen_result = engine.evaluate_generation(
    prompts=test_prompts,
    model_path=MODEL_PATH,
)

for g in gen_result["generations"]:
    print(f"\nPrompt: {g['prompt']}")
    print(f"Response: {g['response'][:300]}...")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Latency Benchmark

# COMMAND ----------

bench_result = engine.benchmark(model_path=MODEL_PATH, num_iterations=20)

print(f"Tokens/sec: {bench_result['tokens_per_second']:.1f}")
print(f"Latency p50: {bench_result['latency_per_generation_ms']['p50']:.1f} ms")
print(f"Latency p95: {bench_result['latency_per_generation_ms']['p95']:.1f} ms")
if "gpu_memory_mb" in bench_result:
    print(f"GPU memory: {bench_result['gpu_memory_mb']:.0f} MB")
