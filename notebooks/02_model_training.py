# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Model Training
# MAGIC
# MAGIC Fine-tune a small language model using the SLM accelerator.
# MAGIC Supports SFT (instruction tuning), DPO, and text classification.

# COMMAND ----------

# MAGIC %pip install -r ../requirements_runtime.txt
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(os.getcwd(), "..", "src"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Config

# COMMAND ----------

from src.config.schema import load_config

CONFIG_PATH = "../configs/sft_phi3_config.yaml"
config = load_config(CONFIG_PATH)

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

engine = TrainingEngine(config)

# Auto-detect GPUs; set num_gpus explicitly to override
metrics = engine.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review Metrics

# COMMAND ----------

print("\nTraining complete. Metrics:")
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick Generation Test

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = os.path.join(config.training.checkpoint_dir, "final_model")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "Explain the concept of fine-tuning in machine learning."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
