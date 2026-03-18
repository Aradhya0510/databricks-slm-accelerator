# Databricks SLM Accelerator

A production-ready framework for fine-tuning **small language models** on Databricks GPU clusters. Config-driven, modular, and extensible — train, evaluate, register, deploy, and monitor SLMs through a single YAML config.

## Supported Tasks

| Task | Trainer | Data Format | Use Case |
|------|---------|-------------|----------|
| **Instruction Tuning (SFT)** | TRL `SFTTrainer` | Alpaca, ShareGPT | Fine-tune for instruction following, chat, domain knowledge |
| **DPO Alignment** | TRL `DPOTrainer` | Preference pairs | Align model outputs with human preferences |
| **Text Classification** | HF `Trainer` | CSV, JSON with text/label | Sentiment, intent, topic classification |

## Supported Models

Any HuggingFace causal LM works out of the box. Pre-configured adapter configs for:

- **Phi-3 / Phi-3.5** (Microsoft) — 3.8B, optimized for efficiency
- **Llama 3.x** (Meta) — 1B, 3B, 8B
- **Mistral / Mixtral** (Mistral AI) — 7B
- **Gemma 2** (Google) — 2B, 7B, 9B
- **Qwen 2.5** (Alibaba) — 0.5B to 7B

## Quick Start

```bash
# Train (auto-detects GPUs)
python jobs/train.py --config_path configs/sft_phi3_config.yaml

# Train on 4 GPUs
python jobs/train.py --config_path configs/sft_phi3_config.yaml --num_gpus 4

# Evaluate
python jobs/evaluate.py --config_path configs/sft_phi3_config.yaml \
    --model_path /tmp/checkpoints/phi3_sft/final_model

# Deploy
python jobs/deploy.py --config_path configs/sft_phi3_config.yaml \
    --run_id <mlflow_run_id> \
    --model_name catalog.schema.phi3_sft \
    --endpoint_name phi3-sft-endpoint

# Monitor
python jobs/monitor.py --endpoint_name phi3-sft-endpoint
```

## Architecture

```
YAML Config → PipelineConfig (Pydantic v2) → TrainingEngine
                                                    │
                                          TaskRegistry dispatches to:
                                                    ├── InstructionTuningTask (SFTTrainer)
                                                    ├── DPOTask (DPOTrainer)
                                                    └── TextClassificationTask (HF Trainer)
                                                    │
                                          Shared model/ package:
                                                    ├── loader.py (quantization + model loading)
                                                    ├── peft_utils.py (LoRA / QLoRA)
                                                    └── adapters.py (model family configs)
```

### Key Design Principles

1. **Config-driven** — One YAML file controls the entire pipeline: model, data, training, serving, monitoring.

2. **Task registry** — Plug in new tasks with `@TaskRegistry.register("task_name")`. Each task implements `BaseTask` (load model, prepare data, create trainer, compute metrics).

3. **Shared model package** — Quantization (4-bit/8-bit via bitsandbytes), PEFT (LoRA/QLoRA), and model-family-specific configs are shared across all tasks — no duplication.

4. **Model adapters as config, not class hierarchies** — Each model family's quirks (LoRA target modules, padding side, special tokens) are captured in a lightweight dataclass, not an inheritance tree.

5. **TRL trainers used directly** — No wrapper layers around `SFTTrainer`/`DPOTrainer`. Each task creates the right trainer with the right config.

6. **Databricks-native** — Unity Catalog Volumes for data/checkpoints, MLflow 3.x for tracking, Model Serving for deployment, system tables for monitoring.

## Project Structure

```
├── src/
│   ├── config/schema.py           # Pydantic v2 config models + YAML loader
│   ├── registry.py                # TaskRegistry with @register decorator
│   ├── model/
│   │   ├── loader.py              # Model + tokenizer loading with quantization
│   │   ├── peft_utils.py          # LoRA/QLoRA configuration and application
│   │   └── adapters.py            # Model family configs (Phi-3, Llama, Mistral, etc.)
│   ├── engine/
│   │   ├── engine.py              # TrainingEngine orchestrator
│   │   └── callbacks.py           # VolumeCheckpoint, EarlyStopping
│   ├── tasks/
│   │   ├── base.py                # BaseTask ABC
│   │   ├── instruction_tuning/    # SFT with Alpaca/ShareGPT formatting
│   │   ├── dpo/                   # DPO with preference pairs
│   │   └── text_classification/   # Sequence classification
│   ├── evaluation/engine.py       # Perplexity, generation quality, benchmarks
│   ├── serving/                   # PyFunc wrappers, UC registration, deployment
│   └── monitoring/                # Endpoint health, request metrics, token stats
├── jobs/                          # CLI entry points
├── notebooks/                     # Interactive Databricks notebooks (01-05)
├── configs/                       # YAML configs per model/task
├── requirements.txt               # Dev dependencies
└── requirements_runtime.txt       # Runtime dependencies
```

## Requirements

- **Databricks Runtime**: 16.4+ ML or 17.3 LTS ML
- **GPU**: A10G, A100, or H100 (single or multi-GPU)
- **Python**: 3.10+

## Data Formats

### Alpaca (instruction tuning)
```json
{"instruction": "Summarize the text.", "input": "Long text here...", "output": "Summary here."}
```

### ShareGPT (multi-turn conversations)
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

### Preference pairs (DPO)
```json
{"prompt": "Question here", "chosen": "Good answer", "rejected": "Bad answer"}
```

### CSV (classification)
```
text,label
"This product is great!",positive
"Terrible experience.",negative
```

## Adding a New Task

1. Create `src/tasks/your_task/__init__.py`
2. Implement the `BaseTask` interface
3. Decorate with `@TaskRegistry.register("your_task")`
4. Import in `src/engine/engine.py`

## Adding a New Model Family

Add an entry to `_FAMILY_CONFIGS` in `src/model/adapters.py` with the model's LoRA target modules, padding side, and special token behavior.
