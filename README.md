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
- **Phi-4 / Phi-4-mini** (Microsoft) — 3.8B, latest-gen small model
- **Llama 3.x** (Meta) — 1B, 3B, 8B (requires gated HF token)
- **Mistral / Mixtral** (Mistral AI) — 7B
- **Gemma 2** (Google) — 2B, 7B, 9B (requires gated HF token)
- **Qwen 2.5 / Qwen 3.5** (Alibaba) — 0.5B to 7B

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
                                                    ├── loader.py (resilient model loading)
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

7. **Resilient model loading** — Both causal-LM and sequence-classification loaders automatically fall back from `trust_remote_code=True` to native transformers classes when a model's custom code raises `ValueError` or `ImportError`. For classification, the score layer is reinitialised when a model's remote code ignores the `num_labels` argument.

8. **Pre-formatted SFT datasets** — Instruction-tuning datasets are converted to chat-templated strings via `dataset.map()` before being passed to `SFTTrainer`, ensuring compatibility across TRL versions and avoiding internal formatting-function pitfalls.

## Project Structure

```
├── src/
│   ├── config/schema.py           # Pydantic v2 config models + YAML loader
│   ├── registry.py                # TaskRegistry with @register decorator
│   ├── model/
│   │   ├── loader.py              # Model + tokenizer loading with quantization
│   │   ├── peft_utils.py          # LoRA/QLoRA configuration and application
│   │   └── adapters.py            # Model family configs (Phi-3, Phi-4, Llama, etc.)
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
└── requirements_runtime.txt       # Databricks runtime dependencies (trl, peft, bitsandbytes)
```

## Requirements

- **Databricks Runtime**: DBR 17.3 LTS ML GPU or later (Spark 4.0, Python 3.12)
- **GPU**: A10G (g5.4xlarge), A100, or H100 — single or multi-GPU
- **Pre-installed by DBR ML**: `transformers`, `accelerate`, `datasets`, `sentencepiece`, `protobuf`, `pydantic`, `torch`, `flash-attn`, `deepspeed`, `mlflow`
- **Installed at runtime**: `trl>=0.12`, `peft>=0.10`, `bitsandbytes>=0.43` (see `requirements_runtime.txt`)

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

## Tested Combinations

The framework has been validated across the following task/model/data-format matrix on DBR 17.3 LTS ML GPU (`g5.4xlarge`, A10G):

| Task Type | Model | Data Format |
|-----------|-------|-------------|
| text_classification | Phi-3.5-mini-instruct | CSV |
| instruction_tuning | Phi-3.5-mini-instruct | Alpaca |
| instruction_tuning | Phi-3.5-mini-instruct | ShareGPT |
| dpo | Phi-3.5-mini-instruct | Preference |
| instruction_tuning | Phi-4-mini-instruct | Alpaca |
| text_classification | Phi-4-mini-instruct | CSV |
| instruction_tuning | Qwen3.5-4B | Alpaca |
| text_classification | Qwen3.5-4B | CSV |

A reusable Databricks Job (`slm-sanity-suite`) runs these as a regression suite. The configs and synthetic data live in the Databricks workspace (not in this repo).

## Adding a New Task

1. Create `src/tasks/your_task/__init__.py`
2. Implement the `BaseTask` interface
3. Decorate with `@TaskRegistry.register("your_task")`
4. Add a lazy import entry in `src/engine/engine.py`

## Adding a New Model Family

Add an entry to `_FAMILY_CONFIGS` in `src/model/adapters.py` with the model's LoRA target modules, padding side, and special token behavior. The `detect_model_family()` function matches model names by substring, so also add a pattern entry if the model name doesn't match an existing family.

## Multi-GPU

Real DDP needs **one process per GPU**. Running the training script as a plain
`python` process and letting HF Trainer see several GPUs gives `nn.DataParallel`
instead — a single process driving every GPU, which is slower and interprets
`per_device_train_batch_size` as the *total* batch rather than the per-GPU
batch. So multi-GPU always goes through `TorchDistributor`:

| `--distributed` | What it does | When |
|---|---|---|
| `auto` (default) | One process per visible GPU on this node | Normal use |
| `single` | Forces one process, pinned to one GPU | Debugging |
| `local` | Single-node multi-process DDP | Explicit form of `auto` |
| `multinode` | Spreads processes across Spark workers | Cluster has workers |

## Completion-only loss

Instruction tuning masks the prompt by default (`training.completion_only_loss`),
so the loss covers only the assistant's response. Training on the prompt as well
teaches the model to reproduce its own template and wastes capacity that small
models do not have to spare. Set it to `false` for whole-sequence training.

## Model artifacts

`mlflow.artifact_format` decides how PEFT weights are persisted:

- **`merged`** (default) folds the adapter into the base weights, so serving is
  an ordinary `from_pretrained` and does not need the base model to stay
  reachable.
- **`adapter`** saves the adapter alone and records the base model id beside it.
  Much smaller, but the base model must remain available at serving time.

## Data sources

Training data can come from a file (`data.train_data_path`) or directly from a
Unity Catalog table (`data.train_table`, as `catalog.schema.table`). The table
path keeps lineage and governance rather than requiring an export to JSONL.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite is offline and CPU-only by design — models are built from configs in
code rather than downloaded — so it runs anywhere and cannot be broken by a
HuggingFace Hub outage.
