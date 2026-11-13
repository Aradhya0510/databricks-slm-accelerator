# Databricks SLM Accelerator

A production-ready framework for fine-tuning **small language models** on [Databricks AI Runtime](https://docs.databricks.com/machine-learning/ai-runtime/), the serverless GPU offering. Config-driven, modular, and extensible — train, evaluate, register, deploy, and monitor SLMs through a single YAML config, with no clusters to provision.

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

The pipeline ships as a Databricks Asset Bundle. Point a config's data paths at
your Unity Catalog volume, then deploy and run it:

```bash
databricks bundle deploy --target dev
databricks bundle run slm_finetuning_pipeline --target dev
```

Every bundle variable can be overridden at deploy time, and the job's
parameters can be overridden for a single run:

```bash
# Train on an 8xH100 node, registering under your own catalog and schema
databricks bundle deploy --target dev \
    --var accelerator=GPU_8xH100 --var catalog=my_catalog --var schema=my_schema

# Train a different config, once, without redeploying
databricks bundle run slm_finetuning_pipeline --target dev \
    --params config_path=configs/dpo_phi3_config.yaml
```

To work interactively instead, open the notebooks in `notebooks/`, attach them
to AI Runtime, and run them in order.

## Architecture

The deployed job runs each stage as a notebook task. The GPU stages run on AI
Runtime, and each provisions its own node, so the model moves between them as
an MLflow run id rather than a file on local disk:

```
train ──run_id──▶ evaluate ──▶ register ──model_version──▶ deploy
(AI Runtime GPU)  (AI Runtime GPU)  (AI Runtime GPU)          (serverless CPU)
```

Deployment waits for the serving endpoint to warm up, which can take half an
hour, so it runs on serverless CPU rather than holding an idle GPU. Monitoring
is a separate scheduled job, also on serverless CPU.

Inside the training stage:

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

6. **Serverless and Databricks-native** — AI Runtime for GPU compute, Unity Catalog Volumes for data/checkpoints, MLflow 3.x for tracking, Model Serving for deployment, system tables for monitoring.

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
│   ├── monitoring/                # Endpoint health, request metrics, token stats
│   └── pipeline.py                # One function per job stage
├── jobs/                          # Job task notebooks (train, evaluate, register, deploy, monitor)
├── notebooks/                     # Interactive Databricks notebooks (01-05)
├── configs/                       # YAML configs per model/task
├── databricks.yml                 # Bundle: the pipeline and monitoring jobs
└── requirements.txt               # Pinned dependencies, installed by every notebook
```

## Requirements

- **A workspace with [AI Runtime](https://docs.databricks.com/machine-learning/ai-runtime/overview#requirements)** (serverless GPU) and serverless compute for notebooks and jobs.
- **Databricks CLI** v1.18 or later, for `databricks bundle`.
- **Unity Catalog**: a volume for training data and checkpoints, and a catalog and schema you can register models in.
- **Model Serving with GPU workloads**, for the deploy stage.

Dependencies come from `requirements.txt`, which each notebook installs with
`%pip`. AI Runtime's Standard environment provides CUDA but not torch or the
Hugging Face stack, and its notebook jobs do not take dependencies from the
job's environment, so the notebooks install them. The versions are pinned to
a set that resolves together on AI Runtime; upgrade them together.

### Choosing an accelerator

| `accelerator` | GPUs | Training |
|---|---|---|
| `GPU_1xA10` (default) | 1 × A10, 24 GB | One process. Fits the shipped 4-bit QLoRA configs. |
| `GPU_1xH100` | 1 × H100, 80 GB | One process. For models or batches an A10 cannot hold. |
| `GPU_8xH100` | 8 × H100, 80 GB each | One process per GPU (DDP). |
| `GPU_8xB300` | 8 × B300, 288 GB each | One process per GPU (DDP). Public Preview. |

AI Runtime notebook jobs run for at most two days. For longer fine-tunes, set
`training.volume_checkpoint_dir` so checkpoints land in a volume.

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

The regression suite covers these task, model and data-format combinations, each on a single A10 GPU:

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

Qwen3.5 needs transformers 5.x: the `qwen3_5` architecture does not exist in the 4.x line.

These run as a regression suite from small synthetic datasets in each supported format; the fixtures are not part of the repo.

## Adding a New Task

1. Create `src/tasks/your_task/__init__.py`
2. Implement the `BaseTask` interface
3. Decorate with `@TaskRegistry.register("your_task")`
4. Add a lazy import entry in `src/engine/engine.py`

## Adding a New Model Family

Add an entry to `_FAMILY_CONFIGS` in `src/model/adapters.py` with the model's LoRA target modules, padding side, and special token behavior. The `detect_model_family()` function matches model names by substring, so also add a pattern entry if the model name doesn't match an existing family.

## Multi-GPU

Real DDP needs **one process per GPU**. Letting a single process see several
GPUs gives `nn.DataParallel` instead — slower, and it reads
`per_device_train_batch_size` as the *total* batch rather than the per-GPU one.

`TrainingEngine.train()` decides from the GPUs it can see. On a single-GPU
accelerator it trains in-process. On an 8×H100 or 8×B300 node it launches one
process per GPU through AI Runtime's
[`@distributed`](https://docs.databricks.com/machine-learning/ai-runtime/connecting/distributed-training),
and every rank logs into the one MLflow run that launch creates. Nothing in the
config changes; select the accelerator in the bundle, or attach the notebook to
it. Training is single-node: `@distributed` does not span nodes.

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
AI Runtime has no local Spark, so tables are read over Spark Connect.

## Testing

```bash
# The CPU build of torch is enough; the suite needs no GPU.
pip install "torch==$(grep -oP '^torch==\K.*' requirements.txt)" \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -e ".[dev]"
pytest
```

The suite is offline and CPU-only by design — models are built from configs in
code rather than downloaded — so it runs anywhere and cannot be broken by a
HuggingFace Hub outage.
