"""Pydantic v2 configuration models for the SLM fine-tuning pipeline.

Single YAML config drives model selection, quantization, PEFT, data loading,
training, evaluation, serving, and monitoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    model_name: str
    task_type: str = "instruction_tuning"
    trust_remote_code: bool = True

    # Quantization
    quantization: str = "4bit"  # "none", "4bit", "8bit"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # PEFT / LoRA
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Generation defaults
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Classification-specific
    num_labels: int = 2

    @field_validator("quantization")
    @classmethod
    def _validate_quantization(cls, v: str) -> str:
        allowed = {"none", "4bit", "8bit"}
        if v not in allowed:
            raise ValueError(f"quantization must be one of {allowed}, got '{v}'")
        return v

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, v: str) -> str:
        allowed = {"instruction_tuning", "dpo", "text_classification"}
        if v not in allowed:
            raise ValueError(f"task_type must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Data config
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    train_data_path: str
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None

    data_format: str = "alpaca"  # alpaca, sharegpt, preference, csv
    max_seq_length: int = 2048
    batch_size: int = 4
    num_workers: int = 4

    # Alpaca format column mappings
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    system_prompt: Optional[str] = None

    # ShareGPT format
    conversations_column: str = "conversations"

    # Text classification
    text_column: str = "text"
    label_column: str = "label"

    # DPO preference format
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"

    # Dataset splitting
    val_split_ratio: float = 0.1

    @field_validator("data_format")
    @classmethod
    def _validate_format(cls, v: str) -> str:
        allowed = {"alpaca", "sharegpt", "preference", "csv"}
        if v not in allowed:
            raise ValueError(f"data_format must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    max_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0

    # Precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    early_stopping_patience: int = 3
    monitor_metric: str = "eval_loss"
    monitor_mode: str = "min"
    checkpoint_dir: str = "/tmp/checkpoints"
    volume_checkpoint_dir: Optional[str] = None
    save_top_k: int = 3
    log_every_n_steps: int = 10

    # SFT-specific
    packing: bool = False
    dataset_text_field: Optional[str] = None

    # DPO-specific
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"

    # DeepSpeed
    deepspeed_config: Optional[str] = None

    # Distributed
    use_gpu: bool = True

    @field_validator("lr_scheduler_type")
    @classmethod
    def _validate_scheduler(cls, v: str) -> str:
        allowed = {"cosine", "linear", "constant", "constant_with_warmup",
                    "cosine_with_restarts", "polynomial"}
        if v not in allowed:
            raise ValueError(f"lr_scheduler_type must be one of {allowed}, got '{v}'")
        return v


# ---------------------------------------------------------------------------
# MLflow config
# ---------------------------------------------------------------------------

class MLflowConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    experiment_name: str = "slm_fine_tuning"
    run_name: str = "default_run"
    log_model: bool = True
    tags: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Output config
# ---------------------------------------------------------------------------

class OutputConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    results_dir: str = "/tmp/results"
    save_predictions: bool = True


# ---------------------------------------------------------------------------
# Serving config
# ---------------------------------------------------------------------------

class ServingConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    registered_model_name: Optional[str] = None
    endpoint_name: Optional[str] = None
    workload_size: str = "Small"
    workload_type: str = "GPU_SMALL"
    scale_to_zero: bool = True


# ---------------------------------------------------------------------------
# Monitoring config
# ---------------------------------------------------------------------------

class MonitoringConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    drift_threshold: float = 0.1
    error_rate_threshold: float = 0.05
    latency_p95_threshold_ms: float = 1000


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = TrainingConfig()
    mlflow: MLflowConfig = MLflowConfig()
    output: OutputConfig = OutputConfig()
    serving: ServingConfig = ServingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load a YAML config and return a validated ``PipelineConfig``."""
    return PipelineConfig.from_yaml(config_path)
