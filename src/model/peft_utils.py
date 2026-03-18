"""PEFT (LoRA / QLoRA) configuration and application utilities."""

from __future__ import annotations

from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from ..config.schema import ModelConfig
from .adapters import get_model_family_config


def build_lora_config(
    model_cfg: ModelConfig,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """Build a ``LoraConfig`` from the pipeline's ``ModelConfig``.

    If ``lora_target_modules`` is not set in config, auto-detects based on
    the model family.
    """
    target_modules = model_cfg.lora_target_modules
    if not target_modules:
        family_cfg = get_model_family_config(model_cfg.model_name)
        target_modules = family_cfg.lora_target_modules

    return LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
    )


def apply_peft(
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> torch.nn.Module:
    """Prepare model for k-bit training (if quantized) and apply LoRA.

    Returns the PeftModel wrapping the original.
    """
    if model_cfg.quantization != "none":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora_config = build_lora_config(model_cfg, task_type=task_type)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
