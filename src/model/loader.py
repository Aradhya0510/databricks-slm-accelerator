"""Unified model and tokenizer loading with optional quantization.

Extracted as a shared utility because quantization + tokenizer config is
identical across all SLM tasks.
"""

from __future__ import annotations

from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..config.schema import ModelConfig
from .adapters import get_model_family_config


# ---------------------------------------------------------------------------
# Quantization config builder
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _build_bnb_config(model_cfg: ModelConfig) -> BitsAndBytesConfig | None:
    """Return a BitsAndBytesConfig for 4-bit or 8-bit, or None."""
    if model_cfg.quantization == "4bit":
        compute_dtype = _DTYPE_MAP.get(model_cfg.bnb_4bit_compute_dtype, torch.bfloat16)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_cfg.use_double_quant,
        )
    if model_cfg.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizer:
    """Load and configure the tokenizer for the given model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        trust_remote_code=model_cfg.trust_remote_code,
    )

    family_cfg = get_model_family_config(model_cfg.model_name)

    tokenizer.padding_side = family_cfg.padding_side

    if tokenizer.pad_token is None:
        if family_cfg.use_eos_as_pad:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_causal_lm(
    model_cfg: ModelConfig,
    tokenizer: PreTrainedTokenizer | None = None,
) -> PreTrainedModel:
    """Load a causal LM with optional quantization.

    Does NOT apply PEFT — that is handled by the task or engine so the task
    can choose the PEFT ``TaskType``.
    """
    bnb_config = _build_bnb_config(model_cfg)
    kwargs = {
        "pretrained_model_name_or_path": model_cfg.model_name,
        "trust_remote_code": model_cfg.trust_remote_code,
        "device_map": "auto" if bnb_config else None,
        "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else None,
    }
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    if tokenizer and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    return model


def load_model_for_sequence_classification(
    model_cfg: ModelConfig,
    tokenizer: PreTrainedTokenizer | None = None,
) -> PreTrainedModel:
    """Load a sequence classification model with optional quantization."""
    bnb_config = _build_bnb_config(model_cfg)
    kwargs = {
        "pretrained_model_name_or_path": model_cfg.model_name,
        "num_labels": model_cfg.num_labels,
        "trust_remote_code": model_cfg.trust_remote_code,
        "device_map": "auto" if bnb_config else None,
    }
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(**kwargs)

    if tokenizer and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.pad_token_id is None and tokenizer:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model


def load_model_and_tokenizer(
    model_cfg: ModelConfig,
    for_classification: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Convenience: load both tokenizer and model in one call."""
    tokenizer = load_tokenizer(model_cfg)
    if for_classification:
        model = load_model_for_sequence_classification(model_cfg, tokenizer)
    else:
        model = load_model_for_causal_lm(model_cfg, tokenizer)
    return model, tokenizer
