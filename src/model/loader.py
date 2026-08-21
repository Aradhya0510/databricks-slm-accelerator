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
from ..utils.environment import (
    is_distributed_launch,
    local_rank,
    resolve_attn_implementation,
)
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


def _resolve_device_map(model_cfg: ModelConfig, quantized: bool):
    """Choose a device placement that is correct for the current launch.

    ``device_map="auto"`` shards one model across every visible GPU.  That is
    right for single-process inference and wrong for training: under a
    distributed launch each rank must hold a *full* replica on its own GPU,
    otherwise every rank tries to shard across all of them.
    """
    import torch

    if not torch.cuda.is_available():
        return None

    if is_distributed_launch():
        # One full replica per rank, pinned to that rank's GPU.
        return {"": local_rank()}

    if quantized:
        # Keep newly-initialised layers on the same device as the quantised
        # backbone rather than letting accelerate split them.
        return {"": 0}

    return None


def _resolve_dtype(model_cfg: ModelConfig):
    """Model dtype for unquantized loads, honouring bf16 support."""
    import torch

    from ..utils.environment import resolve_precision

    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if resolve_precision("auto") == "bf16" else torch.float16


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
        "device_map": _resolve_device_map(model_cfg, quantized=bnb_config is not None),
        # FlashAttention-2 was hardcoded on whenever CUDA was present, with no
        # capability check and no fallback -- it is not part of DBR ML and
        # needs Ampere or newer, so this failed outright on many clusters.
        "attn_implementation": resolve_attn_implementation(
            getattr(model_cfg, "attn_implementation", "auto")
        ),
    }
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = _resolve_dtype(model_cfg)

    for trust_rc in ([True, False] if model_cfg.trust_remote_code else [False]):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                trust_remote_code=trust_rc, **kwargs,
            )
            break
        except (ValueError, ImportError):
            if not trust_rc:
                raise
            print(
                "Warning: AutoModelForCausalLM failed with "
                "trust_remote_code=True, retrying with native config..."
            )

    if tokenizer and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    return model


def load_model_for_sequence_classification(
    model_cfg: ModelConfig,
    tokenizer: PreTrainedTokenizer | None = None,
) -> PreTrainedModel:
    """Load a sequence classification model with optional quantization.

    AutoModelForSequenceClassification does not recognise custom remote-code
    config classes (e.g. Phi-3.5's custom Phi3Config).  We first try with the
    caller's trust_remote_code setting; on failure we retry with it disabled so
    the native transformers Phi3Config (which IS registered) is used instead.
    """
    bnb_config = _build_bnb_config(model_cfg)

    kwargs = {
        "pretrained_model_name_or_path": model_cfg.model_name,
        "num_labels": model_cfg.num_labels,
        "device_map": _resolve_device_map(model_cfg, quantized=bnb_config is not None),
    }
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = _resolve_dtype(model_cfg)

    for trust_rc in ([True, False] if model_cfg.trust_remote_code else [False]):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                trust_remote_code=trust_rc,
                ignore_mismatched_sizes=True,
                **kwargs,
            )
            break
        except (ValueError, ImportError):
            if not trust_rc:
                raise
            print(
                "Warning: AutoModelForSequenceClassification failed with "
                "trust_remote_code=True, retrying with native config..."
            )

    # Ensure the classification head has the correct output dimension.
    # Some models (e.g. Qwen3.5 VL variants) ignore the num_labels kwarg
    # in their remote-code __init__, so the score layer ends up with the
    # wrong out_features.  Reinitialise when this happens.
    if hasattr(model, "score"):
        expected = model_cfg.num_labels
        actual = model.score.out_features if hasattr(model.score, "out_features") else None
        if actual is not None and actual != expected:
            import torch.nn as nn
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            model.score = nn.Linear(
                model.score.in_features, expected, bias=False,
            ).to(device=device, dtype=dtype)
            model.config.num_labels = expected
        elif hasattr(model.config, "num_labels") and model.config.num_labels != expected:
            model.config.num_labels = expected

    # With BitsAndBytes quantization, registered buffers (e.g. rotary
    # embedding inv_freq) can end up on CPU even when device_map puts
    # parameters on GPU.  Move any stray CPU buffers to the target device.
    if bnb_config and torch.cuda.is_available():
        target = torch.device(f"cuda:{local_rank()}")
        for module in model.modules():
            for name, buf in module.named_buffers(recurse=False):
                if buf is not None and buf.device.type == "cpu":
                    module.register_buffer(name, buf.to(target))

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
