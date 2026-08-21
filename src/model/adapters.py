"""Model-family-specific configuration.

SLM model-family differences are captured as lightweight dataclass configs.
Most adaptation is handled by the tokenizer's chat_template; this module
captures the remaining quirks: LoRA target modules, padding side, special
token fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelFamilyConfig:
    """Declarative config for a model family's known quirks."""

    family: str
    lora_target_modules: List[str]
    padding_side: str = "right"
    # Some models need the EOS token set as the pad token
    use_eos_as_pad: bool = False
    # Some models don't ship a default chat template
    default_system_prompt: Optional[str] = None


# ---------------------------------------------------------------------------
# Registry of known model families
# ---------------------------------------------------------------------------

_FAMILY_CONFIGS: Dict[str, ModelFamilyConfig] = {
    "phi-2": ModelFamilyConfig(
        family="phi-2",
        # phi-2 predates the fused projections phi-3 uses; mapping it onto the
        # phi-3 entry meant PEFT could not resolve any target module.
        lora_target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        padding_side="right",
        use_eos_as_pad=True,
    ),
    "phi-3": ModelFamilyConfig(
        family="phi-3",
        lora_target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        padding_side="right",
        use_eos_as_pad=True,
    ),
    "phi-4": ModelFamilyConfig(
        family="phi-4",
        lora_target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        padding_side="right",
        use_eos_as_pad=True,
    ),
    "llama": ModelFamilyConfig(
        family="llama",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        padding_side="right",
        use_eos_as_pad=True,
    ),
    "mistral": ModelFamilyConfig(
        family="mistral",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        padding_side="right",
        use_eos_as_pad=True,
    ),
    "gemma": ModelFamilyConfig(
        family="gemma",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        padding_side="right",
        use_eos_as_pad=True,
    ),
    "qwen": ModelFamilyConfig(
        family="qwen",
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        padding_side="right",
        use_eos_as_pad=False,
    ),
}


def detect_model_family(model_name: str, strict: bool = False) -> str:
    """Infer the model family from a HuggingFace model name/path.

    With ``strict=True`` an unrecognised name raises instead of falling back
    to llama.  The fallback silently hands out llama's projection names, which
    do not exist in Falcon, GPT-2 or MPT, so the failure surfaced deep inside
    PEFT's target resolution rather than here.
    """
    name_lower = model_name.lower()

    patterns = [
        ("phi-4", "phi-4"),
        ("phi-3", "phi-3"), ("phi-2", "phi-2"), ("phi", "phi-3"),
        ("llama", "llama"),
        ("mistral", "mistral"), ("mixtral", "mistral"),
        ("gemma", "gemma"),
        ("qwen", "qwen"),
    ]
    for pattern, family in patterns:
        if pattern in name_lower:
            return family

    if strict:
        raise ValueError(
            f"Unrecognised model family for '{model_name}'. Known families: "
            f"{sorted(_FAMILY_CONFIGS)}. Either add an entry to _FAMILY_CONFIGS "
            f"or set model.lora_target_modules explicitly in your config."
        )

    print(
        f"Warning: unrecognised model family for '{model_name}'; assuming "
        f"llama-style projections. If PEFT cannot find the target modules, "
        f"set model.lora_target_modules explicitly."
    )
    return "llama"


def get_model_family_config(
    model_name: str, strict: bool = False,
) -> ModelFamilyConfig:
    """Return the ``ModelFamilyConfig`` for a model, auto-detecting the family."""
    family = detect_model_family(model_name, strict=strict)
    return _FAMILY_CONFIGS[family]
