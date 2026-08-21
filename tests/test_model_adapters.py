"""Model-family detection and LoRA target selection."""

from __future__ import annotations

import pytest

from src.model.adapters import (
    _FAMILY_CONFIGS,
    detect_model_family,
    get_model_family_config,
)


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("microsoft/Phi-3.5-mini-instruct", "phi-3"),
        ("microsoft/phi-4", "phi-4"),
        ("microsoft/phi-2", "phi-2"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "llama"),
        ("mistralai/Mistral-7B-v0.3", "mistral"),
        ("mistralai/Mixtral-8x7B", "mistral"),
        ("google/gemma-2-9b", "gemma"),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
    ],
)
def test_family_detection(model_name, expected):
    assert detect_model_family(model_name) == expected


def test_phi2_does_not_get_phi3_target_modules():
    """phi-2 predates phi-3's fused projections.

    Mapping phi-2 onto the phi-3 entry meant PEFT could not resolve a single
    target module, failing deep inside get_peft_model.
    """
    phi2 = get_model_family_config("microsoft/phi-2").lora_target_modules
    phi3 = get_model_family_config("microsoft/Phi-3.5-mini").lora_target_modules

    assert phi2 != phi3
    assert "Wqkv" in phi2
    assert "qkv_proj" not in phi2
    assert "qkv_proj" in phi3


def test_more_specific_patterns_win():
    """phi-4 must not be swallowed by the generic 'phi' pattern."""
    assert detect_model_family("microsoft/phi-4") == "phi-4"
    assert detect_model_family("microsoft/Phi-3-mini") == "phi-3"


def test_unknown_family_can_be_made_to_raise():
    """The silent llama fallback hands out projections Falcon does not have."""
    with pytest.raises(ValueError, match="Unrecognised model family"):
        detect_model_family("tiiuae/falcon-7b", strict=True)


def test_unknown_family_falls_back_with_a_warning(capsys):
    assert detect_model_family("tiiuae/falcon-7b") == "llama"
    assert "unrecognised model family" in capsys.readouterr().out.lower()


def test_every_family_declares_target_modules():
    for family, cfg in _FAMILY_CONFIGS.items():
        assert cfg.lora_target_modules, f"{family} has no LoRA target modules"
        assert cfg.padding_side in {"left", "right"}
