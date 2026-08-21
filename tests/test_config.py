"""Config schema: defaults, validation, and cross-field consistency."""

from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from src.config.schema import PipelineConfig, TrainingConfig, load_config

BASE = {
    "model": {"model_name": "microsoft/Phi-3.5-mini-instruct"},
    "data": {"train_data_path": "/Volumes/c/s/v/train.jsonl"},
}


def _write(tmp_path, **overrides):
    cfg = copy.deepcopy(BASE)
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_loads_a_minimal_config(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.model.task_type == "instruction_tuning"
    assert cfg.training.max_epochs > 0


def test_trust_remote_code_defaults_to_false(tmp_path):
    """It executes arbitrary Python from the model repo at load time.

    Models that need it opt in explicitly, so the choice is visible in the
    config that makes it rather than inherited by every fork.
    """
    assert load_config(_write(tmp_path)).model.trust_remote_code is False


def test_trust_remote_code_can_be_opted_into(tmp_path):
    cfg = load_config(_write(tmp_path, model={"trust_remote_code": True}))
    assert cfg.model.trust_remote_code is True


def test_unknown_task_type_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="task_type"):
        load_config(_write(tmp_path, model={"task_type": "summarisation"}))


def test_unknown_quantization_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="quantization"):
        load_config(_write(tmp_path, model={"quantization": "3bit"}))


def test_unknown_dpo_loss_type_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="dpo_loss_type"):
        load_config(_write(tmp_path, training={"dpo_loss_type": "made_up"}))


def test_invalid_monitor_mode_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="monitor_mode"):
        load_config(_write(tmp_path, training={"monitor_mode": "minimise"}))


def test_precision_defaults_to_auto(tmp_path):
    assert load_config(_write(tmp_path)).training.precision == "auto"


def test_legacy_bf16_flag_still_drives_precision():
    """Existing configs set bf16: true; that must keep working."""
    assert TrainingConfig(bf16=True).precision == "bf16"
    assert TrainingConfig(fp16=True).precision == "fp16"
    # An explicit precision wins over the legacy flags.
    assert TrainingConfig(precision="fp32", bf16=True).precision == "fp32"


def test_completion_only_loss_defaults_on(tmp_path):
    assert load_config(_write(tmp_path)).training.completion_only_loss is True


def test_seed_is_available(tmp_path):
    assert isinstance(load_config(_write(tmp_path)).training.seed, int)


def test_artifact_format_is_validated(tmp_path):
    with pytest.raises(ValidationError, match="artifact_format"):
        load_config(_write(tmp_path, mlflow={"artifact_format": "pickle"}))


def test_round_trips_through_dict(tmp_path):
    cfg = load_config(_write(tmp_path))
    again = PipelineConfig(**cfg.to_dict())
    assert again.model.model_name == cfg.model.model_name
