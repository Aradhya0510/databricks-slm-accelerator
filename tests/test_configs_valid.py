"""Every shipped config must load, and must not carry inert settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.schema import load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_there_are_configs_to_check():
    assert CONFIGS, f"no configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads_and_validates(path):
    cfg = load_config(path)
    assert cfg.model.model_name
    assert cfg.model.task_type in {"instruction_tuning", "dpo", "text_classification"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_checkpoints_are_not_written_directly_to_a_volume(path):
    """UC Volumes are a FUSE mount with poor rename/random-write behaviour.

    Training writes to local disk; volume_checkpoint_dir mirrors afterwards.
    """
    cfg = load_config(path)
    assert not cfg.training.checkpoint_dir.startswith("/Volumes/"), (
        f"{path.name} writes checkpoints straight to a Volume."
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_remote_code_is_only_enabled_where_it_is_needed(path):
    """trust_remote_code runs arbitrary Python from the model repo.

    Only families that genuinely ship a custom config class should opt in.
    """
    cfg = load_config(path)
    if cfg.model.trust_remote_code:
        assert "phi" in cfg.model.model_name.lower(), (
            f"{path.name} enables trust_remote_code for "
            f"'{cfg.model.model_name}', which does not need it."
        )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_monitor_metric_matches_monitor_mode(path):
    """A loss tracked with mode 'max' would select the worst checkpoint."""
    cfg = load_config(path)
    metric = cfg.training.monitor_metric
    if "loss" in metric:
        assert cfg.training.monitor_mode == "min", (
            f"{path.name} monitors '{metric}' with mode '{cfg.training.monitor_mode}'"
        )
    if any(k in metric for k in ("accuracy", "f1")):
        assert cfg.training.monitor_mode == "max", (
            f"{path.name} monitors '{metric}' with mode '{cfg.training.monitor_mode}'"
        )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_dpo_configs_use_the_dpo_task(path):
    if "dpo" in path.name:
        assert load_config(path).model.task_type == "dpo"
