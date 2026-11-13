"""Pipeline stages and the engine's choice of how to use the attached GPUs."""

from __future__ import annotations

import pytest

from src import pipeline
from src.engine import TrainingEngine, TrainingResult
from src.engine import engine as engine_module

MINIMAL_CONFIG = """
model:
  model_name: sshleifer/tiny-gpt2
data:
  train_data_path: /Volumes/main/slm/data/train.jsonl
"""


@pytest.fixture
def minimal_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_CONFIG)
    return path


def test_relative_config_paths_resolve_from_the_project_root():
    """A notebook's working directory is jobs/, not the repository root."""
    resolved = pipeline.resolve_config_path("configs/sft_phi3_config.yaml")
    assert resolved == pipeline.PROJECT_ROOT / "configs" / "sft_phi3_config.yaml"
    assert resolved.is_file()


def test_absolute_config_paths_are_kept(minimal_config):
    assert pipeline.resolve_config_path(str(minimal_config)) == minimal_config


def test_register_names_the_missing_model_name(minimal_config):
    with pytest.raises(ValueError, match="serving.registered_model_name"):
        pipeline.register(minimal_config, run_id="abc")


def test_deploy_names_the_missing_endpoint(minimal_config):
    with pytest.raises(ValueError, match="serving.endpoint_name"):
        pipeline.deploy(minimal_config, model_version="1", model_name="main.slm.m")


class _FakeMonitor:
    breaches: list = []

    def __init__(self, endpoint_name, thresholds=None):
        self.endpoint_name = endpoint_name

    def generate_report(self, output_path=None):
        return {
            "endpoint_name": self.endpoint_name,
            "threshold_breaches": self.breaches,
            "status": "BREACH" if self.breaches else "OK",
        }


@pytest.fixture
def fake_monitor(monkeypatch):
    monkeypatch.setattr(
        "src.monitoring.endpoint_monitor.EndpointMonitor", _FakeMonitor,
    )
    return _FakeMonitor


def test_monitor_raises_on_breach_so_the_job_alerts(minimal_config, fake_monitor, monkeypatch):
    monkeypatch.setattr(
        fake_monitor, "breaches",
        [{"metric": "error_rate", "value": 0.2, "threshold": 0.05}],
    )
    with pytest.raises(pipeline.ThresholdBreach, match="error_rate"):
        pipeline.monitor(minimal_config, endpoint_name="ep", fail_on_breach=True)


def test_monitor_reports_a_breach_without_failing_when_asked(minimal_config, fake_monitor, monkeypatch):
    monkeypatch.setattr(
        fake_monitor, "breaches",
        [{"metric": "error_rate", "value": 0.2, "threshold": 0.05}],
    )
    report = pipeline.monitor(minimal_config, endpoint_name="ep", fail_on_breach=False)
    assert report["status"] == "BREACH"


# ---------------------------------------------------------------------------
# GPU dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def engine(minimal_config, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)

    eng = TrainingEngine(pipeline.load_pipeline_config(minimal_config))
    calls = []
    monkeypatch.setattr(eng, "_train_fn", lambda: calls.append("single") or TrainingResult("r1"))
    monkeypatch.setattr(
        eng, "_train_distributed", lambda n: calls.append(f"distributed:{n}") or TrainingResult("r8"),
    )
    eng.calls = calls
    return eng


def test_one_gpu_trains_in_process(engine, monkeypatch):
    monkeypatch.setattr(engine_module, "get_gpu_count", lambda: 1)
    assert engine.train().run_id == "r1"
    assert engine.calls == ["single"]


def test_a_multi_gpu_node_fans_out_one_process_per_gpu(engine, monkeypatch):
    """One process seeing eight GPUs would give DataParallel, not DDP."""
    monkeypatch.setattr(engine_module, "get_gpu_count", lambda: 8)
    assert engine.train().run_id == "r8"
    assert engine.calls == ["distributed:8"]


def test_a_launched_worker_never_fans_out_again(engine, monkeypatch):
    monkeypatch.setattr(engine_module, "get_gpu_count", lambda: 8)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    engine.train()
    assert engine.calls == ["single"]


def test_train_stage_refuses_a_run_without_an_id(minimal_config, monkeypatch):
    """Every later task finds the model through this id."""
    monkeypatch.setattr(TrainingEngine, "train", lambda self: TrainingResult(run_id=None))
    with pytest.raises(RuntimeError, match="run id"):
        pipeline.train(minimal_config)
