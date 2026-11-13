"""The pipeline's stages, one function per job task.

The notebooks under ``jobs/`` are thin wrappers: they read job parameters from
widgets, call one of these, and hand the result to the next task. Keeping the
logic here keeps it importable and testable off Databricks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .config.schema import PipelineConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ThresholdBreach(RuntimeError):
    """A monitored endpoint breached a configured threshold."""


def resolve_config_path(config_path: Union[str, Path]) -> Path:
    """Resolve a config path, taking relative paths from the project root.

    Job parameters are written relative to the repository, but a notebook's
    working directory is wherever the notebook lives.
    """
    path = Path(config_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_pipeline_config(config_path: Union[str, Path]) -> PipelineConfig:
    return load_config(resolve_config_path(config_path))


def _require(value: Optional[str], what: str, config_field: str) -> str:
    if not value:
        raise ValueError(
            f"No {what} given: pass it as a job parameter or set {config_field} "
            f"in the pipeline config."
        )
    return value


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def train(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Fine-tune and log the model. Returns the MLflow run id and metrics."""
    from .engine import TrainingEngine

    result = TrainingEngine(load_pipeline_config(config_path)).train()
    if not result.run_id:
        raise RuntimeError("Training finished without an MLflow run id.")
    return {"run_id": result.run_id, "metrics": result.metrics}


def evaluate(
    config_path: Union[str, Path],
    run_id: str,
    eval_type: str = "all",
) -> Dict[str, Any]:
    """Evaluate the model a training run logged, and record the results on it."""
    import mlflow

    from .evaluation.engine import EvaluationEngine

    engine = EvaluationEngine(load_pipeline_config(config_path))
    results: Dict[str, Any] = {}
    metrics: Dict[str, float] = {}

    if eval_type in ("perplexity", "all"):
        results["perplexity"] = engine.evaluate_perplexity(run_id=run_id)
        metrics["eval_perplexity"] = results["perplexity"]["perplexity"]

    if eval_type in ("generation", "all"):
        results["generation"] = engine.evaluate_generation(run_id=run_id)
        for name, value in (results["generation"].get("rouge") or {}).items():
            metrics[f"eval_{name}"] = value

    if eval_type in ("benchmark", "all"):
        results["benchmark"] = engine.benchmark(run_id=run_id)
        bench = results["benchmark"]
        metrics["eval_tokens_per_second"] = bench["tokens_per_second"]
        metrics["eval_latency_p50_ms"] = bench["latency_per_generation_ms"]["p50"]

    # Without this the evaluation only ever reached stdout, which is lost the
    # moment the job run is cleaned up. The run's own experiment is adopted
    # first: a notebook defaults to its own, and MLflow refuses to reopen a
    # run under a different one.
    experiment_id = mlflow.MlflowClient().get_run(run_id).info.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    return {"metrics": metrics, "results": results}


def register(
    config_path: Union[str, Path],
    run_id: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a training run's model to Unity Catalog."""
    from .serving.registration import register_model

    config = load_pipeline_config(config_path)
    model_name = _require(
        model_name or config.serving.registered_model_name,
        "Unity Catalog model name", "serving.registered_model_name",
    )
    return register_model(
        run_id=run_id,
        registered_model_name=model_name,
        task_type=config.model.task_type,
    )


def deploy(
    config_path: Union[str, Path],
    model_version: str,
    model_name: Optional[str] = None,
    endpoint_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Point a Model Serving endpoint at a registered version and wait for it."""
    from .serving.deployment import deploy_endpoint, wait_for_ready

    config = load_pipeline_config(config_path)
    model_name = _require(
        model_name or config.serving.registered_model_name,
        "Unity Catalog model name", "serving.registered_model_name",
    )
    endpoint_name = _require(
        endpoint_name or config.serving.endpoint_name,
        "serving endpoint name", "serving.endpoint_name",
    )

    deploy_endpoint(
        endpoint_name=endpoint_name,
        registered_model_name=model_name,
        model_version=str(model_version),
        workload_size=config.serving.workload_size,
        workload_type=config.serving.workload_type,
        scale_to_zero=config.serving.scale_to_zero,
    )
    return wait_for_ready(endpoint_name)


def monitor(
    config_path: Union[str, Path],
    endpoint_name: Optional[str] = None,
    output_path: Optional[str] = None,
    fail_on_breach: bool = False,
) -> Dict[str, Any]:
    """Report on an endpoint, raising on a breach so a scheduled job alerts."""
    from .monitoring.endpoint_monitor import EndpointMonitor

    config = load_pipeline_config(config_path)
    endpoint_name = _require(
        endpoint_name or config.serving.endpoint_name,
        "serving endpoint name", "serving.endpoint_name",
    )

    report = EndpointMonitor(endpoint_name, thresholds=config.monitoring).generate_report(
        output_path=output_path,
    )

    breaches = report.get("threshold_breaches", [])
    if breaches and fail_on_breach:
        details = "; ".join(
            f"{b['metric']}={b['value']} exceeds {b['threshold']}" for b in breaches
        )
        raise ThresholdBreach(f"Endpoint '{endpoint_name}' breached thresholds: {details}")
    return report
