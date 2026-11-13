"""Databricks Client for Jobs API, MLflow, and Model Serving."""

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.tracking import MlflowClient

# The job the bundle deploys; its name is prefixed with the target, e.g. "[dev]".
PIPELINE_JOB_NAME = "SLM fine-tuning pipeline"


class DatabricksJobClient:
    """Client for Databricks Jobs API, MLflow, and Serving."""

    def __init__(self):
        self.workspace_client = WorkspaceClient()
        self.mlflow_client = MlflowClient()

    def list_pipeline_jobs(self) -> List[Dict[str, Any]]:
        """The fine-tuning pipeline jobs deployed by the bundle, one per target.

        Matched by substring: the API's ``name`` filter is exact, and bundle
        targets prefix the name (development mode adds the user as well).
        """
        return [
            {"job_id": str(job.job_id), "name": job.settings.name}
            for job in self.workspace_client.jobs.list()
            if job.settings and PIPELINE_JOB_NAME in (job.settings.name or "")
        ]

    def run_pipeline(
        self,
        job_id: str,
        config_path: str,
        model_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> str:
        """Start a pipeline run, overriding its job parameters for this run only.

        Training runs on AI Runtime inside the deployed job, so the app no
        longer defines any compute of its own.
        """
        params = {"config_path": config_path}
        if model_name:
            params["model_name"] = model_name
        if endpoint_name:
            params["endpoint_name"] = endpoint_name

        run = self.workspace_client.jobs.run_now(job_id=int(job_id), job_parameters=params)
        return str(run.run_id)

    def get_job_status(self, run_id: str) -> Dict[str, Any]:
        run = self.workspace_client.jobs.get_run(run_id=int(run_id))
        state = run.state
        start_time = datetime.fromtimestamp(run.start_time / 1000) if run.start_time else None
        end_time = datetime.fromtimestamp(run.end_time / 1000) if run.end_time else None
        duration = (end_time - start_time).total_seconds() if start_time and end_time else None

        return {
            "run_id": run_id,
            "life_cycle_state": str(state.life_cycle_state) if state else "UNKNOWN",
            "result_state": str(state.result_state) if state and state.result_state else "UNKNOWN",
            "state_message": state.state_message if state else "",
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "run_page_url": run.run_page_url,
        }

    def cancel_job(self, run_id: str) -> bool:
        try:
            self.workspace_client.jobs.cancel_run(run_id=int(run_id))
            return True
        except Exception:
            return False

    def get_mlflow_runs(self, experiment_name: str, max_results: int = 100) -> List[Dict[str, Any]]:
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return []
            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )
            return [
                {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
                    "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                }
                for run in runs
            ]
        except Exception:
            return []

    def get_run_metrics_history(self, run_id: str, metric_key: str) -> List[Dict[str, Any]]:
        try:
            history = self.mlflow_client.get_metric_history(run_id, metric_key)
            return [
                {"step": m.step, "value": m.value, "timestamp": datetime.fromtimestamp(m.timestamp / 1000)}
                for m in history
            ]
        except Exception:
            return []

    def get_registered_models(self, max_results: int = 100) -> List[Dict[str, Any]]:
        try:
            models = self.mlflow_client.search_registered_models(max_results=max_results)
            return [
                {
                    "name": model.name,
                    "creation_timestamp": datetime.fromtimestamp(model.creation_timestamp / 1000) if model.creation_timestamp else None,
                    "latest_versions": [
                        {"version": v.version, "aliases": getattr(v, "aliases", []) or [], "run_id": v.run_id}
                        for v in (model.latest_versions or [])
                    ],
                }
                for model in models
            ]
        except Exception:
            return []

    def create_model_serving_endpoint(
        self, endpoint_name: str, model_name: str, model_version: str,
        workload_size: str = "Small", workload_type: str = "GPU_SMALL",
        scale_to_zero: bool = True,
    ) -> Dict[str, Any]:
        from databricks.sdk.service.serving import (
            EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType,
        )

        try:
            served_entity = ServedEntityInput(
                entity_name=model_name, entity_version=model_version,
                workload_size=workload_size,
                workload_type=ServingModelWorkloadType(workload_type),
                scale_to_zero_enabled=scale_to_zero,
            )
            try:
                self.workspace_client.serving_endpoints.get(endpoint_name)
                self.workspace_client.serving_endpoints.update_config(
                    name=endpoint_name, served_entities=[served_entity],
                )
                status = "updated"
            except Exception:
                config = EndpointCoreConfigInput(name=endpoint_name, served_entities=[served_entity])
                self.workspace_client.serving_endpoints.create(name=endpoint_name, config=config)
                status = "created"
            return {"endpoint_name": endpoint_name, "status": status, "model_name": model_name}
        except Exception as e:
            return {"endpoint_name": endpoint_name, "status": "error", "error": str(e)}

    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        try:
            endpoint = self.workspace_client.serving_endpoints.get(endpoint_name)
            state = endpoint.state
            served_models = []
            if endpoint.config and endpoint.config.served_entities:
                for entity in endpoint.config.served_entities:
                    served_models.append({
                        "entity_name": entity.entity_name,
                        "entity_version": entity.entity_version,
                        "workload_size": getattr(entity, "workload_size", "N/A"),
                        "workload_type": getattr(entity, "workload_type", "N/A"),
                    })
            return {
                "endpoint_name": endpoint_name,
                "ready": str(state.ready).split(".")[-1] if state else "UNKNOWN",
                "served_models": served_models,
            }
        except Exception as e:
            return {"endpoint_name": endpoint_name, "ready": "NOT_FOUND", "error": str(e)}

    def query_endpoint(self, endpoint_name: str, prompt: str) -> Dict[str, Any]:
        try:
            resp = self.workspace_client.serving_endpoints.query(
                name=endpoint_name,
                dataframe_records=[{"prompt": prompt}],
            )
            return {"predictions": resp.predictions if hasattr(resp, "predictions") else resp.as_dict()}
        except Exception as e:
            return {"error": str(e)}
