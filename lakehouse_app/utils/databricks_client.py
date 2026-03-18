"""Databricks Client for Jobs API, MLflow, and Model Serving."""

import io
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs
import mlflow
from mlflow.tracking import MlflowClient


class DatabricksJobClient:
    """Client for Databricks Jobs API, MLflow, and Serving."""

    def __init__(self):
        self.workspace_client = WorkspaceClient()
        self.mlflow_client = MlflowClient()

    def create_training_job(
        self,
        job_name: str,
        config_path: str,
        project_path: str,
        num_gpus: Optional[int] = None,
        cluster_config: Optional[Dict[str, Any]] = None,
        existing_cluster_id: Optional[str] = None,
    ) -> str:
        python_params = ["--config_path", config_path]
        if num_gpus is not None:
            python_params.extend(["--num_gpus", str(num_gpus)])

        cluster_kwargs = {}
        if existing_cluster_id:
            cluster_kwargs["existing_cluster_id"] = existing_cluster_id
        else:
            if cluster_config is None:
                cluster_config = {
                    "spark_version": "17.3.x-gpu-ml-scala2.12",
                    "node_type_id": "g5.4xlarge",
                    "num_workers": 0,
                    "data_security_mode": "SINGLE_USER",
                }
            dsm = cluster_config.pop("data_security_mode", None)
            if isinstance(dsm, str):
                dsm = compute.DataSecurityMode(dsm)
            spec = compute.ClusterSpec(**cluster_config)
            if dsm is not None:
                spec.data_security_mode = dsm
            cluster_kwargs["new_cluster"] = spec

        runtime_libs = [
            compute.Library(pypi=compute.PythonPyPiLibrary(package="trl>=0.12")),
            compute.Library(pypi=compute.PythonPyPiLibrary(package="peft>=0.10")),
            compute.Library(pypi=compute.PythonPyPiLibrary(package="bitsandbytes>=0.43")),
        ]

        task = jobs.Task(
            task_key="train_model",
            description="Fine-tune SLM with TRL",
            **cluster_kwargs,
            spark_python_task=jobs.SparkPythonTask(
                python_file=f"{project_path}/jobs/train.py",
                parameters=python_params,
            ),
            libraries=runtime_libs,
            timeout_seconds=0,
        )

        created_job = self.workspace_client.jobs.create(
            name=job_name,
            tasks=[task],
            max_concurrent_runs=1,
        )
        return str(created_job.job_id)

    def run_job(self, job_id: str) -> str:
        run = self.workspace_client.jobs.run_now(job_id=int(job_id))
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
        from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

        try:
            served_entity = ServedEntityInput(
                entity_name=model_name, entity_version=model_version,
                workload_size=workload_size, workload_type=workload_type,
                scale_to_zero_enabled=scale_to_zero,
            )
            try:
                self.workspace_client.serving_endpoints.get(endpoint_name)
                self.workspace_client.serving_endpoints.update_config(
                    name=endpoint_name, served_entities=[served_entity],
                )
                status = "updated"
            except Exception:
                config = EndpointCoreConfigInput(served_entities=[served_entity])
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

    def list_clusters(self) -> List[Dict[str, Any]]:
        try:
            clusters = self.workspace_client.clusters.list()
            return [
                {
                    "cluster_id": c.cluster_id,
                    "cluster_name": c.cluster_name,
                    "state": str(c.state),
                    "node_type_id": c.node_type_id,
                }
                for c in clusters
            ]
        except Exception:
            return []
