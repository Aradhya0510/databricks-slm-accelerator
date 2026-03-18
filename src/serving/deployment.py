"""Endpoint deployment: create, wait, test."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


def deploy_endpoint(
    endpoint_name: str,
    registered_model_name: str,
    model_version: str = "1",
    workload_size: str = "Small",
    workload_type: str = "GPU_SMALL",
    scale_to_zero: bool = True,
) -> Dict[str, Any]:
    """Create or update a Databricks Model Serving endpoint.

    Uses GPU workload type by default since SLMs require GPU for inference.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    w = WorkspaceClient()

    served_entity = ServedEntityInput(
        entity_name=registered_model_name,
        entity_version=model_version,
        workload_size=workload_size,
        workload_type=workload_type,
        scale_to_zero_enabled=scale_to_zero,
    )

    config = EndpointCoreConfigInput(served_entities=[served_entity])

    try:
        endpoint = w.serving_endpoints.create(name=endpoint_name, config=config)
        status = "created"
    except Exception as e:
        if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
            endpoint = w.serving_endpoints.update_config(
                name=endpoint_name, served_entities=[served_entity],
            )
            status = "updated"
        else:
            raise

    print(f"Endpoint '{endpoint_name}' {status}.")
    return {"endpoint_name": endpoint_name, "status": status}


def wait_for_ready(
    endpoint_name: str,
    timeout: int = 3600,
    poll_interval: int = 30,
) -> Dict[str, Any]:
    """Poll until the endpoint is in READY state."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    start = time.time()

    while time.time() - start < timeout:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state

        if state and state.ready == "READY":
            print(f"Endpoint '{endpoint_name}' is READY.")
            return {"endpoint_name": endpoint_name, "state": "READY"}

        config_update = state.config_update if state else None
        print(
            f"Endpoint state: ready={getattr(state, 'ready', '?')}, "
            f"config_update={config_update} — waiting {poll_interval}s..."
        )
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Endpoint '{endpoint_name}' did not become READY within {timeout}s"
    )


def test_endpoint(
    endpoint_name: str,
    test_prompt: str = "Explain machine learning in simple terms.",
) -> Dict[str, Any]:
    """Send a smoke-test request to a deployed endpoint."""
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()

    payload = {"dataframe_records": [{"prompt": test_prompt}]}

    response = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_records=payload["dataframe_records"],
    )

    print(f"Endpoint test response: {response}")
    return {"endpoint_name": endpoint_name, "response": str(response)}
