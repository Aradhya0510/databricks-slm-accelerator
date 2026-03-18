"""EndpointMonitor — health checks, request metrics, token-level stats.

Adapted for SLM endpoints: tracks token usage, generation latency, and
output length distributions in addition to standard request metrics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


class EndpointMonitor:
    """Observability for a deployed Databricks Model Serving endpoint."""

    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self._init_client()

    def _init_client(self) -> None:
        from databricks.sdk import WorkspaceClient
        self.w = WorkspaceClient()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    def get_health(self) -> dict:
        endpoint = self.w.serving_endpoints.get(self.endpoint_name)
        state = endpoint.state

        served_models = []
        if endpoint.config and endpoint.config.served_entities:
            for entity in endpoint.config.served_entities:
                served_models.append({
                    "entity_name": entity.entity_name,
                    "entity_version": entity.entity_version,
                    "workload_size": entity.workload_size,
                    "workload_type": getattr(entity, "workload_type", None),
                    "scale_to_zero": entity.scale_to_zero_enabled,
                })

        return {
            "endpoint_name": self.endpoint_name,
            "ready": getattr(state, "ready", None),
            "config_update": getattr(state, "config_update", None),
            "served_models": served_models,
            "checked_at": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Request metrics
    # ------------------------------------------------------------------
    def get_request_metrics(self, hours: int = 24) -> dict:
        """Query system tables for request-level metrics."""
        sql = f"""
        SELECT
            COUNT(*) AS total_requests,
            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS error_count,
            AVG(execution_time_ms) AS avg_latency_ms,
            PERCENTILE(execution_time_ms, 0.5) AS p50_latency_ms,
            PERCENTILE(execution_time_ms, 0.95) AS p95_latency_ms,
            PERCENTILE(execution_time_ms, 0.99) AS p99_latency_ms
        FROM system.serving.served_model_requests
        WHERE served_entity_name = '{self.endpoint_name}'
          AND request_time >= CURRENT_TIMESTAMP - INTERVAL {hours} HOURS
        """

        try:
            from databricks.sdk.service.sql import StatementState

            result = self.w.statement_execution.execute_statement(
                warehouse_id=self._get_warehouse_id(),
                statement=sql,
            )

            if result.status and result.status.state == StatementState.SUCCEEDED:
                rows = result.result.data_array if result.result else []
                if rows and rows[0]:
                    row = rows[0]
                    total = int(row[0] or 0)
                    errors = int(row[1] or 0)
                    return {
                        "total_requests": total,
                        "error_count": errors,
                        "error_rate": errors / total if total > 0 else 0.0,
                        "avg_latency_ms": float(row[2] or 0),
                        "p50_latency_ms": float(row[3] or 0),
                        "p95_latency_ms": float(row[4] or 0),
                        "p99_latency_ms": float(row[5] or 0),
                        "hours": hours,
                    }
            return {"total_requests": 0, "hours": hours, "note": "No data found"}
        except Exception as e:
            return {"error": str(e), "hours": hours}

    # ------------------------------------------------------------------
    # Token usage stats (SLM-specific)
    # ------------------------------------------------------------------
    def get_token_usage(self, hours: int = 24) -> dict:
        """Approximate token usage from response payloads in system tables."""
        sql = f"""
        SELECT
            response,
            execution_time_ms,
            request_time
        FROM system.serving.served_model_requests
        WHERE served_entity_name = '{self.endpoint_name}'
          AND status_code = 200
          AND request_time >= CURRENT_TIMESTAMP - INTERVAL {hours} HOURS
        ORDER BY request_time DESC
        LIMIT 1000
        """

        try:
            result = self.w.statement_execution.execute_statement(
                warehouse_id=self._get_warehouse_id(),
                statement=sql,
            )

            response_lengths = []
            latencies = []

            if result.result and result.result.data_array:
                for row in result.result.data_array:
                    try:
                        resp = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                        if isinstance(resp, list):
                            for r in resp:
                                response_text = r.get("response", "")
                                # Rough token estimate: ~4 chars per token
                                response_lengths.append(len(response_text) // 4)
                        latencies.append(float(row[1] or 0))
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        continue

            if response_lengths:
                return {
                    "num_responses_sampled": len(response_lengths),
                    "avg_output_tokens_approx": sum(response_lengths) / len(response_lengths),
                    "max_output_tokens_approx": max(response_lengths),
                    "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                    "tokens_per_second_approx": (
                        sum(response_lengths) / (sum(latencies) / 1000)
                        if latencies and sum(latencies) > 0 else 0
                    ),
                    "hours": hours,
                }
            return {"num_responses_sampled": 0, "hours": hours, "note": "No data found"}
        except Exception as e:
            return {"error": str(e), "hours": hours}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def generate_report(self, output_path: Optional[str] = None) -> dict:
        report = {
            "endpoint_name": self.endpoint_name,
            "generated_at": datetime.utcnow().isoformat(),
            "health": self.get_health(),
            "request_metrics_24h": self.get_request_metrics(hours=24),
            "token_usage_24h": self.get_token_usage(hours=24),
        }

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to {output_path}")

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_warehouse_id(self) -> str:
        warehouses = list(self.w.warehouses.list())
        if not warehouses:
            raise RuntimeError("No SQL warehouses found")
        for wh in warehouses:
            if wh.state and wh.state.value == "RUNNING":
                return wh.id
        return warehouses[0].id
