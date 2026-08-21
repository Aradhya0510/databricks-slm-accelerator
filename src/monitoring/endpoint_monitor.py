"""EndpointMonitor — health checks, request metrics, token-level stats.

Adapted for SLM endpoints: tracks token usage, generation latency, and
output length distributions in addition to standard request metrics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class EndpointMonitor:
    """Observability for a deployed Databricks Model Serving endpoint."""

    def __init__(self, endpoint_name: str, thresholds: Optional[Any] = None):
        """
        Args:
            endpoint_name: Serving endpoint to observe.
            thresholds: A ``MonitoringConfig`` (or anything exposing the same
                attributes).  When given, :meth:`generate_report` evaluates the
                collected metrics against it, so a scheduled job can fail on a
                bad endpoint instead of just printing numbers.
        """
        self.endpoint_name = endpoint_name
        self.thresholds = thresholds
        self._init_client()

    def _init_client(self) -> None:
        from databricks.sdk import WorkspaceClient
        self.w = WorkspaceClient()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    def _execute(self, sql: str, parameters: Optional[list] = None):
        """Run a statement with bound parameters.

        Interpolating the endpoint name into SQL made this injectable from
        anywhere the name reaches — the CLI, and the Streamlit monitoring
        form.  The statement runs under the caller's identity against a SQL
        warehouse, so the blast radius was everything that principal can read
        in Unity Catalog.
        """
        from databricks.sdk.service.sql import StatementParameterListItem

        params = None
        if parameters:
            params = [
                StatementParameterListItem(name=name, value=str(value), type=sql_type)
                for name, value, sql_type in parameters
            ]

        return self.w.statement_execution.execute_statement(
            warehouse_id=self._get_warehouse_id(),
            statement=sql,
            parameters=params,
        )

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
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Request metrics
    # ------------------------------------------------------------------
    def get_request_metrics(self, hours: int = 24) -> dict:
        """Query system tables for request-level metrics."""
        sql = """
        SELECT
            COUNT(*) AS total_requests,
            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) AS error_count,
            AVG(execution_time_ms) AS avg_latency_ms,
            PERCENTILE(execution_time_ms, 0.5) AS p50_latency_ms,
            PERCENTILE(execution_time_ms, 0.95) AS p95_latency_ms,
            PERCENTILE(execution_time_ms, 0.99) AS p99_latency_ms
        FROM system.serving.served_model_requests
        WHERE served_entity_name = :endpoint_name
          AND request_time >= CURRENT_TIMESTAMP - make_interval(0, 0, 0, 0, :hours, 0, 0)
        """

        try:
            from databricks.sdk.service.sql import StatementState

            result = self._execute(sql, [
                ("endpoint_name", self.endpoint_name, "STRING"),
                ("hours", int(hours), "INT"),
            ])

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
        sql = """
        SELECT
            response,
            execution_time_ms,
            request_time
        FROM system.serving.served_model_requests
        WHERE served_entity_name = :endpoint_name
          AND status_code = 200
          AND request_time >= CURRENT_TIMESTAMP - make_interval(0, 0, 0, 0, :hours, 0, 0)
        ORDER BY request_time DESC
        LIMIT 1000
        """

        try:
            result = self._execute(sql, [
                ("endpoint_name", self.endpoint_name, "STRING"),
                ("hours", int(hours), "INT"),
            ])

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
    def evaluate_thresholds(self, metrics: dict) -> list:
        """Compare collected metrics against the configured thresholds.

        Without this the thresholds in MonitoringConfig were pure decoration:
        nothing ever compared a measurement to them, so a scheduled monitor
        job could never report that anything was wrong.
        """
        if self.thresholds is None:
            return []

        breaches = []
        for key, attr in (
            ("error_rate", "error_rate_threshold"),
            ("p95_latency_ms", "latency_p95_threshold_ms"),
        ):
            value = metrics.get(key)
            limit = getattr(self.thresholds, attr, None)
            if value is not None and limit is not None and value > limit:
                breaches.append(
                    {"metric": key, "value": value, "threshold": limit}
                )
        return breaches

    def generate_report(self, output_path: Optional[str] = None) -> dict:
        request_metrics = self.get_request_metrics(hours=24)
        breaches = self.evaluate_thresholds(request_metrics)

        report = {
            "endpoint_name": self.endpoint_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "health": self.get_health(),
            "request_metrics_24h": request_metrics,
            "token_usage_24h": self.get_token_usage(hours=24),
            "threshold_breaches": breaches,
            "status": "BREACH" if breaches else "OK",
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
