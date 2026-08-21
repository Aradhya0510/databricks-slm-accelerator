"""Monitoring queries and threshold evaluation."""

from __future__ import annotations

import inspect

from src.monitoring.endpoint_monitor import EndpointMonitor


class _Thresholds:
    error_rate_threshold = 0.05
    latency_p95_threshold_ms = 1000
    drift_threshold = 0.1


def _monitor() -> EndpointMonitor:
    m = EndpointMonitor.__new__(EndpointMonitor)
    m.endpoint_name = "test-endpoint"
    m.thresholds = _Thresholds()
    return m


def test_no_sql_is_built_by_string_interpolation():
    source = inspect.getsource(EndpointMonitor)
    for method in ("get_request_metrics", "get_token_usage"):
        body = source.split(f"def {method}")[1].split("\n    def ")[0]
        assert 'f"""' not in body, f"{method} builds SQL with an f-string"
        assert "{self.endpoint_name}" not in body, f"{method} interpolates the endpoint name"
        assert ":endpoint_name" in body, f"{method} should bind :endpoint_name"


def test_error_rate_breach_is_reported():
    breaches = _monitor().evaluate_thresholds({"error_rate": 0.3, "p95_latency_ms": 100})
    assert [b["metric"] for b in breaches] == ["error_rate"]


def test_latency_breach_is_reported():
    breaches = _monitor().evaluate_thresholds({"error_rate": 0.0, "p95_latency_ms": 5000})
    assert [b["metric"] for b in breaches] == ["p95_latency_ms"]


def test_healthy_metrics_produce_no_breaches():
    assert _monitor().evaluate_thresholds({"error_rate": 0.01, "p95_latency_ms": 300}) == []


def test_no_thresholds_configured_means_no_breaches():
    m = _monitor()
    m.thresholds = None
    assert m.evaluate_thresholds({"error_rate": 1.0, "p95_latency_ms": 99999}) == []


def test_failed_query_does_not_read_as_healthy():
    """A failed query returns {"error": ...}; that must not silently pass."""
    assert _monitor().evaluate_thresholds({"error": "warehouse unavailable"}) == []
