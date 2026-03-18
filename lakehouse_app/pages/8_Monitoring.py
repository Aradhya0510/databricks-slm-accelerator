"""Monitoring — Track endpoint health, request metrics, and token throughput."""

import streamlit as st
import pandas as pd

from components.theme import inject_theme, page_header, section_title, metric_card, status_pill
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

inject_theme()
StateManager.initialize()

page_header("Monitoring", "Track endpoint health, request metrics, and token throughput")


@st.cache_resource
def _get_client():
    return DatabricksJobClient()


tab_health, tab_settings = st.tabs(["Endpoint Health", "Settings"])

with tab_health:
    section_title("Select Endpoint")
    endpoints = StateManager.get("endpoints", [])
    endpoint_names = [ep.get("endpoint_name", "") for ep in endpoints]
    if endpoint_names:
        selected_endpoint = st.selectbox("Endpoint", options=endpoint_names)
    else:
        selected_endpoint = st.text_input("Endpoint Name", value="")

    if st.button("Check Health", type="primary") and selected_endpoint:
        try:
            client = _get_client()
            status = client.get_endpoint_status(selected_endpoint)

            section_title("Status")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Endpoint:** `{status['endpoint_name']}`")
                st.markdown(f"**Ready:** {status_pill(status.get('ready', 'UNKNOWN'))}", unsafe_allow_html=True)
            with c2:
                for model in status.get("served_models", []):
                    st.markdown(f"**Model:** {model.get('entity_name', 'N/A')} v{model.get('entity_version', '?')}")
                    st.markdown(f"**GPU:** {model.get('workload_type', 'N/A')}")

        except Exception as e:
            st.error(str(e))

    section_title("Request Metrics (System Tables)")
    st.markdown(
        "Query Databricks system tables for serving endpoint metrics. "
        "Paste a SQL query or use the presets below."
    )

    preset = st.selectbox("Preset Query", options=[
        "Request count (last 24h)",
        "Avg latency (last 24h)",
        "Error rate (last 24h)",
    ])

    endpoint_for_query = selected_endpoint or "<endpoint_name>"
    presets = {
        "Request count (last 24h)": (
            f"SELECT DATE_TRUNC('hour', request_time) AS hour, COUNT(*) AS requests "
            f"FROM system.serving.served_model_requests "
            f"WHERE served_entity_name = '{endpoint_for_query}' "
            f"AND request_time > CURRENT_TIMESTAMP - INTERVAL 24 HOURS "
            f"GROUP BY 1 ORDER BY 1"
        ),
        "Avg latency (last 24h)": (
            f"SELECT DATE_TRUNC('hour', request_time) AS hour, "
            f"AVG(execution_time_ms) AS avg_latency_ms "
            f"FROM system.serving.served_model_requests "
            f"WHERE served_entity_name = '{endpoint_for_query}' "
            f"AND request_time > CURRENT_TIMESTAMP - INTERVAL 24 HOURS "
            f"GROUP BY 1 ORDER BY 1"
        ),
        "Error rate (last 24h)": (
            f"SELECT DATE_TRUNC('hour', request_time) AS hour, "
            f"SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) / COUNT(*) AS error_rate "
            f"FROM system.serving.served_model_requests "
            f"WHERE served_entity_name = '{endpoint_for_query}' "
            f"AND request_time > CURRENT_TIMESTAMP - INTERVAL 24 HOURS "
            f"GROUP BY 1 ORDER BY 1"
        ),
    }

    query = st.text_area("SQL Query", value=presets.get(preset, ""), height=120)
    st.info("Run this query in the Databricks SQL Editor or via the SDK to see metrics.")

with tab_settings:
    section_title("Monitoring Thresholds")
    config = StateManager.get_current_config()
    mon = config.get("monitoring", {}) if config else {}

    c1, c2, c3 = st.columns(3)
    with c1:
        drift_threshold = st.number_input(
            "Drift Threshold", min_value=0.0, max_value=1.0,
            value=mon.get("drift_threshold", 0.1), step=0.01,
        )
    with c2:
        error_threshold = st.number_input(
            "Error Rate Threshold", min_value=0.0, max_value=1.0,
            value=mon.get("error_rate_threshold", 0.05), step=0.01,
        )
    with c3:
        latency_threshold = st.number_input(
            "Latency P95 Threshold (ms)", min_value=100, max_value=30000,
            value=mon.get("latency_p95_threshold_ms", 2000), step=100,
        )

    section_title("User Preferences")
    prefs = StateManager.get_user_preferences()
    c1, c2 = st.columns(2)
    with c1:
        catalog = st.text_input("Default Catalog", value=prefs.get("default_catalog", "main"))
        schema = st.text_input("Default Schema", value=prefs.get("default_schema", "slm_models"))
    with c2:
        volume = st.text_input("Default Volume", value=prefs.get("default_volume", "slm_data"))
        email = st.text_input("Workspace Email", value=prefs.get("workspace_email") or "")

    if st.button("Save Preferences"):
        StateManager.set_user_preferences({
            "default_catalog": catalog,
            "default_schema": schema,
            "default_volume": volume,
            "workspace_email": email or None,
        })
        st.success("Preferences saved.")
