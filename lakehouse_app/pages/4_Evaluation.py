"""Evaluation — Inspect metrics, view generations, and compare runs."""

import streamlit as st
import pandas as pd

from components.theme import inject_theme, page_header, section_title, metric_card
from components.metrics_display import MetricsDisplay
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

inject_theme()
StateManager.initialize()

page_header(
    "Evaluation",
    "Inspect training metrics, view generation samples, and compare model runs",
)


@st.cache_resource
def _get_client():
    return DatabricksJobClient()


tab_metrics, tab_compare = st.tabs(["Run Metrics", "Compare Runs"])

with tab_metrics:
    config = StateManager.get_current_config()
    experiment_name = st.text_input(
        "Experiment Name",
        value=config.get("mlflow", {}).get("experiment_name", "") if config else "",
    )

    if st.button("Load Runs", type="primary") and experiment_name:
        try:
            client = _get_client()
            runs = client.get_mlflow_runs(experiment_name)
            st.session_state["eval_runs"] = runs
        except Exception as e:
            st.error(str(e))

    runs = st.session_state.get("eval_runs", [])
    if runs:
        section_title("Runs")
        run_options = {f"{r['run_name']} ({r['run_id'][:8]})": r for r in runs}
        selected = st.selectbox("Select Run", options=list(run_options.keys()))
        run = run_options[selected]

        section_title("Metrics")
        metrics = run.get("metrics", {})
        if metrics:
            MetricsDisplay.display_metrics_grid(metrics)
        else:
            st.info("No metrics recorded for this run.")

        section_title("Parameters")
        params = run.get("params", {})
        if params:
            df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
            st.dataframe(df, use_container_width=True)

        section_title("Loss Curve")
        metric_key = st.selectbox(
            "Metric to plot",
            options=list(metrics.keys()) if metrics else ["eval_loss"],
        )
        if st.button("Plot Metric"):
            try:
                client = _get_client()
                history = client.get_run_metrics_history(run["run_id"], metric_key)
                if history:
                    df = pd.DataFrame(history)
                    import plotly.express as px

                    fig = px.line(df, x="step", y="value", title=metric_key)
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Figtree, sans-serif", color="#8A91A8"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No history found.")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Load runs from an MLflow experiment to begin.")

with tab_compare:
    runs = st.session_state.get("eval_runs", [])
    if len(runs) < 2:
        st.info("Load at least 2 runs in the Metrics tab to compare.")
    else:
        section_title("Select Runs to Compare")
        run_names = [f"{r['run_name']} ({r['run_id'][:8]})" for r in runs]
        selected_names = st.multiselect(
            "Runs", options=run_names, default=run_names[:2]
        )
        selected_runs = [
            r for r, n in zip(runs, run_names) if n in selected_names
        ]

        if selected_runs:
            all_metrics = set()
            for r in selected_runs:
                all_metrics.update(r.get("metrics", {}).keys())
            all_metrics = sorted(all_metrics)

            rows = []
            for r in selected_runs:
                row = {"Run": r["run_name"]}
                for m in all_metrics:
                    row[m] = r.get("metrics", {}).get(m, None)
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            if all_metrics:
                compare_metric = st.selectbox("Compare metric", all_metrics)
                import plotly.express as px

                vals = [
                    {
                        "Run": r["run_name"],
                        "Value": r.get("metrics", {}).get(compare_metric),
                    }
                    for r in selected_runs
                ]
                vals = [v for v in vals if v["Value"] is not None]
                if vals:
                    fig = px.bar(
                        pd.DataFrame(vals),
                        x="Run",
                        y="Value",
                        title=compare_metric,
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Figtree, sans-serif", color="#8A91A8"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
