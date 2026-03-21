"""Training — Launch jobs and monitor training progress."""

import streamlit as st
import pandas as pd

from components.theme import (
    inject_theme,
    page_header,
    section_title,
    metric_card,
    status_badge,
    progress_bar,
)
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

inject_theme()
StateManager.initialize()

page_header("Training", "Launch fine-tuning jobs and monitor training progress")

# ── Three metric cards at top ──────────────────────────────────────────────────
training_history = StateManager.get("training_history", [])
active_run = StateManager.get_active_training_run()

c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Total Runs", str(len(training_history)))
with c2:
    badge = status_badge("RUNNING") if active_run else status_badge("QUEUED")
    st.markdown(
        f'<div class="ds-metric">'
        f'<div class="ds-metric-label">CURRENT STATUS</div>'
        f'<div style="margin-top:6px;">{badge}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
with c3:
    last_model = training_history[0].get("model", "—").split("/")[-1] if training_history else "—"
    metric_card("Last Model", last_model)


@st.cache_resource
def _get_client():
    return DatabricksJobClient()


tab_launch, tab_monitor, tab_history = st.tabs(["Launch Job", "Monitor", "History"])

with tab_launch:
    config = StateManager.get_current_config()
    if not config:
        st.warning("No configuration loaded. Go to **Config Setup** first.")
        st.stop()

    section_title("Job Configuration")
    model_name = config.get("model", {}).get("model_name", "N/A")
    task = config.get("model", {}).get("task_type", "N/A")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Task:** {task.replace('_', ' ').title()}")
    with c2:
        quant = config.get("model", {}).get("quantization", "none")
        lora_r = config.get("model", {}).get("lora_r", "N/A")
        st.markdown(f"**Quantization:** {quant}")
        st.markdown(f"**LoRA Rank:** {lora_r}")

    section_title("Cluster Settings")
    job_name = st.text_input(
        "Job Name", value=f"slm-{task}-{model_name.split('/')[-1]}"
    )
    project_path = st.text_input(
        "Project Path (Workspace)",
        value="/Workspace/Users/<email>/databricks-slm-accelerator",
    )
    config_path = StateManager.get("config_path", "")
    config_path = st.text_input("Config Path", value=config_path)

    use_existing = st.checkbox("Use existing cluster")
    cluster_id = ""
    if use_existing:
        cluster_id = st.text_input("Cluster ID")

    num_gpus = (
        st.number_input(
            "Number of GPUs (0 = auto)", min_value=0, max_value=8, value=0
        )
        or None
    )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    if st.button("Launch Training Job", type="primary", use_container_width=True):
        if not config_path:
            st.error("Config path is required.")
        else:
            with st.status("Creating and launching job...", expanded=True) as status:
                try:
                    st.write("Initializing Databricks client...")
                    client = _get_client()
                    st.write("Creating job definition...")
                    job_id = client.create_training_job(
                        job_name=job_name,
                        config_path=config_path,
                        project_path=project_path,
                        num_gpus=num_gpus,
                        existing_cluster_id=cluster_id or None,
                    )
                    st.write("Submitting run...")
                    run_id = client.run_job(job_id)
                    StateManager.set_active_training_run(run_id)
                    StateManager.add_training_run(
                        {
                            "run_id": run_id,
                            "job_id": job_id,
                            "job_name": job_name,
                            "model": model_name,
                            "task": task,
                            "status": "RUNNING",
                        }
                    )
                    status.update(label="Job launched", state="complete")
                    st.success(f"Run ID: `{run_id}`")
                except Exception as e:
                    status.update(label="Launch failed", state="error")
                    st.error(f"Launch failed: {e}")

with tab_monitor:
    section_title("Active Run")
    active_run = StateManager.get_active_training_run()
    if not active_run:
        st.info("No active training run. Launch a job first.")
    else:
        st.markdown(f"**Run ID:** `{active_run}`")
        if st.button("Refresh Status"):
            try:
                client = _get_client()
                job_status = client.get_job_status(active_run)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"**State:** {status_badge(job_status['life_cycle_state'])}",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(f"**Result:** {job_status['result_state']}")
                with c3:
                    if job_status.get("duration_seconds"):
                        st.markdown(
                            f"**Duration:** {job_status['duration_seconds']:.0f}s"
                        )
                if job_status.get("run_page_url"):
                    st.markdown(
                        f"[View in Databricks]({job_status['run_page_url']})"
                    )
                if job_status["life_cycle_state"] in (
                    "TERMINATED",
                    "SKIPPED",
                    "INTERNAL_ERROR",
                ):
                    StateManager.set_active_training_run(None)
            except Exception as e:
                st.error(str(e))

        if st.button("Cancel Run", type="secondary"):
            try:
                client = _get_client()
                if client.cancel_job(active_run):
                    StateManager.set_active_training_run(None)
                    st.success("Run cancelled.")
            except Exception as e:
                st.error(str(e))

    section_title("MLflow Metrics")
    config = StateManager.get_current_config()
    experiment_name = ""
    if config:
        experiment_name = config.get("mlflow", {}).get("experiment_name", "")
    experiment_name = st.text_input("Experiment Name", value=experiment_name)

    metric_key = st.selectbox(
        "Metric", options=["train_loss", "eval_loss", "learning_rate"]
    )
    run_id_for_metrics = st.text_input("MLflow Run ID", value="")

    if st.button("Load Metrics") and run_id_for_metrics:
        try:
            client = _get_client()
            history = client.get_run_metrics_history(run_id_for_metrics, metric_key)
            if history:
                df = pd.DataFrame(history)
                import plotly.express as px

                fig = px.line(df, x="step", y="value", title=f"{metric_key} over steps")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Figtree, sans-serif", color="#8A91A8"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No metric history found.")
        except Exception as e:
            st.error(str(e))

with tab_history:
    section_title("Training History")
    history = StateManager.get("training_history", [])
    if history:
        for run in history:
            with st.expander(
                f"{run.get('job_name', 'Run')} — {run.get('timestamp', '')}"
            ):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Run ID:** `{run.get('run_id', 'N/A')}`")
                c2.markdown(f"**Model:** {run.get('model', 'N/A')}")
                c3.markdown(f"**Task:** {run.get('task', 'N/A')}")
    else:
        st.info("No training history yet.")
