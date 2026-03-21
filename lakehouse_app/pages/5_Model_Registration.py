"""Model Registration — Register models to Unity Catalog."""

import streamlit as st

from components.theme import inject_theme, page_header, section_title, status_badge, metric_card
from components.metrics_display import MetricsDisplay
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

inject_theme()
StateManager.initialize()

page_header(
    "Model Registration",
    "Register fine-tuned models to Unity Catalog with versioning and lineage",
)

# ── Three metric cards at top ──────────────────────────────────────────────────
registered = StateManager.get("registered_models", [])
c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Registered Models", str(len(registered)))
with c2:
    latest = registered[0].get("name", "—").split(".")[-1] if registered else "—"
    metric_card("Latest Model", latest)
with c3:
    latest_ver = str(registered[0].get("version", "—")) if registered else "—"
    metric_card("Latest Version", latest_ver)


@st.cache_resource
def _get_client():
    return DatabricksJobClient()


tab_register, tab_browse = st.tabs(["Register Model", "Browse Registry"])

with tab_register:
    section_title("Model Details")
    c1, c2 = st.columns(2)
    with c1:
        catalog = st.text_input("Catalog", value="main")
        schema = st.text_input("Schema", value="slm_models")
    with c2:
        model_short_name = st.text_input("Model Name", value="phi3-sft")
        model_version_alias = st.text_input("Alias (optional)", value="champion")

    full_model_name = f"{catalog}.{schema}.{model_short_name}"
    st.markdown(f"**Registered model name:** `{full_model_name}`")

    section_title("Source Run")
    run_id = st.text_input(
        "MLflow Run ID",
        value="",
        help="Run ID from which to register the model artifacts",
    )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    if st.button("Register Model", type="primary", use_container_width=True):
        if not run_id:
            st.error("MLflow Run ID is required.")
        else:
            with st.status("Registering model...", expanded=True) as reg_status:
                try:
                    st.write("Connecting to MLflow...")
                    from mlflow.tracking import MlflowClient

                    client = MlflowClient()
                    model_uri = f"runs:/{run_id}/model"
                    st.write(f"Creating model version for `{full_model_name}`...")
                    result = client.create_model_version(
                        name=full_model_name,
                        source=model_uri,
                        run_id=run_id,
                    )
                    if model_version_alias:
                        st.write(f"Setting alias `{model_version_alias}`...")
                        client.set_registered_model_alias(
                            full_model_name,
                            model_version_alias,
                            result.version,
                        )
                    StateManager.add_registered_model(
                        {
                            "name": full_model_name,
                            "version": result.version,
                            "run_id": run_id,
                            "alias": model_version_alias,
                        }
                    )
                    reg_status.update(label="Registration complete", state="complete")
                    st.success(
                        f"Registered `{full_model_name}` version {result.version}"
                    )
                except Exception as e:
                    reg_status.update(label="Registration failed", state="error")
                    st.error(f"Registration failed: {e}")

with tab_browse:
    section_title("Registered Models")
    if st.button("Refresh Models"):
        try:
            client = _get_client()
            models = client.get_registered_models()
            st.session_state["registry_models"] = models
        except Exception as e:
            st.error(str(e))

    models = st.session_state.get("registry_models", [])
    if models:
        for model in models:
            with st.expander(model["name"]):
                st.markdown(
                    f"**Created:** {model.get('creation_timestamp', 'N/A')}"
                )
                for v in model.get("latest_versions", []):
                    aliases = ", ".join(v.get("aliases", [])) or "none"
                    st.markdown(
                        f"- Version **{v['version']}** — Aliases: {aliases} "
                        f"— Run: `{v.get('run_id', 'N/A')}`"
                    )
    else:
        st.info("Click **Refresh Models** to load the model registry.")

    section_title("Session-Registered Models")
    local_models = StateManager.get("registered_models", [])
    if local_models:
        for m in local_models:
            MetricsDisplay.display_model_card(
                {
                    "name": m.get("name"),
                    "task": m.get("alias", "N/A"),
                    "version": m.get("version", "N/A"),
                }
            )
    else:
        st.info("No models registered in this session.")
