"""Deployment — Deploy models to Databricks Model Serving endpoints."""

import streamlit as st

from components.theme import inject_theme, page_header, section_title, status_badge, metric_card
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

inject_theme()
StateManager.initialize()

page_header("Deployment", "Deploy fine-tuned models to GPU Model Serving endpoints")

# ── Three metric cards at top ──────────────────────────────────────────────────
endpoints = StateManager.get("endpoints", [])
c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Active Endpoints", str(len(endpoints)))
with c2:
    latest_ep = endpoints[0].get("endpoint_name", "—") if endpoints else "—"
    metric_card("Latest Endpoint", latest_ep)
with c3:
    latest_status = endpoints[0].get("status", "—").upper() if endpoints else "—"
    metric_card("Latest Status", latest_status)


@st.cache_resource
def _get_client():
    return DatabricksJobClient()


tab_deploy, tab_manage = st.tabs(["Deploy Endpoint", "Manage Endpoints"])

with tab_deploy:
    section_title("Endpoint Configuration")
    c1, c2 = st.columns(2)
    with c1:
        endpoint_name = st.text_input("Endpoint Name", value="slm-phi3-sft")
        model_name = st.text_input(
            "Registered Model Name", value="main.slm_models.phi3-sft"
        )
    with c2:
        model_version = st.text_input("Model Version", value="1")
        workload_size = st.selectbox(
            "Workload Size", options=["Small", "Medium", "Large"]
        )

    workload_type = st.selectbox(
        "Workload Type",
        options=["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"],
        help="GPU compute type for the endpoint",
    )
    scale_to_zero = st.checkbox("Scale to Zero", value=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    if st.button("Deploy Endpoint", type="primary", use_container_width=True):
        with st.status("Deploying endpoint...", expanded=True) as deploy_status:
            try:
                st.write("Connecting to Model Serving...")
                client = _get_client()
                st.write(f"Deploying `{endpoint_name}`...")
                result = client.create_model_serving_endpoint(
                    endpoint_name=endpoint_name,
                    model_name=model_name,
                    model_version=model_version,
                    workload_size=workload_size,
                    workload_type=workload_type,
                    scale_to_zero=scale_to_zero,
                )
                if result.get("status") == "error":
                    deploy_status.update(label="Deployment failed", state="error")
                    st.error(f"Deployment failed: {result.get('error')}")
                else:
                    StateManager.add_endpoint(result)
                    deploy_status.update(
                        label="Deployment complete", state="complete"
                    )
                    st.success(
                        f"Endpoint `{endpoint_name}` {result['status']} successfully."
                    )
            except Exception as e:
                deploy_status.update(label="Deployment failed", state="error")
                st.error(f"Deployment failed: {e}")

with tab_manage:
    section_title("Endpoint Status")
    check_endpoint = st.text_input("Endpoint Name to Check", value="")
    if st.button("Check Status") and check_endpoint:
        try:
            client = _get_client()
            ep_status = client.get_endpoint_status(check_endpoint)

            st.markdown(
                f'<div class="ds-detail-card">'
                f'<div class="detail-header">'
                f'<span class="detail-name">{ep_status["endpoint_name"]}</span>'
                f'{status_badge(ep_status.get("ready", "UNKNOWN"))}'
                f"</div>",
                unsafe_allow_html=True,
            )
            for model in ep_status.get("served_models", []):
                st.markdown(
                    f'<div class="detail-row">'
                    f'<span class="detail-label">Model</span>'
                    f'<span class="detail-value">'
                    f"{model.get('entity_name', 'N/A')} v{model.get('entity_version', '?')}"
                    f"</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(str(e))

    section_title("Session Endpoints")
    endpoints = StateManager.get("endpoints", [])
    if endpoints:
        for ep in endpoints:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**{ep.get('endpoint_name')}**")
            c2.markdown(ep.get("model_name", "N/A"))
            c3.markdown(
                status_badge(ep.get("status", "UNKNOWN")),
                unsafe_allow_html=True,
            )
    else:
        st.info("No endpoints deployed in this session.")
