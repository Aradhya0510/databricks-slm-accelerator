"""Deployment — Deploy models to Databricks Model Serving endpoints."""

import streamlit as st

from components.theme import inject_theme, page_header, section_title, status_pill
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

inject_theme()
StateManager.initialize()

page_header("Deployment", "Deploy fine-tuned models to GPU Model Serving endpoints")


@st.cache_resource
def _get_client():
    return DatabricksJobClient()


tab_deploy, tab_manage = st.tabs(["Deploy Endpoint", "Manage Endpoints"])

with tab_deploy:
    section_title("Endpoint Configuration")
    c1, c2 = st.columns(2)
    with c1:
        endpoint_name = st.text_input("Endpoint Name", value="slm-phi3-sft")
        model_name = st.text_input("Registered Model Name", value="main.slm_models.phi3-sft")
    with c2:
        model_version = st.text_input("Model Version", value="1")
        workload_size = st.selectbox("Workload Size", options=["Small", "Medium", "Large"])

    workload_type = st.selectbox(
        "Workload Type",
        options=["GPU_SMALL", "GPU_MEDIUM", "GPU_LARGE"],
        help="GPU compute type for the endpoint",
    )
    scale_to_zero = st.checkbox("Scale to Zero", value=True)

    st.markdown("")
    if st.button("Deploy Endpoint", type="primary", use_container_width=True):
        with st.spinner("Deploying..."):
            try:
                client = _get_client()
                result = client.create_model_serving_endpoint(
                    endpoint_name=endpoint_name,
                    model_name=model_name,
                    model_version=model_version,
                    workload_size=workload_size,
                    workload_type=workload_type,
                    scale_to_zero=scale_to_zero,
                )
                if result.get("status") == "error":
                    st.error(f"Deployment failed: {result.get('error')}")
                else:
                    StateManager.add_endpoint(result)
                    st.success(f"Endpoint `{endpoint_name}` {result['status']} successfully.")
            except Exception as e:
                st.error(f"Deployment failed: {e}")

with tab_manage:
    section_title("Endpoint Status")
    check_endpoint = st.text_input("Endpoint Name to Check", value="")
    if st.button("Check Status") and check_endpoint:
        try:
            client = _get_client()
            status = client.get_endpoint_status(check_endpoint)
            st.markdown(
                f'<div class="endpoint-card">'
                f'<div class="endpoint-header">'
                f'<span class="endpoint-name">{status["endpoint_name"]}</span>'
                f'{status_pill(status.get("ready", "UNKNOWN"))}'
                f'</div>',
                unsafe_allow_html=True,
            )
            for model in status.get("served_models", []):
                st.markdown(
                    f'<div class="detail-row">'
                    f'<span class="detail-label">Model</span>'
                    f'<span class="detail-value">{model.get("entity_name", "N/A")} v{model.get("entity_version", "?")}</span>'
                    f'</div>',
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
            c1.write(f"**{ep.get('endpoint_name')}**")
            c2.write(ep.get("model_name", "N/A"))
            c3.markdown(status_pill(ep.get("status", "UNKNOWN")), unsafe_allow_html=True)
    else:
        st.info("No endpoints deployed in this session.")
