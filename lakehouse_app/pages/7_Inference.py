"""Inference — Test deployed models interactively."""

import json
import streamlit as st

from components.theme import inject_theme, page_header, section_title
from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient

inject_theme()
StateManager.initialize()

page_header("Inference", "Test deployed models with interactive text prompts")


@st.cache_resource
def _get_client():
    return DatabricksJobClient()


tab_single, tab_batch = st.tabs(["Single Prompt", "Batch"])

with tab_single:
    section_title("Endpoint")
    endpoints = StateManager.get("endpoints", [])
    endpoint_options = [ep.get("endpoint_name", "") for ep in endpoints] if endpoints else []
    if endpoint_options:
        endpoint_name = st.selectbox("Select Endpoint", options=endpoint_options)
    else:
        endpoint_name = st.text_input("Endpoint Name", value="")

    section_title("Prompt")
    system_prompt = st.text_area("System Prompt (optional)", value="You are a helpful assistant.", height=60)
    user_prompt = st.text_area("User Prompt", value="", height=150, placeholder="Ask the model something...")

    with st.expander("Generation Parameters"):
        c1, c2, c3 = st.columns(3)
        with c1:
            max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=512)
        with c2:
            temperature = st.slider("Temperature", 0.0, 2.0, value=0.7, step=0.1)
        with c3:
            top_p = st.slider("Top P", 0.0, 1.0, value=0.9, step=0.05)

    if st.button("Generate", type="primary", use_container_width=True):
        if not endpoint_name:
            st.error("Endpoint name is required.")
        elif not user_prompt:
            st.error("User prompt is required.")
        else:
            with st.spinner("Generating..."):
                try:
                    client = _get_client()
                    prompt_text = user_prompt
                    if system_prompt:
                        prompt_text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"

                    result = client.query_endpoint(endpoint_name, prompt_text)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        section_title("Response")
                        predictions = result.get("predictions", result)
                        if isinstance(predictions, list) and len(predictions) > 0:
                            response_text = predictions[0] if isinstance(predictions[0], str) else json.dumps(predictions[0], indent=2)
                        elif isinstance(predictions, dict):
                            response_text = json.dumps(predictions, indent=2)
                        else:
                            response_text = str(predictions)
                        st.markdown(
                            f'<div class="glass-card"><pre style="color:#E6EDF3;white-space:pre-wrap;">{response_text}</pre></div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"Inference failed: {e}")

with tab_batch:
    section_title("Batch Inference")
    st.info("Upload a JSONL file with one prompt per line to run batch inference.")

    endpoint_name_batch = st.text_input("Endpoint Name", value=endpoint_name if endpoint_name else "", key="batch_ep")
    uploaded = st.file_uploader("Upload JSONL", type=["jsonl"])

    if uploaded and st.button("Run Batch", type="primary"):
        if not endpoint_name_batch:
            st.error("Endpoint name is required.")
        else:
            lines = uploaded.read().decode("utf-8").strip().split("\n")
            prompts = []
            for line in lines:
                try:
                    record = json.loads(line)
                    prompts.append(record.get("prompt", record.get("text", line)))
                except json.JSONDecodeError:
                    prompts.append(line)

            results = []
            progress = st.progress(0)
            client = _get_client()
            for i, prompt in enumerate(prompts):
                resp = client.query_endpoint(endpoint_name_batch, prompt)
                results.append({"prompt": prompt[:100], "response": str(resp.get("predictions", resp.get("error", "")))[:200]})
                progress.progress((i + 1) / len(prompts))

            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
