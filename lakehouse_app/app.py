"""Databricks SLM Pipeline — Lakehouse App main entry point and dashboard."""

import streamlit as st

st.set_page_config(
    page_title="SLM Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.theme import inject_theme
from utils.state_manager import StateManager

inject_theme()
StateManager.initialize()


def _quick_stat(label, value, icon=""):
    return (
        f'<div class="glass-card" style="text-align:center;padding:1.2rem;">'
        f'<div style="font-size:1.3rem;margin-bottom:0.3rem;">{icon}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:#fff;">{value}</div>'
        f'<div style="font-size:0.78rem;color:#8B949E;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-top:0.2rem;">{label}</div>'
        f'</div>'
    )


def main():
    st.markdown(
        '<div class="hero-banner">'
        "<h1>SLM Fine-Tuning Pipeline</h1>"
        "<p>End-to-end fine-tuning, evaluation, and deployment for small language models on Databricks</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    config = StateManager.get_current_config()
    training_history = StateManager.get("training_history", [])
    endpoints = StateManager.get("endpoints", [])
    active_run = StateManager.get_active_training_run()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        task = config.get("model", {}).get("task_type", "---").replace("_", " ").title() if config else "---"
        st.markdown(_quick_stat("Active Task", task), unsafe_allow_html=True)
    with c2:
        model = config.get("model", {}).get("model_name", "---").split("/")[-1] if config else "---"
        st.markdown(_quick_stat("Model", model), unsafe_allow_html=True)
    with c3:
        st.markdown(_quick_stat("Training Runs", str(len(training_history))), unsafe_allow_html=True)
    with c4:
        st.markdown(_quick_stat("Endpoints", str(len(endpoints))), unsafe_allow_html=True)

    st.markdown("")

    if active_run:
        st.markdown(
            f'<div class="glass-card" style="display:flex;align-items:center;gap:1rem;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:#FFAA00;'
            f'animation:pulse 1.5s infinite;"></div>'
            f'<div><strong style="color:#E6EDF3;">Training in progress</strong>'
            f'<span style="color:#8B949E;margin-left:0.5rem;">Run ID: {str(active_run)[:12]}...</span></div>'
            f'<style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}</style>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    nav_items = [
        ("Config Setup", "Build YAML configs with an interactive form. Select models, LoRA settings, and data formats.",
         "pages/1_Config_Setup.py"),
        ("Data Explorer", "Preview training data, check formatting, and analyze token distributions.",
         "pages/2_Data_Explorer.py"),
        ("Training", "Launch fine-tuning jobs, monitor loss curves and training progress.",
         "pages/3_Training.py"),
        ("Evaluation", "Evaluate perplexity, generation quality, and compare runs.",
         "pages/4_Evaluation.py"),
        ("Registration", "Register models to Unity Catalog with versioning and lineage.",
         "pages/5_Model_Registration.py"),
        ("Deployment", "Deploy to GPU Model Serving endpoints.",
         "pages/6_Deployment.py"),
        ("Inference", "Test deployed models interactively with text prompts.",
         "pages/7_Inference.py"),
        ("Monitoring", "Track endpoint health, request metrics, and token throughput.",
         "pages/8_Monitoring.py"),
    ]

    cols = st.columns(4)
    for idx, (title, desc, page) in enumerate(nav_items):
        with cols[idx % 4]:
            st.markdown(
                f'<div class="nav-card">'
                f"<h3>{title}</h3>"
                f"<p>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title}", key=f"nav_{idx}", use_container_width=True):
                st.switch_page(page)

    st.markdown("")
    st.markdown('<div class="section-title">Supported Tasks</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="glass-card">'
            '<strong style="color:#E6EDF3;">Instruction Tuning (SFT)</strong>'
            '<p style="color:#8B949E;font-size:0.88rem;margin:0.4rem 0 0 0;">'
            "Phi-3, Llama 3, Mistral, Gemma, Qwen &mdash; Alpaca/ShareGPT format, QLoRA</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="glass-card">'
            '<strong style="color:#E6EDF3;">DPO Alignment</strong>'
            '<p style="color:#8B949E;font-size:0.88rem;margin:0.4rem 0 0 0;">'
            "Align model outputs with human preferences using prompt/chosen/rejected pairs</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="glass-card">'
            '<strong style="color:#E6EDF3;">Text Classification</strong>'
            '<p style="color:#8B949E;font-size:0.88rem;margin:0.4rem 0 0 0;">'
            "Sentiment, intent, topic classification with LM backbone + classification head</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="text-align:center;color:#8B949E;padding:2rem 0 1rem 0;font-size:0.82rem;">'
        "Built on Databricks Lakehouse &bull; TRL &bull; PEFT &bull; MLflow 3"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
