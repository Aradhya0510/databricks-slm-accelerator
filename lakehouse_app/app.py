"""Databricks SLM Pipeline — Lakehouse App main dashboard."""

import streamlit as st

st.set_page_config(
    page_title="SLM Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.theme import (
    inject_theme,
    metric_card,
    section_title,
    status_badge,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
)
from utils.state_manager import StateManager

inject_theme()
StateManager.initialize()


def main():
    # ── Topbar ──────────────────────────────────────────────────────────────
    active_run = StateManager.get_active_training_run()
    badge_html = status_badge("RUNNING") if active_run else ""
    st.markdown(
        f'<div class="ds-topbar">'
        f"<h1>SLM Fine-Tuning Pipeline</h1>"
        f"{badge_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Three metric cards (always the first element below topbar) ──────────
    config = StateManager.get_current_config()
    training_history = StateManager.get("training_history", [])
    endpoints = StateManager.get("endpoints", [])

    c1, c2, c3 = st.columns(3)
    with c1:
        task = (
            config.get("model", {}).get("task_type", "---").replace("_", " ").title()
            if config
            else "---"
        )
        metric_card("Active Task", task)
    with c2:
        metric_card("Training Runs", str(len(training_history)))
    with c3:
        metric_card("Endpoints", str(len(endpoints)))

    # ── Active-run banner ───────────────────────────────────────────────────
    if active_run:
        st.markdown(
            f'<div class="ds-active-run">'
            f'<div class="pulse-dot"></div>'
            f'<span class="run-label">Training in progress</span>'
            f'<span class="run-id">Run ID: {str(active_run)[:12]}...</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Navigation cards ────────────────────────────────────────────────────
    section_title("Pipeline Stages")

    nav_items = [
        (
            "Config Setup",
            "Build YAML configs with an interactive form. Select models, LoRA settings, and data formats.",
            "pages/1_Config_Setup.py",
        ),
        (
            "Data Explorer",
            "Preview training data, check formatting, and analyze token distributions.",
            "pages/2_Data_Explorer.py",
        ),
        (
            "Training",
            "Launch fine-tuning jobs, monitor loss curves and training progress.",
            "pages/3_Training.py",
        ),
        (
            "Evaluation",
            "Evaluate perplexity, generation quality, and compare runs.",
            "pages/4_Evaluation.py",
        ),
        (
            "Registration",
            "Register models to Unity Catalog with versioning and lineage.",
            "pages/5_Model_Registration.py",
        ),
        (
            "Deployment",
            "Deploy to GPU Model Serving endpoints.",
            "pages/6_Deployment.py",
        ),
        (
            "Inference",
            "Test deployed models interactively with text prompts.",
            "pages/7_Inference.py",
        ),
        (
            "Monitoring",
            "Track endpoint health, request metrics, and token throughput.",
            "pages/8_Monitoring.py",
        ),
    ]

    for row_start in range(0, len(nav_items), 4):
        row = nav_items[row_start : row_start + 4]
        cols = st.columns(4)
        for idx, (title, desc, page) in enumerate(row):
            with cols[idx]:
                st.markdown(
                    f'<div class="ds-nav-card">'
                    f"<h3>{title}</h3>"
                    f"<p>{desc}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"Open {title}",
                    key=f"nav_{row_start + idx}",
                    use_container_width=True,
                ):
                    st.switch_page(page)

    # ── Supported tasks ─────────────────────────────────────────────────────
    section_title("Supported Tasks")

    c1, c2, c3 = st.columns(3)
    tasks = [
        (
            "Instruction Tuning (SFT)",
            "Phi-3, Phi-4, Llama 3, Mistral, Gemma, Qwen — Alpaca/ShareGPT format, QLoRA",
        ),
        (
            "DPO Alignment",
            "Align model outputs with human preferences using prompt/chosen/rejected pairs",
        ),
        (
            "Text Classification",
            "Sentiment, intent, topic classification with LM backbone + classification head",
        ),
    ]
    for col, (name, detail) in zip([c1, c2, c3], tasks):
        with col:
            st.markdown(
                f'<div class="ds-card">'
                f"<h3>{name}</h3>"
                f"<p>{detail}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="ds-footer">'
        "BUILT ON DATABRICKS LAKEHOUSE &middot; TRL &middot; PEFT &middot; MLFLOW 3"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
