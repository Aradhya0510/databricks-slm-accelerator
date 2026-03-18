"""Metrics display components for SLM training results."""

import streamlit as st
from typing import Any, Dict, List, Optional


class MetricsDisplay:
    """Reusable metric display components."""

    @staticmethod
    def display_metrics_grid(metrics: Dict[str, float], columns: int = 4):
        cols = st.columns(columns)
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                display_name = key.replace("eval_", "").replace("_", " ").title()
                if isinstance(value, float):
                    st.metric(display_name, f"{value:.4f}")
                else:
                    st.metric(display_name, str(value))

    @staticmethod
    def display_training_summary(run_info: Dict[str, Any]):
        st.markdown(
            f'<div class="glass-card">'
            f'<strong style="color:#E6EDF3;">Training Summary</strong>'
            f'<div style="margin-top:0.8rem;">'
            f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid rgba(139,148,158,0.1);">'
            f'<span style="color:#8B949E;">Run ID</span>'
            f'<span style="color:#E6EDF3;font-weight:500;">{run_info.get("run_id", "N/A")}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid rgba(139,148,158,0.1);">'
            f'<span style="color:#8B949E;">Status</span>'
            f'<span style="color:#E6EDF3;font-weight:500;">{run_info.get("status", "N/A")}</span></div>'
            f'<div style="display:flex;justify-content:space-between;padding:0.3rem 0;">'
            f'<span style="color:#8B949E;">Duration</span>'
            f'<span style="color:#E6EDF3;font-weight:500;">{run_info.get("duration", "N/A")}</span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    @staticmethod
    def display_model_card(model_info: Dict[str, Any]):
        st.markdown(
            f'<div class="glass-card">'
            f'<strong style="color:#E6EDF3;">{model_info.get("name", "Model")}</strong>'
            f'<div style="margin-top:0.6rem;font-size:0.88rem;">'
            f'<div style="color:#8B949E;">Task: {model_info.get("task", "N/A")}</div>'
            f'<div style="color:#8B949E;">Version: {model_info.get("version", "N/A")}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
