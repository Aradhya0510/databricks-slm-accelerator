"""Metrics display components following the ML Accelerator Design System."""

import streamlit as st
from typing import Any, Dict, List, Optional

from components.theme import (
    metric_card,
    status_badge,
    section_title,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_TERTIARY,
    BG_SURFACE,
    BG_RAISED,
    BORDER_SUBTLE,
)


class MetricsDisplay:
    """Reusable metric and model display components."""

    @staticmethod
    def display_metrics_grid(metrics: Dict[str, float], columns: int = 3):
        """Render a row of st.metric() cards — always 3 per row by default."""
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
        """Render a training-run summary card."""
        rows = [
            ("Run ID", run_info.get("run_id", "N/A")),
            ("Status", run_info.get("status", "N/A")),
            ("Duration", run_info.get("duration", "N/A")),
        ]
        detail_rows = "".join(
            f'<div class="detail-row">'
            f'<span class="detail-label">{label}</span>'
            f'<span class="detail-value">{value}</span>'
            f"</div>"
            for label, value in rows
        )
        st.markdown(
            f'<div class="ds-detail-card">'
            f'<div class="detail-name">Training Summary</div>'
            f"{detail_rows}"
            f"</div>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def display_model_card(model_info: Dict[str, Any]):
        """Render a model info card."""
        name = model_info.get("name", "Model")
        task = model_info.get("task", "N/A")
        version = model_info.get("version", "N/A")
        st.markdown(
            f'<div class="ds-detail-card">'
            f'<div class="detail-name">{name}</div>'
            f'<div class="detail-row">'
            f'<span class="detail-label">Task</span>'
            f'<span class="detail-value">{task}</span>'
            f"</div>"
            f'<div class="detail-row">'
            f'<span class="detail-label">Version</span>'
            f'<span class="detail-value">{version}</span>'
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
