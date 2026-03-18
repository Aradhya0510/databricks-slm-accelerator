"""Data Explorer — Preview and analyze training data."""

import json

import pandas as pd
import streamlit as st

from components.theme import inject_theme, page_header, section_title, metric_card
from utils.state_manager import StateManager

inject_theme()
StateManager.initialize()

page_header("Data Explorer", "Preview training data, check formatting, and analyze token distributions")


def _try_load_data(path: str) -> pd.DataFrame:
    """Attempt to load data from a local or /Volumes path."""
    if not path:
        return pd.DataFrame()
    try:
        if path.endswith(".jsonl"):
            return pd.read_json(path, lines=True)
        elif path.endswith(".json"):
            return pd.read_json(path)
        elif path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


config = StateManager.get_current_config()

tab_preview, tab_stats = st.tabs(["Data Preview", "Statistics"])

with tab_preview:
    data_path = st.text_input(
        "Training Data Path",
        value=config.get("data", {}).get("train_data_path", "") if config else "",
    )
    if st.button("Load Data", type="primary"):
        df = _try_load_data(data_path)
        if df.empty:
            st.warning("Could not load data. Check the path and format.")
        else:
            st.session_state["explorer_df"] = df

    df = st.session_state.get("explorer_df", pd.DataFrame())
    if not df.empty:
        section_title("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total Examples", f"{len(df):,}")
        with c2:
            metric_card("Columns", str(len(df.columns)))
        with c3:
            metric_card("Format", data_path.rsplit(".", 1)[-1].upper() if data_path else "—")

        section_title("Sample Records")
        n = st.slider("Rows to display", 5, min(50, len(df)), 10)
        st.dataframe(df.head(n), use_container_width=True)

        section_title("Column Details")
        for col in df.columns:
            with st.expander(f"Column: `{col}`"):
                st.write(f"**Dtype:** {df[col].dtype}")
                st.write(f"**Non-null:** {df[col].notna().sum()} / {len(df)}")
                if df[col].dtype == "object":
                    lengths = df[col].dropna().astype(str).str.len()
                    st.write(f"**Avg length:** {lengths.mean():.0f} chars")
                    st.write(f"**Max length:** {lengths.max()} chars")
                    st.text_area("Example value", value=str(df[col].dropna().iloc[0])[:500] if len(df[col].dropna()) > 0 else "", height=100, key=f"sample_{col}")

with tab_stats:
    df = st.session_state.get("explorer_df", pd.DataFrame())
    if df.empty:
        st.info("Load data in the preview tab first.")
    else:
        section_title("Token Length Estimates")
        st.caption("Approximate token counts (chars / 4)")

        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if text_cols:
            selected_col = st.selectbox("Text column", text_cols)
            lengths = df[selected_col].dropna().astype(str).str.len() / 4
            import plotly.express as px
            fig = px.histogram(lengths, nbins=50, title=f"Token length distribution — {selected_col}")
            fig.update_layout(
                xaxis_title="Estimated tokens",
                yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Mean", f"{lengths.mean():.0f}")
            with c2:
                metric_card("Median", f"{lengths.median():.0f}")
            with c3:
                metric_card("P95", f"{lengths.quantile(0.95):.0f}")
            with c4:
                metric_card("Max", f"{lengths.max():.0f}")
        else:
            st.info("No text columns found in the dataset.")
