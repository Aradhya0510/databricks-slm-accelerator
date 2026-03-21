"""Design-system theme for the SLM Pipeline App (v1.0).

Implements the ML Accelerator Design System: dark-first, data-native,
with Syne / Figtree / IBM Plex Mono typography and four-tier elevation.
"""

import streamlit as st

# ── Colour tokens ──────────────────────────────────────────────────────────────
BG_BASE = "#0D0F12"
BG_SURFACE = "#141720"
BG_RAISED = "#1C2030"
BG_OVERLAY = "#232840"

ACCENT_PRIMARY = "#00C2A8"
ACCENT_WARM = "#F4A742"
ACCENT_ALERT = "#F25C5C"
ACCENT_INFO = "#5B8AF5"

TEXT_PRIMARY = "#EDF0F7"
TEXT_SECONDARY = "#8A91A8"
TEXT_TERTIARY = "#4E566A"

BORDER_SUBTLE = "rgba(255,255,255,0.06)"
BORDER_ACCENT = "rgba(0,194,168,0.30)"

# ── Global CSS ─────────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;500;600;700&family=Figtree:wght@300;400;500;600&display=swap');

/* ── Base typography ───────────────────────────────── */
html, body, [class*="st-"] {{
    font-family: 'Figtree', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    line-height: 1.6;
}}
h1, h2, h3 {{
    font-family: 'Syne', sans-serif !important;
    line-height: 1.1;
}}
code, pre, .stCode {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px;
    line-height: 1.7;
}}

/* ── Sidebar (200px, bg-base) ──────────────────────── */
section[data-testid="stSidebar"] {{
    width: 200px !important;
    min-width: 200px !important;
    max-width: 250px !important;
    background: {BG_BASE};
    border-right: 1px solid {BORDER_SUBTLE};
}}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {ACCENT_PRIMARY};
    font-family: 'Syne', sans-serif !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
    font-family: 'Figtree', sans-serif !important;
    font-size: 13px;
    color: {TEXT_SECONDARY};
    padding: 6px 12px;
    border-radius: 6px;
    border-left: 2px solid transparent;
    transition: all 0.15s ease;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: rgba(0,194,168,0.08);
    color: {ACCENT_PRIMARY};
    border-left-color: {ACCENT_PRIMARY};
}}

/* ── Page header ───────────────────────────────────── */
.page-header {{
    padding: 32px 0 16px 0;
    border-bottom: 1px solid {BORDER_SUBTLE};
    margin-bottom: 32px;
}}
.page-header h1 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 28px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin: 0;
}}
.page-header p {{
    font-family: 'Figtree', sans-serif;
    color: {TEXT_SECONDARY};
    font-size: 14px;
    margin: 4px 0 0 0;
}}

/* ── Section title ─────────────────────────────────── */
.ds-section-title {{
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}

/* ── Metric card ───────────────────────────────────── */
.ds-metric {{
    background: {BG_RAISED};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 16px 24px;
    text-align: left;
}}
.ds-metric .ds-metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    color: {TEXT_TERTIARY};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 6px 0;
}}
.ds-metric .ds-metric-value {{
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
    margin: 0;
}}
.ds-metric .ds-metric-delta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    margin-top: 4px;
}}
.ds-metric .ds-metric-delta.positive {{ color: {ACCENT_PRIMARY}; }}
.ds-metric .ds-metric-delta.negative {{ color: {ACCENT_ALERT}; }}

/* ── Status badge ──────────────────────────────────── */
.ds-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 4px 10px;
    border-radius: 4px;
}}
.ds-badge .dot {{
    width: 5px;
    height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.ds-badge.running  {{ background: rgba(0,194,168,0.10); color: {ACCENT_PRIMARY}; }}
.ds-badge.running .dot  {{ background: {ACCENT_PRIMARY}; }}
.ds-badge.queued   {{ background: rgba(244,167,66,0.10); color: {ACCENT_WARM}; }}
.ds-badge.queued .dot   {{ background: {ACCENT_WARM}; }}
.ds-badge.failed   {{ background: rgba(242,92,92,0.10); color: {ACCENT_ALERT}; }}
.ds-badge.failed .dot   {{ background: {ACCENT_ALERT}; }}
.ds-badge.registered {{ background: rgba(91,138,245,0.10); color: {ACCENT_INFO}; }}
.ds-badge.registered .dot {{ background: {ACCENT_INFO}; }}

/* ── Card (bg-surface) ─────────────────────────────── */
.ds-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 16px;
}}
.ds-card h3 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 14px;
    font-weight: 500;
    color: {TEXT_PRIMARY};
    margin: 0 0 8px 0;
}}
.ds-card p {{
    font-family: 'Figtree', sans-serif;
    font-size: 14px;
    color: {TEXT_SECONDARY};
    margin: 0;
    line-height: 1.6;
}}

/* ── Code / config block ───────────────────────────── */
.ds-code-block {{
    background: {BG_RAISED};
    border-left: 2px solid {ACCENT_PRIMARY};
    border-radius: 4px;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    line-height: 1.7;
    color: {TEXT_PRIMARY};
    overflow-x: auto;
}}

/* ── Progress bar ──────────────────────────────────── */
.ds-progress-track {{
    background: {BG_OVERLAY};
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    margin: 8px 0;
}}
.ds-progress-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
}}
.ds-progress-fill.training {{ background: {ACCENT_PRIMARY}; }}
.ds-progress-fill.data     {{ background: {ACCENT_WARM}; }}
.ds-progress-fill.gpu      {{ background: {ACCENT_INFO}; }}

/* ── Data table overrides ──────────────────────────── */
.stDataFrame {{
    border: 1px solid {BORDER_SUBTLE} !important;
    border-radius: 8px;
}}
.stMetric label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_TERTIARY} !important;
}}
.stMetric [data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}

/* ── Buttons ───────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: {ACCENT_PRIMARY} !important;
    color: {BG_BASE} !important;
    border: none;
    border-radius: 8px;
    font-family: 'Figtree', sans-serif;
    font-weight: 500;
    font-size: 13px;
    padding: 8px 16px;
    transition: all 0.15s ease;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    filter: brightness(1.1);
}}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {{
    background: {BG_RAISED} !important;
    color: {TEXT_SECONDARY} !important;
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    font-family: 'Figtree', sans-serif;
    font-weight: 500;
    font-size: 13px;
    padding: 8px 16px;
}}

/* ── Tabs ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 6px;
    padding: 8px 16px;
    color: {TEXT_SECONDARY};
    font-family: 'Figtree', sans-serif;
    font-weight: 500;
    font-size: 13px;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(0,194,168,0.08) !important;
    border-color: {ACCENT_PRIMARY} !important;
    color: {ACCENT_PRIMARY} !important;
}}

/* ── Expander ──────────────────────────────────────── */
details {{
    border: 1px solid {BORDER_SUBTLE} !important;
    border-radius: 8px !important;
    background: {BG_SURFACE} !important;
}}

/* ── Nav card (dashboard) ──────────────────────────── */
.ds-nav-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 24px;
    cursor: pointer;
    transition: border-color 0.15s ease;
    height: 100%;
}}
.ds-nav-card:hover {{
    border-color: {ACCENT_PRIMARY};
}}
.ds-nav-card h3 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 14px;
    font-weight: 500;
    color: {TEXT_PRIMARY};
    margin: 0 0 8px 0;
}}
.ds-nav-card p {{
    font-family: 'Figtree', sans-serif;
    font-size: 13px;
    color: {TEXT_SECONDARY};
    margin: 0;
    line-height: 1.5;
}}

/* ── Endpoint / detail card ────────────────────────── */
.ds-detail-card {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER_SUBTLE};
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 16px;
}}
.ds-detail-card .detail-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}
.ds-detail-card .detail-name {{
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: {TEXT_PRIMARY};
}}
.ds-detail-card .detail-row {{
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid {BORDER_SUBTLE};
    font-family: 'Figtree', sans-serif;
    font-size: 12px;
}}
.ds-detail-card .detail-label {{
    color: {TEXT_SECONDARY};
}}
.ds-detail-card .detail-value {{
    color: {TEXT_PRIMARY};
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
}}

/* ── Topbar status area ────────────────────────────── */
.ds-topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 0;
    border-bottom: 1px solid {BORDER_SUBTLE};
    margin-bottom: 24px;
}}
.ds-topbar h1 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 18px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    margin: 0;
}}

/* ── Input overrides ───────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {{
    background: {BG_RAISED} !important;
    border-color: {BORDER_SUBTLE} !important;
    font-family: 'Figtree', sans-serif;
    font-size: 13px;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {ACCENT_PRIMARY} !important;
    outline: 2px solid rgba(0,194,168,0.25);
}}

/* ── Active-run banner ─────────────────────────────── */
.ds-active-run {{
    background: rgba(0,194,168,0.06);
    border: 1px solid {BORDER_ACCENT};
    border-radius: 8px;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
}}
.ds-active-run .pulse-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {ACCENT_PRIMARY};
    animation: pulse 1.5s infinite;
    flex-shrink: 0;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}
.ds-active-run .run-label {{
    font-family: 'Figtree', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: {TEXT_PRIMARY};
}}
.ds-active-run .run-id {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: {TEXT_SECONDARY};
    margin-left: 8px;
}}

/* ── Footer ────────────────────────────────────────── */
.ds-footer {{
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: {TEXT_TERTIARY};
    padding: 32px 0 16px 0;
    letter-spacing: 0.02em;
}}
</style>
"""


def inject_theme():
    """Inject global CSS into the Streamlit page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """Render a page header with Syne 700 title and optional subtitle."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def section_title(text: str):
    """Render a section heading (Syne 600, 18px)."""
    st.markdown(
        f'<div class="ds-section-title">{text}</div>', unsafe_allow_html=True
    )


def metric_card(label: str, value, delta: str = "", delta_positive: bool = True):
    """Render a metric card matching the design-system spec."""
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        arrow = "\u25B2" if delta_positive else "\u25BC"
        delta_html = f'<div class="ds-metric-delta {cls}">{arrow} {delta}</div>'
    st.markdown(
        f'<div class="ds-metric">'
        f'<div class="ds-metric-label">{label}</div>'
        f'<div class="ds-metric-value">{value}</div>'
        f"{delta_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """Return an HTML status badge with dot indicator.

    Four canonical states: RUNNING (teal), QUEUED (amber), FAILED (red),
    REGISTERED (blue). Unmapped statuses fall back to QUEUED styling.
    """
    status_upper = status.upper()
    mapping = {
        "READY": "running",
        "NOT_UPDATING": "running",
        "SUCCESS": "running",
        "FINISHED": "running",
        "RUNNING": "running",
        "PENDING": "queued",
        "UPDATING": "queued",
        "QUEUED": "queued",
        "FAILED": "failed",
        "ERROR": "failed",
        "CANCELLED": "failed",
        "TERMINATED": "failed",
        "REGISTERED": "registered",
    }
    cls = mapping.get(status_upper, "queued")
    return (
        f'<span class="ds-badge {cls}">'
        f'<span class="dot"></span>{status}'
        f"</span>"
    )


# Keep backward-compatible alias
status_pill = status_badge


def progress_bar(percent: float, variant: str = "training") -> str:
    """Return an HTML progress bar. variant: training | data | gpu."""
    clamped = max(0.0, min(100.0, percent))
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div class="ds-progress-track" style="flex:1;">'
        f'<div class="ds-progress-fill {variant}" style="width:{clamped:.1f}%;"></div>'
        f"</div>"
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
        f"color:{TEXT_SECONDARY};min-width:36px;text-align:right;\">"
        f"{clamped:.0f}%</span>"
        f"</div>"
    )


def code_block(text: str):
    """Render a config/code block with left accent border."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="ds-code-block"><pre style="margin:0;white-space:pre-wrap;">{escaped}</pre></div>',
        unsafe_allow_html=True,
    )
