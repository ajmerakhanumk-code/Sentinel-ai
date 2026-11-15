"""
UI Style Injector and HTML Components for Sentinel AI Command Center
"""

import streamlit as st
from pathlib import Path
import config

def inject_custom_styles():
    """Injects high-fidelity command center CSS stylesheet into Streamlit."""
    css_file = config.CSS_DIR / "command_center.css"
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("Command center CSS stylesheet not found; running with default styles.")

def render_header(active_mode: str = "SYSTEM READY"):
    """Renders top command-centre telemetry header."""
    header_html = f"""
    <div class="sentinel-header">
        <div class="sentinel-brand">
            <div class="sentinel-logo-badge">SENTINEL-AI // v{config.APP_VERSION}</div>
            <div>
                <h1 class="sentinel-title">{config.APP_NAME}</h1>
                <div class="sentinel-subtitle">MULTIMODAL SENTIMENT ANALYSIS & OBSERVABLE CUE INTELLIGENCE</div>
            </div>
        </div>
        <div>
            <span class="status-pill online">
                <span class="status-dot"></span>
                NODE ACTIVE : {active_mode}
            </span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_disclaimer():
    """Renders academic & ethical disclosure banner."""
    html = f"""
    <div class="disclaimer-banner">
        <strong>[ACADEMIC NOTICE & ETHICAL BOUNDS]</strong><br>
        {config.ETHICAL_DISCLAIMER}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, color_class: str = "cyan", subtext: str = ""):
    """Renders a command center HUD metric tile."""
    sub_html = f'<div style="font-size:0.7rem; color:#64748B; margin-top:0.25rem; font-family:JetBrains Mono;">{subtext}</div>' if subtext else ""
    html = f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color_class}">{value}</div>
        {sub_html}
    </div>
    """
    return html

def render_hud_panel_open(title: str):
    """Returns HTML to open a HUD panel."""
    return f"""
    <div class="hud-panel">
        <div class="hud-title">{title}</div>
    """

def render_hud_panel_close():
    """Returns HTML to close a HUD panel."""
    return "</div>"
