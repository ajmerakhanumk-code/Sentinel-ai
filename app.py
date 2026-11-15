"""
SENTINEL AI - Multimodal Sentiment Analysis Command Center
Academic Engineering Master Application Entry Point
Windows 11 CPU-Optimized Execution Target
"""

import streamlit as st
import time

# 1. Page Configuration (Must be first Streamlit call)
st.set_page_config(
    page_title="SENTINEL AI // Multimodal Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Styles & Session Management
from src.ui.styles import inject_custom_styles, render_header
from src.utils.session_state import init_session_state
import config

# Inject dark command center aesthetic CSS
inject_custom_styles()
init_session_state()

# 3. Telemetry Header
render_header(active_mode="NODE ONLINE // SENTINEL-01")

# 4. Command Center Main Navigation Tabs
tab_dash, tab_text, tab_webcam, tab_comb, tab_ins, tab_demo = st.tabs([
    "DASHBOARD",
    "TEXT INTELLIGENCE",
    "LIVE WEBCAM CUES",
    "COMBINED MULTIMODAL",
    "INSIGHTS & HISTORY",
    "DEMONSTRATION MODE"
])

# 5. Route Views to Respective Workstations
with tab_dash:
    from src.ui.views_dashboard import render_dashboard_view
    render_dashboard_view()

with tab_text:
    from src.ui.views_text import render_text_view
    render_text_view()

with tab_webcam:
    from src.ui.views_webcam import render_webcam_view
    render_webcam_view()

with tab_comb:
    from src.ui.views_combined import render_combined_view
    render_combined_view()

with tab_ins:
    from src.ui.views_insights import render_insights_view
    render_insights_view()

with tab_demo:
    from src.ui.views_demo import render_demo_view
    render_demo_view()

# 6. Command Center Bottom Telemetry Strip
st.markdown("<br><hr style='border-color:#212C45;'>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; align-items:center; font-family:'JetBrains Mono'; font-size:0.72rem; color:#64748B; padding:0.4rem 0;">
        <div>
            <span>SYSTEM: {config.APP_NAME} v{config.APP_VERSION}</span> | 
            <span>TARGET: Windows 11 (8GB DDR4 CPU Profile)</span> | 
            <span>MODELS: TF-IDF + LogReg | VADER | FACS Action Unit Engine</span>
        </div>
        <div>
            <span style="color:#00E676;">● SYSTEM HEALTH: OPTIMAL (LOW LATENCY)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
