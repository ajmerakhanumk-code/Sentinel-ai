"""
Sentinel AI - Dashboard View
Main command center telemetry overview, system status, and activity feed.
"""

import streamlit as st
import config
from src.ui.styles import render_metric_card

def render_dashboard_view():
    """Renders the executive command center dashboard."""
    st.markdown("### SYSTEM OVERVIEW & TELEMETRY")

    # Top Telemetry Row
    col1, col2, col3, col4 = st.columns(4)

    total_inf = st.session_state.get("total_inferences", 0)
    history_len = len(st.session_state.get("history", []))

    with col1:
        st.markdown(
            render_metric_card("CORE STATUS", "OPERATIONAL", "green", "Node: SENTINEL-01"),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            render_metric_card("TEXT ENGINE", "ONLINE", "cyan", "TF-IDF + LogReg / VADER"),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            render_metric_card("VISION SENSOR", "STANDBY", "cyan", "FACS Morphometric AU"),
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            render_metric_card("SESSION INFERENCES", str(total_inf), "amber", f"Records Logged: {history_len}"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture & Operational Principles Card
    col_arch, col_act = st.columns([3, 2])

    with col_arch:
        st.markdown(
            """
            <div class="hud-panel">
                <div class="hud-title">SYSTEM ARCHITECTURE & CAPABILITIES</div>
                <div style="font-size:0.88rem; line-height:1.6; color:#94A3B8;">
                    <strong>SENTINEL AI</strong> is an academic multimodal intelligence platform engineered to perform 
                    triangulated sentiment and affective display evaluation. It continuously evaluates two independent observation channels:
                    <ul style="margin-top:0.4rem; padding-left:1.2rem; color:#CBD5E1;">
                        <li><strong>Text Channel:</strong> Dual-pipeline sentiment scoring leveraging a trained supervised 
                        TF-IDF + Logistic Regression model alongside a VADER rule-based lexicon baseline.</li>
                        <li><strong>Vision Channel:</strong> CPU-optimized facial Action Unit (AU) morphometry 
                        measuring zygomaticus smile curvature (AU12), corrugator brow furrowing (AU4), and lip depression (AU15).</li>
                        <li><strong>Multimodal Fusion:</strong> Transparent cognitive divergence analysis identifying 
                        congruence, sarcasm, masked hesitation, and affective leakage without opaque score averaging.</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="hud-panel">
                <div class="hud-title">ACADEMIC & ETHICAL BOUNDS</div>
                <div style="font-size:0.82rem; line-height:1.5; color:#94A3B8;">
                    In accordance with affective computing standards (Barrett et al., 2019), facial cues detected by 
                    computer vision represent <em>observable physical actions</em> (Action Units) rather than internal subjective emotional reality. 
                    Facial movements can be voluntary, masked, or culturally contextual. SENTINEL AI transparently flags this boundary.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_act:
        st.markdown(
            """
            <div class="hud-panel">
                <div class="hud-title">RECENT SESSION ACTIVITY</div>
            """,
            unsafe_allow_html=True
        )

        history = st.session_state.get("history", [])
        if history:
            st.markdown('<div style="max-height: 280px; overflow-y: auto;">', unsafe_allow_html=True)
            for item in reversed(history[-5:]):
                t_color = "#00E676" if item["text_sentiment"] == "Positive" else ("#FF3366" if item["text_sentiment"] == "Negative" else "#00F0FF")
                st.markdown(
                    f"""
                    <div style="border-bottom:1px solid #212C45; padding:0.5rem 0; font-family:'JetBrains Mono'; font-size:0.75rem;">
                        <span style="color:#64748B;">[{item['timestamp']}]</span> 
                        <strong style="color:{t_color};">[{item['text_sentiment']}]</strong> 
                        <span style="color:#E2E8F0;">{item['text'][:32]}...</span><br>
                        <span style="color:#94A3B8; font-size:0.7rem;">Face: {item['face_sentiment']} | Match: {item['agreement']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#64748B; font-size:0.8rem; font-family:JetBrains Mono; padding:1.5rem 0; text-align:center;">'
                'No inferences recorded in current session.<br>Run an analysis in the Text or Vision panels to log telemetry.'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
