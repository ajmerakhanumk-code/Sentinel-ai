"""
Sentinel AI - Combined Multimodal Intelligence View
Cross-modal agreement matrix, sarcasm detection, uncertainty estimation,
and dual-modality probability radar comparison.
"""

import streamlit as st
from src.multimodal.fusion_engine import fusion_engine
from src.ui.components import (
    create_confidence_gauge,
    create_bimodal_radar_chart,
    render_agreement_badge
)
from src.ui.styles import render_disclaimer
import config

def render_combined_view():
    """Renders the combined multimodal intelligence view."""
    st.markdown("### COMBINED MULTIMODAL INTELLIGENCE & CONCORDANCE")

    render_disclaimer()

    text_res = st.session_state.get("text_results")
    vision_res = st.session_state.get("vision_results")

    # Re-evaluate fusion
    fusion_data = fusion_engine.evaluate_fusion(text_res, vision_res)

    status_code = fusion_data.get("status_code", "INCOMPLETE")

    if status_code == "INCOMPLETE":
        st.markdown(
            """
            <div class="hud-panel" style="text-align:center; padding:2rem;">
                <div style="font-family:'JetBrains Mono'; font-size:1.1rem; color:#FFB300; margin-bottom:0.5rem;">
                    ⚠ INSUFFICIENT MULTIMODAL TELEMETRY
                </div>
                <div style="font-size:0.85rem; color:#94A3B8; max-width:600px; margin:0 auto; line-height:1.6;">
                    Combined intelligence requires sensory inputs. Neither text sentiment nor facial cue data 
                    has been evaluated yet in this session.<br><br>
                    Please navigate to <strong>Text Analysis</strong> to enter text, or <strong>Live Webcam</strong> 
                    to capture observable facial cues. Alternatively, launch a scenario in <strong>Demonstration Mode</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Modality Status Overview Row
    col_t_card, col_v_card = st.columns(2)

    with col_t_card:
        if text_res and text_res.get("status") == "SUCCESS":
            t_label = text_res["label"]
            t_color = "#00E676" if t_label == "Positive" else ("#FF3366" if t_label == "Negative" else "#00F0FF")
            st.markdown(
                f"""
                <div class="hud-panel">
                    <div class="hud-title">MODALITY A: LINGUISTIC TEXT SENTIMENT</div>
                    <div style="font-size:1.4rem; font-weight:700; color:{t_color}; font-family:'JetBrains Mono'; margin-bottom:0.3rem;">
                        {t_label.upper()} ({text_res['confidence']*100:.1f}%)
                    </div>
                    <div style="font-size:0.8rem; color:#94A3B8; background:#0E131F; padding:0.5rem; border-radius:4px; font-style:italic;">
                        "{text_res['text']}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="hud-panel">
                    <div class="hud-title">MODALITY A: LINGUISTIC TEXT SENTIMENT</div>
                    <div style="color:#64748B; font-family:'JetBrains Mono'; font-size:0.9rem;">
                        [ NO TEXT INPUT EVALUATED ]
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_v_card:
        if vision_res and vision_res.get("status") == "SUCCESS":
            v_label = vision_res["label"]
            v_color = "#00E676" if v_label == "Positive" else ("#FF3366" if v_label == "Negative" else "#00F0FF")
            st.markdown(
                f"""
                <div class="hud-panel">
                    <div class="hud-title">MODALITY B: OBSERVABLE FACIAL CUE (FACS)</div>
                    <div style="font-size:1.4rem; font-weight:700; color:{v_color}; font-family:'JetBrains Mono'; margin-bottom:0.3rem;">
                        {v_label.upper()} ({vision_res['confidence']*100:.1f}%)
                    </div>
                    <div style="font-size:0.8rem; color:#94A3B8; background:#0E131F; padding:0.5rem; border-radius:4px;">
                        Primary: {vision_res['cues'][0] if vision_res.get('cues') else 'Baseline geometry'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="hud-panel">
                    <div class="hud-title">MODALITY B: OBSERVABLE FACIAL CUE (FACS)</div>
                    <div style="color:#64748B; font-family:'JetBrains Mono'; font-size:0.9rem;">
                        [ NO CAMERA CUE EVALUATED / NO FACE DETECTED ]
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Combined Multimodal Synthesis Panel
    st.markdown("#### CROSS-MODAL FUSION & CONGRUENCE")

    col_diag, col_radar = st.columns([3, 2])

    with col_diag:
        is_agree = fusion_data.get("is_agreement", False)
        agree_text = fusion_data.get("agreement_status", "PARTIAL")

        render_agreement_badge(is_agree, agree_text)

        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="hud-panel">
                <div class="hud-title">MULTIMODAL DIAGNOSTIC INTERPRETATION</div>
                <div style="margin-bottom:0.75rem;">
                    <div style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono';">COMBINED INTERPRETATION</div>
                    <div style="font-size:1.35rem; font-weight:700; color:#F1F5F9; font-family:'JetBrains Mono';">
                        {fusion_data['combined_sentiment'].upper()}
                    </div>
                </div>
                <div style="font-size:0.86rem; line-height:1.6; color:#CBD5E1; background:#0E131F; padding:0.8rem; border-radius:6px; border:1px solid #212C45; margin-bottom:0.75rem;">
                    <strong>Cognitive Divergence Assessment:</strong><br>
                    {fusion_data['diagnostic_summary']}
                </div>
                <div style="font-size:0.8rem; color:#00F0FF; font-family:'JetBrains Mono';">
                    ● Actionable Insight: {fusion_data['actionable_insight']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Uncertainty breakdown panel
        st.markdown(
            f"""
            <div class="hud-panel">
                <div class="hud-title">UNCERTAINTY & CREDIBILITY PROFILE</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; font-family:'JetBrains Mono'; font-size:0.8rem;">
                    <div style="background:#0E131F; padding:0.6rem; border-radius:4px; border:1px solid #212C45;">
                        <span style="color:#64748B;">UNCERTAINTY LEVEL:</span><br>
                        <strong style="color:#F1F5F9;">{fusion_data['uncertainty_level']}</strong>
                    </div>
                    <div style="background:#0E131F; padding:0.6rem; border-radius:4px; border:1px solid #212C45;">
                        <span style="color:#64748B;">FUSION METHOD:</span><br>
                        <strong style="color:#00F0FF;">Bayesian Concordance Weighting</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_radar:
        # Gauge
        gauge_fig = create_confidence_gauge(
            fusion_data["confidence"],
            title="MULTIMODAL CREDIBILITY",
            color="#00E676" if is_agree else "#FFB300"
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # If both modalities are available, render dual-modality radar chart!
        if text_res and vision_res and text_res.get("probabilities") and vision_res.get("probabilities"):
            st.markdown(
                """
                <div class="hud-panel">
                    <div class="hud-title">BIMODAL PROBABILITY RADAR</div>
                """,
                unsafe_allow_html=True
            )
            radar_fig = create_bimodal_radar_chart(
                text_res["probabilities"],
                vision_res["probabilities"]
            )
            st.plotly_chart(radar_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
