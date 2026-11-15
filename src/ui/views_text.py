"""
Sentinel AI - Text Sentiment Analysis View
Interactive text entry, dual model execution (TF-IDF + VADER),
probability charts, token attribution, and diagnostics.
"""

import streamlit as st
from src.text_analysis.classifier import text_engine
from src.multimodal.fusion_engine import fusion_engine
from src.utils.session_state import log_analysis
from src.ui.components import create_confidence_gauge, create_probability_bar_chart

def render_text_view():
    """Renders the text sentiment analysis workstation."""
    st.markdown("### TEXT SENTIMENT INTELLIGENCE")

    # Sample Presets Row
    st.markdown(
        '<div style="font-size:0.78rem; color:#94A3B8; font-family:JetBrains Mono; margin-bottom:0.4rem;">'
        'QUICK LOAD BENCHMARK SAMPLES:</div>',
        unsafe_allow_html=True
    )
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    preset_text = None
    if col_p1.button("Positive Benchmark", use_container_width=True):
        preset_text = "The system performance exceeded our expectations significantly and was wonderful!"
    if col_p2.button("Negative Benchmark", use_container_width=True):
        preset_text = "The application crashed abruptly with an unhandled memory exception and severe bug."
    if col_p3.button("Neutral Benchmark", use_container_width=True):
        preset_text = "The server returned status code 200 indicating successful transmission of telemetry."
    if col_p4.button("Ironic / Sarcastic", use_container_width=True):
        preset_text = "Oh wonderful, another catastrophic database corruption error right before the demo."

    if preset_text:
        st.session_state.current_text = preset_text

    # Main Text Input Area
    user_input = st.text_area(
        label="Input Observation or Statement for Sentiment Analysis:",
        value=st.session_state.get("current_text", ""),
        height=130,
        placeholder="Type or paste natural language text here (e.g., user feedback, operational report, chat message)...",
        key="text_input_area"
    )

    col_btn, col_clear = st.columns([4, 1])
    with col_btn:
        analyze_clicked = st.button("RUN TEXT SENTIMENT ANALYSIS", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear Text", use_container_width=True):
            st.session_state.current_text = ""
            st.rerun()

    if analyze_clicked or (user_input and st.session_state.get("text_results") is None):
        if user_input.strip():
            with st.spinner("Executing TF-IDF + Logistic Regression & VADER pipelines..."):
                res = text_engine.analyze(user_input)
                st.session_state.text_results = res
                st.session_state.current_text = user_input

                # Trigger fusion recalculation
                vision_res = st.session_state.get("vision_results")
                fused = fusion_engine.evaluate_fusion(res, vision_res)
                st.session_state.combined_results = fused

                # Log session record
                log_analysis({
                    "text": user_input,
                    "text_sentiment": res["label"],
                    "text_conf": res["confidence"],
                    "face_sentiment": vision_res["label"] if vision_res else "N/A",
                    "face_conf": vision_res["confidence"] if vision_res else 0.0,
                    "agreement": fused["agreement_status"],
                    "modality": fused["modality_status"]
                })
        else:
            st.warning("Please enter valid text or select one of the benchmark samples above.")

    # Render Results if available
    res = st.session_state.get("text_results")
    if res and res.get("status") == "SUCCESS":
        st.markdown("<hr style='border-color:#212C45; margin:1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### ANALYSIS RESULTS & MODEL INFERENCE")

        col_main, col_charts = st.columns([3, 2])

        with col_main:
            # Classification Badge
            label = res["label"]
            badge_color = "#00E676" if label == "Positive" else ("#FF3366" if label == "Negative" else "#00F0FF")
            st.markdown(
                f"""
                <div class="hud-panel">
                    <div class="hud-title">SUPERVISED CLASSIFIER OUTPUT</div>
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;">
                        <div>
                            <div style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono';">PREDICTED CLASS</div>
                            <div style="font-size:2rem; font-weight:800; color:{badge_color}; font-family:'JetBrains Mono';">
                                {label.upper()}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.75rem; color:#64748B; font-family:'JetBrains Mono';">CONFIDENCE</div>
                            <div style="font-size:1.8rem; font-weight:700; color:#F1F5F9; font-family:'JetBrains Mono';">
                                {res['confidence']*100:.1f}%
                            </div>
                        </div>
                    </div>
                    <div style="font-size:0.85rem; line-height:1.5; color:#CBD5E1; background:#0E131F; padding:0.8rem; border-radius:6px; border:1px solid #212C45;">
                        <strong>Diagnostic Explanation:</strong><br>{res['explanation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Token Attributions
            if res.get("attributions"):
                st.markdown(
                    """
                    <div class="hud-panel">
                        <div class="hud-title">LEXICAL FEATURE ATTRIBUTION (TOP DRIVERS)</div>
                        <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                    """,
                    unsafe_allow_html=True
                )
                for attr in res["attributions"]:
                    w_color = "#00E676" if attr["importance"] > 0 else "#FF3366"
                    st.markdown(
                        f"""
                        <span style="background:rgba(14,19,31,0.8); border:1px solid #212C45; border-radius:4px; padding:0.35rem 0.65rem; font-family:'JetBrains Mono'; font-size:0.78rem;">
                            <strong style="color:#F1F5F9;">{attr['token']}</strong>: 
                            <span style="color:{w_color};">{attr['importance']:+.3f}</span>
                        </span>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("</div></div>", unsafe_allow_html=True)

        with col_charts:
            # Probability Distribution Chart
            st.markdown(
                """
                <div class="hud-panel">
                    <div class="hud-title">CLASS PROBABILITIES</div>
                """,
                unsafe_allow_html=True
            )
            prob_fig = create_probability_bar_chart(res["probabilities"])
            st.plotly_chart(prob_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Confidence Gauge
            gauge_fig = create_confidence_gauge(res["confidence"], title="MODEL CERTAINTY", color=badge_color)
            st.plotly_chart(gauge_fig, use_container_width=True)

        # Baseline Comparison: VADER vs Supervised Classifier
        vader = res["vader"]
        v_color = "#00E676" if vader["label"] == "Positive" else ("#FF3366" if vader["label"] == "Negative" else "#00F0FF")
        consensus = "AGREEMENT" if res["models_agree"] else "DIVERGENCE"
        c_badge = "#00E676" if res["models_agree"] else "#FFB300"

        st.markdown(
            f"""
            <div class="hud-panel">
                <div class="hud-title">DUAL-MODEL VALIDATION: SUPERVISED ML vs VADER BASELINE</div>
                <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1rem; margin-bottom:0.75rem;">
                    <div style="background:#0E131F; border:1px solid #212C45; padding:0.75rem; border-radius:6px;">
                        <div style="font-size:0.7rem; color:#64748B; font-family:'JetBrains Mono';">SUPERVISED TF-IDF PIPELINE</div>
                        <div style="font-size:1.1rem; font-weight:700; color:{badge_color}; font-family:'JetBrains Mono';">{label.upper()}</div>
                        <div style="font-size:0.75rem; color:#94A3B8;">Confidence: {res['confidence']*100:.1f}%</div>
                    </div>
                    <div style="background:#0E131F; border:1px solid #212C45; padding:0.75rem; border-radius:6px;">
                        <div style="font-size:0.7rem; color:#64748B; font-family:'JetBrains Mono';">VADER LEXICON BASELINE</div>
                        <div style="font-size:1.1rem; font-weight:700; color:{v_color}; font-family:'JetBrains Mono';">{vader['label'].upper()}</div>
                        <div style="font-size:0.75rem; color:#94A3B8;">Compound: {vader['compound']:+.3f}</div>
                    </div>
                    <div style="background:#0E131F; border:1px solid #212C45; padding:0.75rem; border-radius:6px;">
                        <div style="font-size:0.7rem; color:#64748B; font-family:'JetBrains Mono';">MODEL CONSENSUS</div>
                        <div style="font-size:1.1rem; font-weight:700; color:{c_badge}; font-family:'JetBrains Mono';">{consensus}</div>
                        <div style="font-size:0.75rem; color:#94A3B8;">{'Both models concur' if res['models_agree'] else 'Subtle/domain difference'}</div>
                    </div>
                </div>
                <div style="font-size:0.8rem; color:#94A3B8; font-family:'JetBrains Mono';">
                    VADER Pos Density: {vader['scores']['pos']*100:.1f}% | Neu: {vader['scores']['neu']*100:.1f}% | Neg: {vader['scores']['neg']*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
