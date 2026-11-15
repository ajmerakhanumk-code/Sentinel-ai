"""
Sentinel AI - Guided Demonstration Mode
Preloaded academic scenarios demonstrating real model execution across
congruent, sarcastic, ambivalent, and neutral multimodal states.
"""

import streamlit as st
from src.text_analysis.classifier import text_engine
from src.vision_analysis.expression_classifier import expression_engine
from src.multimodal.fusion_engine import fusion_engine
from src.utils.session_state import log_analysis

DEMO_SCENARIOS = {
    "SCENARIO_1": {
        "title": "Scenario 1: Sincere Praise (Congruent Positive)",
        "badge": "CONGRUENT POSITIVE",
        "description": "High-confidence alignment between enthusiastic text and genuine smiling facial cues.",
        "text": "The system performance exceeded our expectations significantly and was wonderful!",
        "features": {
            "smile_curvature": 0.32,
            "mouth_aspect_ratio": 3.4,
            "brow_furrow_index": 0.09,
            "eye_aperture_ratio": 0.23,
            "lip_corner_depression": 0.03
        },
        "expected": "Congruent Positive (High Mutual Reinforcement)"
    },
    "SCENARIO_2": {
        "title": "Scenario 2: Sarcastic Feedback (Cognitive Divergence)",
        "badge": "SARCASM / IRONY",
        "description": "Negative linguistic text paired with positive smiling facial cues (AU12).",
        "text": "Oh wonderful, another catastrophic database corruption error right before the live presentation.",
        "features": {
            "smile_curvature": 0.29,
            "mouth_aspect_ratio": 3.2,
            "brow_furrow_index": 0.14,
            "eye_aperture_ratio": 0.25,
            "lip_corner_depression": 0.04
        },
        "expected": "Incongruent Divergence (Sarcasm / Politeness Facade)"
    },
    "SCENARIO_3": {
        "title": "Scenario 3: Masked Hesitation (Cognitive Strain)",
        "badge": "MASKED STRAIN",
        "description": "Positive verbal affirmation accompanied by furrowed brow (AU4) and downturned lips (AU15).",
        "text": "I feel great about our project progress and team dedication, everything is totally fine.",
        "features": {
            "smile_curvature": -0.22,
            "mouth_aspect_ratio": 2.1,
            "brow_furrow_index": 0.70,
            "eye_aperture_ratio": 0.31,
            "lip_corner_depression": 0.48
        },
        "expected": "Incongruent Divergence (Ambivalence / Cognitive Conflict)"
    },
    "SCENARIO_4": {
        "title": "Scenario 4: Technical Telemetry Report (Congruent Neutral)",
        "badge": "CONGRUENT NEUTRAL",
        "description": "Objective factual statement with rest-state baseline facial musculature.",
        "text": "The server returned status code 200 indicating successful transmission of telemetry.",
        "features": {
            "smile_curvature": 0.01,
            "mouth_aspect_ratio": 2.5,
            "brow_furrow_index": 0.16,
            "eye_aperture_ratio": 0.32,
            "lip_corner_depression": 0.07
        },
        "expected": "Congruent Neutral (Objective Information)"
    },
    "SCENARIO_5": {
        "title": "Scenario 5: Critical Defect Incident (Congruent Negative)",
        "badge": "CONGRUENT NEGATIVE",
        "description": "Severe critical complaint aligned with intense corrugator furrowing and displeasure.",
        "text": "The application crashed abruptly with an unhandled memory exception and severe bug.",
        "features": {
            "smile_curvature": -0.28,
            "mouth_aspect_ratio": 1.9,
            "brow_furrow_index": 0.76,
            "eye_aperture_ratio": 0.29,
            "lip_corner_depression": 0.54
        },
        "expected": "Congruent Negative (Mutual Reinforcement)"
    }
}

def render_demo_view():
    """Renders the demonstration mode interface."""
    st.markdown("### GUIDED DEMONSTRATION WORKSTATION")
    st.markdown(
        """
        <div style="font-size:0.85rem; color:#94A3B8; margin-bottom:1.5rem; line-height:1.5;">
            Select any preloaded scenario below. The application will execute the <strong>real TF-IDF text pipeline</strong> 
            and the <strong>real FACS Action Unit expression classifier</strong> to demonstrate live cross-modal synthesis, 
            agreement checks, and sarcasm detection.
        </div>
        """,
        unsafe_allow_html=True
    )

    for sc_id, sc in DEMO_SCENARIOS.items():
        st.markdown(
            f"""
            <div class="hud-panel">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                    <div style="font-size:1.05rem; font-weight:700; color:#F1F5F9; font-family:'JetBrains Mono';">
                        {sc['title']}
                    </div>
                    <span class="status-pill idle">{sc['badge']}</span>
                </div>
                <div style="font-size:0.82rem; color:#94A3B8; margin-bottom:0.75rem;">
                    {sc['description']}
                </div>
                <div style="background:#0E131F; border:1px solid #212C45; padding:0.6rem; border-radius:4px; font-size:0.82rem; color:#CBD5E1; margin-bottom:0.75rem;">
                    <strong>Text Payload:</strong> "{sc['text']}"
                </div>
            """,
            unsafe_allow_html=True
        )

        col_run, col_exp = st.columns([1, 3])
        with col_run:
            if st.button(f"EXECUTE {sc_id}", key=f"btn_{sc_id}", use_container_width=True):
                # 1. Execute Real Text Engine
                t_res = text_engine.analyze(sc["text"])
                st.session_state.current_text = sc["text"]
                st.session_state.text_results = t_res

                # 2. Execute Real Vision Engine
                v_res = expression_engine.classify_cues(sc["features"])
                st.session_state.vision_results = v_res

                # 3. Execute Real Multimodal Fusion Engine
                f_res = fusion_engine.evaluate_fusion(t_res, v_res)
                st.session_state.combined_results = f_res

                # 4. Log Session Record
                log_analysis({
                    "text": sc["text"],
                    "text_sentiment": t_res["label"],
                    "text_conf": t_res["confidence"],
                    "face_sentiment": v_res["label"],
                    "face_conf": v_res["confidence"],
                    "agreement": f_res["agreement_status"],
                    "modality": "BIMODAL"
                })

                st.session_state.demo_scenario = sc_id
                st.success(f"Executed {sc_id}! Telemetry propagated to Text, Vision, and Combined Intelligence tabs.")
                st.rerun()

        with col_exp:
            st.markdown(
                f'<div style="font-size:0.78rem; color:#00F0FF; font-family:JetBrains Mono; padding-top:0.4rem;">'
                f'Expected Cognitive Synthesis: {sc["expected"]}</div>',
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)
