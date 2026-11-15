"""
Sentinel AI - Live Webcam & Facial Cue Analysis View
Browser camera capture, local snapshot fallback, face detection,
FACS Action Unit morphometrics, and expression classification.
"""

import streamlit as st
import numpy as np
from PIL import Image

from src.vision_analysis.camera_handler import decode_image_buffer, process_frame
from src.multimodal.fusion_engine import fusion_engine
from src.utils.session_state import log_analysis
from src.ui.components import create_confidence_gauge, create_probability_bar_chart
from src.ui.styles import render_disclaimer
import config

def render_webcam_view():
    """Renders the webcam facial cue analysis workstation."""
    st.markdown("### LIVE WEBCAM & OBSERVABLE FACIAL CUE INTELLIGENCE")

    render_disclaimer()

    col_ctrl, col_info = st.columns([1, 2])
    with col_ctrl:
        camera_toggle = st.toggle("ACTIVATE WEBCAM SENSOR", value=st.session_state.get("webcam_active", False))
        st.session_state.webcam_active = camera_toggle

    with col_info:
        if camera_toggle:
            st.markdown(
                '<div style="font-family:JetBrains Mono; font-size:0.8rem; color:#00E676; padding-top:0.4rem;">'
                '● CAMERA SUBSYSTEM ONLINE: Ready to capture live observable facial cues.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="font-family:JetBrains Mono; font-size:0.8rem; color:#94A3B8; padding-top:0.4rem;">'
                '○ CAMERA SUBSYSTEM OFFLINE: Activate toggle to request browser camera permission.</div>',
                unsafe_allow_html=True
            )

    captured_bytes = None
    input_source = "None"

    if camera_toggle:
        st.markdown(
            '<div style="font-size:0.8rem; color:#94A3B8; font-family:JetBrains Mono; margin:0.75rem 0 0.25rem 0;">'
            'BROWSER CAMERA INPUT [Live Feed & Instant Cue Capture]:</div>',
            unsafe_allow_html=True
        )
        cam_frame = st.camera_input("Sentinel Optical Sensor Input", label_visibility="collapsed")
        if cam_frame is not None:
            captured_bytes = cam_frame.getvalue()
            input_source = "Browser Webcam"

    # Fallback image upload for benchmarking or systems without hardware webcam
    with st.expander("OPTIONAL: Load Image Snapshot from Disk (Fallback / Benchmark Mode)"):
        uploaded_file = st.file_uploader("Upload portrait frame (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            captured_bytes = uploaded_file.getvalue()
            input_source = "Uploaded Image Snapshot"

    # Process frame if available
    if captured_bytes is not None:
        frame_bgr = decode_image_buffer(captured_bytes)
        if frame_bgr is not None:
            with st.spinner("Processing facial geometry and FACS Action Units..."):
                analysis_res = process_frame(frame_bgr)
                st.session_state.vision_results = analysis_res["classification"]

                # Recalculate multimodal fusion
                text_res = st.session_state.get("text_results")
                fused = fusion_engine.evaluate_fusion(text_res, analysis_res["classification"])
                st.session_state.combined_results = fused

                # Log session record
                log_analysis({
                    "text": text_res["text"] if text_res else "N/A",
                    "text_sentiment": text_res["label"] if text_res else "N/A",
                    "text_conf": text_res["confidence"] if text_res else 0.0,
                    "face_sentiment": analysis_res["classification"]["label"],
                    "face_conf": analysis_res["classification"]["confidence"],
                    "agreement": fused["agreement_status"],
                    "modality": fused["modality_status"]
                })

            st.markdown("<hr style='border-color:#212C45; margin:1.5rem 0;'>", unsafe_allow_html=True)
            st.markdown(f"#### VISION INFERENCE RESULTS [{input_source.upper()}]")

            col_annotated, col_telemetry = st.columns([3, 2])

            with col_annotated:
                st.markdown(
                    """
                    <div class="hud-panel">
                        <div class="hud-title">OPTICAL HUD OVERLAY & TRACKING</div>
                    """,
                    unsafe_allow_html=True
                )
                if analysis_res["annotated_frame_rgb"] is not None:
                    st.image(
                        analysis_res["annotated_frame_rgb"],
                        use_container_width=True,
                        caption="Sentinel AI HUD Telemetry with Target Tracking Brackets"
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            with col_telemetry:
                cls_data = analysis_res["classification"]
                v_label = cls_data["label"]
                v_conf = cls_data["confidence"]
                v_color = "#00E676" if v_label == "Positive" else ("#FF3366" if v_label == "Negative" else "#00F0FF")

                if analysis_res["face_detected"]:
                    st.markdown(
                        f"""
                        <div class="hud-panel">
                            <div class="hud-title">FACS ACTION UNIT CLASSIFICATION</div>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                                <div>
                                    <div style="font-size:0.72rem; color:#64748B; font-family:'JetBrains Mono';">OBSERVABLE CUE</div>
                                    <div style="font-size:1.8rem; font-weight:800; color:{v_color}; font-family:'JetBrains Mono';">
                                        {v_label.upper()}
                                    </div>
                                </div>
                                <div style="text-align:right;">
                                    <div style="font-size:0.72rem; color:#64748B; font-family:'JetBrains Mono';">CERTAINTY</div>
                                    <div style="font-size:1.6rem; font-weight:700; color:#F1F5F9; font-family:'JetBrains Mono';">
                                        {v_conf*100:.1f}%
                                    </div>
                                </div>
                            </div>
                            <div style="font-size:0.82rem; line-height:1.5; color:#CBD5E1; background:#0E131F; padding:0.6rem; border-radius:4px; border:1px solid #212C45;">
                                {cls_data['explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Action Unit breakdown
                    if analysis_res.get("features"):
                        fts = analysis_res["features"]
                        st.markdown(
                            f"""
                            <div class="hud-panel">
                                <div class="hud-title">EXTRACTED FACS MORPHOMETRIC METRICS</div>
                                <div style="font-family:'JetBrains Mono'; font-size:0.78rem; line-height:1.8; color:#94A3B8;">
                                    <div>AU12 Lip Corner Puller (Smile): <strong style="color:#00F0FF;">{fts.get('smile_curvature', 0.0):+.3f}</strong></div>
                                    <div>AU4 Corrugator Supercilii (Furrow): <strong style="color:#00F0FF;">{fts.get('brow_furrow_index', 0.0):.3f}</strong></div>
                                    <div>AU15 Lip Depressor (Downturn): <strong style="color:#00F0FF;">{fts.get('lip_corner_depression', 0.0):.3f}</strong></div>
                                    <div>AU25/26 Oral Aperture Ratio: <strong style="color:#00F0FF;">{fts.get('mouth_aspect_ratio', 2.5):.2f}</strong></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Probabilities bar chart
                    prob_fig = create_probability_bar_chart(cls_data["probabilities"])
                    st.plotly_chart(prob_fig, use_container_width=True)
                else:
                    st.markdown(
                        """
                        <div class="hud-panel">
                            <div class="hud-title" style="color:#FF3366;">DETECTION STATUS</div>
                            <div style="background:rgba(255,51,102,0.1); border:1px solid #FF3366; border-radius:6px; padding:1rem; color:#FF3366; font-family:'JetBrains Mono'; font-size:0.85rem;">
                                ⚠ NO HUMAN FACE DETECTED<br>
                                <span style="font-size:0.75rem; color:#E2E8F0;">
                                Ensure adequate lighting and position your face clearly in the camera field of view.
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.markdown(
            '<div style="text-align:center; padding:2.5rem; background:#0E131F; border:1px dashed #212C45; border-radius:8px; margin-top:1rem;">'
            '<div style="font-family:JetBrains Mono; font-size:0.95rem; color:#94A3B8;">'
            'Awaiting Video Frame Capture.<br>'
            '<span style="font-size:0.8rem; color:#64748B;">Activate the camera toggle above and take a snapshot to begin facial cue analysis.</span>'
            '</div></div>',
            unsafe_allow_html=True
        )
