"""
Session state manager for Sentinel AI.
Maintains history, modality inputs, and runtime telemetry.
"""

import streamlit as st
import time

def init_session_state():
    """Initializes all necessary session state variables if they do not exist."""
    defaults = {
        "session_start_time": time.time(),
        "history": [],
        "current_text": "",
        "text_results": None,
        "vision_results": None,
        "combined_results": None,
        "active_tab_index": 0,
        "total_inferences": 0,
        "last_inference_time_ms": 0.0,
        "webcam_active": False,
        "demo_scenario": None,
    }
    
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def log_analysis(entry: dict):
    """Appends an analysis record to the session history."""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    record = {
        "timestamp": time.strftime("%H:%M:%S"),
        "text": entry.get("text", "N/A"),
        "text_sentiment": entry.get("text_sentiment", "N/A"),
        "text_conf": entry.get("text_conf", 0.0),
        "face_sentiment": entry.get("face_sentiment", "N/A"),
        "face_conf": entry.get("face_conf", 0.0),
        "agreement": entry.get("agreement", "N/A"),
        "modality": entry.get("modality", "Unimodal"),
    }
    st.session_state.history.append(record)
    st.session_state.total_inferences += 1

def clear_session_history():
    """Clears analysis history."""
    st.session_state.history = []
