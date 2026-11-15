"""
SENTINEL AI - Configuration & System Parameters
Academic Engineering Multimodal Sentiment Intelligence Platform
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
DATA_DIR = ASSETS_DIR / "data"
CSS_DIR = ASSETS_DIR / "css"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSS_DIR.mkdir(parents=True, exist_ok=True)

# System Metadata
APP_NAME = "SENTINEL AI"
APP_SUBTITLE = "Multimodal Sentiment Analysis Engine"
APP_VERSION = "2.4.0"
SYSTEM_ID = "SENTINEL-NODE-01"

# Target Sentiment Classes
CLASSES = ["Positive", "Negative", "Neutral"]

# Model Asset Paths
TEXT_MODEL_PATH = MODELS_DIR / "text_sentiment_pipeline.joblib"
TEXT_METRICS_PATH = MODELS_DIR / "text_model_metrics.json"
TRAINING_DATA_PATH = DATA_DIR / "sentiment_dataset.csv"
FER_ONNX_PATH = MODELS_DIR / "fer_lightweight.onnx"

# VADER Sentiment Thresholds
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# Vision Processing Constraints (optimized for 8GB RAM CPU execution)
VISION_FPS_LIMIT = 15
FACE_DETECTION_CONFIDENCE = 0.55
FACIAL_MESH_MAX_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Visual Command-Centre Palette
THEME = {
    "bg_primary": "#0A0D14",
    "bg_secondary": "#111624",
    "bg_panel": "#171E30",
    "bg_card": "#1E263C",
    "border_subtle": "#25314C",
    "border_active": "#00F0FF",
    "accent_cyan": "#00F0FF",
    "accent_green": "#00E676",
    "accent_amber": "#FFB300",
    "accent_red": "#FF3366",
    "text_bright": "#F1F5F9",
    "text_muted": "#8A99AD",
    "text_dim": "#57677F",
}

# Ethical & Academic Disclaimers
ETHICAL_DISCLAIMER = (
    "ACADEMIC NOTICE: Facial expressions detected via computer vision represent observable "
    "muscular action units (FACS cues) and do NOT constitute biometric ground-truth emotion. "
    "Affective displays can be voluntary, culturally modulated, or masked. Results must be "
    "interpreted strictly as behavioral correlates."
)
