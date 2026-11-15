"""
Sentinel AI - Vision Facial Expression Classifier
Evaluates extracted FACS Action Unit features using the trained scikit-learn model.
Provides calibrated probabilities, feature breakdown, and academic cue explanations.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

import config

class FacialExpressionEngine:
    """Manages inference, feature evaluation, and academic reporting for facial cues."""

    def __init__(self):
        self.pipeline = None
        self.metrics = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Loads serialized model or initiates training if missing."""
        model_path = config.MODELS_DIR / "vision_expression_pipeline.joblib"
        if model_path.exists():
            try:
                self.pipeline = joblib.load(model_path)
            except Exception as e:
                print(f"[Warning] Failed to load vision model: {e}. Retraining...")
                self._train()
        else:
            self._train()

        metrics_path = config.MODELS_DIR / "vision_model_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = None

    def _train(self):
        """Triggers vision model training."""
        from src.vision_analysis.train_vision_model import train_and_evaluate_vision_model
        self.metrics = train_and_evaluate_vision_model()
        model_path = config.MODELS_DIR / "vision_expression_pipeline.joblib"
        self.pipeline = joblib.load(model_path)

    def classify_cues(self, features: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Classifies facial expression based on normalized FACS Action Unit metrics.
        Returns label, confidence, probability distribution, and physical cue interpretations.
        """
        if not features:
            return {
                "status": "NO_FACE_DETECTED",
                "label": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.333, "Negative": 0.333, "Neutral": 0.334},
                "features": {},
                "cues": ["Sensor active but no face within detection bounds."],
                "explanation": "No facial cues available for classification. Face must be centered in frame.",
                "academic_notice": config.ETHICAL_DISCLAIMER
            }

        # Format input vector
        feature_order = [
            "smile_curvature", "mouth_aspect_ratio", "brow_furrow_index",
            "eye_aperture_ratio", "lip_corner_depression"
        ]
        x_vec = np.array([[features.get(f, 0.0) for f in feature_order]], dtype=float)

        classes = list(self.pipeline.named_steps["classifier"].classes_)
        proba_list = self.pipeline.predict_proba(x_vec)[0]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba_list)}

        pred_label = str(self.pipeline.predict(x_vec)[0])
        confidence = round(float(probabilities[pred_label]), 4)

        # Interpret observable physical cues (FACS)
        cues = []
        sc = features.get("smile_curvature", 0.0)
        bf = features.get("brow_furrow_index", 0.0)
        mar = features.get("mouth_aspect_ratio", 2.5)
        lcd = features.get("lip_corner_depression", 0.0)

        if sc > 0.12:
            cues.append(f"Zygomaticus major contraction detected (AU12 smile curvature: {sc:+.2f})")
        elif sc < -0.08:
            cues.append(f"Depressor anguli oris activation observed (AU15 downturn: {sc:+.2f})")

        if bf > 0.35:
            cues.append(f"Corrugator supercilii tension noted (AU4 brow furrow: {bf:.2f})")
        else:
            cues.append(f"Brow musculature relaxed (AU4 index: {bf:.2f})")

        if mar > 3.0:
            cues.append(f"Oral aperture widened (AU25/26 aspect ratio: {mar:.2f})")

        if not cues:
            cues.append("Rest-state facial geometry within baseline neutral tolerance.")

        # Construct explanation
        explanation = (
            f"FACS Action Unit classifier categorized observable cue as '{pred_label.upper()}' "
            f"with {confidence*100:.1f}% confidence. Primary markers: {'; '.join(cues[:2])}."
        )

        return {
            "status": "SUCCESS",
            "label": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": features,
            "cues": cues,
            "explanation": explanation,
            "academic_notice": config.ETHICAL_DISCLAIMER
        }

# Global singleton expression engine
expression_engine = FacialExpressionEngine()
