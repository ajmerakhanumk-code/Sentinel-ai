"""
Sentinel AI - Unified Text Sentiment Analysis Classifier
Integrates the trained TF-IDF + Logistic Regression pipeline with VADER baseline,
providing token attribution, confidence scores, and explanatory diagnostics.
"""

import re
import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

import config
from src.text_analysis.vader_analyzer import analyze_vader_sentiment

class TextSentimentEngine:
    """Manages the lifecycle, inference, and explanation for text sentiment models."""

    def __init__(self):
        self.pipeline = None
        self.metrics = None
        self._load_or_initialize_model()

    def _load_or_initialize_model(self):
        """Loads serialized pipeline or triggers training if model file is missing."""
        if config.TEXT_MODEL_PATH.exists():
            try:
                self.pipeline = joblib.load(config.TEXT_MODEL_PATH)
            except Exception as e:
                print(f"[Warning] Failed to load model from disk: {e}. Retraining...")
                self._train_and_save()
        else:
            self._train_and_save()

        # Load metrics if available
        if config.TEXT_METRICS_PATH.exists():
            try:
                with open(config.TEXT_METRICS_PATH, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                self.metrics = None

    def _train_and_save(self):
        """Calls training pipeline module to generate model file."""
        from src.text_analysis.train_text_model import train_and_evaluate_model
        self.metrics = train_and_evaluate_model()
        self.pipeline = joblib.load(config.TEXT_MODEL_PATH)

    def _clean_text(self, text: str) -> str:
        """Sanitizes text, trims whitespace and normalizes spaces."""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_token_attributions(self, text: str, predicted_label: str) -> List[Dict[str, Any]]:
        """
        Calculates feature attribution weights for terms present in input text
        based on the logistic regression coefficients for the predicted class.
        """
        if not self.pipeline:
            return []

        try:
            vectorizer = self.pipeline.named_steps["tfidf"]
            classifier = self.pipeline.named_steps["classifier"]

            feature_names = np.array(vectorizer.get_feature_names_out())
            class_idx = list(classifier.classes_).index(predicted_label)
            class_coefs = classifier.coef_[class_idx]

            # Transform input
            X_vec = vectorizer.transform([text])
            non_zero_indices = X_vec.nonzero()[1]

            attributions = []
            for idx in non_zero_indices:
                feature = feature_names[idx]
                tfidf_val = X_vec[0, idx]
                weight = class_coefs[idx] * tfidf_val
                attributions.append({
                    "token": feature,
                    "importance": round(float(weight), 4),
                    "impact": "Supports" if weight > 0 else "Opposes"
                })

            # Sort by absolute magnitude
            attributions.sort(key=lambda x: abs(x["importance"]), reverse=True)
            return attributions[:5]
        except Exception:
            return []

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Performs dual-model sentiment analysis on input text.
        Returns prediction, confidence, probability distribution,
        lexical attribution, and VADER baseline comparison.
        """
        cleaned = self._clean_text(text)

        # Handle empty or whitespace input gracefully
        if not cleaned:
            return {
                "status": "EMPTY_INPUT",
                "text": text,
                "label": "Neutral",
                "confidence": 0.0,
                "probabilities": {"Positive": 0.333, "Negative": 0.333, "Neutral": 0.334},
                "tokens_analyzed": 0,
                "explanation": "No alphanumeric content provided for analysis. Neutral baseline assigned.",
                "attributions": [],
                "vader": analyze_vader_sentiment(""),
                "models_agree": True
            }

        # 1. ML Model Inference
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        proba_list = self.pipeline.predict_proba([cleaned])[0]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba_list)}

        pred_label = str(self.pipeline.predict([cleaned])[0])
        confidence = round(float(probabilities[pred_label]), 4)

        # 2. VADER Baseline Comparison
        vader_res = analyze_vader_sentiment(cleaned)
        models_agree = (pred_label == vader_res["label"])

        # 3. Token attributions
        attributions = self._extract_token_attributions(cleaned, pred_label)

        # 4. Construct human-interpretable explanation
        if models_agree:
            explanation = (
                f"Both the trained TF-IDF classifier ({confidence*100:.1f}% confidence) and the "
                f"VADER baseline agree on '{pred_label.upper()}'. Key lexical cues: "
                f"{', '.join([f'{a['token']} ({a['importance']:+.2f})' for a in attributions[:3]]) or 'context semantics'}."
            )
        else:
            explanation = (
                f"Model divergence detected: Supervised classifier predicts '{pred_label.upper()}' "
                f"({confidence*100:.1f}% conf), whereas VADER predicts '{vader_res['label'].upper()}' "
                f"(compound {vader_res['compound']:+.2f}). This often occurs in nuanced, technical, "
                f"or domain-specific phrasing."
            )

        token_count = len(cleaned.split())

        return {
            "status": "SUCCESS",
            "text": cleaned,
            "label": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "tokens_analyzed": token_count,
            "explanation": explanation,
            "attributions": attributions,
            "vader": vader_res,
            "models_agree": models_agree
        }

# Global singleton engine instance
text_engine = TextSentimentEngine()
