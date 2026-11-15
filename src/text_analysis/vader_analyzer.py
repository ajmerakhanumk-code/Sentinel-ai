"""
Sentinel AI - VADER Sentiment Analysis Module
Lexicon and rule-based sentiment intensity analysis baseline.
"""

from typing import Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config

# Initialize singleton analyzer instance
_vader_analyzer = SentimentIntensityAnalyzer()

def analyze_vader_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes input text using VADER rule-based sentiment analyzer.
    Returns compound score, class label, confidence, and score breakdown.
    """
    if not text or not text.strip():
        return {
            "label": "Neutral",
            "compound": 0.0,
            "confidence": 0.0,
            "scores": {"pos": 0.0, "neu": 1.0, "neg": 0.0},
            "explanation": "No text provided; default neutral baseline assigned."
        }
    
    scores = _vader_analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    # Determine classification according to standardized VADER conventions
    if compound >= config.VADER_POS_THRESHOLD:
        label = "Positive"
        # Confidence scaled from compound strength [0.05, 1.0] -> [0.55, 0.99]
        confidence = min(0.99, max(0.55, abs(compound)))
        explanation = (
            f"VADER classified text as POSITIVE with compound valence score of {compound:+.3f}. "
            f"Positive lexicon density: {scores['pos']*100:.1f}%, Neutral: {scores['neu']*100:.1f}%, "
            f"Negative: {scores['neg']*100:.1f}%."
        )
    elif compound <= config.VADER_NEG_THRESHOLD:
        label = "Negative"
        confidence = min(0.99, max(0.55, abs(compound)))
        explanation = (
            f"VADER classified text as NEGATIVE with compound valence score of {compound:+.3f}. "
            f"Negative lexicon density: {scores['neg']*100:.1f}%, Neutral: {scores['neu']*100:.1f}%, "
            f"Positive: {scores['pos']*100:.1f}%."
        )
    else:
        label = "Neutral"
        # For neutral texts, high neutrality score indicates high confidence in neutral label
        confidence = min(0.95, max(0.50, scores["neu"]))
        explanation = (
            f"VADER classified text as NEUTRAL with compound valence score of {compound:+.3f} "
            f"(within neutral bounds [-0.05, +0.05]). Factual/neutral lexicon density: {scores['neu']*100:.1f}%."
        )
    
    return {
        "label": label,
        "compound": round(compound, 4),
        "confidence": round(confidence, 4),
        "scores": {
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4)
        },
        "explanation": explanation
    }
