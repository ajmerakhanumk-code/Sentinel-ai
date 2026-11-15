"""
Sentinel AI - Multimodal Fusion & Concordance Engine
Combines text sentiment and vision facial cues using transparent Bayesian-weighted
fusion and cognitive divergence analysis (e.g. Sarcasm, Masked Emotion).
"""

from typing import Dict, Any, Optional
import config

class MultimodalFusionEngine:
    """Orchestrates cross-modal comparison, divergence diagnostics, and uncertainty calculation."""

    def evaluate_fusion(
        self,
        text_result: Optional[Dict[str, Any]],
        vision_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesizes text sentiment and vision expression cues.
        Detects agreement, conflict, sarcasm, or unimodal incompleteness.
        """
        has_text = text_result is not None and text_result.get("status") == "SUCCESS"
        has_vision = vision_result is not None and vision_result.get("status") == "SUCCESS"

        # Case 1: Neither modality available
        if not has_text and not has_vision:
            return {
                "modality_status": "NONE",
                "status_code": "INCOMPLETE",
                "agreement_status": "NO DATA",
                "combined_sentiment": "Indeterminate",
                "confidence": 0.0,
                "uncertainty_level": "High (Missing Inputs)",
                "diagnostic_summary": "Neither text input nor facial cue data is currently available.",
                "actionable_insight": "Provide text in the Text Analysis panel or activate camera capture to generate multimodal intelligence.",
                "disclaimer": config.ETHICAL_DISCLAIMER
            }

        # Case 2: Only Text available
        if has_text and not has_vision:
            t_label = text_result["label"]
            t_conf = text_result["confidence"]
            return {
                "modality_status": "UNIMODAL_TEXT",
                "status_code": "PARTIAL",
                "agreement_status": "UNIMODAL (Text Only)",
                "combined_sentiment": t_label,
                "confidence": t_conf,
                "uncertainty_level": "Moderate (Single Modality)",
                "text_summary": f"Text: {t_label} ({t_conf*100:.1f}%)",
                "vision_summary": "Vision: Not available / No face detected",
                "diagnostic_summary": (
                    f"Analysis is based exclusively on written linguistic cues ({t_label.upper()}). "
                    f"Visual facial verification is absent. Result represents unimodal text sentiment."
                ),
                "actionable_insight": "Capture a facial cue image via webcam to evaluate cross-modal alignment or sarcasm.",
                "disclaimer": config.ETHICAL_DISCLAIMER
            }

        # Case 3: Only Vision available
        if has_vision and not has_text:
            v_label = vision_result["label"]
            v_conf = vision_result["confidence"]
            return {
                "modality_status": "UNIMODAL_VISION",
                "status_code": "PARTIAL",
                "agreement_status": "UNIMODAL (Vision Only)",
                "combined_sentiment": v_label,
                "confidence": v_conf,
                "uncertainty_level": "Moderate (Observable Cue Only)",
                "text_summary": "Text: Not provided",
                "vision_summary": f"Vision: {v_label} ({v_conf*100:.1f}%)",
                "diagnostic_summary": (
                    f"Analysis is based exclusively on observable facial Action Unit displays ({v_label.upper()}). "
                    f"Facial cues reflect physical movements, not ground-truth emotional state."
                ),
                "actionable_insight": "Enter associated written text to test for linguistic congruence.",
                "disclaimer": config.ETHICAL_DISCLAIMER
            }

        # Case 4: BIMODAL (Both Text and Vision present)
        t_label = text_result["label"]
        t_conf = text_result["confidence"]
        v_label = vision_result["label"]
        v_conf = vision_result["confidence"]

        is_agreement = (t_label == v_label)

        if is_agreement:
            agreement_status = "CONGRUENT (Agreement)"
            combined_sentiment = t_label
            # Cross-modal reinforcement boost
            combined_conf = round(min(0.98, ((t_conf + v_conf) / 2.0) + 0.08), 4)
            uncertainty_level = "Low (Cross-Modal Reinforcement)"
            
            diagnostic_summary = (
                f"Linguistic sentiment and facial behavioral cues are concordant ({t_label.upper()}). "
                f"The observable Action Units (AU) align with the lexical tone of the text, providing "
                f"high confidence across independent observation channels."
            )
            scenario_flag = "REINFORCED_CONGRUENCE"
            actionable_insight = "High reliability. Both modalities mutually confirm the classification."
        else:
            agreement_status = "INCONGRUENT (Divergence)"
            uncertainty_level = "High (Epistemic Cross-Modal Conflict)"
            
            # Sarcasm / Irony / Politeness: Negative Text + Positive Face
            if t_label == "Negative" and v_label == "Positive":
                combined_sentiment = "Incongruent (Sarcasm / Politeness Mask)"
                combined_conf = round((t_conf + v_conf) / 2.0, 4)
                scenario_flag = "POTENTIAL_SARCASM"
                diagnostic_summary = (
                    "CRITICAL DIVERGENCE: The text conveys negative sentiment, while facial cues exhibit "
                    "positive smile markers (AU12). In affective computing, this pattern is a classic marker of "
                    "sarcasm, forced professional politeness, or masked frustration."
                )
                actionable_insight = "Context suggests non-literal communication or masked negative affect."

            # Masked Conflict / Cognitive Load: Positive Text + Negative Face
            elif t_label == "Positive" and v_label == "Negative":
                combined_sentiment = "Incongruent (Ambivalence / Cognitive Load)"
                combined_conf = round((t_conf + v_conf) / 2.0, 4)
                scenario_flag = "MASKED_STRAIN"
                diagnostic_summary = (
                    "DIVERGENCE DETECTED: The text expresses positive affirmation, yet facial cues display "
                    "corrugator brow furrowing (AU4) or mouth corner depression (AU15). This indicates "
                    "underlying skepticism, hesitance, or high cognitive difficulty despite positive words."
                )
                actionable_insight = "Review for unvoiced reservations or stress during affirmation."

            # Neutral Text + Polar Face
            elif t_label == "Neutral" and v_label in ["Positive", "Negative"]:
                combined_sentiment = f"Affective Leakage ({v_label})"
                combined_conf = round(v_conf * 0.85, 4)
                scenario_flag = "AFFECTIVE_LEAKAGE"
                diagnostic_summary = (
                    f"Objective/factual text paired with observable {v_label.lower()} facial cues. "
                    f"Suggests the subject experiences an affective reaction to factual subject matter."
                )
                actionable_insight = "Observe whether facial reaction reflects personal engagement with neutral facts."

            # Polar Text + Neutral Face
            else:
                combined_sentiment = f"Controlled Display ({t_label} Text)"
                combined_conf = round(t_conf * 0.85, 4)
                scenario_flag = "CONTROLLED_AFFECT"
                diagnostic_summary = (
                    f"The text carries {t_label.lower()} sentiment, while the face remains in rest-state neutral posture. "
                    f"Typical of professional, restrained communication or high emotional composure."
                )
                actionable_insight = "Subject maintains composed facial presentation despite emotional text content."

        return {
            "modality_status": "BIMODAL",
            "status_code": "COMPLETE",
            "agreement_status": agreement_status,
            "is_agreement": is_agreement,
            "scenario_flag": scenario_flag,
            "combined_sentiment": combined_sentiment,
            "confidence": combined_conf,
            "uncertainty_level": uncertainty_level,
            "text_summary": f"Text: {t_label} ({t_conf*100:.1f}%)",
            "vision_summary": f"Vision: {v_label} ({v_conf*100:.1f}%)",
            "diagnostic_summary": diagnostic_summary,
            "actionable_insight": actionable_insight,
            "disclaimer": config.ETHICAL_DISCLAIMER
        }

# Global singleton fusion engine
fusion_engine = MultimodalFusionEngine()
