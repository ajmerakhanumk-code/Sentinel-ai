"""
Unit Tests for Sentinel AI Multimodal Fusion Engine
"""

import unittest
from src.multimodal.fusion_engine import fusion_engine

class TestMultimodalFusion(unittest.TestCase):

    def setUp(self):
        self.text_pos = {"status": "SUCCESS", "label": "Positive", "confidence": 0.85}
        self.text_neg = {"status": "SUCCESS", "label": "Negative", "confidence": 0.88}
        self.text_neu = {"status": "SUCCESS", "label": "Neutral", "confidence": 0.75}

        self.face_pos = {"status": "SUCCESS", "label": "Positive", "confidence": 0.82}
        self.face_neg = {"status": "SUCCESS", "label": "Negative", "confidence": 0.80}
        self.face_neu = {"status": "SUCCESS", "label": "Neutral", "confidence": 0.70}

    def test_congruent_positive(self):
        """Verifies multimodal agreement on positive inputs."""
        res = fusion_engine.evaluate_fusion(self.text_pos, self.face_pos)
        self.assertEqual(res["modality_status"], "BIMODAL")
        self.assertTrue(res["is_agreement"])
        self.assertEqual(res["combined_sentiment"], "Positive")
        self.assertGreater(res["confidence"], 0.85)

    def test_sarcasm_divergence(self):
        """Verifies sarcasm detection for negative text + positive face."""
        res = fusion_engine.evaluate_fusion(self.text_neg, self.face_pos)
        self.assertEqual(res["modality_status"], "BIMODAL")
        self.assertFalse(res["is_agreement"])
        self.assertEqual(res["scenario_flag"], "POTENTIAL_SARCASM")
        self.assertIn("Sarcasm", res["combined_sentiment"])

    def test_strain_divergence(self):
        """Verifies strain/ambivalence detection for positive text + negative face."""
        res = fusion_engine.evaluate_fusion(self.text_pos, self.face_neg)
        self.assertEqual(res["modality_status"], "BIMODAL")
        self.assertFalse(res["is_agreement"])
        self.assertEqual(res["scenario_flag"], "MASKED_STRAIN")

    def test_unimodal_text(self):
        """Verifies partial unimodal output when vision is missing."""
        res = fusion_engine.evaluate_fusion(self.text_pos, None)
        self.assertEqual(res["modality_status"], "UNIMODAL_TEXT")
        self.assertEqual(res["combined_sentiment"], "Positive")
        self.assertIn("Partial", res["status_code"].capitalize())

    def test_unimodal_vision(self):
        """Verifies partial unimodal output when text is missing."""
        res = fusion_engine.evaluate_fusion(None, self.face_neg)
        self.assertEqual(res["modality_status"], "UNIMODAL_VISION")
        self.assertEqual(res["combined_sentiment"], "Negative")

    def test_neither_modality(self):
        """Verifies no-data handling."""
        res = fusion_engine.evaluate_fusion(None, None)
        self.assertEqual(res["modality_status"], "NONE")
        self.assertEqual(res["confidence"], 0.0)

if __name__ == "__main__":
    unittest.main()
