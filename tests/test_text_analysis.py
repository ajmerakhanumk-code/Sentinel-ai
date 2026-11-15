"""
Unit Tests for Sentinel AI Text Sentiment Analysis Module
Validates positive, negative, neutral, empty, and edge cases.
"""

import unittest
from src.text_analysis.classifier import text_engine
from src.text_analysis.vader_analyzer import analyze_vader_sentiment
import config

class TestTextSentimentModule(unittest.TestCase):

    def test_model_loaded(self):
        """Verifies that the trained pipeline is initialized and loaded."""
        self.assertIsNotNone(text_engine.pipeline)
        self.assertTrue(config.TEXT_MODEL_PATH.exists())

    def test_positive_text(self):
        """Verifies correct classification of clear positive input."""
        res = text_engine.analyze("The system performance exceeded our expectations and was absolutely wonderful!")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["label"], "Positive")
        self.assertGreater(res["confidence"], 0.40)
        self.assertIn("Positive", res["probabilities"])

    def test_negative_text(self):
        """Verifies correct classification of clear negative input."""
        res = text_engine.analyze("The application crashed abruptly with severe bugs and terrible errors.")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["label"], "Negative")
        self.assertGreater(res["confidence"], 0.40)

    def test_neutral_text(self):
        """Verifies correct classification of factual neutral input."""
        res = text_engine.analyze("The server returned status code 200 indicating successful transmission.")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["label"], "Neutral")

    def test_empty_input(self):
        """Verifies graceful handling of empty string and whitespace."""
        res_empty = text_engine.analyze("")
        self.assertEqual(res_empty["status"], "EMPTY_INPUT")
        self.assertEqual(res_empty["label"], "Neutral")

        res_space = text_engine.analyze("    \n\t  ")
        self.assertEqual(res_space["status"], "EMPTY_INPUT")

    def test_unusual_input(self):
        """Verifies handling of special characters, symbols, and punctuation."""
        res = text_engine.analyze("### 404_ERR_UNKNOWN_SYMBOL ??!!! @@ 12345")
        self.assertIn(res["status"], ["SUCCESS", "EMPTY_INPUT"])
        self.assertIn(res["label"], ["Positive", "Negative", "Neutral"])

    def test_vader_baseline(self):
        """Verifies VADER polarity analyzer output structure."""
        v = analyze_vader_sentiment("This is a superb and delightful experience.")
        self.assertEqual(v["label"], "Positive")
        self.assertGreater(v["compound"], 0.05)
        self.assertIn("pos", v["scores"])

if __name__ == "__main__":
    unittest.main()
