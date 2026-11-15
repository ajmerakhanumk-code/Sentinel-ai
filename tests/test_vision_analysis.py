"""
Unit Tests for Sentinel AI Vision & Facial Cue Module
Validates normal laptop sitting distances, indoor lighting variations,
FACS Action Unit classifications, and blank frame handling.
"""

import unittest
import numpy as np
import cv2

from src.vision_analysis.detector import face_detector
from src.vision_analysis.expression_classifier import expression_engine
from src.vision_analysis.camera_handler import process_frame

class TestVisionModule(unittest.TestCase):

    def setUp(self):
        # 1. Normal sitting distance face frame (640x480 frame, face is ~90x120px = ~2.5% of frame)
        self.normal_distance_frame = np.full((480, 640, 3), 45, dtype=np.uint8) # indoor dim room
        # Add realistic skin tone face at normal distance
        cv2.ellipse(self.normal_distance_frame, (320, 230), (45, 60), 0, 0, 360, (110, 145, 195), -1)
        # Add eyes & mouth internal texture
        cv2.circle(self.normal_distance_frame, (305, 215), 5, (30, 30, 40), -1)
        cv2.circle(self.normal_distance_frame, (335, 215), 5, (30, 30, 40), -1)
        cv2.ellipse(self.normal_distance_frame, (320, 260), (16, 5), 0, 0, 180, (40, 50, 100), -1)

        # 2. 720p laptop webcam frame at sitting distance (~120x160px face = ~2% of 1280x720 frame)
        self.hd_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
        cv2.ellipse(self.hd_frame, (640, 330), (60, 80), 0, 0, 360, (105, 140, 190), -1)
        cv2.circle(self.hd_frame, (620, 310), 7, (30, 30, 40), -1)
        cv2.circle(self.hd_frame, (660, 310), 7, (30, 30, 40), -1)
        cv2.ellipse(self.hd_frame, (640, 370), (22, 7), 0, 0, 180, (40, 50, 100), -1)

        # 3. Blank background frame (no face)
        self.blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_expression_engine_loaded(self):
        """Verifies that the vision pipeline is loaded."""
        self.assertIsNotNone(expression_engine.pipeline)

    def test_normal_distance_face_detection(self):
        """Verifies face detection at normal laptop sitting distance (~60cm)."""
        bbox = face_detector.detect_face(self.normal_distance_frame)
        self.assertIsNotNone(bbox, "Face at normal distance should be detected.")
        x, y, w, h = bbox
        self.assertGreater(w, 20)
        self.assertGreater(h, 25)

    def test_hd_webcam_face_detection(self):
        """Verifies face detection on 720p resolution at normal sitting distance."""
        bbox = face_detector.detect_face(self.hd_frame)
        self.assertIsNotNone(bbox, "Face in 720p frame at normal distance should be detected.")

    def test_positive_cue_classification(self):
        """Verifies classification of positive smile features."""
        features = {
            "smile_curvature": 0.35,
            "mouth_aspect_ratio": 3.6,
            "brow_furrow_index": 0.08,
            "eye_aperture_ratio": 0.22,
            "lip_corner_depression": 0.02
        }
        res = expression_engine.classify_cues(features)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["label"], "Positive")
        self.assertGreater(res["confidence"], 0.50)

    def test_negative_cue_classification(self):
        """Verifies classification of negative frown/furrow features."""
        features = {
            "smile_curvature": -0.25,
            "mouth_aspect_ratio": 2.0,
            "brow_furrow_index": 0.75,
            "eye_aperture_ratio": 0.30,
            "lip_corner_depression": 0.55
        }
        res = expression_engine.classify_cues(features)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["label"], "Negative")
        self.assertGreater(res["confidence"], 0.50)

    def test_neutral_cue_classification(self):
        """Verifies classification of neutral baseline features."""
        features = {
            "smile_curvature": 0.01,
            "mouth_aspect_ratio": 2.5,
            "brow_furrow_index": 0.15,
            "eye_aperture_ratio": 0.32,
            "lip_corner_depression": 0.06
        }
        res = expression_engine.classify_cues(features)
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["label"], "Neutral")

    def test_no_face_handling(self):
        """Verifies handling when no face is present."""
        bbox = face_detector.detect_face(self.blank_frame)
        self.assertIsNone(bbox)
        res = expression_engine.classify_cues(None)
        self.assertEqual(res["status"], "NO_FACE_DETECTED")
        self.assertEqual(res["confidence"], 0.0)

    def test_process_frame_pipeline(self):
        """Verifies complete process_frame pipeline at normal distance."""
        res = process_frame(self.normal_distance_frame)
        self.assertEqual(res["face_detected"], True)
        self.assertIsNotNone(res["bbox"])
        self.assertIsNotNone(res["annotated_frame_rgb"])

if __name__ == "__main__":
    unittest.main()
