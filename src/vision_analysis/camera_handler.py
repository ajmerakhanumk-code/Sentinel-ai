"""
Sentinel AI - Camera & Video Frame Processor
Handles browser camera snapshots, frame decoding, detection pipeline execution,
and HUD overlay rendering with zero persistent frame storage.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import io

from src.vision_analysis.detector import face_detector
from src.vision_analysis.expression_classifier import expression_engine

def decode_image_buffer(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decodes image byte buffer into OpenCV BGR numpy array."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame_bgr
    except Exception as e:
        print(f"[Error] Failed to decode image buffer: {e}")
        return None

def process_frame(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Executes complete vision analysis pipeline on a single BGR frame:
    1. Face detection
    2. Action unit extraction
    3. Expression classification
    4. HUD overlay rendering
    Returns analysis dictionary with annotated frame in RGB format.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "status": "INVALID_FRAME",
            "face_detected": False,
            "bbox": None,
            "classification": expression_engine.classify_cues(None),
            "annotated_frame_rgb": None
        }

    # 1. Detect Face
    bbox = face_detector.detect_face(frame_bgr)
    face_detected = bbox is not None

    if face_detected:
        # 2. Extract Action Units
        features = face_detector.extract_action_units(frame_bgr, bbox)
        # 3. Classify Expression
        classification = expression_engine.classify_cues(features)
    else:
        features = None
        classification = expression_engine.classify_cues(None)

    # 4. Render HUD Overlay
    label = classification.get("label", "Neutral")
    conf = classification.get("confidence", 0.0)
    annotated_bgr = face_detector.render_hud_overlay(
        frame_bgr, bbox, label=label, conf=conf, features=features
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return {
        "status": "SUCCESS" if face_detected else "NO_FACE_DETECTED",
        "face_detected": face_detected,
        "bbox": bbox,
        "features": features,
        "classification": classification,
        "annotated_frame_rgb": annotated_rgb
    }

def test_local_camera_availability(camera_index: int = 0) -> bool:
    """Tests if a local hardware webcam device is accessible via OpenCV."""
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return False
        ret, frame = cap.read()
        cap.release()
        return bool(ret and frame is not None and frame.size > 0)
    except Exception:
        return False
