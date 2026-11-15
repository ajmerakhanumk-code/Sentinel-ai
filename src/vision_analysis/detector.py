"""
Sentinel AI - Enhanced Face Geometry Detector & FACS Action Unit Extractor
Optimized for normal laptop sitting distances (50-80cm) and diverse indoor lighting.
Features CLAHE illumination equalization, dual-space color segmentation,
biometric anthropometric validation, and canonical scale normalization.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class FaceGeometryDetector:
    """Robust, illumination-adaptive face detector and FACS Action Unit morphometric analyzer."""

    def __init__(self):
        # YCrCb color space bounds (calibrated for varied indoor lighting)
        self.lower_ycrcb = np.array([0, 130, 72], dtype=np.uint8)
        self.upper_ycrcb = np.array([255, 185, 138], dtype=np.uint8)

        # HSV color space bounds (secondary verification for warm/cool lamps)
        self.lower_hsv1 = np.array([0, 20, 30], dtype=np.uint8)
        self.upper_hsv1 = np.array([28, 255, 255], dtype=np.uint8)
        self.lower_hsv2 = np.array([165, 20, 30], dtype=np.uint8)
        self.upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)

        # CLAHE for dynamic indoor lighting contrast normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Recent detection cache for tracking stability
        self._prev_bbox: Optional[Tuple[int, int, int, int]] = None
        self._smooth_alpha = 0.65 # Weight for current detection vs previous

    def _equalize_illumination(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Applies CLAHE on luminance channel to neutralize harsh indoor shadows."""
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = self.clahe.apply(y)
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    def detect_face(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects human face bounding box (x, y, w, h) at normal laptop sitting distances.
        Uses dual-space segmentation with adaptive morphological reconstruction.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h_img, w_img = frame_bgr.shape[:2]

        # 1. Illumination Normalization
        frame_balanced = self._equalize_illumination(frame_bgr)

        # 2. Dual-space Chrominance Segmentation (YCrCb + HSV)
        ycrcb = cv2.cvtColor(frame_balanced, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb)

        hsv = cv2.cvtColor(frame_balanced, cv2.COLOR_BGR2HSV)
        mask_hsv1 = cv2.inRange(hsv, self.lower_hsv1, self.upper_hsv1)
        mask_hsv2 = cv2.inRange(hsv, self.lower_hsv2, self.upper_hsv2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # Intersect masks for high precision, then add relaxed YCrCb core
        skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        skin_mask = cv2.bitwise_or(skin_mask, cv2.erode(mask_ycrcb, np.ones((3, 3), np.uint8)))

        # 3. Distance-Adaptive Morphology: clean noise without eroding small faces
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 4. Find Contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Min area threshold calibrated for normal sitting distance (~0.3% of frame area)
        # e.g., in 640x480 (307k px), min_area = 920 px (a 30x35 px face is detected)
        # in 1280x720 (921k px), min_area = 2760 px (a 45x60 px face is detected)
        min_area = (w_img * h_img) * 0.003
        max_area = (w_img * h_img) * 0.85

        frame_cx, frame_cy = w_img / 2.0, h_img / 2.0
        best_box = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(h) / max(1, w)

            # Human face aspect ratio bounds: roughly 1.05 to 2.2
            if not (1.0 <= aspect_ratio <= 2.2):
                continue

            # Minimum dimension check (at least 28px wide/high)
            if w < 28 or h < 32:
                continue

            # Biometric validation: Internal Texture / Gradient Check
            face_crop = frame_bgr[y:y+h, x:x+w]
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray_crop, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
            texture_variance = float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)))

            # Real faces have eyes, eyebrows, nose, and lips creating distinct gradient texture
            if texture_variance < 12.0:
                continue

            # Centrality & Size Scoring:
            # Users sitting in front of laptops are located in the upper-middle frame
            box_cx = x + w / 2.0
            box_cy = y + h / 2.0
            dist_from_center_x = abs(box_cx - frame_cx) / max(1, frame_cx)
            dist_from_center_y = abs(box_cy - frame_cy) / max(1, frame_cy)

            center_score = max(0.0, 1.0 - (dist_from_center_x * 0.5 + dist_from_center_y * 0.5))
            area_score = min(1.0, area / ((w_img * h_img) * 0.15))
            solidity = float(area) / max(1, w * h)

            composite_score = (area_score * 0.35) + (center_score * 0.35) + (solidity * 0.15) + (min(1.0, texture_variance / 60.0) * 0.15)

            if composite_score > best_score:
                best_score = composite_score
                best_box = (x, y, w, h)

        if best_box is None:
            return None

        # 5. Temporal Smoothing (EMA) to prevent frame-to-frame bounding box flicker
        if self._prev_bbox is not None:
            px, py, pw, ph = self._prev_bbox
            cx, cy, cw, ch = best_box
            # If distance between successive boxes is reasonable, smooth
            if abs(cx - px) < w_img * 0.25 and abs(cy - py) < h_img * 0.25:
                sx = int(self._smooth_alpha * cx + (1 - self._smooth_alpha) * px)
                sy = int(self._smooth_alpha * cy + (1 - self._smooth_alpha) * py)
                sw = int(self._smooth_alpha * cw + (1 - self._smooth_alpha) * pw)
                sh = int(self._smooth_alpha * ch + (1 - self._smooth_alpha) * ph)
                best_box = (sx, sy, sw, sh)

        self._prev_bbox = best_box
        return best_box

    def extract_action_units(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Extracts continuous FACS Action Unit metrics from canonicalized face ROI:
        - AU12: smile_curvature (Zygomaticus major)
        - AU25/26: mouth_aspect_ratio (Oral aperture)
        - AU4: brow_furrow_index (Corrugator supercilii)
        - AU6: eye_aperture_ratio (Orbicularis oculi)
        - AU15: lip_corner_depression (Depressor anguli oris)
        """
        x, y, w, h = bbox
        h_img, w_img = frame_bgr.shape[:2]

        # Clamp bounding box within frame borders
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        face_roi = frame_bgr[y:y+h, x:x+w]
        if face_roi.size == 0 or w < 10 or h < 10:
            return {
                "smile_curvature": 0.0,
                "mouth_aspect_ratio": 2.5,
                "brow_furrow_index": 0.2,
                "eye_aperture_ratio": 0.3,
                "lip_corner_depression": 0.1
            }

        # Canonical Rescaling (200x240 pixels) ensures distance-invariant metric extraction
        canonical_w, canonical_h = 200, 240
        face_canonical = cv2.resize(face_roi, (canonical_w, canonical_h), interpolation=cv2.INTER_AREA)

        gray_face = cv2.cvtColor(face_canonical, cv2.COLOR_BGR2GRAY)
        gray_face = self.clahe.apply(gray_face)

        # 1. Mouth Region (Lower 38% of canonical face)
        mouth_y1 = int(canonical_h * 0.62)
        mouth_roi = gray_face[mouth_y1:canonical_h, :]
        mh, mw = mouth_roi.shape[:2]

        # Otsu thresholding in mouth region
        _, mouth_thresh = cv2.threshold(mouth_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        m_contours, _ = cv2.findContours(mouth_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mouth_aspect_ratio = 2.5
        smile_curvature = 0.0
        lip_depression = 0.08

        if m_contours:
            # Largest contour in lower region corresponds to mouth/lips
            largest_m = max(m_contours, key=cv2.contourArea)
            mx, my, mw_box, mh_box = cv2.boundingRect(largest_m)
            if mw_box > mw * 0.18:
                mouth_aspect_ratio = float(mw_box) / max(1, mh_box)

                # Vertical center vs lip corner heights
                left_pt = largest_m[largest_m[:, :, 0].argmin()][0]
                right_pt = largest_m[largest_m[:, :, 0].argmax()][0]
                center_y = my + mh_box / 2.0
                corners_avg_y = (left_pt[1] + right_pt[1]) / 2.0

                # Upward elevation of corners: smile curvature
                diff = center_y - corners_avg_y
                smile_curvature = float(diff / max(1, mh_box))

                if smile_curvature < -0.06:
                    lip_depression = float(abs(smile_curvature) * 1.6)
                else:
                    lip_depression = 0.05

        # 2. Brow / Glabella Region (Top 18% to 38% of canonical face)
        brow_y1 = int(canonical_h * 0.16)
        brow_y2 = int(canonical_h * 0.38)
        brow_roi = gray_face[brow_y1:brow_y2, int(canonical_w*0.25):int(canonical_w*0.75)]

        # Corrugator furrow index: vertical edge density between eyebrows
        sobel_x = cv2.Sobel(brow_roi, cv2.CV_64F, 1, 0, ksize=3)
        brow_furrow_index = float(np.mean(np.abs(sobel_x)) / 90.0)
        brow_furrow_index = min(1.0, max(0.0, brow_furrow_index))

        # 3. Eye Aperture Region (Top 26% to 48% of canonical face)
        eye_y1 = int(canonical_h * 0.26)
        eye_y2 = int(canonical_h * 0.48)
        eye_roi = gray_face[eye_y1:eye_y2, :]
        _, eye_thresh = cv2.threshold(eye_roi, 45, 255, cv2.THRESH_BINARY_INV)
        eye_aspect_ratio = float(np.count_nonzero(eye_thresh) / max(1, eye_thresh.size))
        eye_aperture_ratio = min(0.5, max(0.15, eye_aspect_ratio))

        # Physiological bounding
        smile_curvature = min(0.6, max(-0.5, smile_curvature))
        mouth_aspect_ratio = min(5.0, max(1.2, mouth_aspect_ratio))

        return {
            "smile_curvature": round(smile_curvature, 4),
            "mouth_aspect_ratio": round(mouth_aspect_ratio, 4),
            "brow_furrow_index": round(brow_furrow_index, 4),
            "eye_aperture_ratio": round(eye_aperture_ratio, 4),
            "lip_corner_depression": round(lip_depression, 4)
        }

    def render_hud_overlay(
        self,
        frame_bgr: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        label: str = "SEARCHING",
        conf: float = 0.0,
        features: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Renders dark cyber command center HUD tracking brackets and telemetry onto frame."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # Top diagnostic bar
        cv2.rectangle(vis, (0, 0), (w, 36), (10, 13, 20), -1)
        cv2.line(vis, (0, 36), (w, 36), (0, 240, 255), 1)
        cv2.putText(
            vis, "SENTINEL-VISION // CUE ANALYZER", (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1, cv2.LINE_AA
        )

        if bbox is not None:
            bx, by, bw, bh = bbox

            # Theme color based on classification
            if label == "Positive":
                theme_color = (118, 230, 0) # Green BGR
            elif label == "Negative":
                theme_color = (102, 51, 255) # Red BGR
            else:
                theme_color = (255, 240, 0) # Cyan BGR

            # Glowing corner brackets
            corner_len = min(24, int(bw * 0.18))
            thick = 2

            # Top-Left
            cv2.line(vis, (bx, by), (bx + corner_len, by), theme_color, thick)
            cv2.line(vis, (bx, by), (bx, by + corner_len), theme_color, thick)
            # Top-Right
            cv2.line(vis, (bx + bw, by), (bx + bw - corner_len, by), theme_color, thick)
            cv2.line(vis, (bx + bw, by), (bx + bw, by + corner_len), theme_color, thick)
            # Bottom-Left
            cv2.line(vis, (bx, by + bh), (bx + corner_len, by + bh), theme_color, thick)
            cv2.line(vis, (bx, by + bh), (bx, by + bh - corner_len), theme_color, thick)
            # Bottom-Right
            cv2.line(vis, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), theme_color, thick)
            cv2.line(vis, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), theme_color, thick)

            # Center target crosshair
            cx, cy = bx + bw // 2, by + bh // 2
            cv2.drawMarker(vis, (cx, cy), theme_color, cv2.MARKER_CROSS, 14, 1)

            # Target Telemetry Plate
            cv2.rectangle(vis, (bx, max(0, by - 26)), (bx + 200, max(24, by - 2)), (14, 19, 31), -1)
            cv2.rectangle(vis, (bx, max(0, by - 26)), (bx + 200, max(24, by - 2)), theme_color, 1)
            status_text = f"TARGET: {label.upper()} ({conf*100:.0f}%)"
            cv2.putText(
                vis, status_text, (bx + 6, max(18, by - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1, cv2.LINE_AA
            )

            # Bottom AU HUD metrics if provided
            if features:
                sc = features.get("smile_curvature", 0.0)
                bf = features.get("brow_furrow_index", 0.0)
                au_text = f"AU12(Smile): {sc:+.2f} | AU4(Furrow): {bf:.2f}"
                cv2.rectangle(vis, (bx, min(h - 4, by + bh + 4)), (bx + bw, min(h, by + bh + 24)), (10, 13, 20), -1)
                cv2.putText(
                    vis, au_text, (bx + 4, min(h - 6, by + bh + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA
                )
        else:
            # No face detected indicator
            cv2.putText(
                vis, "[ NO FACE DETECTED IN SENSOR RANGE ]", (w // 2 - 160, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (102, 51, 255), 1, cv2.LINE_AA
            )

        return vis

# Singleton detector instance
face_detector = FaceGeometryDetector()
