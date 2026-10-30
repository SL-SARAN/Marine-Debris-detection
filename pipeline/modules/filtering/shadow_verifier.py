import cv2
import numpy as np

class AcousticShadowVerifier:
    """
    Physics-Consistent Acoustic Shadow Verifier for Side-Scan Sonar (SSS).
    
    Verifies that candidate object detections exhibit genuine acoustic physics:
    1. A high-reflectance backscatter highlight facing the towfish transducer.
    2. A trailing acoustic shadow cast along the acoustic ray propagation vector.
    3. Significant Highlight-to-Shadow Contrast Ratio (HSCR).
    
    Filters out natural seafloor clutter, rock mounds, and speckle anomalies
    that lack physical acoustic shadow coupling.
    """
    def __init__(self, min_hscr=0.35, min_shadow_ratio=0.08, shadow_thresh_pct=35, highlight_thresh_pct=75):
        self.min_hscr = min_hscr
        self.min_shadow_ratio = min_shadow_ratio
        self.shadow_thresh_pct = shadow_thresh_pct
        self.highlight_thresh_pct = highlight_thresh_pct

    def verify_candidate(self, image, bbox, nadir_x=None, sensor_channel=None):
        """
        Evaluates acoustic physics compliance for a single detection candidate.
        
        Parameters:
            image: 2D grayscale image (uint8)
            bbox: [x1, y1, x2, y2]
            nadir_x: X-coordinate of towfish track (defaults to image center if None)
            sensor_channel: 'port', 'starboard', or None (auto-detected from nadir_x)
            
        Returns:
            verification_dict containing:
            - is_verified: bool (passes physical shadow check)
            - hscr: float (Highlight-to-Shadow Contrast Ratio)
            - physics_score: float [0.0, 1.0]
            - highlight_area_ratio: float
            - shadow_area_ratio: float
            - shadow_geometry_valid: bool
            - message: str
        """
        h_img, w_img = image.shape[:2]
        if nadir_x is None:
            nadir_x = w_img // 2

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 - x1 < 6 or y2 - y1 < 6:
            return {
                "is_verified": False,
                "hscr": 0.0,
                "physics_score": 0.1,
                "highlight_area_ratio": 0.0,
                "shadow_area_ratio": 0.0,
                "shadow_geometry_valid": False,
                "message": "Candidate ROI too small for acoustic shadow analysis"
            }

        # Determine acoustic ray propagation direction
        box_center_x = (x1 + x2) / 2.0
        if sensor_channel == "port":
            # Port side: transducer on right, sound travels right-to-left -> shadow to left
            shadow_dir = -1  # left
        elif sensor_channel == "starboard":
            # Starboard side: transducer on left, sound travels left-to-right -> shadow to right
            shadow_dir = 1   # right
        else:
            # Auto-infer from nadir track position
            shadow_dir = 1 if box_center_x >= nadir_x else -1

        # Extract ROI with a margin to capture extended acoustic shadows
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.15)
        
        rx1 = max(0, x1 - pad_x if shadow_dir == -1 else x1)
        rx2 = min(w_img, x2 + pad_x if shadow_dir == 1 else x2)
        ry1 = max(0, y1 - pad_y)
        ry2 = min(h_img, y2 + pad_y)

        roi = image[ry1:ry2, rx1:rx2].astype(np.float32)
        
        # Local ambient background estimate (excluding candidate box)
        ambient_mean = np.median(roi)
        ambient_std = np.std(roi) + 1e-6

        # Adaptive thresholding for acoustic highlight and shadow
        thresh_shadow = np.percentile(roi, self.shadow_thresh_pct)
        thresh_highlight = np.percentile(roi, self.highlight_thresh_pct)

        highlight_mask = roi >= thresh_highlight
        shadow_mask = roi <= thresh_shadow

        highlight_area = np.sum(highlight_mask)
        shadow_area = np.sum(shadow_mask)
        total_pixels = roi.size

        highlight_ratio = highlight_area / (total_pixels + 1e-6)
        shadow_ratio = shadow_area / (total_pixels + 1e-6)

        mean_highlight = np.mean(roi[highlight_mask]) if highlight_area > 0 else ambient_mean
        mean_shadow = np.mean(roi[shadow_mask]) if shadow_area > 0 else ambient_mean

        # Highlight-to-Shadow Contrast Ratio (HSCR)
        # In true acoustic targets, highlight is bright, shadow is near zero
        hscr = (mean_highlight - mean_shadow) / (ambient_mean + 1e-6)

        # Geometric Validation: Check if shadow center is cast behind highlight center
        shadow_geometry_valid = False
        if highlight_area > 4 and shadow_area > 4:
            # Find centroid of highlight vs shadow
            hy, hx = np.where(highlight_mask)
            sy, sx = np.where(shadow_mask)
            
            mean_hx = np.mean(hx)
            mean_sx = np.mean(sx)
            
            # If shadow_dir == 1 (rightwards ray), shadow centroid must be to the right of highlight
            # If shadow_dir == -1 (leftwards ray), shadow centroid must be to the left of highlight
            offset_x = (mean_sx - mean_hx) * shadow_dir
            if offset_x > 0.5:  # Shadow is physically cast away from sensor
                shadow_geometry_valid = True
            elif offset_x > -1.5:  # Margin for irregular rope netting
                shadow_geometry_valid = True

        # Compute physics compliance score [0.0, 1.0]
        score_hscr = np.clip(hscr / 1.2, 0.0, 1.0)
        score_shadow = np.clip(shadow_ratio / 0.20, 0.0, 1.0)
        score_highlight = np.clip(highlight_ratio / 0.15, 0.0, 1.0)
        geom_bonus = 0.20 if shadow_geometry_valid else -0.15

        physics_score = float(np.clip(
            0.40 * score_hscr + 0.25 * score_shadow + 0.15 * score_highlight + geom_bonus,
            0.05, 1.0
        ))

        # Decision rule: requires sufficient HSCR and shadow presence
        is_verified = (
            hscr >= self.min_hscr and
            shadow_ratio >= self.min_shadow_ratio and
            physics_score >= 0.45
        )

        msg = "Verified Acoustic Target" if is_verified else "Suspected Natural Seafloor Clutter / Weak Shadow"

        return {
            "is_verified": bool(is_verified),
            "hscr": float(round(hscr, 3)),
            "physics_score": float(round(physics_score, 3)),
            "highlight_area_ratio": float(round(highlight_ratio, 3)),
            "shadow_area_ratio": float(round(shadow_ratio, 3)),
            "shadow_geometry_valid": bool(shadow_geometry_valid),
            "mean_highlight": float(round(mean_highlight, 1)),
            "mean_shadow": float(round(mean_shadow, 1)),
            "message": msg
        }

    def filter_detections(self, image, detections, nadir_x=None, sensor_channel=None):
        """
        Runs physics verification across all candidate detections in an image.
        Augments each detection with acoustic physics metrics.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        verified_detections = []
        for det in detections:
            v_res = self.verify_candidate(gray, det["bbox"], nadir_x, sensor_channel)
            det_copy = dict(det)
            det_copy["physics"] = v_res
            det_copy["is_physics_verified"] = v_res["is_verified"]
            det_copy["hscr"] = v_res["hscr"]
            det_copy["physics_score"] = v_res["physics_score"]
            verified_detections.append(det_copy)

        return verified_detections
