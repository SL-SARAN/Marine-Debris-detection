import numpy as np
from scipy.optimize import minimize


class ConfidenceCalibrator:
    """
    Composite Ranking & Calibration Engine for Side-Scan Sonar Object Detections.

    IMPORTANT ARCHITECTURAL NOTE:
    Until representative ground-truth validation labels are collected and temperature
    scaling is empirically validated on held-out test data, detection scores are
    presented as an UNCALIBRATED COMPOSITE SCORE (0 - 100), combining YOLO detector
    confidence and acoustic physics shadow verification as a ranking aid.
    It must NOT be presented or interpreted as an empirical probability of correctness.

    Implements:
    1. Composite score calculation combining detector score and acoustic physics evidence.
    2. Temperature Scaling / Platt Scaling framework (to be fitted once ground-truth
       labels are provided).
    3. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) diagnostic tools.
    4. Reliability Diagram data generation.
    """

    def __init__(self, temperature=1.0, physics_weight=0.35):
        """
        Parameters:
            temperature: Scaling parameter T > 0 (T=1.0 by default until fitted on real labels)
            physics_weight: Relative weighting of acoustic physics verification vs model score
        """
        self.temperature = float(temperature)
        self.physics_weight = float(physics_weight)
        self.model_weight = 1.0 - self.physics_weight

    def logit(self, p, eps=1e-5):
        """Converts probability [eps, 1-eps] to logit."""
        p_clipped = np.clip(p, eps, 1.0 - eps)
        return np.log(p_clipped / (1.0 - p_clipped))

    def sigmoid(self, z):
        """Standard sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def scale_probability(self, raw_conf):
        """
        Applies temperature scaling to raw confidence:
        p_calibrated = sigmoid( logit(p_raw) / T )
        """
        z = self.logit(raw_conf)
        calibrated = self.sigmoid(z / self.temperature)
        return float(np.clip(calibrated, 0.01, 0.99))

    def fit_temperature(self, raw_confs, ground_truth_labels):
        """
        Learns optimal temperature T on a representative validation set by minimizing
        Negative Log-Likelihood (NLL).

        NOTE: Must only be called with genuine ground-truth detection matches
        (True Positives = 1, False Positives = 0), NEVER synthetic random labels.
        """
        raw_confs = np.array(raw_confs, dtype=np.float32)
        labels = np.array(ground_truth_labels, dtype=np.float32)
        if len(raw_confs) < 10:
            print("[ConfidenceCalibrator] Insufficient validation samples to fit temperature.")
            return self.temperature

        logits = self.logit(raw_confs)

        def nll_loss(t_param):
            t = t_param[0]
            if t <= 0.01:
                return 1e6
            scaled_p = self.sigmoid(logits / t)
            loss = -np.mean(labels * np.log(scaled_p + 1e-7) + (1.0 - labels) * np.log(1.0 - scaled_p + 1e-7))
            return loss

        res = minimize(nll_loss, x0=[1.0], method="Nelder-Mead")
        self.temperature = float(max(0.1, res.x[0]))
        print(f"[ConfidenceCalibrator] Fitted Temperature T = {self.temperature:.3f} (NLL: {res.fun:.4f})")
        return self.temperature

    def compute_ece(self, confidences, true_labels, num_bins=10):
        """
        Computes Expected Calibration Error (ECE) and bin statistics for Reliability Diagrams.
        ECE = sum_m (|B_m| / N) * | acc(B_m) - conf(B_m) |
        """
        confs = np.array(confidences, dtype=np.float32)
        labels = np.array(true_labels, dtype=np.float32)
        n = len(confs)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        mce = 0.0
        diagram_data = []

        for i in range(num_bins):
            b_min, b_max = bin_boundaries[i], bin_boundaries[i+1]
            mask = (confs >= b_min) & (confs < b_max if i < num_bins - 1 else confs <= b_max)
            count = np.sum(mask)

            if count > 0:
                bin_acc = float(np.mean(labels[mask]))
                bin_conf = float(np.mean(confs[mask]))
                diff = abs(bin_acc - bin_conf)
                ece += (count / n) * diff
                mce = max(mce, diff)

                diagram_data.append({
                    "bin_range": f"{int(b_min*100)}-{int(b_max*100)}%",
                    "count": int(count),
                    "confidence": round(bin_conf, 3),
                    "accuracy": round(bin_acc, 3),
                    "gap": round(diff, 3)
                })
            else:
                diagram_data.append({
                    "bin_range": f"{int(b_min*100)}-{int(b_max*100)}%",
                    "count": 0,
                    "confidence": round((b_min + b_max) / 2, 3),
                    "accuracy": round((b_min + b_max) / 2, 3),
                    "gap": 0.0
                })

        return {
            "ece": float(round(ece, 4)),
            "mce": float(round(mce, 4)),
            "diagram": diagram_data
        }

    def calibrate_detection(self, detection):
        """
        Calculates an uncalibrated composite ranking score (0% to 100%)
        combining detector confidence and acoustic physics verification.

        Explicitly flags that probability calibration is pending ground-truth validation.
        """
        raw_conf = float(detection.get("raw_confidence", 0.5))
        physics = detection.get("physics", {})
        physics_score = float(physics.get("physics_score", 0.5))
        is_physics_verified = bool(physics.get("is_verified", True))

        # Composite heuristic ranking calculation
        if is_physics_verified:
            fused_score = (self.model_weight * raw_conf) + (self.physics_weight * physics_score)
        else:
            # Physics shadow check failed: penalize candidate
            fused_score = (self.model_weight * raw_conf * 0.65) + (self.physics_weight * physics_score * 0.40)

        composite_pct = float(round(np.clip(fused_score * 100.0, 1.0, 99.9), 1))

        # Operational Review Statuses (Ranking aid, NOT probability tiers)
        if composite_pct >= 75.0 and is_physics_verified:
            review_status = "high_priority_review"
            legacy_tier = "CONFIRMED_ANOMALY"  # Deprecated alias
        elif composite_pct >= 45.0:
            review_status = "standard_review"
            legacy_tier = "PROBABLE_HAZARD"    # Deprecated alias
        else:
            review_status = "low_score_review"
            legacy_tier = "LOW_CONFIDENCE_SUSPECT"  # Deprecated alias

        return {
            "detector_score": float(round(raw_conf, 3)),
            "physics_score": float(round(physics_score, 3)),
            "composite_score_pct": composite_pct,
            "score_type": "uncalibrated_composite",
            "calibration_status": "pending_ground_truth",
            "review_status": review_status,
            "score_disclaimer": "Score is a ranking aid; probability calibration is pending ground-truth validation.",
            # Backwards compatibility aliases (marked uncalibrated)
            "calibrated_confidence_pct": composite_pct,
            "confidence_tier": legacy_tier,
            "is_confirmed": (review_status == "high_priority_review"),
        }

    def calibrate_all_detections(self, detections):
        """
        Calculates composite scores for an entire list of detections.
        """
        calibrated_list = []
        for det in detections:
            c_info = self.calibrate_detection(det)
            det_merged = dict(det)
            det_merged.update(c_info)
            calibrated_list.append(det_merged)
        return calibrated_list
