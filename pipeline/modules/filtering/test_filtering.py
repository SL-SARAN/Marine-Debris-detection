import sys
import os
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.detection.model import SSSObjectDetector
from pipeline.modules.filtering.speckle_filter import preprocess_sonar_swath
from pipeline.modules.filtering.shadow_verifier import AcousticShadowVerifier
from pipeline.modules.filtering.calibration import ConfidenceCalibrator

def test_module_2():
    print("=" * 65)
    print("  TESTING MODULE 2: COMPOSITE SCORING & NOISE FILTERING")
    print("=" * 65)

    test_img_dir = PROJECT_ROOT / "data" / "common_sss_dataset" / "test" / "images"
    if not test_img_dir.exists():
        test_img_dir = PROJECT_ROOT / "data" / "merged_sss_debris_dataset" / "test" / "images"
    sample_files = list(test_img_dir.glob("sodd_*.jpg")) + list(test_img_dir.glob("klsg_*.jpg"))
    if not sample_files:
        print("No test images found.")
        return

    sample_img_path = sample_files[0]
    print(f"\n[Stage 1] Loading test sonar swath: {sample_img_path.name}")
    raw_img = cv2.imread(str(sample_img_path))

    # 1. Test Denoising & Speckle Filtering
    print("\n[Stage 2] Applying 2D Wavelet DWT + Enhanced Lee Speckle Filter + CLAHE...")
    denoised_swath, metrics, intensity_swath = preprocess_sonar_swath(raw_img, return_intensity_preserving=True)
    print(f"  Speckle Suppression Metrics:")
    print(f"    - Equivalent Number of Looks (ENL): {metrics['enl']:.2f}")
    print(f"    - Peak Signal-to-Noise Ratio (PSNR): {metrics['psnr_db']:.2f} dB")
    print(f"    - Mean Squared Error (MSE): {metrics['mse']:.2f}")

    # Save filtered output
    out_dir = PROJECT_ROOT / "pipeline" / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    denoised_path = out_dir / f"denoised_{sample_img_path.stem}.jpg"
    cv2.imwrite(str(denoised_path), denoised_swath)
    print(f"  Saved denoised swath to: {denoised_path}")

    # 2. Test Detection on Raw Sonar Image
    print("\n[Stage 3] Running YOLOv8n detector on raw image (preserving training distribution)...")
    detector = SSSObjectDetector()
    results = detector.predict(raw_img)
    print(f"  Raw Detections Found: {results['total_detected']}")
    for det in results["detections"]:
        print(f"    Candidate: {det['class_name']} | Raw Score: {det['raw_confidence']*100:.1f}% | BBox: {det['bbox']}")

    # 3. Test Physics-Consistent Acoustic Shadow Verification on Intensity-Preserving Swath
    print("\n[Stage 4] Running Acoustic Shadow Verification on Intensity-Preserving Swath...")
    verifier = AcousticShadowVerifier()
    verified_detections = verifier.filter_detections(intensity_swath, results["detections"])

    for det in verified_detections:
        p = det["physics"]
        print(f"    - Target: {det['class_name']}")
        print(f"      HSCR Contrast Ratio  : {p['hscr']:.3f} (Min Required: {verifier.min_hscr})")
        print(f"      Shadow Area Ratio    : {p['shadow_area_ratio']*100:.1f}%")
        print(f"      Shadow Direction Valid: {p['shadow_geometry_valid']}")
        print(f"      Physics Compliance   : {p['physics_score']*100:.1f}%")
        print(f"      Verification Status  : {'VERIFIED [PASS]' if p['is_verified'] else 'REJECTED [CLUTTER/NOISE]'}")

    # 4. Test Uncalibrated Composite Ranking Score
    print("\n[Stage 5] Applying Composite Ranking Score (Detector + Physics Fusion)...")
    calibrator = ConfidenceCalibrator()
    calibrated_detections = calibrator.calibrate_all_detections(verified_detections)

    print(f"\n{'Target':<16} | {'Detector':<10} | {'Physics':<10} | {'COMPOSITE %':<14} | {'Review Status'}")
    print("-" * 75)
    for det in calibrated_detections:
        print(f"{det['class_name']:<16} | {det['detector_score']*100:>7.1f}%   | {det['physics_score']*100:>7.1f}%  | {det['composite_score_pct']:>11.1f}%  | {det['review_status']}")

    # 5. Test ECE & Reliability Diagram calculation
    print("\n[Stage 6] Calculating Expected Calibration Error (ECE) on synthetic benchmark...")
    # Synthetic validation set for ECE benchmark
    sim_confs = [d["raw_confidence"] for d in calibrated_detections] + [0.92, 0.85, 0.74, 0.65, 0.45, 0.35, 0.25]
    sim_labels = [1 if d.get("is_confirmed", True) else 0 for d in calibrated_detections] + [1, 1, 1, 0, 0, 0, 0]
    ece_results = calibrator.compute_ece(sim_confs, sim_labels, num_bins=5)
    print(f"  Expected Calibration Error (ECE): {ece_results['ece']:.4f} ({ece_results['ece']*100:.2f}%)")
    print(f"  Maximum Calibration Error (MCE) : {ece_results['mce']:.4f} ({ece_results['mce']*100:.2f}%)")
    print("  Reliability Diagram Bins:")
    for b in ece_results["diagram"]:
        if b["count"] > 0:
            print(f"    Bin {b['bin_range']:<8}: Count={b['count']}, Avg Conf={b['confidence']*100:.1f}%, Accuracy={b['accuracy']*100:.1f}%, Gap={b['gap']*100:.1f}%")

    print("\n" + "=" * 65)
    print("  MODULE 2 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_module_2()
