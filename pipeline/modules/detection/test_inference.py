import os
import sys
import glob
from pathlib import Path
import cv2

# Add root project directory to path
sys.path.insert(0, r"c:\Users\saran\Downloads\SODD")

def run_test_inference():
    from pipeline.modules.detection.model import SSSObjectDetector

    print("=== TESTING SSS OBJECT DETECTOR ===")
    detector = SSSObjectDetector()

    # Find sample test images
    test_img_dir = Path(r"c:\Users\saran\Downloads\SODD\merged_sss_debris_dataset\test\images")
    sodd_samples = list(test_img_dir.glob("sodd_*.png")) + list(test_img_dir.glob("sodd_*.jpg"))
    klsg_samples = list(test_img_dir.glob("klsg_*.jpg")) + list(test_img_dir.glob("klsg_*.png"))
    
    test_images = sodd_samples[:3] + klsg_samples[:3]
    if not test_images:
        print("No test images found.")
        return

    print(f"Testing {len(test_images)} sample images ({len(sodd_samples[:3])} ghost nets, {len(klsg_samples[:3])} debris)...")

    out_dir = Path(r"c:\Users\saran\Downloads\SODD\pipeline\test_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(test_images[:3]):
        print(f"\nProcessing sample {i+1}: {img_path.name}")
        img = cv2.imread(str(img_path))
        results = detector.predict(img)

        print(f"  Detected: {results['total_detected']} anomalies")
        print(f"  Latency : Preprocess: {results['latency']['preprocess_ms']:.2f}ms | Inference: {results['latency']['inference_ms']:.2f}ms | Postprocess: {results['latency']['postprocess_ms']:.2f}ms")
        for det in results["detections"]:
            print(f"    - {det['class_name']} (conf: {det['raw_confidence']*100:.1f}%) at bbox {det['bbox']}")

        vis = detector.draw_detections(img, results["detections"])
        save_file = out_dir / f"pred_{img_path.stem}.jpg"
        cv2.imwrite(str(save_file), vis)
        print(f"  Saved visual output to: {save_file}")

    print(f"\nTest completed successfully. Visual outputs saved in {out_dir}")

if __name__ == "__main__":
    run_test_inference()
