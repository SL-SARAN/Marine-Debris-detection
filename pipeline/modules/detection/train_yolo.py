import os
import sys
import time
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def train_detector(
    data_yaml=r"D:\MarineDebris\data\common_sss_dataset\data.yaml",
    model_name="yolov8n.pt",
    epochs=35,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/detect",
    name="yolov8n_common_sss"
):

    print("=" * 60)
    print("  SSS MARINE ANOMALY & GHOST NET DETECTOR TRAINING")
    print("=" * 60)
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device       : {torch.cuda.get_device_name(0)}")
        print(f"CUDA VRAM Total   : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("WARNING: CUDA not detected, falling back to CPU.")
        device = "cpu"

    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    print(f"\nConfiguration:")
    print(f"  Dataset YAML    : {data_path}")
    print(f"  Base Architecture: {model_name}")
    print(f"  Input Resolution: {imgsz}x{imgsz}")
    print(f"  Batch Size      : {batch}")
    print(f"  Target Epochs   : {epochs}")
    print(f"  Device          : {device}")

    # Load baseline model
    print(f"\nLoading baseline {model_name}...")
    model = YOLO(model_name)

    # Begin Training
    print("\nStarting model training...")
    t0 = time.time()
    train_results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        workers=4,
        patience=10,
        cls=1.0,
        save=True,
        plots=True,
        verbose=True
    )
    train_duration = time.time() - t0
    print(f"\nTraining completed in {train_duration / 60:.2f} minutes.")

    # Validation on Test Split
    print("\nRunning comprehensive evaluation on TEST split...")
    test_metrics = model.val(data=str(data_path), split="test", imgsz=imgsz, device=device)

    print("\n" + "=" * 60)
    print("  TEST EVALUATION BENCHMARK RESULTS")
    print("=" * 60)
    
    # Class-wise metrics
    class_names = test_metrics.names
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10} | {'mAP@0.5:0.95':<12}")
    print("-" * 70)
    
    for i, name_str in class_names.items():
        p = test_metrics.box.p[i] if i < len(test_metrics.box.p) else 0.0
        r = test_metrics.box.r[i] if i < len(test_metrics.box.r) else 0.0
        ap50 = test_metrics.box.ap50[i] if i < len(test_metrics.box.ap50) else 0.0
        ap = test_metrics.box.ap[i] if i < len(test_metrics.box.ap) else 0.0
        print(f"{name_str:<20} | {p:<10.3f} | {r:<10.3f} | {ap50:<10.3f} | {ap:<12.3f}")

    print("-" * 70)
    overall_p = test_metrics.box.mp
    overall_r = test_metrics.box.mr
    overall_map50 = test_metrics.box.map50
    overall_map = test_metrics.box.map
    print(f"{'OVERALL (ALL CLASSES)':<20} | {overall_p:<10.3f} | {overall_r:<10.3f} | {overall_map50:<10.3f} | {overall_map:<12.3f}")

    # Latency Benchmark
    speed = test_metrics.speed
    pre_ms = speed.get('preprocess', 0.0)
    inf_ms = speed.get('inference', 0.0)
    post_ms = speed.get('postprocess', 0.0)
    total_ms = pre_ms + inf_ms + post_ms
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    print("\n" + "=" * 60)
    print("  EDGE INFERENCE LATENCY BENCHMARK")
    print("=" * 60)
    print(f"Preprocess Latency  : {pre_ms:.2f} ms")
    print(f"Inference Latency   : {inf_ms:.2f} ms")
    print(f"Postprocess Latency : {post_ms:.2f} ms")
    print(f"Total Frame Latency : {total_ms:.2f} ms ({fps:.1f} FPS)")

    # Save best checkpoint to pipeline/models/
    target_weights_dir = Path(r"c:\Users\saran\Downloads\SODD\pipeline\models")
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    best_weights_src = Path(model.trainer.save_dir) / "weights" / "best.pt"
    
    if best_weights_src.exists():
        dest_best = target_weights_dir / "best.pt"
        shutil.copy2(best_weights_src, dest_best)
        print(f"\nSuccessfully deployed best prototype weights to: {dest_best}")

    # Export to ONNX for edge deployment (Jetson / Embedded AUV)
    try:
        print("\n" + "=" * 60)
        print("  EDGE RUNTIME EXPORT: ONNX FORMAT")
        print("=" * 60)
        best_model = YOLO(str(dest_best))
        onnx_file = best_model.export(format="onnx", imgsz=imgsz, dynamic=False, opset=12)
        if onnx_file and Path(onnx_file).exists():
            dest_onnx = target_weights_dir / "best.onnx"
            shutil.copy2(onnx_file, dest_onnx)
            print(f"Successfully deployed ONNX edge weights to: {dest_onnx}")
    except Exception as e:
        print(f"Notice: ONNX export encountered ({e}) - continuing with PyTorch weights.")

    # Confidence Calibration Note
    print("\n" + "=" * 60)
    print("  CONFIDENCE CALIBRATION STATUS")
    print("=" * 60)
    print("  Notice: Temperature-scaling calibration must be evaluated on genuine")
    print("  ground-truth labeled detections (matching predicted bounding boxes")
    print("  to annotated truth labels on a separate held-out validation set).")
    print("  Synthetic confidence/label samples have been removed to avoid presenting")
    print("  uncalibrated scores as probabilities.")
    print("  Current operational scores are designated as: uncalibrated_composite.")
    print("  TODO: Once empirical TP/FP detection matching is evaluated on real")
    print("  validation sonar tracks, fit optimal temperature T* and compute ECE/MCE.")
    print("=" * 60)

    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n SSS Marine Debris Detector")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default=0, help="Device (0 or cpu)")
    args = parser.parse_args()

    train_detector(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )

