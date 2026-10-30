"""
Trial training pass on the 7-class common_sss_dataset (with cylinder).
YOLOv8n, P3-P5 architecture unchanged, 640x640 input.
Reports per-class precision/recall/F1.
"""
import multiprocessing
multiprocessing.freeze_support()

from ultralytics import YOLO
from pathlib import Path
import json
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_YAML = 'd:/MarineDebris/data/common_sss_dataset/data.yaml'
MODEL = 'yolov8n.pt'  # Pretrained base weights
PROJECT = 'd:/MarineDebris/runs/detect'
NAME = 'yolov8n_7class_cylinder_trial'
EPOCHS = 50
IMGSZ = 640
BATCH = 16
PATIENCE = 15  # Early stopping patience
WORKERS = 0    # Windows-safe: avoid multiprocessing spawn issues

CLASS_NAMES = {
    0: 'ghost_net',
    1: 'shipwreck',
    2: 'aircraft',
    3: 'seafloor_debris',
    4: 'pipeline_cable',
    5: 'engineering_platform',
    6: 'cylinder',
}


def main():
    # =============================================================================
    # TRAINING
    # =============================================================================
    print("=" * 70)
    print("  TRAINING: YOLOv8n 7-Class Trial (P3-P5 unchanged)")
    print("=" * 70)
    print(f"  Data:    {DATA_YAML}")
    print(f"  Epochs:  {EPOCHS} (patience={PATIENCE})")
    print(f"  ImgSz:   {IMGSZ}")
    print(f"  Batch:   {BATCH}")
    print(f"  Workers: {WORKERS}")
    print()

    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        verbose=True,
        workers=WORKERS,
        # Keep P3-P5 architecture as-is
        # Standard augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Optimizer
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
    )

    # =============================================================================
    # VALIDATION & PER-CLASS METRICS
    # =============================================================================
    print("\n" + "=" * 70)
    print("  VALIDATION: Per-Class Metrics on TEST split")
    print("=" * 70)

    # Run validation on best weights against test split
    best_model_path = Path(PROJECT) / NAME / 'weights' / 'best.pt'
    val_model = YOLO(str(best_model_path))
    val_results = val_model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        split='test',
        workers=WORKERS,
    )

    # Extract per-class metrics
    box = val_results.box

    precisions = box.p      # shape (nc,)
    recalls = box.r         # shape (nc,)
    ap50 = box.ap50         # shape (nc,)
    ap = box.ap             # shape (nc,) — AP@0.5:0.95
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)

    print(f"\n  {'Class':>22} | {'P':>7} | {'R':>7} | {'F1':>7} | {'AP@50':>7} | {'AP@50:95':>9}")
    print(f"  {'-'*22}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}")

    metrics_report = {}
    for i in range(len(CLASS_NAMES)):
        p = float(precisions[i]) if i < len(precisions) else 0.0
        r = float(recalls[i]) if i < len(recalls) else 0.0
        f1 = float(f1_scores[i]) if i < len(f1_scores) else 0.0
        a50 = float(ap50[i]) if i < len(ap50) else 0.0
        a = float(ap[i]) if i < len(ap) else 0.0

        name = CLASS_NAMES[i]
        print(f"  {name:>22} | {p:>7.4f} | {r:>7.4f} | {f1:>7.4f} | {a50:>7.4f} | {a:>9.4f}")

        metrics_report[name] = {
            'precision': p,
            'recall': r,
            'f1': f1,
            'ap50': a50,
            'ap50_95': a,
        }

    # Aggregate
    mean_p = float(np.mean(precisions))
    mean_r = float(np.mean(recalls))
    mean_f1 = float(np.mean(f1_scores))
    mean_ap50 = float(np.mean(ap50))
    mean_ap = float(np.mean(ap))

    print(f"  {'-'*22}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}")
    print(f"  {'MEAN (all classes)':>22} | {mean_p:>7.4f} | {mean_r:>7.4f} | {mean_f1:>7.4f} | {mean_ap50:>7.4f} | {mean_ap:>9.4f}")

    metrics_report['_mean'] = {
        'precision': mean_p,
        'recall': mean_r,
        'f1': mean_f1,
        'ap50': mean_ap50,
        'ap50_95': mean_ap,
    }

    # Save metrics to JSON
    metrics_path = Path(PROJECT) / NAME / 'per_class_metrics.json'
    metrics_path.write_text(json.dumps(metrics_report, indent=2))
    print(f"\n  Metrics saved to: {metrics_path}")
    print(f"  Best weights:     {best_model_path}")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
