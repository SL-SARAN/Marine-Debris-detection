import os
import shutil
import json
import multiprocessing
from pathlib import Path
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path('d:/MarineDebris/runs/detect/yolov8n_7class_cylinder_trial/weights/best.pt')
DATA_YAML = Path('d:/MarineDebris/data/common_sss_dataset/data.yaml')
PROJECT = Path('d:/MarineDebris/runs/detect')
NAME = 'eval_7class_test_consistent'
ART_DIR = Path(r"C:\Users\saran\.gemini\antigravity-ide\brain\be1b0cf5-47fc-4995-9d57-d19f185ff34f")

def main():
    print("=" * 70)
    print("  CONSISTENT 7-CLASS TEST EVALUATION")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Data:    {DATA_YAML}")
    print(f"  Target:  {PROJECT / NAME}")
    print("=" * 70)

    model = YOLO(str(MODEL_PATH))

    # Run validation explicitly directed into our local directory with workers=0
    val_results = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        split='test',
        project=str(PROJECT),
        name=NAME,
        exist_ok=True,
        plots=True,
        verbose=True,
        save_json=True,
        workers=0,
    )

    eval_dir = PROJECT / NAME
    print(f"\nResults saved in: {eval_dir}")

    # Extract metrics
    box = val_results.box
    precisions = box.p
    recalls = box.r
    ap50 = box.ap50
    ap = box.ap
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-16)

    class_names = model.names
    print(f"\n{'Class ID':<8} | {'Class Name':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'AP@50':<10} | {'AP@50:95':<10}")
    print("-" * 92)

    consistent_metrics = {}
    for i in range(len(class_names)):
        p = float(precisions[i]) if i < len(precisions) else 0.0
        r = float(recalls[i]) if i < len(recalls) else 0.0
        f1 = float(f1_scores[i]) if i < len(f1_scores) else 0.0
        a50 = float(ap50[i]) if i < len(ap50) else 0.0
        a = float(ap[i]) if i < len(ap) else 0.0
        cname = class_names[i]
        print(f"{i:<8} | {cname:<22} | {p:<10.4f} | {r:<10.4f} | {f1:<10.4f} | {a50:<10.4f} | {a:<10.4f}")
        consistent_metrics[cname] = {
            'class_id': i,
            'precision': p,
            'recall': r,
            'f1': f1,
            'ap50': a50,
            'ap50_95': a
        }

    print("-" * 92)
    mean_p = float(np.mean(precisions))
    mean_r = float(np.mean(recalls))
    mean_f1 = float(np.mean(f1_scores))
    mean_a50 = float(np.mean(ap50))
    mean_a = float(np.mean(ap))
    print(f"{'MEAN':<8} | {'All 7 Classes':<22} | {mean_p:<10.4f} | {mean_r:<10.4f} | {mean_f1:<10.4f} | {mean_a50:<10.4f} | {mean_a:<10.4f}")

    consistent_metrics['_mean'] = {
        'precision': mean_p,
        'recall': mean_r,
        'f1': mean_f1,
        'ap50': mean_a50,
        'ap50_95': mean_a
    }

    # Copy files directly from this run to artifact directory
    artifacts_to_copy = [
        ("confusion_matrix_normalized.png", "eval_confusion_matrix_normalized.png"),
        ("confusion_matrix.png", "eval_confusion_matrix_raw.png"),
        ("BoxPR_curve.png", "eval_box_pr_curve.png"),
        ("BoxF1_curve.png", "eval_box_f1_curve.png"),
        ("val_batch0_pred.jpg", "eval_test_detections_batch0.jpg"),
        ("val_batch1_pred.jpg", "eval_test_detections_batch1.jpg"),
        ("val_batch2_pred.jpg", "eval_test_detections_batch2.jpg"),
    ]

    for src_name, dst_name in artifacts_to_copy:
        src_file = eval_dir / src_name
        if src_file.exists():
            shutil.copy2(src_file, ART_DIR / dst_name)
            print(f"Copied {src_name} -> {dst_name}")
        else:
            print(f"Warning: {src_name} not found in {eval_dir}")

    # Save JSON
    out_json = ART_DIR / "consistent_test_metrics.json"
    out_json.write_text(json.dumps(consistent_metrics, indent=2))
    print(f"Saved consistent metrics to {out_json}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

