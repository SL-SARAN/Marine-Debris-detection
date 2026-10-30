"""
src/audit_cylinder_dataset.py
=============================
Diagnostic audit of the Roboflow 'cylider2' dataset located in unprocessed_data/cylinder.

Audit Objectives:
1. Identify dataset modality (side-scan sonar, optical, or other acoustic sensor).
2. Document original provenance, citations, and claim of content.
3. Quantify image count, instance count, split distribution, and YOLO format.
4. Calculate elongation (major/minor) and aspect ratio distributions for 'cylider' vs 'manta'.
5. Generate visual contact sheets (40 samples with contextual padding and overlaid boxes)
   to assess whether the 'cylider' label matches genuine cylindrical object geometry
   or is confounded by acoustic shadow tails or non-cylindrical shapes.
6. Diagnostic only: do NOT merge or relabel into the training pipeline.
"""

import sys
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = {0: "cylider", 1: "manta"}
SPLITS = ["train", "valid", "test"]


def parse_cylinder_dataset(data_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses all YOLO annotations and image metadata across train, valid, and test splits.
    """
    records = []
    meta = {
        "dataset_name": "cylider2 (Roboflow Universe v6)",
        "splits": {},
        "total_images": 0,
        "total_boxes": 0,
        "class_counts": {0: 0, 1: 0},
        "image_shapes": set(),
        "min_coords": [float("inf")] * 4,
        "max_coords": [float("-inf")] * 4,
    }

    for split in SPLITS:
        img_dir = data_dir / split / "images"
        lbl_dir = data_dir / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[WARN] Missing directory for split: {split}")
            continue

        img_files = sorted(list(img_dir.glob("*.jpg")))
        lbl_files = sorted(list(lbl_dir.glob("*.txt")))

        meta["splits"][split] = {
            "images": len(img_files),
            "labels": len(lbl_files),
            "cylider": 0,
            "manta": 0,
        }
        meta["total_images"] += len(img_files)

        img_map = {f.stem: f for f in img_files}

        for lbl_path in lbl_files:
            content = lbl_path.read_text().strip()
            if not content:
                continue

            img_path = img_map.get(lbl_path.stem)
            if not img_path:
                print(f"[WARN] No image for label {lbl_path.name}")
                continue

            # Load image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]
            meta["image_shapes"].add((w_img, h_img))

            for line in content.splitlines():
                parts = [float(p) for p in line.strip().split()]
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                xc_norm, yc_norm, nw_norm, nh_norm = parts[1:5]

                meta["total_boxes"] += 1
                meta["class_counts"][cls_id] = meta["class_counts"].get(cls_id, 0) + 1
                if cls_id == 0:
                    meta["splits"][split]["cylider"] += 1
                elif cls_id == 1:
                    meta["splits"][split]["manta"] += 1

                for i, val in enumerate([xc_norm, yc_norm, nw_norm, nh_norm]):
                    meta["min_coords"][i] = min(meta["min_coords"][i], val)
                    meta["max_coords"][i] = max(meta["max_coords"][i], val)

                # Pixel dimensions
                bw_px = nw_norm * w_img
                bh_px = nh_norm * h_img

                x1 = int(round((xc_norm * w_img) - (bw_px / 2.0)))
                y1 = int(round((yc_norm * h_img) - (bh_px / 2.0)))
                x2 = int(round((xc_norm * w_img) + (bw_px / 2.0)))
                y2 = int(round((yc_norm * h_img) + (bh_px / 2.0)))

                if bw_px > 0 and bh_px > 0:
                    elongation = max(bw_px, bh_px) / min(bw_px, bh_px)
                    aspect_ratio_wh = bw_px / bh_px
                else:
                    elongation = 1.0
                    aspect_ratio_wh = 1.0

                records.append({
                    "split": split,
                    "file_stem": lbl_path.stem,
                    "img_file": str(img_path),
                    "lbl_file": str(lbl_path),
                    "img_w": w_img,
                    "img_h": h_img,
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "xc_norm": xc_norm,
                    "yc_norm": yc_norm,
                    "nw_norm": nw_norm,
                    "nh_norm": nh_norm,
                    "bw_px": bw_px,
                    "bh_px": bh_px,
                    "box_x1": max(0, x1),
                    "box_y1": max(0, y1),
                    "box_x2": min(w_img, x2),
                    "box_y2": min(h_img, y2),
                    "elongation": elongation,
                    "aspect_ratio_wh": aspect_ratio_wh,
                    "orientation": "horizontal" if bw_px >= bh_px else "vertical",
                })

    meta["image_shapes"] = list(meta["image_shapes"])
    return records, meta


def compute_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes distribution metrics and threshold percentages."""
    stats = {}
    thresholds = [1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]

    for cls_id, cls_name in CLASS_NAMES.items():
        cls_records = [r for r in records if r["class_id"] == cls_id]
        if not cls_records:
            continue

        elong_vals = np.array([r["elongation"] for r in cls_records])
        wh_vals = np.array([r["aspect_ratio_wh"] for r in cls_records])
        bw_vals = np.array([r["bw_px"] for r in cls_records])
        bh_vals = np.array([r["bh_px"] for r in cls_records])

        th_data = {}
        for th in thresholds:
            cnt = int(np.sum(elong_vals >= th))
            pct = float((cnt / len(cls_records)) * 100.0)
            th_data[f"gte_{th}"] = {"count": cnt, "total": len(cls_records), "percentage": round(pct, 2)}

        stats[cls_name] = {
            "count": len(cls_records),
            "elongation": {
                "mean": round(float(np.mean(elong_vals)), 3),
                "std": round(float(np.std(elong_vals)), 3),
                "median": round(float(np.median(elong_vals)), 3),
                "p25": round(float(np.percentile(elong_vals, 25)), 3),
                "p75": round(float(np.percentile(elong_vals, 75)), 3),
                "min": round(float(np.min(elong_vals)), 3),
                "max": round(float(np.max(elong_vals)), 3),
            },
            "aspect_ratio_wh": {
                "mean": round(float(np.mean(wh_vals)), 3),
                "median": round(float(np.median(wh_vals)), 3),
                "min": round(float(np.min(wh_vals)), 3),
                "max": round(float(np.max(wh_vals)), 3),
            },
            "box_width_px": {
                "mean": round(float(np.mean(bw_vals)), 1),
                "median": round(float(np.median(bw_vals)), 1),
                "min": round(float(np.min(bw_vals)), 1),
                "max": round(float(np.max(bw_vals)), 1),
            },
            "box_height_px": {
                "mean": round(float(np.mean(bh_vals)), 1),
                "median": round(float(np.median(bh_vals)), 1),
                "min": round(float(np.min(bh_vals)), 1),
                "max": round(float(np.max(bh_vals)), 1),
            },
            "threshold_percentages": th_data,
        }

    return stats


def plot_histograms(records: List[Dict[str, Any]], output_path: Path):
    """Generates comparison plots between 'cylider' and 'manta'."""
    cyl = [r for r in records if r["class_id"] == 0]
    manta = [r for r in records if r["class_id"] == 1]

    cyl_elong = [r["elongation"] for r in cyl]
    manta_elong = [r["elongation"] for r in manta]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # Panel 1: Histogram
    ax1 = axes[0, 0]
    bins = np.linspace(1.0, 3.5, 25)
    ax1.hist(cyl_elong, bins=bins, alpha=0.65, color="#1f77b4", label=f"Cylider (n={len(cyl_elong)})", edgecolor="black")
    ax1.hist(manta_elong, bins=bins, alpha=0.55, color="#ff7f0e", label=f"Manta (n={len(manta_elong)})", edgecolor="black")
    ax1.axvline(2.0, color="#d62728", linestyle="--", linewidth=2.0, label="Cylinder Candidate Thresh (2.0)")
    ax1.axvline(1.5, color="#2ca02c", linestyle=":", linewidth=1.8, label="Moderate Elongation (1.5)")
    ax1.set_title("A. Bounding Box Elongation (max(w,h) / min(w,h))", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Elongation Ratio", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=10, loc="upper right")

    # Panel 2: CDF
    ax2 = axes[0, 1]
    s_cyl = np.sort(cyl_elong)
    s_manta = np.sort(manta_elong)
    cdf_cyl = np.arange(1, len(s_cyl) + 1) / len(s_cyl)
    cdf_manta = np.arange(1, len(s_manta) + 1) / len(s_manta)
    ax2.plot(s_cyl, cdf_cyl, color="#1f77b4", linewidth=2.5, label="Cylider CDF")
    ax2.plot(s_manta, cdf_manta, color="#ff7f0e", linewidth=2.5, linestyle="--", label="Manta CDF")
    ax2.axvline(1.5, color="#2ca02c", linestyle=":", linewidth=1.5, label="Elongation = 1.5")
    ax2.axvline(2.0, color="#d62728", linestyle="--", linewidth=1.5, label="Elongation = 2.0")
    pct_cyl_2 = (np.sum(s_cyl >= 2.0) / len(s_cyl)) * 100
    pct_manta_2 = (np.sum(s_manta >= 2.0) / len(s_manta)) * 100
    ax2.annotate(f"Cylider >= 2.0: {pct_cyl_2:.1f}%", xy=(2.0, 1.0 - pct_cyl_2/100),
                 xytext=(2.2, 0.6), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5), fontsize=10, fontweight="bold")
    ax2.set_title("B. Cumulative Distribution Function (CDF)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Elongation Ratio", fontsize=11)
    ax2.set_ylabel("Cumulative Fraction", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=10, loc="lower right")

    # Panel 3: Width vs Height
    ax3 = axes[1, 0]
    ax3.scatter([r["bw_px"] for r in cyl], [r["bh_px"] for r in cyl], alpha=0.6, color="#1f77b4", s=25, label="Cylider")
    ax3.scatter([r["bw_px"] for r in manta], [r["bh_px"] for r in manta], alpha=0.5, color="#ff7f0e", s=25, label="Manta")
    max_d = 100
    xg = np.linspace(0, max_d, 50)
    ax3.plot(xg, xg, color="gray", linestyle=":", label="1:1 Aspect (Square / Circle)")
    ax3.plot(xg, 0.5 * xg, color="purple", linestyle="--", alpha=0.7, label="2:1 Aspect")
    ax3.plot(xg, 2.0 * xg, color="brown", linestyle="--", alpha=0.7, label="1:2 Aspect")
    ax3.set_xlim(0, max_d)
    ax3.set_ylim(0, max_d)
    ax3.set_title("C. Bounding Box Dimensions (Width vs Height in Pixels)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Width (pixels)", fontsize=11)
    ax3.set_ylabel("Height (pixels)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9, loc="upper right")

    # Panel 4: Box Size vs Image Scale
    ax4 = axes[1, 1]
    cyl_areas = [r["bw_px"] * r["bh_px"] for r in cyl]
    manta_areas = [r["bw_px"] * r["bh_px"] for r in manta]
    ax4.boxplot([cyl_areas, manta_areas], tick_labels=["Cylider", "Manta"], patch_artist=True,
                boxprops=dict(facecolor="#c6dbef", color="#1f77b4"),
                medianprops=dict(color="red", linewidth=2))
    ax4.set_title("D. Object Bounding Box Area (Pixels^2 on 1280x1280 Canvas)", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Bounding Box Area (px^2)", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Saved cylinder histograms to: {output_path}")


def create_contact_sheet(
    records: List[Dict[str, Any]],
    output_path: Path,
    title: str,
    target_class_id: int,
    grid_size: Tuple[int, int] = (8, 5),  # 40 samples
    cell_size: int = 240,
    box_color: Tuple[int, int, int] = (0, 255, 120),
    random_seed: int = 42,
):
    """
    Creates a 40-sample contact sheet showing sonar crops with 80% contextual margin
    around each bounding box to inspect acoustic highlight vs shadow morphology.
    """
    cls_records = [r for r in records if r["class_id"] == target_class_id]
    if not cls_records:
        return

    n_samples = grid_size[0] * grid_size[1]
    rng = random.Random(random_seed)

    # Stratified: 10 top elongated + 30 random representative samples
    cls_records.sort(key=lambda x: x["elongation"], reverse=True)
    top_samples = cls_records[:10]
    remaining = cls_records[10:]
    random_samples = rng.sample(remaining, min(n_samples - len(top_samples), len(remaining)))
    selected = top_samples + random_samples

    cols, rows = grid_size
    header_h = 42
    footer_h = 36
    cell_h = cell_size + footer_h
    sheet_w = cols * cell_size
    sheet_h = rows * cell_h + header_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:] = (24, 24, 28)

    # Main header
    cv2.putText(
        sheet,
        title,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 245),
        2,
        cv2.LINE_AA,
    )

    for idx, rec in enumerate(selected):
        if idx >= n_samples:
            break
        r_idx = idx // cols
        c_idx = idx % cols
        x_offset = c_idx * cell_size
        y_offset = header_h + r_idx * cell_h

        img = cv2.imread(rec["img_file"])
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]

        bx1, by1, bx2, by2 = rec["box_x1"], rec["box_y1"], rec["box_x2"], rec["box_y2"]
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # Generous contextual padding (80% margin, min 35px)
        pad_x = max(35, int(bw * 0.85))
        pad_y = max(35, int(bh * 0.85))

        crop_x1 = max(0, bx1 - pad_x)
        crop_y1 = max(0, by1 - pad_y)
        crop_x2 = min(w_orig, bx2 + pad_x)
        crop_y2 = min(h_orig, by2 + pad_y)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # Bounding box in crop coordinates
        box_crop_x1 = bx1 - crop_x1
        box_crop_y1 = by1 - crop_y1
        box_crop_x2 = bx2 - crop_x1
        box_crop_y2 = by2 - crop_y1

        cv2.rectangle(
            crop,
            (box_crop_x1, box_crop_y1),
            (box_crop_x2, box_crop_y2),
            box_color,
            2,
        )

        resized_crop = cv2.resize(crop, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        sheet[y_offset : y_offset + cell_size, x_offset : x_offset + cell_size] = resized_crop

        # Cell border
        cv2.rectangle(
            sheet,
            (x_offset, y_offset),
            (x_offset + cell_size - 1, y_offset + cell_size - 1),
            (60, 60, 70),
            1,
        )

        # Footer
        foot_y = y_offset + cell_size
        cv2.rectangle(
            sheet,
            (x_offset, foot_y),
            (x_offset + cell_size, foot_y + footer_h),
            (35, 35, 42),
            -1,
        )

        f_stem = rec["file_stem"].split(".")[0]
        if len(f_stem) > 16:
            f_stem = f_stem[:14] + ".."
        line1 = f"{f_stem} [{rec['split']}]"
        line2 = f"E:{rec['elongation']:.2f} | {rec['bw_px']:.0f}x{rec['bh_px']:.0f}px"

        elong_color = (100, 255, 100) if rec["elongation"] >= 2.0 else ((100, 200, 255) if rec["elongation"] >= 1.5 else (180, 180, 180))

        cv2.putText(
            sheet,
            line1,
            (x_offset + 6, foot_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (210, 210, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            line2,
            (x_offset + 6, foot_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            elong_color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), sheet)
    print(f"[OK] Saved contact sheet ({len(selected)} samples) to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit unprocessed_data/cylinder dataset.")
    parser.add_argument("--data-dir", type=str, default="unprocessed_data/cylinder", help="Path to cylinder dataset")
    parser.add_argument("--output-dir", type=str, default="data/diagnostics", help="Diagnostic output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" CYLINDER2 (ROBOFLOW) SIDE-SCAN SONAR DATASET AUDIT ")
    print("=" * 80)

    records, meta = parse_cylinder_dataset(data_dir)
    print(f"Total images audited:    {meta['total_images']}")
    print(f"Total objects audited:   {meta['total_boxes']}")
    print(f"Image dimensions:        {meta['image_shapes']}")
    print(f"Class counts:            {meta['class_counts']} (0=cylider, 1=manta)")
    print(f"Splits breakdown:")
    for sp, d in meta["splits"].items():
        print(f"  {sp}: {d['images']} images (cylider={d['cylider']}, manta={d['manta']})")

    stats = compute_statistics(records)
    cyl_s = stats.get("cylider", {})
    manta_s = stats.get("manta", {})

    print("\n--- 'CYLIDER' (Class 0) ELONGATION SUMMARY ---")
    print(f"  Count:  {cyl_s['count']}")
    print(f"  Mean:   {cyl_s['elongation']['mean']:.2f} +/- {cyl_s['elongation']['std']:.2f}")
    print(f"  Median: {cyl_s['elongation']['median']:.2f} (IQR: {cyl_s['elongation']['p25']:.2f} - {cyl_s['elongation']['p75']:.2f})")
    print(f"  Range:  [{cyl_s['elongation']['min']:.2f}, {cyl_s['elongation']['max']:.2f}]")
    print("  Thresholds:")
    for k, v in cyl_s["threshold_percentages"].items():
        print(f"    Elongation >= {k.replace('gte_', '')}: {v['count']:3d} / {v['total']:3d} ({v['percentage']:5.2f}%)")

    print("\n--- 'MANTA' (Class 1) ELONGATION SUMMARY ---")
    print(f"  Count:  {manta_s['count']}")
    print(f"  Mean:   {manta_s['elongation']['mean']:.2f} +/- {manta_s['elongation']['std']:.2f}")
    print(f"  Median: {manta_s['elongation']['median']:.2f}")
    print(f"  Elongation >= 2.0: {manta_s['threshold_percentages']['gte_2.0']['count']} / {manta_s['threshold_percentages']['gte_2.0']['total']} ({manta_s['threshold_percentages']['gte_2.0']['percentage']:.2f}%)")

    # Generate visualizations
    print("\n--- GENERATING VISUALIZATIONS ---")
    hist_p = output_dir / "cylinder2_aspect_ratio_histograms.png"
    plot_histograms(records, hist_p)

    cyl_sheet_p = output_dir / "cylinder2_contact_sheet.png"
    create_contact_sheet(
        records=records,
        output_path=cyl_sheet_p,
        title="Roboflow 'cylider2' Dataset - Class 0: 'cylider' (40 Samples with Context)",
        target_class_id=0,
        grid_size=(8, 5),  # 40 samples
        box_color=(0, 255, 120),
    )

    manta_sheet_p = output_dir / "cylinder2_manta_contact_sheet.png"
    create_contact_sheet(
        records=records,
        output_path=manta_sheet_p,
        title="Roboflow 'cylider2' Dataset - Class 1: 'manta' (30 Samples with Context)",
        target_class_id=1,
        grid_size=(6, 5),  # 30 samples
        box_color=(0, 180, 255),
    )

    # Save summary report
    rep_p = output_dir / "cylinder2_audit_report.json"
    with open(rep_p, "w") as f:
        json.dump({"meta": meta, "statistics": stats}, f, indent=2)
    print(f"[OK] Full diagnostic summary saved to: {rep_p}")
    print("\n" + "=" * 80)
    print(" AUDIT COMPLETE ")
    print("=" * 80)


if __name__ == "__main__":
    main()
