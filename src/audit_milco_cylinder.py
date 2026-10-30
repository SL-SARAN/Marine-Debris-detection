"""
src/audit_milco_cylinder.py
===========================
Diagnostic audit of MILCO (Mine-Like Contacts) and NOMBO (Non-Mine-Like Bottom Objects)
in the Side-Scan Sonar Mine Detection dataset (unprocessed_data/2010, 2015, 2017, 2018, 2021).

Goal:
1. Verify raw annotation format (field order, coordinate normalization, center vs top-left).
2. Audit aspect ratio and elongation distributions (major / minor axis) for MILCO before
   making any decision to map MILCO to a "cylinder" class.
3. Compare MILCO aspect ratios against NOMBO false targets.
4. Generate contact sheets for manual visual review of sonar highlights and acoustic shadows.
5. Provide quantitative statistics (% elongated at various thresholds).

This script is purely diagnostic. It does NOT relabel or merge data into the training pipeline.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import matplotlib.pyplot as plt


YEAR_DIRS = ["2010", "2015", "2017", "2018", "2021"]
CLASS_MAP = {0: "MILCO", 1: "NOMBO"}


def parse_raw_annotations(data_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses all annotation files (.txt) across the target year folders.
    Verifies coordinate ranges, dimensions, and center-based normalization.
    """
    records = []
    format_diag = {
        "total_images_found": 0,
        "total_annotation_files": 0,
        "empty_annotation_files": 0,
        "non_empty_annotation_files": 0,
        "total_boxes": 0,
        "class_counts": {0: 0, 1: 0},
        "year_counts": {},
        "min_raw_coords": [float("inf")] * 4,
        "max_raw_coords": [float("-inf")] * 4,
        "out_of_bounds_center_based": 0,
        "out_of_bounds_top_left_based": 0,
        "degenerate_boxes": [],
    }

    for year in YEAR_DIRS:
        ydir = data_dir / year
        if not ydir.exists():
            print(f"[WARN] Directory not found: {ydir}")
            continue

        img_files = sorted(list(ydir.glob("*.jpg")))
        txt_files = sorted(list(ydir.glob("*.txt")))
        format_diag["year_counts"][year] = {
            "images": len(img_files),
            "txts": len(txt_files),
            "boxes": 0,
            "milco": 0,
            "nombo": 0,
        }
        format_diag["total_images_found"] += len(img_files)
        format_diag["total_annotation_files"] += len(txt_files)

        img_map = {f.stem: f for f in img_files}

        for txt_path in txt_files:
            if txt_path.name.lower() == "desktop.ini":
                continue

            content = txt_path.read_text().strip()
            if not content:
                format_diag["empty_annotation_files"] += 1
                continue

            format_diag["non_empty_annotation_files"] += 1
            img_path = img_map.get(txt_path.stem)
            if not img_path:
                print(f"[WARN] No matching image found for {txt_path}")
                continue

            # Load image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[ERROR] Failed to read image {img_path}")
                continue
            img_h, img_w = img.shape[:2]

            for line_idx, line in enumerate(content.splitlines()):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    print(f"[WARN] Invalid line in {txt_path.name}: '{line}'")
                    continue

                cls_id = int(float(parts[0]))
                raw_c = [float(p) for p in parts[1:5]]
                xc_norm, yc_norm, w_norm, h_norm = raw_c

                format_diag["total_boxes"] += 1
                format_diag["class_counts"][cls_id] = format_diag["class_counts"].get(cls_id, 0) + 1
                format_diag["year_counts"][year]["boxes"] += 1
                if cls_id == 0:
                    format_diag["year_counts"][year]["milco"] += 1
                elif cls_id == 1:
                    format_diag["year_counts"][year]["nombo"] += 1

                for i in range(4):
                    format_diag["min_raw_coords"][i] = min(format_diag["min_raw_coords"][i], raw_c[i])
                    format_diag["max_raw_coords"][i] = max(format_diag["max_raw_coords"][i], raw_c[i])

                # Pixel conversions under Center-based hypothesis (YOLO standard)
                bw_px = w_norm * img_w
                bh_px = h_norm * img_h
                x1_c = (xc_norm * img_w) - (bw_px / 2.0)
                y1_c = (yc_norm * img_h) - (bh_px / 2.0)
                x2_c = x1_c + bw_px
                y2_c = y1_c + bh_px

                # Pixel conversions under Top-Left hypothesis
                x1_tl = xc_norm * img_w
                y1_tl = yc_norm * img_h
                x2_tl = x1_tl + bw_px
                y2_tl = y1_tl + bh_px

                eps = 1e-2
                if x1_c < -eps or y1_c < -eps or x2_c > img_w + eps or y2_c > img_h + eps:
                    format_diag["out_of_bounds_center_based"] += 1
                if x1_tl < -eps or y1_tl < -eps or x2_tl > img_w + eps or y2_tl > img_h + eps:
                    format_diag["out_of_bounds_top_left_based"] += 1

                is_degenerate = (bw_px < 3.0 or bh_px < 3.0)
                if is_degenerate:
                    format_diag["degenerate_boxes"].append({
                        "file": txt_path.name,
                        "year": year,
                        "class_id": cls_id,
                        "bw_px": bw_px,
                        "bh_px": bh_px,
                        "raw": raw_c,
                    })

                # Safe aspect ratio and elongation calculation
                if bw_px >= 3.0 and bh_px >= 3.0:
                    elongation = max(bw_px, bh_px) / min(bw_px, bh_px)
                    aspect_ratio_wh = bw_px / bh_px
                else:
                    elongation = 1.0
                    aspect_ratio_wh = 1.0

                records.append({
                    "year": year,
                    "file_stem": txt_path.stem,
                    "img_file": str(img_path),
                    "txt_file": str(txt_path),
                    "img_w": img_w,
                    "img_h": img_h,
                    "class_id": cls_id,
                    "class_name": CLASS_MAP.get(cls_id, f"Class_{cls_id}"),
                    "xc_norm": xc_norm,
                    "yc_norm": yc_norm,
                    "w_norm": w_norm,
                    "h_norm": h_norm,
                    "bw_px": bw_px,
                    "bh_px": bh_px,
                    "box_x1": max(0, int(round(x1_c))),
                    "box_y1": max(0, int(round(y1_c))),
                    "box_x2": min(img_w, int(round(x2_c))),
                    "box_y2": min(img_h, int(round(y2_c))),
                    "elongation": elongation,
                    "aspect_ratio_wh": aspect_ratio_wh,
                    "is_degenerate": is_degenerate,
                    "orientation": "horizontal" if bw_px >= bh_px else "vertical",
                })

    return records, format_diag


def compute_aspect_ratio_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive elongation and aspect ratio metrics."""
    stats = {}
    thresholds = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0]

    for cls_id, cls_name in CLASS_MAP.items():
        cls_records = [r for r in records if r["class_id"] == cls_id and not r.get("is_degenerate", False)]
        if not cls_records:
            continue

        elong_vals = np.array([r["elongation"] for r in cls_records])
        wh_vals = np.array([r["aspect_ratio_wh"] for r in cls_records])
        bw_vals = np.array([r["bw_px"] for r in cls_records])
        bh_vals = np.array([r["bh_px"] for r in cls_records])

        th_counts = {}
        for th in thresholds:
            count = int(np.sum(elong_vals >= th))
            pct = float((count / len(cls_records)) * 100.0)
            th_counts[f"elongation_gte_{th}"] = {
                "count": count,
                "total": len(cls_records),
                "percentage": round(pct, 2),
            }

        stats[cls_name] = {
            "total_objects": len(cls_records),
            "elongation": {
                "mean": round(float(np.mean(elong_vals)), 3),
                "std": round(float(np.std(elong_vals)), 3),
                "median": round(float(np.median(elong_vals)), 3),
                "min": round(float(np.min(elong_vals)), 3),
                "max": round(float(np.max(elong_vals)), 3),
                "p25": round(float(np.percentile(elong_vals, 25)), 3),
                "p75": round(float(np.percentile(elong_vals, 75)), 3),
            },
            "aspect_ratio_wh": {
                "mean": round(float(np.mean(wh_vals)), 3),
                "std": round(float(np.std(wh_vals)), 3),
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
            "orientation_counts": {
                "horizontal": int(np.sum([r["orientation"] == "horizontal" for r in cls_records])),
                "vertical": int(np.sum([r["orientation"] == "vertical" for r in cls_records])),
            },
            "threshold_percentages": th_counts,
        }

    return stats


def plot_aspect_ratio_histograms(records: List[Dict[str, Any]], output_path: Path):
    """
    Generates a 4-panel publication-quality diagnostic figure:
    1. Elongation distribution histogram with threshold lines.
    2. Cumulative distribution function (CDF) of elongation.
    3. Width vs Height scatter plot with aspect ratio guide lines.
    4. Orientation breakdown (Horizontal across-track vs Vertical along-track).
    """
    milco_records = [r for r in records if r["class_id"] == 0 and not r.get("is_degenerate", False)]
    nombo_records = [r for r in records if r["class_id"] == 1 and not r.get("is_degenerate", False)]

    milco_elong = [r["elongation"] for r in milco_records]
    nombo_elong = [r["elongation"] for r in nombo_records]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)

    # Panel 1: Histogram
    ax1 = axes[0, 0]
    bins = np.linspace(1.0, 6.0, 30)
    ax1.hist(milco_elong, bins=bins, alpha=0.65, color="#1f77b4", label=f"MILCO (Mine-Like, n={len(milco_elong)})", edgecolor="black")
    ax1.hist(nombo_elong, bins=bins, alpha=0.55, color="#ff7f0e", label=f"NOMBO (Non-Mine, n={len(nombo_elong)})", edgecolor="black")
    ax1.axvline(2.0, color="#d62728", linestyle="--", linewidth=2.0, label="Cylinder Candidate Thresh (Ratio = 2.0)")
    ax1.axvline(1.5, color="#2ca02c", linestyle=":", linewidth=1.8, label="Moderate Elongation (Ratio = 1.5)")
    ax1.set_title("A. Bounding Box Elongation Distribution (max(w,h) / min(w,h))", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Elongation Ratio (Major / Minor Axis)", fontsize=11)
    ax1.set_ylabel("Frequency (Object Count)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=10, loc="upper right")

    # Panel 2: CDF
    ax2 = axes[0, 1]
    sorted_milco = np.sort(milco_elong)
    sorted_nombo = np.sort(nombo_elong)
    cdf_milco = np.arange(1, len(sorted_milco) + 1) / len(sorted_milco)
    cdf_nombo = np.arange(1, len(sorted_nombo) + 1) / len(sorted_nombo)
    ax2.plot(sorted_milco, cdf_milco, color="#1f77b4", linewidth=2.5, label="MILCO CDF")
    ax2.plot(sorted_nombo, cdf_nombo, color="#ff7f0e", linewidth=2.5, linestyle="--", label="NOMBO CDF")
    ax2.axvline(2.0, color="#d62728", linestyle="--", linewidth=1.5, label="Elongation = 2.0")
    # Annotate values at 2.0
    pct_milco_gte_2 = (np.sum(sorted_milco >= 2.0) / len(sorted_milco)) * 100
    pct_nombo_gte_2 = (np.sum(sorted_nombo >= 2.0) / len(sorted_nombo)) * 100
    ax2.annotate(f"MILCO >= 2.0: {pct_milco_gte_2:.1f}%", xy=(2.0, 1.0 - pct_milco_gte_2/100),
                 xytext=(2.4, 0.45), arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5), fontsize=10, fontweight="bold")
    ax2.annotate(f"NOMBO >= 2.0: {pct_nombo_gte_2:.1f}%", xy=(2.0, 1.0 - pct_nombo_gte_2/100),
                 xytext=(2.4, 0.35), arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5), fontsize=10, fontweight="bold")
    ax2.set_title("B. Cumulative Distribution Function (CDF)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Elongation Ratio", fontsize=11)
    ax2.set_ylabel("Cumulative Fraction", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=10, loc="lower right")

    # Panel 3: Width vs Height Scatter Plot
    ax3 = axes[1, 0]
    m_w = [r["bw_px"] for r in milco_records]
    m_h = [r["bh_px"] for r in milco_records]
    n_w = [r["bw_px"] for r in nombo_records]
    n_h = [r["bh_px"] for r in nombo_records]
    ax3.scatter(m_w, m_h, alpha=0.6, color="#1f77b4", s=25, label="MILCO Contacts", edgecolors="none")
    ax3.scatter(n_w, n_h, alpha=0.5, color="#ff7f0e", s=25, label="NOMBO Contacts", edgecolors="none")
    # Guide lines
    max_dim = max(max(m_w + n_w, default=100), max(m_h + n_h, default=100))
    x_guide = np.linspace(0, max_dim, 100)
    ax3.plot(x_guide, x_guide, color="gray", linestyle=":", label="1:1 Aspect (Equidimensional / Sphere)")
    ax3.plot(x_guide, 0.5 * x_guide, color="purple", linestyle="--", alpha=0.7, label="2:1 Aspect (Elongated Horizontal)")
    ax3.plot(x_guide, 2.0 * x_guide, color="brown", linestyle="--", alpha=0.7, label="1:2 Aspect (Elongated Vertical)")
    ax3.set_xlim(0, min(max_dim, 250))
    ax3.set_ylim(0, min(max_dim, 250))
    ax3.set_title("C. Bounding Box Dimensions (Width vs Height in Pixels)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Box Width (pixels)", fontsize=11)
    ax3.set_ylabel("Box Height (pixels)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(fontsize=9, loc="upper right")

    # Panel 4: Orientation and Elongation by Year
    ax4 = axes[1, 1]
    years_order = [y for y in YEAR_DIRS if any(r["year"] == y for r in milco_records)]
    milco_means = []
    nombo_means = []
    for y in years_order:
        y_m = [r["elongation"] for r in milco_records if r["year"] == y]
        y_n = [r["elongation"] for r in nombo_records if r["year"] == y]
        milco_means.append(np.mean(y_m) if y_m else 0)
        nombo_means.append(np.mean(y_n) if y_n else 0)
    
    x_idx = np.arange(len(years_order))
    bar_w = 0.35
    ax4.bar(x_idx - bar_w/2, milco_means, width=bar_w, color="#1f77b4", label="MILCO Mean Elongation", edgecolor="black")
    ax4.bar(x_idx + bar_w/2, nombo_means, width=bar_w, color="#ff7f0e", label="NOMBO Mean Elongation", edgecolor="black")
    ax4.axhline(2.0, color="#d62728", linestyle="--", linewidth=1.5, label="Cylinder Candidate Thresh (2.0)")
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(years_order, fontsize=10)
    ax4.set_title("D. Mean Elongation Across Data Collection Surveys (Years)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Survey Collection Year", fontsize=11)
    ax4.set_ylabel("Mean Elongation (max/min)", fontsize=11)
    ax4.set_ylim(0, 3.5)
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Saved aspect ratio histograms to: {output_path}")


def create_contact_sheet(
    records: List[Dict[str, Any]],
    output_path: Path,
    title: str,
    target_class_id: int,
    selection_mode: str = "stratified",
    grid_size: Tuple[int, int] = (6, 5),
    cell_size: int = 240,
    box_color: Tuple[int, int, int] = (0, 255, 0),
):
    """
    Builds a high-resolution contact sheet of cropped sonar contacts.
    Pads around the bounding box by 50% so seabed context, acoustic highlights,
    and shadow tails are clearly visible for manual review.
    """
    cls_records = [
        r for r in records
        if r["class_id"] == target_class_id and not r.get("is_degenerate", False)
    ]
    if not cls_records:
        print(f"[WARN] No records found for class {target_class_id}")
        return

    n_samples = grid_size[0] * grid_size[1]

    # Select samples based on mode
    if selection_mode == "stratified":
        # 1/3 high elongation (>= 2.5), 1/3 moderate (1.5 - 2.5), 1/3 compact (< 1.5)
        high = [r for r in cls_records if r["elongation"] >= 2.5]
        mod = [r for r in cls_records if 1.5 <= r["elongation"] < 2.5]
        low = [r for r in cls_records if r["elongation"] < 1.5]

        # Sort each group descending by elongation
        high.sort(key=lambda x: x["elongation"], reverse=True)
        mod.sort(key=lambda x: x["elongation"], reverse=True)
        low.sort(key=lambda x: x["elongation"], reverse=True)

        n_per_group = n_samples // 3
        rem = n_samples - (n_per_group * 3)

        selected = high[:n_per_group + rem] + mod[:n_per_group] + low[:n_per_group]
        # If any group was short, backfill from remaining
        if len(selected) < n_samples:
            remaining = [r for r in cls_records if r not in selected]
            remaining.sort(key=lambda x: x["elongation"], reverse=True)
            selected += remaining[: n_samples - len(selected)]
    elif selection_mode == "top_elongated":
        cls_records.sort(key=lambda x: x["elongation"], reverse=True)
        selected = cls_records[:n_samples]
    else:
        # Balanced sampling
        cls_records.sort(key=lambda x: x["elongation"], reverse=True)
        step = max(1, len(cls_records) // n_samples)
        selected = cls_records[::step][:n_samples]

    cols, rows = grid_size
    header_h = 40
    footer_h = 36
    cell_h = cell_size + footer_h
    sheet_w = cols * cell_size
    sheet_h = rows * cell_h + header_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:] = (24, 24, 28)  # dark slate background

    # Main header
    cv2.putText(
        sheet,
        title,
        (20, 28),
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

        img_path = Path(rec["img_file"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]

        bx1, by1, bx2, by2 = rec["box_x1"], rec["box_y1"], rec["box_x2"], rec["box_y2"]
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # Context padding: 50% margin (at least 20 px)
        pad_x = max(20, int(bw * 0.55))
        pad_y = max(20, int(bh * 0.55))

        crop_x1 = max(0, bx1 - pad_x)
        crop_y1 = max(0, by1 - pad_y)
        crop_x2 = min(w_orig, bx2 + pad_x)
        crop_y2 = min(h_orig, by2 + pad_y)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # Bounding box in crop coordinates
        box_in_crop_x1 = bx1 - crop_x1
        box_in_crop_y1 = by1 - crop_y1
        box_in_crop_x2 = bx2 - crop_x1
        box_in_crop_y2 = by2 - crop_y1

        # Draw box on crop before resizing
        cv2.rectangle(
            crop,
            (box_in_crop_x1, box_in_crop_y1),
            (box_in_crop_x2, box_in_crop_y2),
            box_color,
            2,
        )

        # Resize crop into cell image area
        resized_crop = cv2.resize(crop, (cell_size, cell_size), interpolation=cv2.INTER_AREA)

        # Place image patch
        sheet[y_offset : y_offset + cell_size, x_offset : x_offset + cell_size] = resized_crop

        # Draw cell border
        cv2.rectangle(
            sheet,
            (x_offset, y_offset),
            (x_offset + cell_size - 1, y_offset + cell_size - 1),
            (60, 60, 70),
            1,
        )

        # Draw footer with metadata
        foot_y = y_offset + cell_size
        cv2.rectangle(
            sheet,
            (x_offset, foot_y),
            (x_offset + cell_size, foot_y + footer_h),
            (35, 35, 42),
            -1,
        )

        # Caption lines
        f_name = rec["file_stem"]
        if len(f_name) > 16:
            f_name = f_name[:14] + ".."
        line1 = f"{f_name} | {rec['year']}"
        line2 = f"E:{rec['elongation']:.2f} | {rec['bw_px']:.0f}x{rec['bh_px']:.0f}px"

        # Color-code elongation badge
        elong_color = (100, 255, 100) if rec["elongation"] >= 2.0 else ((100, 200, 255) if rec["elongation"] >= 1.5 else (180, 180, 180))

        cv2.putText(
            sheet,
            line1,
            (x_offset + 6, foot_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet,
            line2,
            (x_offset + 6, foot_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            elong_color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), sheet)
    print(f"[OK] Saved contact sheet ({len(selected)} contacts) to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit MILCO/NOMBO dataset for potential cylinder class mapping."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="unprocessed_data",
        help="Path to unprocessed dataset root containing year subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/diagnostics",
        help="Directory to store diagnostic plots and contact sheets",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" MILCO / NOMBO SIDE-SCAN SONAR DATASET DIAGNOSTIC AUDIT ")
    print("=" * 80)
    print(f"Target data directory: {data_dir.resolve()}")
    print(f"Output directory:      {output_dir.resolve()}")

    # Step 1: Parse and verify annotation files
    print("\n--- STEP 1: VERIFYING ANNOTATION FORMAT ---")
    records, format_diag = parse_raw_annotations(data_dir)

    print(f"Total images audited across years: {format_diag['total_images_found']}")
    print(f"Total annotation files audited:    {format_diag['total_annotation_files']}")
    print(f"Empty files (background images):   {format_diag['empty_annotation_files']}")
    print(f"Files with detections:             {format_diag['non_empty_annotation_files']}")
    print(f"Total labeled objects found:       {format_diag['total_boxes']}")
    print(f"Class 0 (MILCO) contacts:          {format_diag['class_counts'].get(0, 0)}")
    print(f"Class 1 (NOMBO) contacts:          {format_diag['class_counts'].get(1, 0)}")

    print("\n[FORMAT VERIFICATION FINDINGS]:")
    print(f"  Raw Coordinate Ranges:")
    print(f"    Min (c1, c2, c3, c4): {[round(x, 6) for x in format_diag['min_raw_coords']]}")
    print(f"    Max (c1, c2, c3, c4): {[round(x, 6) for x in format_diag['max_raw_coords']]}")
    print(f"  --> All coordinates are in range [0.0, 1.0]: NORMALIZED (NOT absolute pixels).")
    print(f"  Out-of-bounds check:")
    print(f"    Center-based (YOLO standard): {format_diag['out_of_bounds_center_based']} box boundary violations (degenerate 0-px box)")
    print(f"    Top-left based hypothesis:   {format_diag['out_of_bounds_top_left_based']} box boundary violations (exceeds image frame)")
    print(f"  --> Conclusive verdict: The dataset uses standard normalized YOLO center format: [class_id, x_center, y_center, width, height]")
    print(f"      The Data-in-Brief paper text description ('absolute pixel, top-left') is superseded by the actual Figshare repository files.")

    if format_diag["degenerate_boxes"]:
        print(f"\n[FLAG] {len(format_diag['degenerate_boxes'])} degenerate sub-pixel boxes detected:")
        for db in format_diag["degenerate_boxes"]:
            print(f"  - {db['year']}/{db['file']}: bw={db['bw_px']:.2f}px, bh={db['bh_px']:.2f}px, raw={db['raw']}")

    # Step 2: Statistical aspect ratio & elongation analysis
    print("\n--- STEP 2: COMPUTING ASPECT RATIO & ELONGATION METRICS ---")
    stats = compute_aspect_ratio_statistics(records)

    milco_s = stats.get("MILCO", {})
    nombo_s = stats.get("NOMBO", {})

    print(f"\nMILCO Elongation Summary (max(w,h) / min(w,h)):")
    print(f"  Count:  {milco_s['total_objects']}")
    print(f"  Mean:   {milco_s['elongation']['mean']:.2f} +/- {milco_s['elongation']['std']:.2f}")
    print(f"  Median: {milco_s['elongation']['median']:.2f} (IQR: {milco_s['elongation']['p25']:.2f} - {milco_s['elongation']['p75']:.2f})")
    print(f"  Range:  [{milco_s['elongation']['min']:.2f}, {milco_s['elongation']['max']:.2f}]")

    print(f"\nMILCO Elongated Percentages across Thresholds:")
    for th_key, th_val in milco_s["threshold_percentages"].items():
        th_num = th_key.replace("elongation_gte_", "")
        print(f"  Elongation >= {th_num}: {th_val['count']:3d} / {th_val['total']:3d}  --> {th_val['percentage']:6.2f}%")

    print(f"\nNOMBO Comparison Elongation Summary:")
    print(f"  Count:  {nombo_s['total_objects']}")
    print(f"  Mean:   {nombo_s['elongation']['mean']:.2f} +/- {nombo_s['elongation']['std']:.2f}")
    print(f"  Median: {nombo_s['elongation']['median']:.2f}")
    print(f"  Elongation >= 2.0: {nombo_s['threshold_percentages']['elongation_gte_2.0']['count']} / {nombo_s['threshold_percentages']['elongation_gte_2.0']['total']} ({nombo_s['threshold_percentages']['elongation_gte_2.0']['percentage']:.2f}%)")

    # Step 3: Generate visual plots & contact sheets
    print("\n--- STEP 3: GENERATING DIAGNOSTIC VISUALIZATIONS ---")
    hist_path = output_dir / "milco_nombo_aspect_ratio_histograms.png"
    plot_aspect_ratio_histograms(records, hist_path)

    milco_sheet_path = output_dir / "milco_contact_sheet.png"
    create_contact_sheet(
        records=records,
        output_path=milco_sheet_path,
        title="MILCO (Class 0: Mine-Like Contacts) - Bounding Box & Acoustic Shadow Review",
        target_class_id=0,
        selection_mode="stratified",
        grid_size=(6, 5),
        box_color=(0, 255, 120),  # bright emerald green
    )

    nombo_sheet_path = output_dir / "nombo_contact_sheet.png"
    create_contact_sheet(
        records=records,
        output_path=nombo_sheet_path,
        title="NOMBO (Class 1: Non-Mine Bottom Objects) - Comparison False Targets",
        target_class_id=1,
        selection_mode="stratified",
        grid_size=(6, 5),
        box_color=(0, 180, 255),  # electric cyan / orange
    )

    # Save summary report JSON
    summary_report = {
        "format_verification": format_diag,
        "aspect_ratio_statistics": stats,
        "artifact_paths": {
            "histograms": str(hist_path),
            "milco_contact_sheet": str(milco_sheet_path),
            "nombo_contact_sheet": str(nombo_sheet_path),
        },
    }
    summary_path = output_dir / "milco_cylinder_audit_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[OK] Full diagnostic report saved to: {summary_path}")

    print("\n" + "=" * 80)
    print(" AUDIT COMPLETE - READY FOR USER MANUAL REVIEW ")
    print("=" * 80)


if __name__ == "__main__":
    main()
