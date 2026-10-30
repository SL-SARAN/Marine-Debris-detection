"""
extract_structural_params.py
===========================
Structural Parameter Extraction from Optical Ghost Net Reference Imagery.

IMPORTANT DESIGN CONSTRAINT:
This script performs STRUCTURAL / GEOMETRIC ANALYSIS ONLY.
It extracts numerical parameter bounds (knot spacing ratios, strand thickness
ratios, drape aspect ratios, tangle densities) from photographic reference imagery.

It does NOT perform style-transfer, image-to-image translation, or visual synthesis.
Optical light physics and sonar acoustic grazing angle physics are fundamentally
different. These extracted parameters are exported as a JSON distribution to drive
the physics-based procedural mesh generator.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure optional OpenCV and PIL imports handle gracefully
try:
    import cv2
except ImportError:
    cv2 = None


def extract_from_image(image_path: Path):
    """
    Extract structural geometric properties from a single optical reference image.
    Uses edge/ridge filtering, contour analysis, and spatial autocorrelation.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for structural image analysis.")

    # Load in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [Warning] Could not decode image: {image_path}")
        return None

    h, w = img.shape
    # Normalize size for consistent measurement
    target_dim = 800
    scale = target_dim / max(h, w)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h_r, w_r = img_resized.shape
    diag = np.sqrt(h_r**2 + w_r**2)

    # 1. Multi-scale Edge & Ridge Detection (Isolating filament lines)
    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Morphological closing to group adjacent strands
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 2. Net Footprint & Bounding Contour
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter out tiny noise contours, find main net mass
    net_contours = [c for c in contours if cv2.contourArea(c) > (0.01 * h_r * w_r)]
    if not net_contours:
        net_contours = contours

    all_pts = np.vstack(net_contours)
    x, y, bw, bh = cv2.boundingRect(all_pts)
    net_span = max(bw, bh)
    if net_span < 10:
        return None

    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

    # 3. Tangle Density (Net Area vs Convex Hull Area)
    hull = cv2.convexHull(all_pts)
    hull_area = cv2.contourArea(hull)
    strand_pixel_count = np.count_nonzero(edges[y:y+bh, x:x+bw])
    tangle_density = float(strand_pixel_count) / float(hull_area) if hull_area > 0 else 0.15
    tangle_density = float(np.clip(tangle_density, 0.05, 0.85))

    # 4. Strand Thickness Estimation (Distance Transform on inverted edges)
    dist_trans = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    # Mean ridge distance represents typical cord half-width at edge boundaries
    strand_thickness = float(np.median(dist_trans[dist_trans > 0])) if np.any(dist_trans > 0) else 1.5
    strand_thickness_ratio = float(np.clip(strand_thickness / net_span, 0.003, 0.035))

    # 5. Mesh / Knot Spacing via Distance Transform peaks & Corner Detection
    corners = cv2.goodFeaturesToTrack(
        edges[y:y+bh, x:x+bw],
        maxCorners=150,
        qualityLevel=0.08,
        minDistance=max(4, int(net_span * 0.02))
    )
    if corners is not None and len(corners) >= 4:
        pts = corners[:, 0, :]
        # Pairwise distance to nearest neighbor (approximate mesh unit cell size)
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        median_spacing = float(np.median(min_dists))
        knot_spacing_ratio = float(np.clip(median_spacing / net_span, 0.02, 0.18))
    else:
        knot_spacing_ratio = 0.065

    # 6. Drape Classification based on aspect ratio and contour circularity
    perimeter = cv2.arcLength(hull, True)
    circularity = 4 * np.pi * (hull_area / (perimeter**2)) if perimeter > 0 else 0.5
    if circularity > 0.65:
        drape_mode = "pooled"      # Compact clump on seabed
    elif aspect_ratio > 2.2 or aspect_ratio < 0.45:
        drape_mode = "stretched"   # Stretched along current
    else:
        drape_mode = "snagged"     # Irregular / snagged over mound

    return {
        "knot_spacing_ratio": round(knot_spacing_ratio, 4),
        "strand_thickness_ratio": round(strand_thickness_ratio, 5),
        "aspect_ratio": round(aspect_ratio, 3),
        "tangle_density": round(tangle_density, 3),
        "drape_mode": drape_mode
    }


def analyze_optical_reference_dataset(input_dir: Path, output_file: Path, max_samples: int = 300):
    """
    Analyze optical reference folder and export parameter distributions.
    """
    print(f"[*] Scanning optical reference directory: {input_dir}")
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in valid_exts and p.is_file()]

    if not image_files:
        print(f"[!] No optical reference images found in: {input_dir}")
        print("    --> Applying default estimated parameters (no optical reference images provided yet;")
        print("        values are approximate placeholders, not sourced from a published standard).")
        print("    --> Once optical photos are placed in the folder, re-run to update with measured metrics.")
        
        # Engineering estimate placeholders pending optical reference measurements
        params = {
            "source": "Default estimated parameters (no optical reference images provided yet; values are approximate placeholders, not sourced from a published standard)",
            "sample_count": 0,
            "knot_spacing_ratio": {"min": 0.025, "max": 0.120, "mean": 0.065, "std": 0.022},
            "strand_thickness_ratio": {"min": 0.004, "max": 0.025, "mean": 0.012, "std": 0.005},
            "aspect_ratio": {"min": 0.55, "max": 2.80, "mean": 1.45, "std": 0.50},
            "tangle_density": {"min": 0.12, "max": 0.70, "mean": 0.38, "std": 0.14},
            "drape_mode_distribution": {"pooled": 0.40, "snagged": 0.40, "stretched": 0.20},
            "shadow_softness_factor": {"min": 1.2, "max": 3.0, "mean": 2.1}
        }
    else:
        total_found = len(image_files)
        if total_found > max_samples:
            print(f"[*] Found {total_found} images. Subsampling {max_samples} representative images for fast geometric analysis...")
            # Deterministic uniform sampling
            indices = np.linspace(0, total_found - 1, max_samples, dtype=int)
            image_files = [image_files[i] for i in indices]
        else:
            print(f"[*] Analyzing all {total_found} optical reference images...")

        extracted_samples = []
        drape_counts = {"pooled": 0, "snagged": 0, "stretched": 0}

        for idx, img_path in enumerate(image_files, 1):
            try:
                res = extract_from_image(img_path)
                if res:
                    extracted_samples.append(res)
                    drape_counts[res["drape_mode"]] += 1
            except Exception:
                pass
            if idx % 50 == 0 or idx == len(image_files):
                print(f"  --> Processed {idx}/{len(image_files)} reference images ({len(extracted_samples)} valid structural profiles extracted)...")

        if not extracted_samples:
            print("[!] Could not extract features from images. Falling back to default estimated parameters.")
            return analyze_optical_reference_dataset(Path("non_existent"), output_file, max_samples)

        # Aggregate statistics
        def get_stats(key):
            vals = [s[key] for s in extracted_samples]
            return {
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4)
            }

        total = len(extracted_samples)
        params = {
            "source": f"Measured from {total} optical reference images in {input_dir.name} (subsampled from {total_found} total)",
            "sample_count": total,
            "knot_spacing_ratio": get_stats("knot_spacing_ratio"),
            "strand_thickness_ratio": get_stats("strand_thickness_ratio"),
            "aspect_ratio": get_stats("aspect_ratio"),
            "tangle_density": get_stats("tangle_density"),
            "drape_mode_distribution": {k: round(v / total, 3) for k, v in drape_counts.items()},
            "shadow_softness_factor": {"min": 1.2, "max": 3.0, "mean": 2.1}
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(params, f, indent=2)

    print("\n" + "="*60)
    print("STRUCTURAL PARAMETER EXTRACTION SUMMARY")
    print("="*60)
    print(f"Reference Source : {params['source']}")
    print(f"Knot Spacing     : {params['knot_spacing_ratio']['mean']} ± {params['knot_spacing_ratio']['std']} (range: {params['knot_spacing_ratio']['min']} - {params['knot_spacing_ratio']['max']})")
    print(f"Strand Thickness : {params['strand_thickness_ratio']['mean']} ± {params['strand_thickness_ratio']['std']} (range: {params['strand_thickness_ratio']['min']} - {params['strand_thickness_ratio']['max']})")
    print(f"Aspect Ratio     : {params['aspect_ratio']['mean']} ± {params['aspect_ratio']['std']}")
    print(f"Tangle Density   : {params['tangle_density']['mean']} ± {params['tangle_density']['std']}")
    print(f"Drape Modes      : {params['drape_mode_distribution']}")
    print(f"[OK] Parameter distribution saved to: {output_file}")
    print("="*60 + "\n")
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structural parameters from optical ghost net reference images.")
    parser.add_argument("--input_dir", type=str, default="data/optical_ghost_net_reference",
                        help="Path to optical ghost net reference photos.")
    parser.add_argument("--output_json", type=str, default="data/structural_mesh_parameters.json",
                        help="Path to output structural parameters JSON.")
    parser.add_argument("--max_samples", type=int, default=300,
                        help="Maximum number of images to subsample for structural analysis.")
    args = parser.parse_args()

    analyze_optical_reference_dataset(Path(args.input_dir), Path(args.output_json), args.max_samples)
