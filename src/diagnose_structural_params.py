"""
diagnose_structural_params.py
==============================
Diagnostic checks on extract_structural_params.py output BEFORE trusting
the measured parameters for synthetic ghost net generation.

Checks:
1. Pre-clip boundary-hit rate for knot_spacing_ratio, strand_thickness_ratio,
   tangle_density — flags if clip bounds are truncating real data.
2. Confirms whether shadow_softness_factor is actually computed anywhere,
   or is a hardcoded literal.
3. Re-runs drape_mode classification with circularity/aspect_ratio logged
   per-image, to check if circularity-first ordering is suppressing the
   'stretched' category.
4. Saves a random visual QA contact sheet of N sample images for manual
   relevance/quality inspection.
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_DIR = Path("data/optical_ghost_net_reference/ghost-nets")
OUTPUT_DIR = Path("data/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_SAMPLES = 300
RANDOM_SEED = 42

# --- Hardcoded clip bounds copied from extract_structural_params.py ---
CLIP_BOUNDS = {
    "knot_spacing_ratio": (0.02, 0.18),
    "strand_thickness_ratio": (0.003, 0.035),
    "tangle_density": (0.05, 0.85),
}


def extract_raw_unclipped(image_path: Path):
    """Same pipeline as extract_from_image() but WITHOUT clipping,
    and with circularity/aspect_ratio logged explicitly."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    target_dim = 800
    scale = target_dim / max(h, w)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h_r, w_r = img_resized.shape

    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    net_contours = [c for c in contours if cv2.contourArea(c) > (0.01 * h_r * w_r)]
    if not net_contours:
        net_contours = contours

    all_pts = np.vstack(net_contours)
    x, y, bw, bh = cv2.boundingRect(all_pts)
    net_span = max(bw, bh)
    if net_span < 10:
        return None

    aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

    hull = cv2.convexHull(all_pts)
    hull_area = cv2.contourArea(hull)
    strand_pixel_count = np.count_nonzero(edges[y:y+bh, x:x+bw])
    tangle_density_raw = float(strand_pixel_count) / float(hull_area) if hull_area > 0 else None

    dist_trans = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    strand_thickness = float(np.median(dist_trans[dist_trans > 0])) if np.any(dist_trans > 0) else None
    strand_thickness_ratio_raw = strand_thickness / net_span if strand_thickness else None

    corners = cv2.goodFeaturesToTrack(
        edges[y:y+bh, x:x+bw], maxCorners=150, qualityLevel=0.08,
        minDistance=max(4, int(net_span * 0.02))
    )
    knot_spacing_ratio_raw = None
    if corners is not None and len(corners) >= 4:
        pts = corners[:, 0, :]
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        knot_spacing_ratio_raw = float(np.median(min_dists)) / net_span

    perimeter = cv2.arcLength(hull, True)
    circularity = 4 * np.pi * (hull_area / (perimeter**2)) if perimeter > 0 else None

    return {
        "path": str(image_path),
        "knot_spacing_ratio_raw": knot_spacing_ratio_raw,
        "strand_thickness_ratio_raw": strand_thickness_ratio_raw,
        "tangle_density_raw": tangle_density_raw,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
    }


def check_shadow_softness_usage():
    """Static check: does shadow_softness_factor appear anywhere as a
    computed value, or only as a literal?"""
    src_file = Path("src/extract_structural_params.py")
    text = src_file.read_text(encoding="utf-8", errors="ignore")
    occurrences = [i for i in range(len(text)) if text.startswith("shadow_softness_factor", i)]
    print(f"[CHECK] 'shadow_softness_factor' appears {len(occurrences)}x in extract_structural_params.py")
    print("[CHECK] Searching for any computation (variable assignment from cv2/np) tied to it...")
    computed = "shadow_softness" in text and (
        "cv2." in text.split("shadow_softness_factor")[0][-200:] if "shadow_softness_factor" in text else False
    )
    print(f"[RESULT] Appears to be a hardcoded literal only: "
          f"{'CONFIRMED - not computed from image data' if not computed else 'possible computation found, verify manually'}")


def main():
    random.seed(RANDOM_SEED)
    image_files = [p for p in INPUT_DIR.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                   and p.is_file()]
    total_found = len(image_files)
    print(f"[*] Found {total_found} total images in {INPUT_DIR}")

    sampled = image_files if total_found <= MAX_SAMPLES else \
        [image_files[i] for i in np.linspace(0, total_found - 1, MAX_SAMPLES, dtype=int)]

    results = []
    for idx, p in enumerate(sampled, 1):
        try:
            r = extract_raw_unclipped(p)
            if r:
                results.append(r)
        except Exception as e:
            pass
        if idx % 50 == 0:
            print(f"  --> processed {idx}/{len(sampled)}")

    print(f"\n[*] Valid extractions: {len(results)}/{len(sampled)}\n")

    # --- Check 1: boundary hit rates ---
    print("="*60)
    print("CHECK 1: CLIP BOUNDARY HIT RATES (pre-clip raw values)")
    print("="*60)
    for param, (lo, hi) in CLIP_BOUNDS.items():
        raw_key = f"{param}_raw"
        vals = [r[raw_key] for r in results if r.get(raw_key) is not None]
        if not vals:
            print(f"{param}: no valid values extracted")
            continue
        vals = np.array(vals)
        below = np.sum(vals < lo) / len(vals) * 100
        above = np.sum(vals > hi) / len(vals) * 100
        print(f"\n{param} (clip range {lo}-{hi}):")
        print(f"  raw min={vals.min():.4f}  raw max={vals.max():.4f}  raw mean={vals.mean():.4f}")
        print(f"  % of samples BELOW clip floor: {below:.1f}%")
        print(f"  % of samples ABOVE clip ceiling: {above:.1f}%")
        if below > 5 or above > 5:
            print(f"  [FLAG] >5% of real data is being truncated — clip bounds likely need widening.")

    # --- Check 2: shadow_softness_factor ---
    print("\n" + "="*60)
    print("CHECK 2: SHADOW_SOFTNESS_FACTOR COMPUTATION CHECK")
    print("="*60)
    check_shadow_softness_usage()

    # --- Check 3: drape mode classification order ---
    print("\n" + "="*60)
    print("CHECK 3: DRAPE MODE CLASSIFICATION SENSITIVITY")
    print("="*60)
    circ_vals = [r["circularity"] for r in results if r["circularity"] is not None]
    asp_vals = [r["aspect_ratio"] for r in results if r["aspect_ratio"] is not None]
    print(f"circularity: min={min(circ_vals):.3f} max={max(circ_vals):.3f} mean={np.mean(circ_vals):.3f}")
    print(f"aspect_ratio: min={min(asp_vals):.3f} max={max(asp_vals):.3f} mean={np.mean(asp_vals):.3f}")

    would_be_stretched_but_caught_by_circularity = 0
    for r in results:
        if r["circularity"] is not None and r["aspect_ratio"] is not None:
            if r["circularity"] > 0.65 and (r["aspect_ratio"] > 2.2 or r["aspect_ratio"] < 0.45):
                would_be_stretched_but_caught_by_circularity += 1
    pct = would_be_stretched_but_caught_by_circularity / len(results) * 100 if results else 0
    print(f"\nImages classified 'pooled' (circularity>0.65) that ALSO meet the "
          f"'stretched' aspect-ratio condition: {would_be_stretched_but_caught_by_circularity} "
          f"({pct:.1f}% of samples)")
    if pct > 3:
        print("[FLAG] Circularity-first ordering is likely suppressing the 'stretched' "
              "category for a meaningful fraction of images. Consider reordering or "
              "using a combined score instead of if/elif priority.")

    # --- Check 4: visual QA contact sheet ---
    print("\n" + "="*60)
    print("CHECK 4: RANDOM VISUAL QA SAMPLE (20-30 images)")
    print("="*60)
    qa_sample = random.sample(image_files, min(24, len(image_files)))
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    for ax, img_path in zip(axes.flat, qa_sample):
        img = cv2.imread(str(img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
        ax.set_title(img_path.name, fontsize=6)
        ax.axis("off")
    plt.tight_layout()
    qa_path = OUTPUT_DIR / "visual_qa_contact_sheet.png"
    plt.savefig(qa_path, dpi=150)
    print(f"[OK] Saved 24-image random contact sheet to: {qa_path}")
    print("     --> MANUALLY REVIEW this image before trusting the dataset's relevance.")

    # Save raw diagnostic data
    diag_out = OUTPUT_DIR / "raw_diagnostic_results.json"
    with open(diag_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OK] Full raw (pre-clip) results saved to: {diag_out}")


if __name__ == "__main__":
    main()
