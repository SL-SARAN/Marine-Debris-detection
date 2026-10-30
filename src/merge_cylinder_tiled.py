"""
Merge cylider2 cylinder instances into common_sss_dataset as class 6,
implement tiled 640x640 cropping with overlap, and prepare for training.
"""
import os
import shutil
import cv2
import numpy as np
import random
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
CYL_BASE = Path('d:/MarineDebris/unprocessed_data/cylinder')
COMMON_DS = Path('d:/MarineDebris/data/common_sss_dataset')
TILE_SIZE = 640
TILE_OVERLAP = 128  # 20% overlap = 128px on each side
STRIDE = TILE_SIZE - TILE_OVERLAP  # 512px stride
MIN_BOX_AREA_PX = 16  # Minimum box area in pixels to keep after clipping
MIN_BOX_VISIBILITY = 0.3  # Object must have ≥30% visible area to be included in tile
CYL_CLASS_ID_SRC = 0   # cylinder in cylider2 data.yaml
MANTA_CLASS_ID_SRC = 1  # manta in cylider2 data.yaml
CYL_CLASS_ID_DST = 6   # cylinder in common_sss_dataset

# Split mapping: cylider2 split -> common_sss_dataset split
SPLIT_MAP = {
    'train': 'train',
    'valid': 'validation',
    'test': 'test',
}

img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

# =============================================================================
# PHASE 1: IDENTIFY AND FILTER CYLINDER-ONLY IMAGES
# =============================================================================
def identify_cylinder_images():
    """Identify images containing cylinder annotations, excluding manta-only images."""
    print("=" * 70)
    print("  PHASE 1: FILTERING CYLINDER INSTANCES (EXCLUDING MANTA)")
    print("=" * 70)
    
    results = {}  # split -> list of (img_path, label_lines_cylinder_only)
    stats = Counter()
    
    for src_split, dst_split in SPLIT_MAP.items():
        img_dir = CYL_BASE / src_split / 'images'
        lbl_dir = CYL_BASE / src_split / 'labels'
        
        if not img_dir.exists():
            print(f"  [SKIP] {src_split} split not found")
            continue
        
        split_results = []
        
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in img_exts:
                continue
            
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            
            content = lbl_path.read_text().strip()
            if not content:
                continue
            
            cylinder_lines = []
            manta_lines = []
            
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id == CYL_CLASS_ID_SRC:
                    # Remap to class 6
                    cylinder_lines.append(f"{CYL_CLASS_ID_DST} {' '.join(parts[1:])}")
                elif cls_id == MANTA_CLASS_ID_SRC:
                    manta_lines.append(line.strip())
            
            stats['total_images'] += 1
            stats['total_cylinder_boxes'] += len(cylinder_lines)
            stats['total_manta_boxes'] += len(manta_lines)
            
            if cylinder_lines:
                split_results.append((img_path, cylinder_lines, dst_split))
                stats['images_with_cylinder'] += 1
            else:
                stats['manta_only_dropped'] += 1
        
        results[dst_split] = split_results
        print(f"  [{src_split:>5} -> {dst_split}] {len(split_results)} images with cylinder annotations")
    
    print(f"\n  Summary:")
    print(f"    Total images scanned:       {stats['total_images']}")
    print(f"    Images with cylinder:        {stats['images_with_cylinder']}")
    print(f"    Manta-only images dropped:   {stats['manta_only_dropped']}")
    print(f"    Cylinder boxes kept:         {stats['total_cylinder_boxes']}")
    print(f"    Manta boxes dropped:         {stats['total_manta_boxes']}")
    
    return results, stats


# =============================================================================
# PHASE 2: TILED 640x640 CROPPING WITH OVERLAP
# =============================================================================
def compute_tiles(img_w, img_h, tile_size, stride):
    """Compute tile origins (x, y) for a given image size."""
    tiles = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            # Clamp to image boundary
            x1 = min(x, max(0, img_w - tile_size))
            y1 = min(y, max(0, img_h - tile_size))
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            
            # Clamp if image is smaller than tile
            x2 = min(x2, img_w)
            y2 = min(y2, img_h)
            
            tile = (x1, y1, x2, y2)
            if tile not in tiles:
                tiles.append(tile)
    return tiles


def assign_box_to_tile(box_cx, box_cy, box_w, box_h, img_w, img_h, tiles):
    """
    DEDUPLICATION STRATEGY: Center-point assignment.
    
    Each annotation is assigned to exactly ONE tile — the tile whose region
    contains the bounding box CENTER. This guarantees:
      - No double-counting: each object appears in exactly one tile's label
      - Objects straddling tile boundaries are assigned to whichever tile
        contains their center point
      - The box is then clipped to the tile's boundary, so partial visibility
        is preserved but the object is only counted once
    
    Returns: list of (tile_index, clipped_box_in_tile_coords) — always 0 or 1 items.
    """
    # Convert normalized to pixel coords
    cx_px = box_cx * img_w
    cy_px = box_cy * img_h
    w_px = box_w * img_w
    h_px = box_h * img_h
    
    x1_box = cx_px - w_px / 2
    y1_box = cy_px - h_px / 2
    x2_box = cx_px + w_px / 2
    y2_box = cy_px + h_px / 2
    
    original_area = w_px * h_px
    
    assignments = []
    
    for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile_w = tx2 - tx1
        tile_h = ty2 - ty1
        
        # Check if center falls within this tile
        if tx1 <= cx_px < tx2 and ty1 <= cy_px < ty2:
            # Clip box to tile boundary
            clipped_x1 = max(x1_box, tx1)
            clipped_y1 = max(y1_box, ty1)
            clipped_x2 = min(x2_box, tx2)
            clipped_y2 = min(y2_box, ty2)
            
            clipped_w = clipped_x2 - clipped_x1
            clipped_h = clipped_y2 - clipped_y1
            
            if clipped_w <= 0 or clipped_h <= 0:
                continue
            
            clipped_area = clipped_w * clipped_h
            visibility = clipped_area / original_area if original_area > 0 else 0
            
            if visibility < MIN_BOX_VISIBILITY:
                continue
            
            if clipped_area < MIN_BOX_AREA_PX:
                continue
            
            # Convert to normalized coordinates relative to tile
            new_cx = (clipped_x1 + clipped_x2) / 2 - tx1
            new_cy = (clipped_y1 + clipped_y2) / 2 - ty1
            
            norm_cx = new_cx / tile_w
            norm_cy = new_cy / tile_h
            norm_w = clipped_w / tile_w
            norm_h = clipped_h / tile_h
            
            # Clamp to [0, 1]
            norm_cx = max(0.0, min(1.0, norm_cx))
            norm_cy = max(0.0, min(1.0, norm_cy))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            assignments.append((i, norm_cx, norm_cy, norm_w, norm_h, visibility))
            break  # Center can only be in one tile — stop searching
    
    return assignments


def tile_and_merge_cylinder_images(filtered_images, stats):
    """Tile cylinder images and merge into common_sss_dataset."""
    print("\n" + "=" * 70)
    print("  PHASE 2: TILED 640x640 CROPPING + MERGE INTO COMMON DATASET")
    print("=" * 70)
    
    tile_stats = Counter()
    
    for dst_split, image_list in filtered_images.items():
        dst_img_dir = COMMON_DS / dst_split / 'images'
        dst_lbl_dir = COMMON_DS / dst_split / 'labels'
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, cyl_label_lines, _ in image_list:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [WARN] Could not read {img_path.name}, skipping")
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Parse cylinder annotations
            boxes = []
            for line in cyl_label_lines:
                parts = line.strip().split()
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls_id, cx, cy, w, h))
            
            # Determine if tiling is needed
            # Tile if image is larger than tile_size in either dimension
            needs_tiling = img_w > TILE_SIZE or img_h > TILE_SIZE
            
            if needs_tiling:
                tiles = compute_tiles(img_w, img_h, TILE_SIZE, STRIDE)
                
                for tile_idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                    tile_w = tx2 - tx1
                    tile_h = ty2 - ty1
                    tile_img = img[ty1:ty2, tx1:tx2]
                    
                    # Pad if tile is smaller than TILE_SIZE (edge cases)
                    if tile_w < TILE_SIZE or tile_h < TILE_SIZE:
                        padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                        padded[:tile_h, :tile_w] = tile_img
                        tile_img = padded
                    
                    # Assign boxes to this tile via center-point rule
                    tile_labels = []
                    for cls_id, cx, cy, w, h in boxes:
                        assignments = assign_box_to_tile(cx, cy, w, h, img_w, img_h, [(tx1, ty1, tx2, ty2)])
                        for _, ncx, ncy, nw, nh, vis in assignments:
                            # Recheck center is in THIS tile (redundant but safe)
                            cx_px = cx * img_w
                            cy_px = cy * img_h
                            if tx1 <= cx_px < tx2 and ty1 <= cy_px < ty2:
                                tile_labels.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                    
                    # Only save tiles that contain at least one cylinder annotation
                    if tile_labels:
                        tile_name = f"cyl_{img_path.stem}_t{tile_idx}"
                        tile_img_path = dst_img_dir / f"{tile_name}.jpg"
                        tile_lbl_path = dst_lbl_dir / f"{tile_name}.txt"
                        
                        cv2.imwrite(str(tile_img_path), tile_img)
                        tile_lbl_path.write_text("\n".join(tile_labels) + "\n")
                        
                        tile_stats[f'{dst_split}_tiles'] += 1
                        tile_stats[f'{dst_split}_boxes'] += len(tile_labels)
            else:
                # Image fits within 640x640 — just copy directly
                # Resize to 640x640 if needed
                if img_w != TILE_SIZE or img_h != TILE_SIZE:
                    img = cv2.resize(img, (TILE_SIZE, TILE_SIZE))
                
                out_name = f"cyl_{img_path.stem}"
                dst_img_path = dst_img_dir / f"{out_name}.jpg"
                dst_lbl_path = dst_lbl_dir / f"{out_name}.txt"
                
                cv2.imwrite(str(dst_img_path), img)
                dst_lbl_path.write_text("\n".join(cyl_label_lines) + "\n")
                
                tile_stats[f'{dst_split}_direct'] += 1
                tile_stats[f'{dst_split}_boxes'] += len(cyl_label_lines)
    
    print(f"\n  Tiling Results:")
    for split in ['train', 'validation', 'test']:
        tiles = tile_stats.get(f'{split}_tiles', 0)
        direct = tile_stats.get(f'{split}_direct', 0)
        boxes = tile_stats.get(f'{split}_boxes', 0)
        print(f"    [{split.upper():>10}] Tiled crops: {tiles}  |  Direct copies: {direct}  |  Total boxes: {boxes}")
    
    return tile_stats


# =============================================================================
# PHASE 3: UPDATE data.yaml
# =============================================================================
def update_data_yaml():
    """Update data.yaml to include class 6: cylinder."""
    print("\n" + "=" * 70)
    print("  PHASE 3: UPDATING data.yaml FOR 7-CLASS TAXONOMY")
    print("=" * 70)
    
    yaml_content = f"""# Unified Multi-Source Side-Scan Sonar (SSS) Marine Debris Dataset
# Merged Sources:
#  1. SODD_synthetic_sss (Acoustic physics synthetic ghost nets)
#  2. KLSG Sonar Benchmark (Shipwrecks, Aircraft, Seafloor Debris)
#  3. Marine_PULSE Sonar (Subsea Pipelines/Cables, Engineering Platforms, Natural Seabed)
#  4. cylider2 / SOCA-YOLO (Cylinder mines from side-scan sonar)

path: {str(COMMON_DS).replace(chr(92), '/')}
train: train/images
val: validation/images
test: test/images

nc: 7
names:
  0: ghost_net
  1: shipwreck
  2: aircraft
  3: seafloor_debris
  4: pipeline_cable
  5: engineering_platform
  6: cylinder
"""
    yaml_path = COMMON_DS / 'data.yaml'
    yaml_path.write_text(yaml_content)
    print(f"  Written: {yaml_path}")
    print(f"  Classes: 7 (added class 6: cylinder)")


# =============================================================================
# PHASE 4: FINAL DATASET VALIDATION
# =============================================================================
def validate_final_dataset():
    """Validate the merged dataset before training."""
    print("\n" + "=" * 70)
    print("  PHASE 4: POST-MERGE VALIDATION")
    print("=" * 70)
    
    class_dist = defaultdict(Counter)
    pair_issues = []
    total_images = Counter()
    empty_labels = Counter()
    
    for split in ['train', 'validation', 'test']:
        img_dir = COMMON_DS / split / 'images'
        lbl_dir = COMMON_DS / split / 'labels'
        
        imgs = {f.stem for f in img_dir.iterdir() if f.suffix.lower() in img_exts}
        lbls = {f.stem for f in lbl_dir.iterdir() if f.suffix.lower() == '.txt'}
        
        orphan_imgs = imgs - lbls
        orphan_lbls = lbls - imgs
        
        if orphan_imgs:
            pair_issues.append(f"  [{split}] {len(orphan_imgs)} images without labels")
        if orphan_lbls:
            pair_issues.append(f"  [{split}] {len(orphan_lbls)} labels without images")
        
        total_images[split] = len(imgs)
        
        for lbl_file in lbl_dir.glob('*.txt'):
            content = lbl_file.read_text().strip()
            if not content:
                empty_labels[split] += 1
                continue
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    class_dist[split][cls_id] += 1
    
    class_names = {
        0: 'ghost_net', 1: 'shipwreck', 2: 'aircraft',
        3: 'seafloor_debris', 4: 'pipeline_cable', 5: 'engineering_platform',
        6: 'cylinder'
    }
    
    print(f"\n  {'Split':>12} | {'Images':>7} | {'BG':>4} |", end="")
    for cid in range(7):
        print(f" {class_names[cid][:8]:>8} |", end="")
    print()
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*4}-+" + "+".join([f"-{'-'*8}-" for _ in range(7)]) + "+")
    
    for split in ['train', 'validation', 'test']:
        print(f"  {split:>12} | {total_images[split]:>7} | {empty_labels[split]:>4} |", end="")
        for cid in range(7):
            print(f" {class_dist[split][cid]:>8} |", end="")
        print()
    
    if pair_issues:
        print(f"\n  [FAIL] Pair issues found:")
        for issue in pair_issues:
            print(f"    {issue}")
    else:
        print(f"\n  [PASS] All image-label pairs consistent")
    
    return class_dist, total_images


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print(f"\n{'#' * 70}")
    print(f"  CYLINDER MERGE + TILED CROPPING PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")
    
    # Phase 1: Filter
    filtered, filter_stats = identify_cylinder_images()
    
    # Phase 2: Tile and merge
    tile_stats = tile_and_merge_cylinder_images(filtered, filter_stats)
    
    # Phase 3: Update data.yaml
    update_data_yaml()
    
    # Phase 4: Validate
    class_dist, img_counts = validate_final_dataset()
    
    print(f"\n{'#' * 70}")
    print(f"  MERGE COMPLETE — READY FOR TRAINING")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}\n")
