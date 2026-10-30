import os
import shutil
from pathlib import Path
from collections import Counter
import random

# Set random seed for reproducible train/val split of Marine_PULSE
random.seed(42)

def clean_obsolete_files():
    print("=" * 65)
    print("  STEP 1: CLEANING UP UNUSED SCRATCH & TEMPORARY FILES")
    print("=" * 65)

    base_dir = Path(r"c:\Users\saran\Downloads\SODD")
    
    files_to_remove = [
        "check_klsg.py",
        "inspect_datasets.py",
        "scratch_xtf.py",
        "test_api.py",
        "test_enhanced.py",
        "test_realistic.py",
        "test_synthesis.py",
        "verify_env.py",
        "generate_perfect_sss.py",
        "prepare_merged_dataset.py",
        "yolov8n.pt"  # base weight; fine-tuned model is in pipeline/models/best.pt
    ]

    for fname in files_to_remove:
        fpath = base_dir / fname
        if fpath.exists():
            try:
                fpath.unlink()
                print(f"  [Removed file] {fname}")
            except Exception as e:
                print(f"  [Could not remove {fname}]: {e}")

    # Remove __MACOSX directory if exists
    macosx_dir = base_dir / "__MACOSX"
    if macosx_dir.exists():
        try:
            shutil.rmtree(macosx_dir)
            print("  [Removed directory] __MACOSX")
        except Exception as e:
            print(f"  [Could not remove __MACOSX]: {e}")

    # Remove empty synthetic_ghost_nets if present
    empty_dir = Path(r"D:\MarineDebris\data\synthetic_ghost_nets")
    if empty_dir.exists() and not any(empty_dir.iterdir()):
        try:
            empty_dir.rmdir()
            print("  [Removed empty directory] synthetic_ghost_nets")
        except Exception:
            pass

    print("  Cleanup completed successfully!\n")

def build_common_dataset():
    print("=" * 65)
    print("  STEP 2: PREPARING UNIFIED COMMON SSS DATASET")
    print("=" * 65)

    base_dir = Path(r"D:\MarineDebris\data")
    common_dir = base_dir / "common_sss_dataset"
    sodd_dir = base_dir / "SODD_synthetic_sss"
    klsg_dir = base_dir / "KLSG"
    pulse_dir = base_dir / "Marine_PULSE"

    # Target class taxonomy:
    # 0: ghost_net (from SODD_synthetic_sss)
    # 1: shipwreck (from KLSG class 3)
    # 2: aircraft (from KLSG class 0)
    # 3: seafloor_debris (from KLSG class 2 'other')
    # 4: pipeline_cable (from Marine_PULSE 'pipeline or cable')
    # 5: engineering_platform (from Marine_PULSE 'engineering platform')
    # Note: KLSG 'fish', Marine_PULSE 'seabed surface' & 'residual mound'
    # are included as negative background samples (empty labels) to eliminate false alarms!

    class_names = {
        0: "ghost_net",
        1: "shipwreck",
        2: "aircraft",
        3: "seafloor_debris",
        4: "pipeline_cable",
        5: "engineering_platform"
    }

    klsg_mapping = {
        3: 1,  # shipwreck
        0: 2,  # aircraft
        2: 3   # other -> seafloor_debris
    }

    # Reset / create common dataset folder
    for split in ["train", "validation", "test"]:
        (common_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (common_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    summary_stats = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter()
    }
    image_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }
    bg_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    # -------------------------------------------------------------
    # 1. INGEST SODD SYNTHETIC GHOST NETS
    # -------------------------------------------------------------
    print("[1/3] Ingesting Synthetic SSS Ghost Net Dataset...")
    sodd_splits = [("train", "train"), ("validation", "validation"), ("test", "test")]
    for src_split, tgt_split in sodd_splits:
        img_dir = sodd_dir / src_split / "images"
        lbl_dir = sodd_dir / src_split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*.*"):
            dest_img = common_dir / tgt_split / "images" / f"sodd_{img_path.name}"
            shutil.copy2(img_path, dest_img)
            image_counts[tgt_split] += 1

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            dest_lbl = common_dir / tgt_split / "labels" / f"sodd_{img_path.stem}.txt"
            if lbl_path.exists():
                lines = lbl_path.read_text().strip().splitlines()
                valid_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        # Class 0: ghost_net
                        valid_lines.append(f"0 {' '.join(parts[1:])}")
                        summary_stats[tgt_split][0] += 1
                dest_lbl.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""))
                if not valid_lines:
                    bg_counts[tgt_split] += 1
            else:
                dest_lbl.write_text("")
                bg_counts[tgt_split] += 1

    # -------------------------------------------------------------
    # 2. INGEST KLSG SONAR BENCHMARK
    # -------------------------------------------------------------
    print("[2/3] Ingesting KLSG Sonar Benchmark Dataset...")
    klsg_splits = [("train", "train"), ("valid", "validation"), ("test", "test")]
    for src_split, tgt_split in klsg_splits:
        img_dir = klsg_dir / src_split / "images"
        lbl_dir = klsg_dir / src_split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.glob("*.*"):
            dest_img = common_dir / tgt_split / "images" / f"klsg_{img_path.name}"
            shutil.copy2(img_path, dest_img)
            image_counts[tgt_split] += 1

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            dest_lbl = common_dir / tgt_split / "labels" / f"klsg_{img_path.stem}.txt"
            if lbl_path.exists():
                lines = lbl_path.read_text().strip().splitlines()
                valid_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        src_cls = int(parts[0])
                        if src_cls in klsg_mapping:
                            tgt_cls = klsg_mapping[src_cls]
                            valid_lines.append(f"{tgt_cls} {' '.join(parts[1:])}")
                            summary_stats[tgt_split][tgt_cls] += 1
                dest_lbl.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""))
                if not valid_lines:
                    bg_counts[tgt_split] += 1
            else:
                dest_lbl.write_text("")
                bg_counts[tgt_split] += 1

    # -------------------------------------------------------------
    # 3. INGEST MARINE_PULSE SONAR DATASET
    # -------------------------------------------------------------
    print("[3/3] Ingesting Marine_PULSE Sonar Dataset...")
    # Marine_PULSE has 'train' (split into 85% train, 15% validation) and 'test' (test)
    if pulse_dir.exists():
        # Categories mapping
        # "pipeline or cable" -> class 4
        # "engineering platform" -> class 5
        # "seabed surface" -> background (empty)
        # "underwater residual mound" -> background (empty)
        pulse_cats = {
            "pipeline or cable": 4,
            "engineering platform": 5,
            "seabed surface": None,
            "underwater residual mound": None
        }

        # Process Test split
        test_dir = pulse_dir / "test"
        if test_dir.exists():
            for cat_name, cid in pulse_cats.items():
                cat_folder = test_dir / cat_name
                if cat_folder.exists():
                    for img_file in cat_folder.glob("*.*"):
                        safe_stem = f"pulse_test_{cat_name.replace(' ', '_')}_{img_file.name}"
                        dest_img = common_dir / "test" / "images" / safe_stem
                        shutil.copy2(img_file, dest_img)
                        image_counts["test"] += 1

                        dest_lbl = common_dir / "test" / "labels" / f"{dest_img.stem}.txt"
                        if cid is not None:
                            # Standard bounding box for cropped anomaly patch
                            dest_lbl.write_text(f"{cid} 0.5 0.5 0.90 0.85\n")
                            summary_stats["test"][cid] += 1
                        else:
                            dest_lbl.write_text("")
                            bg_counts["test"] += 1

        # Process Train split (split 85% train / 15% validation)
        train_dir = pulse_dir / "train"
        if train_dir.exists():
            for cat_name, cid in pulse_cats.items():
                cat_folder = train_dir / cat_name
                if cat_folder.exists():
                    img_list = list(cat_folder.glob("*.*"))
                    random.shuffle(img_list)
                    n_val = max(1, int(len(img_list) * 0.15))
                    val_imgs = img_list[:n_val]
                    train_imgs = img_list[n_val:]

                    for img_file in train_imgs:
                        safe_stem = f"pulse_train_{cat_name.replace(' ', '_')}_{img_file.name}"
                        dest_img = common_dir / "train" / "images" / safe_stem
                        shutil.copy2(img_file, dest_img)
                        image_counts["train"] += 1

                        dest_lbl = common_dir / "train" / "labels" / f"{dest_img.stem}.txt"
                        if cid is not None:
                            dest_lbl.write_text(f"{cid} 0.5 0.5 0.90 0.85\n")
                            summary_stats["train"][cid] += 1
                        else:
                            dest_lbl.write_text("")
                            bg_counts["train"] += 1

                    for img_file in val_imgs:
                        safe_stem = f"pulse_val_{cat_name.replace(' ', '_')}_{img_file.name}"
                        dest_img = common_dir / "validation" / "images" / safe_stem
                        shutil.copy2(img_file, dest_img)
                        image_counts["validation"] += 1

                        dest_lbl = common_dir / "validation" / "labels" / f"{dest_img.stem}.txt"
                        if cid is not None:
                            dest_lbl.write_text(f"{cid} 0.5 0.5 0.90 0.85\n")
                            summary_stats["validation"][cid] += 1
                        else:
                            dest_lbl.write_text("")
                            bg_counts["validation"] += 1

    # Generate data.yaml
    yaml_content = f"""# Unified Multi-Source Side-Scan Sonar (SSS) Marine Debris Dataset
# Merged Sources:
#  1. SODD_synthetic_sss (Acoustic physics synthetic ghost nets)
#  2. KLSG Sonar Benchmark (Shipwrecks, Aircraft, Seafloor Debris)
#  3. Marine_PULSE Sonar (Subsea Pipelines/Cables, Engineering Platforms, Natural Seabed)

path: {str(common_dir).replace('\\', '/')}
train: train/images
val: validation/images
test: test/images

nc: 6
names:
  0: ghost_net
  1: shipwreck
  2: aircraft
  3: seafloor_debris
  4: pipeline_cable
  5: engineering_platform
"""
    yaml_path = common_dir / "data.yaml"
    yaml_path.write_text(yaml_content)

    # Generate README.md
    readme_content = f"""# Unified Common Side-Scan Sonar (SSS) Marine Debris Dataset

This common dataset consolidates three primary sonar benchmark datasets into a single, standardized YOLO-format repository:
1. **Synthetic Ghost Net Dataset (SODD_synthetic_sss)**: High-resolution acoustic waterfall images with ray-cast shadows, specular highlights, and Rayleigh speckle.
2. **KLSG Sonar Benchmark**: Curated side-scan sonar imagery of shipwrecks, submerged aircraft, and seafloor debris.
3. **Marine_PULSE Sonar Dataset**: High-frequency sonar swaths of subsea pipelines, underwater cables, and offshore engineering platforms.

---

## Dataset Statistics

| Split | Images | Negative (Background) Samples | Labeled Target Boxes |
| :--- | :--- | :--- | :--- |
| **Train** | {image_counts['train']} | {bg_counts['train']} | {sum(summary_stats['train'].values())} |
| **Validation** | {image_counts['validation']} | {bg_counts['validation']} | {sum(summary_stats['validation'].values())} |
| **Test** | {image_counts['test']} | {bg_counts['test']} | {sum(summary_stats['test'].values())} |
| **TOTAL** | **{sum(image_counts.values())}** | **{sum(bg_counts.values())}** | **{sum(sum(s.values()) for s in summary_stats.values())}** |

---

## Class Breakdown

| Class ID | Target Class | Train Boxes | Validation Boxes | Test Boxes | Total Boxes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `ghost_net` | {summary_stats['train'][0]} | {summary_stats['validation'][0]} | {summary_stats['test'][0]} | {summary_stats['train'][0] + summary_stats['validation'][0] + summary_stats['test'][0]} |
| 1 | `shipwreck` | {summary_stats['train'][1]} | {summary_stats['validation'][1]} | {summary_stats['test'][1]} | {summary_stats['train'][1] + summary_stats['validation'][1] + summary_stats['test'][1]} |
| 2 | `aircraft` | {summary_stats['train'][2]} | {summary_stats['validation'][2]} | {summary_stats['test'][2]} | {summary_stats['train'][2] + summary_stats['validation'][2] + summary_stats['test'][2]} |
| 3 | `seafloor_debris` | {summary_stats['train'][3]} | {summary_stats['validation'][3]} | {summary_stats['test'][3]} | {summary_stats['train'][3] + summary_stats['validation'][3] + summary_stats['test'][3]} |
| 4 | `pipeline_cable` | {summary_stats['train'][4]} | {summary_stats['validation'][4]} | {summary_stats['test'][4]} | {summary_stats['train'][4] + summary_stats['validation'][4] + summary_stats['test'][4]} |
| 5 | `engineering_platform` | {summary_stats['train'][5]} | {summary_stats['validation'][5]} | {summary_stats['test'][5]} | {summary_stats['train'][5] + summary_stats['validation'][5] + summary_stats['test'][5]} |

---

## Negative / Background Clutter Samples
To ensure the detector resists false alarms from natural seabed topology, **{sum(bg_counts.values())} negative background images** (containing natural sand ripples, rock mounds, and fish clutter) are included with empty annotation files.
"""
    readme_path = common_dir / "README.md"
    readme_path.write_text(readme_content)

    print("\n" + "=" * 65)
    print("  COMMON DATASET CREATED SUCCESSFULLY!")
    print("=" * 65)
    print(f"Location: {common_dir}")
    print(f"Total Images: {sum(image_counts.values())}")
    for split in ["train", "validation", "test"]:
        print(f"  [{split.upper()}] Images: {image_counts[split]} (Backgrounds: {bg_counts[split]})")
        for cid, cname in class_names.items():
            print(f"    Class {cid} ({cname}): {summary_stats[split][cid]} boxes")
    print(f"\nConfiguration: {yaml_path}")
    print(f"Documentation: {readme_path}")

if __name__ == "__main__":
    clean_obsolete_files()
    build_common_dataset()
