import os
import glob
import shutil

src_base = r"D:\MarineDebris\data\SODD_nets"
dst_base = r"D:\MarineDebris\data\SODD_nets"

os.makedirs(dst_base, exist_ok=True)

stats = {}

for split in ["train", "validation", "test"]:
    src_lbl_dir = os.path.join(src_base, split, "labels")
    src_img_dir = os.path.join(src_base, split, "images")
    
    dst_lbl_dir = os.path.join(dst_base, split, "labels")
    dst_img_dir = os.path.join(dst_base, split, "images")
    
    os.makedirs(dst_lbl_dir, exist_ok=True)
    os.makedirs(dst_img_dir, exist_ok=True)
    
    img_count = 0
    box_count = 0
    
    lbl_files = sorted(glob.glob(os.path.join(src_lbl_dir, "*.txt")))
    for lbl_path in lbl_files:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        net_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "3":
                # Remap class 3 -> 0 for single class dataset
                net_lines.append(" ".join(["0"] + parts[1:]) + "\n")
                
        if net_lines:
            base_name = os.path.splitext(os.path.basename(lbl_path))[0]
            src_img = os.path.join(src_img_dir, base_name + ".jpg")
            if not os.path.exists(src_img):
                raise FileNotFoundError(f"Image not found: {src_img}")
                
            dst_img = os.path.join(dst_img_dir, base_name + ".jpg")
            dst_lbl = os.path.join(dst_lbl_dir, base_name + ".txt")
            
            shutil.copy2(src_img, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.writelines(net_lines)
                
            img_count += 1
            box_count += len(net_lines)
            
    stats[split] = {"images": img_count, "boxes": box_count}

yaml_path_clean = dst_base.replace("\\", "/")
yaml_content = f"""# SODD Net-Only Dataset (YOLO format)
path: {yaml_path_clean}
train: train/images
val: validation/images
test: test/images

names:
  0: net
"""

with open(os.path.join(dst_base, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

total_imgs = sum(s["images"] for s in stats.values())
total_boxes = sum(s["boxes"] for s in stats.values())

readme_content = f"""# SODD Net-Only Dataset

Extracted from the Underwater Sonar/Optical Debris Dataset (SODD).
This dataset isolates the **net** object class (originally class ID `3`), remapped to class ID `0` for standard single-class object detection models (e.g., YOLOv8 / YOLOv11 / RT-DETR).

## Dataset Summary
- **Train**: {stats['train']['images']} images ({stats['train']['boxes']} net annotations)
- **Validation**: {stats['validation']['images']} images ({stats['validation']['boxes']} net annotations)
- **Test**: {stats['test']['images']} images ({stats['test']['boxes']} net annotations)
- **Total**: {total_imgs} images ({total_boxes} net bounding boxes)

## Directory Structure
```
SODD_nets/
├── data.yaml
├── README.md
├── train/
│   ├── images/
│   └── labels/
├── validation/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Classes
- `0`: net
"""

with open(os.path.join(dst_base, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)

print("Done extracting nets dataset!")
for k, v in stats.items():
    print(f"  {k}: {v['images']} images, {v['boxes']} boxes")
print(f"Total: {total_imgs} images, {total_boxes} boxes")
