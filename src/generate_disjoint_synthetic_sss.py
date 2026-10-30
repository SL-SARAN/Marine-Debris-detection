import os
import glob
import cv2
import numpy as np
import concurrent.futures
from pathlib import Path

def create_sonar_lut():
    x_nodes = np.array([0, 35, 110, 185, 255], dtype=np.float32)
    b_nodes = np.array([0,  4,  20,  60, 215], dtype=np.float32)
    g_nodes = np.array([0, 12,  75, 155, 240], dtype=np.float32)
    r_nodes = np.array([0, 28, 150, 230, 255], dtype=np.float32)
    xp = np.arange(256)
    b_lut = np.interp(xp, x_nodes, b_nodes).astype(np.uint8)
    g_lut = np.interp(xp, x_nodes, g_nodes).astype(np.uint8)
    r_lut = np.interp(xp, x_nodes, r_nodes).astype(np.uint8)
    return np.stack([b_lut, g_lut, r_lut], axis=-1)

SONAR_LUT = create_sonar_lut()

def generate_disjoint_seabed_swath(h, w, split, sonar_side="left", rng=None):
    """
    Generates genuinely disjoint seabed acoustic backscatter per split.
    Train: High-frequency sand ripples
    Validation: Coarse gravel roughness with macro-topography
    Test: Fine silt / muddy seafloor with low attenuation & non-linear depressions
    """
    rng = rng or np.random.default_rng(42)
    x = np.linspace(0, 1, w)
    if sonar_side == "left":
        base_tvg = 0.46 + 0.12 * np.sin(np.sqrt(x) * np.pi) - 0.14 * x
    else:
        base_tvg = 0.46 + 0.12 * np.sin(np.sqrt(1.0 - x) * np.pi) - 0.14 * (1.0 - x)
        
    seabed = np.tile(base_tvg, (h, 1))
    
    if split == "train":
        # Sand ripples
        freq1 = rng.uniform(0.025, 0.040)
        freq2 = rng.uniform(0.008, 0.015)
        phase = rng.uniform(0, 2 * np.pi)
        yy, xx = np.indices((h, w))
        angle = rng.uniform(-0.25, 0.25)
        proj = xx * np.cos(angle) + yy * np.sin(angle)
        r1 = np.sin(proj * freq1 + phase)
        r2 = np.sin(proj * freq2 + phase * 0.7)
        ripples = 0.08 * np.sign(r1) * (np.abs(r1) ** 0.85) + 0.03 * r2
        seabed += ripples
        
    elif split == "validation":
        # Coarse gravel with heavy roughness
        roughness = cv2.resize(rng.normal(0, 0.16, (h // 6, w // 6)), (w, h), interpolation=cv2.INTER_CUBIC)
        macro = cv2.resize(rng.normal(0, 0.05, (h // 12, w // 12)), (w, h), interpolation=cv2.INTER_CUBIC)
        seabed += (roughness + macro)
        
    elif split == "test":
        # Fine silt / muddy bottom: lower backscatter, wide depression trenches
        seabed *= 0.80
        depressions = cv2.resize(rng.normal(0, 0.08, (h // 30, w // 30)), (w, h), interpolation=cv2.INTER_CUBIC)
        seabed += depressions

    # Along-track gain variation
    ping_jitter = rng.normal(1.0, 0.02, (h, 1))
    seabed *= ping_jitter
    return np.clip(seabed, 0.12, 0.88)

def extract_net_acoustic_profile(rgb_crop, global_y1, global_x1):
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape
    
    osd_mask = np.full((ch, cw), 255, dtype=np.uint8)
    for r in range(ch):
        gy = global_y1 + r
        if gy < 120:
            for c in range(cw):
                gx = global_x1 + c
                if gx < 850:
                    osd_mask[r, c] = 0
                    
    taper = np.ones((ch, cw), dtype=np.float32)
    pad = min(16, ch // 4, cw // 4)
    if pad > 1:
        for i in range(pad):
            v = np.sin((i / pad) * (np.pi / 2))
            taper[i, :] = np.minimum(taper[i, :], v)
            taper[ch - 1 - i, :] = np.minimum(taper[ch - 1 - i, :], v)
            taper[:, i] = np.minimum(taper[:, i], v)
            taper[:, cw - 1 - i] = np.minimum(taper[:, cw - 1 - i], v)
            
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    filtered = cv2.bilateralFilter(enhanced, 7, 45, 45)
    
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = (grad / (grad.max() + 1e-5) * 255).astype(np.uint8)
    
    salience = cv2.addWeighted(filtered, 0.55, grad, 0.45, 0)
    thresh_val, _ = cv2.threshold(salience, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, net_mask = cv2.threshold(salience, int(thresh_val * 0.72), 255, cv2.THRESH_BINARY)
    net_mask &= osd_mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    net_mask = cv2.morphologyEx(net_mask, cv2.MORPH_OPEN, kernel)
    net_mask = cv2.morphologyEx(net_mask, cv2.MORPH_CLOSE, kernel)
    
    net_mask_float = (net_mask.astype(np.float32) / 255.0) * taper
    net_mask = (net_mask_float > 0.15).astype(np.uint8) * 255
    
    dist = cv2.distanceTransform(net_mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    else:
        dist = np.zeros_like(gray, dtype=np.float32)
        
    relief = (0.65 * dist + 0.35 * (filtered.astype(np.float32) / 255.0)) * taper
    relief[net_mask == 0] = 0.0
    tex_norm = (filtered.astype(np.float32) / 255.0)
    return tex_norm, net_mask, relief

def process_single_image_disjoint(task):
    img_path, lbl_path, dst_img_path, dst_lbl_path, split, seed_val = task
    rng = np.random.default_rng(seed_val)
    
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    
    with open(lbl_path, "r", encoding="utf-8") as f:
        boxes = [list(map(float, line.strip().split()[1:])) for line in f if line.strip()]
    if not boxes:
        return None
        
    sonar_side = "left" if rng.uniform() > 0.5 else "right"
    bg = generate_disjoint_seabed_swath(h, w, split=split, sonar_side=sonar_side, rng=rng)
    synthetic = bg.copy()
    
    global_shadow_mask = np.zeros((h, w), dtype=np.float32)
    global_highlight_mask = np.zeros((h, w), dtype=np.float32)
    global_highlight_val = np.zeros((h, w), dtype=np.float32)
    
    # Genuinely disjoint shadow geometry per split:
    if split == "train":
        shadow_mult_range = (1.2, 1.8)     # Standard mid-depth grazing
        speckle_sigma = 0.14
    elif split == "validation":
        shadow_mult_range = (2.4, 3.6)     # Medium-low grazing (longer shadows)
        speckle_sigma = 0.19
    else:  # test
        shadow_mult_range = (4.0, 5.5)     # Ultra-low grazing (very long shadows, high occlusion)
        speckle_sigma = 0.24
        
    updated_yolo_boxes = []
    
    for box in boxes:
        xc, yc, bw, bh = box
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w, int((xc + bw / 2) * w))
        y2 = min(h, int((yc + bh / 2) * h))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
            
        tex, mask, relief = extract_net_acoustic_profile(crop, y1, x1)
        ch, cw = crop.shape[:2]
        if np.sum(mask > 0) < 25:
            continue
            
        sobel_x = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3)
        facing = np.maximum(0, -sobel_x * 2.2) if sonar_side == "left" else np.maximum(0, sobel_x * 2.2)
        highlight = 0.72 + 0.20 * tex + 0.30 * facing
        highlight = np.clip(highlight, 0.45, 1.0)
        
        shadow_mult = rng.uniform(*shadow_mult_range)
        max_s_len = int(cw * shadow_mult)
        
        min_footprint_x = w
        max_footprint_x = 0
        min_footprint_y = h
        max_footprint_y = 0
        
        for r in range(ch):
            row_y = y1 + r
            if row_y >= h:
                continue
            active_cols = np.where(mask[r, :] > 0)[0]
            if len(active_cols) == 0:
                continue
                
            min_footprint_y = min(min_footprint_y, row_y)
            max_footprint_y = max(max_footprint_y, row_y)
            global_active_x = x1 + active_cols
            min_footprint_x = min(min_footprint_x, np.min(global_active_x))
            max_footprint_x = max(max_footprint_x, np.max(global_active_x))
            
            r_height = np.max(relief[r, active_cols])
            s_len = int(r_height * max_s_len * rng.uniform(0.9, 1.15))
            s_len = max(5, min(s_len, max_s_len))
            
            if sonar_side == "left":
                start_x = x1 + active_cols[-1]
                end_x = min(w, start_x + s_len)
                if end_x > start_x:
                    global_shadow_mask[row_y, start_x:end_x] = 1.0
                    max_footprint_x = max(max_footprint_x, end_x)
            else:
                start_x = x1 + active_cols[0]
                end_x = max(0, start_x - s_len)
                if start_x > end_x:
                    global_shadow_mask[row_y, end_x:start_x] = 1.0
                    min_footprint_x = min(min_footprint_x, end_x)
                    
        alpha = (mask > 0).astype(np.float32)
        global_highlight_mask[y1:y2, x1:x2] = np.maximum(global_highlight_mask[y1:y2, x1:x2], alpha)
        global_highlight_val[y1:y2, x1:x2] = np.where(mask > 0, highlight, global_highlight_val[y1:y2, x1:x2])
        
        if max_footprint_x > min_footprint_x and max_footprint_y > min_footprint_y:
            n_xc = ((min_footprint_x + max_footprint_x) / 2.0) / w
            n_yc = ((min_footprint_y + max_footprint_y) / 2.0) / h
            n_bw = (max_footprint_x - min_footprint_x) / w
            n_bh = (max_footprint_y - min_footprint_y) / h
            updated_yolo_boxes.append(f"0 {min(1.0, max(0.0, n_xc)):.6f} {min(1.0, max(0.0, n_yc)):.6f} {min(1.0, max(0.001, n_bw)):.6f} {min(1.0, max(0.001, n_bh)):.6f}\n")
            
    if not updated_yolo_boxes:
        return None
        
    global_shadow_soft = cv2.GaussianBlur(global_shadow_mask, (7, 7), 1.5)
    synthetic = synthetic * (1.0 - global_shadow_soft * 0.96) + 0.03 * (global_shadow_soft * 0.96)
    
    hl_alpha_soft = cv2.GaussianBlur(global_highlight_mask, (5, 5), 1.0)
    synthetic = synthetic * (1.0 - hl_alpha_soft * 0.92) + global_highlight_val * (hl_alpha_soft * 0.92)
    
    synthetic = cv2.GaussianBlur(synthetic, (3, 3), 0.4)
    # Rayleigh speckle with disjoint sigma per split
    rayleigh = rng.rayleigh(scale=1.0, size=synthetic.shape)
    norm = 1.0 + speckle_sigma * (rayleigh - np.sqrt(np.pi / 2.0))
    synthetic = np.clip(synthetic * norm, 0.0, 1.0)
    
    # SSS Color LUT via direct lookup
    u8 = (np.clip(synthetic, 0, 1) * 255).astype(np.uint8)
    sonar_copper = SONAR_LUT[u8]
    
    cv2.imwrite(dst_img_path, sonar_copper)
    with open(dst_lbl_path, "w", encoding="utf-8") as f:
        f.writelines(updated_yolo_boxes)
    return len(updated_yolo_boxes)

def generate_disjoint_dataset():
    src_base = r"D:\MarineDebris\data\SODD_nets"
    dst_base = r"D:\MarineDebris\data\SODD_synthetic_sss"
    
    splits = ["train", "validation", "test"]
    split_seed_offsets = {"train": 10000, "validation": 20000, "test": 30000}
    
    for split in splits:
        src_img_dir = os.path.join(src_base, split, "images")
        src_lbl_dir = os.path.join(src_base, split, "labels")
        dst_img_dir = os.path.join(dst_base, split, "images")
        dst_lbl_dir = os.path.join(dst_base, split, "labels")
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        lbl_files = sorted(glob.glob(os.path.join(src_lbl_dir, "*.txt")))
        tasks = []
        for idx, lbl_path in enumerate(lbl_files):
            stem = os.path.splitext(os.path.basename(lbl_path))[0]
            img_path = os.path.join(src_img_dir, stem + ".jpg")
            if os.path.exists(img_path):
                dst_img = os.path.join(dst_img_dir, stem + ".jpg")
                dst_lbl = os.path.join(dst_lbl_dir, stem + ".txt")
                seed_val = split_seed_offsets[split] + idx * 7 + 1
                tasks.append((img_path, lbl_path, dst_img, dst_lbl, split, seed_val))
                
        print(f"Synthesizing {split} split ({len(tasks)} images) with disjoint parameter regime...")
        completed = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(process_single_image_disjoint, tasks):
                if res is not None:
                    completed += 1
        print(f"Finished {split}: {completed} disjoint images synthesized.")

if __name__ == "__main__":
    generate_disjoint_dataset()
