import sys
import unittest
import cv2
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.filtering.speckle_filter import (
    enhanced_lee_filter,
    wavelet_denoise_2d,
    adaptive_clahe,
    preprocess_sonar_swath
)

def extract_homogeneous_patch(img_gray, patch_size=32):
    """
    Automatically searches for the most homogeneous (lowest variance / mean ratio)
    seabed patch in the image to measure true multiplicative speckle ENL.
    """
    h, w = img_gray.shape
    best_var_ratio = float("inf")
    best_patch = None

    step = max(8, patch_size // 2)
    for y in range(0, h - patch_size, step):
        for x in range(0, w - patch_size, step):
            patch = img_gray[y:y+patch_size, x:x+patch_size].astype(np.float32)
            m = np.mean(patch)
            v = np.var(patch)
            if m > 15.0 and v > 1.0:  # Avoid completely black/clipped regions
                ratio = v / (m ** 2)
                if ratio < best_var_ratio:
                    best_var_ratio = ratio
                    best_patch = (y, x)

    if best_patch is None:
        y, x = h // 4, w // 4
        return (y, x, patch_size)
    return (best_patch[0], best_patch[1], patch_size)

def compute_patch_enl(img_gray, patch_coords):
    """
    Computes Equivalent Number of Looks (ENL):
    ENL = (mean / std)^2
    """
    y, x, s = patch_coords
    patch = img_gray[y:y+s, x:x+s].astype(np.float32)
    m = np.mean(patch)
    s_val = np.std(patch)
    if s_val < 1e-4:
        return 50.0
    return float((m / s_val) ** 2)

def compute_ssi(raw, filtered):
    """
    Speckle Suppression Index (SSI):
    SSI = (std_f / mean_f) / (std_r / mean_r)
    SSI < 1 indicates speckle reduction. Lower is better.
    """
    rf = raw.astype(np.float32)
    ff = filtered.astype(np.float32)
    mr, sr = np.mean(rf), np.std(rf)
    mf, sf = np.mean(ff), np.std(ff)
    if mr < 1e-3 or sr < 1e-3 or mf < 1e-3:
        return 1.0
    return float((sf / mf) / (sr / mr))

def compute_psnr(raw, filtered):
    """Peak Signal-to-Noise Ratio (PSNR) in dB."""
    rf = raw.astype(np.float32)
    ff = filtered.astype(np.float32)
    mse = np.mean((rf - ff) ** 2)
    if mse < 1e-6:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))

class TestSpeckleFilterBenchmark(unittest.TestCase):
    """
    Empirical quantitative evaluation of acoustic speckle reduction
    on 10 REAL side-scan sonar survey images (KLSG and Marine_PULSE).
    """
    def setUp(self):
        test_dir = PROJECT_ROOT / "data" / "common_sss_dataset" / "test" / "images"
        self.real_images = []
        
        # Collect real KLSG sonar debris images
        klsg_imgs = sorted(test_dir.glob("klsg_*.jpg"))[:5]
        # Collect real Marine_PULSE subsea infrastructure images
        pulse_imgs = sorted(test_dir.glob("pulse_*.jpg"))[:5]

        self.sample_paths = klsg_imgs + pulse_imgs

    def test_despeckle_real_samples(self):
        self.assertGreaterEqual(len(self.sample_paths), 10, "Must have at least 10 real test images")

        print("\n" + "=" * 90)
        print("  SPECKLE FILTER BENCHMARK: 10 REAL SONAR IMAGES (NO SYNTHETIC)")
        print("=" * 90)
        print(f"{'Image Source / Filename':<32} | {'Raw ENL':<8} | {'Filt ENL':<8} | {'ENL Gain':<8} | {'PSNR (dB)':<9} | {'SSI':<6} | {'Status'}")
        print("-" * 90)

        raw_enls = []
        filt_enls = []
        psnrs = []
        ssis = []

        for p in self.sample_paths:
            img_bgr = cv2.imread(str(p))
            self.assertIsNotNone(img_bgr, f"Could not load {p}")
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            h, w = img_gray.shape

            # Representative homogeneous seafloor ROI (central quadrant)
            y1, y2 = int(h * 0.25), int(h * 0.75)
            x1, x2 = int(w * 0.25), int(w * 0.75)
            roi_raw = img_gray[y1:y2, x1:x2].astype(np.float32)
            m_raw, s_raw = np.mean(roi_raw), np.std(roi_raw)
            raw_enl = float((m_raw / (s_raw + 1e-6)) ** 2)

            # Stage 1: Wavelet DWT de-speckling (sym4 BayesShrink)
            w_img = wavelet_denoise_2d(img_gray, wavelet="sym4", level=2)
            # Stage 2: Enhanced Lee adaptive spatial filter
            despeckled = enhanced_lee_filter(w_img, window_size=7, cu=0.52)

            roi_flt = despeckled[y1:y2, x1:x2].astype(np.float32)
            m_flt, s_flt = np.mean(roi_flt), np.std(roi_flt)
            filt_enl = float((m_flt / (s_flt + 1e-6)) ** 2)

            psnr = compute_psnr(img_gray, despeckled)
            ssi = compute_ssi(img_gray, despeckled)
            enl_gain = filt_enl / max(1e-3, raw_enl)

            raw_enls.append(raw_enl)
            filt_enls.append(filt_enl)
            psnrs.append(psnr)
            ssis.append(ssi)

            short_name = p.name[:30] + ".." if len(p.name) > 30 else p.name
            status = "PASS" if filt_enl > raw_enl and psnr >= 14.0 else "FAIL"
            print(f"{short_name:<32} | {raw_enl:<8.2f} | {filt_enl:<8.2f} | {enl_gain:<8.2f}x | {psnr:<9.2f} | {ssi:<6.3f} | {status}")

            # Unit assertion: filtered ENL must strictly exceed raw ENL (speckle variance reduced)
            self.assertGreater(filt_enl, raw_enl, f"Filtered ENL must exceed raw ENL on {p.name}")
            # Unit assertion: PSNR must be within acceptable structural fidelity range (>= 14 dB for raw sonar)
            self.assertGreaterEqual(psnr, 14.0, f"PSNR too low on {p.name}")

        mean_raw_enl = np.mean(raw_enls)
        mean_filt_enl = np.mean(filt_enls)
        mean_psnr = np.mean(psnrs)
        mean_ssi = np.mean(ssis)
        mean_gain = mean_filt_enl / mean_raw_enl

        print("-" * 90)
        print(f"{'MEAN OVER ALL 10 REAL IMAGES':<32} | {mean_raw_enl:<8.2f} | {mean_filt_enl:<8.2f} | {mean_gain:<8.2f}x | {mean_psnr:<9.2f} | {mean_ssi:<6.3f} | {'PASS'}")
        print("=" * 90)

if __name__ == "__main__":
    unittest.main()
