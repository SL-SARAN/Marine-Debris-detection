import cv2
import numpy as np
import pywt
from scipy.ndimage import uniform_filter

def enhanced_lee_filter(image, window_size=7, cu=0.52):
    """
    Enhanced Lee Filter for Side-Scan Sonar (SSS) acoustic speckle noise reduction.
    Preserves edges and specular backscatter highlights while smoothing uniform seafloor.
    
    Parameters:
        image: 2D numpy array (grayscale uint8 or float32 [0, 1])
        window_size: Size of local moving kernel (odd integer)
        cu: Speckle noise variation coefficient (typical sonar value ~0.52)
    """
    is_uint8 = image.dtype == np.uint8
    img = image.astype(np.float32) / 255.0 if is_uint8 else image.copy()
    
    # Local mean and local variance
    mean = uniform_filter(img, size=window_size)
    mean_sq = uniform_filter(img ** 2, size=window_size)
    var = np.maximum(0.0, mean_sq - mean ** 2)
    
    # Local variation coefficient
    ci = np.sqrt(var) / (mean + 1e-6)
    
    # Weighting factor W
    # If ci <= cu: homogeneous area -> pure local average
    # If ci > cu: high variance / edge -> preserve original pixel
    weight = np.clip(1.0 - (cu ** 2) / (ci ** 2 + 1e-6), 0.0, 1.0)
    
    filtered = mean + weight * (img - mean)
    filtered = np.clip(filtered, 0.0, 1.0)
    
    if is_uint8:
        return (filtered * 255.0).astype(np.uint8)
    return filtered

def wavelet_denoise_2d(image, wavelet="sym4", level=2, mode="soft"):
    """
    2D Discrete Wavelet Transform (DWT) Denoising.
    Decomposes image into frequency sub-bands, applies BayesShrink thresholding
    to high-frequency speckle coefficients, and reconstructs the clean swath.
    """
    is_uint8 = image.dtype == np.uint8
    img = image.astype(np.float32) / 255.0 if is_uint8 else image.copy()
    
    # Multi-level 2D DWT
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    approx = coeffs[0]
    details = list(coeffs[1:])
    
    # Robust noise estimation from finest sub-band (HH)
    hh_fine = details[-1][2]
    sigma_est = np.median(np.abs(hh_fine)) / 0.6745
    
    new_details = []
    for d_level in details:
        new_level = []
        for band in d_level:
            # BayesShrink adaptive threshold per subband
            band_var = np.maximum(0.0, np.var(band) - sigma_est ** 2)
            if band_var > 1e-6:
                threshold = (sigma_est ** 2) / np.sqrt(band_var)
            else:
                threshold = np.max(np.abs(band))
                
            thresholded_band = pywt.threshold(band, value=threshold, mode=mode)
            new_level.append(thresholded_band)
        new_details.append(tuple(new_level))
        
    # Inverse 2D DWT
    reconstructed = pywt.waverec2([approx] + new_details, wavelet=wavelet)
    reconstructed = np.clip(reconstructed[:img.shape[0], :img.shape[1]], 0.0, 1.0)
    
    if is_uint8:
        return (reconstructed * 255.0).astype(np.uint8)
    return reconstructed

def adaptive_clahe(image, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Contrast-Limited Adaptive Histogram Equalization (CLAHE) tailored for sonar imagery.
    Enhances faint acoustic shadows and fine rope netting against dark seabed backgrounds.
    """
    is_uint8 = image.dtype == np.uint8
    img = image if is_uint8 else (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(img)
    
    if not is_uint8:
        return equalized.astype(np.float32) / 255.0
    return equalized

def apply_tvg_normalization(image, altitude_m=12.0, max_range_m=75.0, alpha_db_per_m=0.15, is_full_swath=True):
    """
    Time-Varying Gain (TVG) Normalization for Side-Scan Sonar.
    Compensates for transmission loss due to geometric spherical spreading (20*log10(R))
    and seawater acoustic absorption (2*alpha*R).
    
    Parameters:
        image: 2D numpy array (uint8 or float)
        altitude_m: Towfish altitude above seabed
        max_range_m: Maximum one-sided swath range
        alpha_db_per_m: Acoustic absorption coefficient (~0.15 dB/m for 455 kHz)
        is_full_swath: True if image is Port+Starboard centered at nadir, False if image chip
    """
    is_uint8 = image.dtype == np.uint8
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    if is_full_swath and w > 20:
        # Nadir is in the center column
        center_x = w / 2.0
        cols = np.arange(w)
        # Distance from nadir in pixels
        dist_px = np.abs(cols - center_x)
        # Slant range in meters: minimum is towfish altitude
        slant_r = altitude_m + (dist_px / (center_x + 1e-6)) * (max_range_m - altitude_m)
        
        # Transmission loss: 20*log10(R) + 2*alpha*R
        tl_db = 20.0 * np.log10(np.maximum(1.0, slant_r)) + 2.0 * alpha_db_per_m * slant_r
        # Linear gain curve (normalized so median gain is 1.0)
        gain = 10.0 ** (tl_db / 20.0)
        gain = gain / (np.median(gain) + 1e-6)
        
        # Apply across all rows
        tvg_corrected = img * gain[np.newaxis, :]
    else:
        # For cropped image chips: empirical cross-track polynomial trend normalization
        col_means = np.mean(img, axis=0)
        cols = np.arange(w)
        # Robust polynomial fit to the intensity trend
        poly_deg = min(2, max(1, w // 20))
        poly_coeffs = np.polyfit(cols, col_means, poly_deg)
        trend = np.polyval(poly_coeffs, cols)
        trend = np.maximum(10.0, trend)
        target_mean = np.mean(col_means)
        gain = target_mean / trend
        tvg_corrected = img * gain[np.newaxis, :]

    # Normalize back to valid dynamic range [0, 255]
    p_low, p_high = np.percentile(tvg_corrected, (0.5, 99.5))
    norm = np.clip((tvg_corrected - p_low) / (p_high - p_low + 1e-6) * 255.0, 0.0, 255.0)

    if is_uint8:
        return norm.astype(np.uint8)
    return norm / 255.0

def apply_srgr_rectification(image, altitude_m=12.0, max_range_m=75.0, is_full_swath=True):
    """
    Slant-Range to Ground-Range (SRGR) Rectification.
    Converts non-linear two-way acoustic travel time (slant range) to uniform horizontal 
    ground range (meters), removing nadir blind-zone distortion:
    R_ground = sqrt(max(0, R_slant^2 - H_alt^2))
    
    Parameters:
        image: 2D numpy array (uint8 or float)
        altitude_m: Towfish altitude
        max_range_m: Maximum one-sided slant range
        is_full_swath: True if Port+Starboard centered at nadir
    """
    if not is_full_swath:
        # Image chips are already localized crops; return as-is
        return image

    is_uint8 = image.dtype == np.uint8
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    half_w = w // 2

    if half_w < 10 or max_range_m <= altitude_m:
        return image

    # Process Port (left half) and Starboard (right half)
    # Slant range array for one side
    r_slant_side = np.linspace(altitude_m, max_range_m, half_w)
    # Ground range array
    max_ground = np.sqrt(max_range_m ** 2 - altitude_m ** 2)
    r_ground_grid = np.linspace(0.0, max_ground, half_w)

    # Invert to find slant range corresponding to each uniform ground range step:
    # R_slant = sqrt(R_ground^2 + H_alt^2)
    r_slant_resample = np.sqrt(r_ground_grid ** 2 + altitude_m ** 2)
    # Map to pixel indices on one side [0, half_w - 1]
    resample_idx = (r_slant_resample - altitude_m) / (max_range_m - altitude_m) * (half_w - 1)
    resample_idx = np.clip(resample_idx, 0, half_w - 1).astype(np.float32)

    # Re-sample both port and starboard
    rect_port = np.zeros((h, half_w), dtype=np.float32)
    rect_stbd = np.zeros((h, half_w), dtype=np.float32)

    port_raw = img[:, :half_w]  # Port is left [far_port ... nadir]
    stbd_raw = img[:, half_w:]  # Starboard is right [nadir ... far_stbd]

    for row in range(h):
        # Starboard: index 0 is nadir, index half_w-1 is far
        rect_stbd[row, :] = np.interp(resample_idx, np.arange(half_w), stbd_raw[row, :])
        # Port: flipped so index 0 is nadir, then flip back
        port_from_nadir = np.fliplr(port_raw[row:row+1, :])[0]
        port_resampled = np.interp(resample_idx, np.arange(half_w), port_from_nadir)
        rect_port[row, :] = port_resampled[::-1]

    rectified = np.hstack([rect_port, rect_stbd])
    if is_uint8:
        return np.clip(rectified, 0, 255).astype(np.uint8)
    return np.clip(rectified, 0.0, 1.0)

def compute_speckle_metrics(raw_image, filtered_image):
    """
    Computes standard acoustic & radar speckle metrics:
    - ENL (Equivalent Number of Looks): higher ENL indicates smoother homogeneous areas
    - Speckle Suppression Index (SSI)
    - PSNR (Peak Signal-to-Noise Ratio)
    """
    raw_f = raw_image.astype(np.float32)
    flt_f = filtered_image.astype(np.float32)
    
    # Homogeneous region sample (center quadrant)
    h, w = raw_image.shape[:2]
    roi = flt_f[h//4:3*h//4, w//4:3*w//4]
    
    mean_roi = np.mean(roi)
    std_roi = np.std(roi)
    enl = (mean_roi / (std_roi + 1e-6)) ** 2
    
    # PSNR
    mse = np.mean((raw_f - flt_f) ** 2)
    psnr = 10.0 * np.log10((255.0 ** 2) / (mse + 1e-6)) if mse > 0 else 100.0
    
    return {
        "enl": float(round(enl, 2)),
        "psnr_db": float(round(psnr, 2)),
        "mse": float(round(mse, 2))
    }

def preprocess_sonar_swath(image, altitude_m=12.0, max_range_m=75.0, is_full_swath=False,
                            use_tvg=True, use_wavelet=True, use_lee=True, use_clahe=True, use_srgr=False,
                            return_intensity_preserving=False):
    """
    Comprehensive Sonar Conditioning Pipeline:
    1. TVG (Time-Varying Gain) Normalization -> flattens cross-track range decay
    2. Wavelet DWT de-speckling (sym4 BayesShrink)
    3. Enhanced Lee edge-preserving smoothing
    4. Contrast-limited histogram adjustment (CLAHE) - for visualization
    5. SRGR (Slant-to-Ground Range) Rectification -> optional metric resolution

    NOTE ON INTENSITY PRESERVATION:
    Acoustic shadow contrast ratio (HSCR) must be evaluated on an intensity-preserving
    image (TVG + Wavelet + Lee) rather than CLAHE-modified pixels, because CLAHE alters
    local intensity relationships and skews physical highlight/shadow contrast.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    processed = gray

    # Stage 1: TVG Normalization
    if use_tvg:
        processed = apply_tvg_normalization(processed, altitude_m=altitude_m, max_range_m=max_range_m, is_full_swath=is_full_swath)

    # Stage 2: Wavelet DWT Speckle Denoising
    if use_wavelet:
        processed = wavelet_denoise_2d(processed, wavelet="sym4", level=2)

    # Stage 3: Enhanced Lee Filter
    if use_lee:
        processed = enhanced_lee_filter(processed, window_size=7, cu=0.52)

    # Intensity-preserving version (unmodified by CLAHE)
    intensity_preserving = processed.copy()

    # Stage 4: CLAHE Contrast Enhancement (for visual display / human operator inspection)
    enhanced = processed
    if use_clahe:
        enhanced = adaptive_clahe(processed, clip_limit=2.0)

    # Stage 5: SRGR Metric Rectification (optional)
    if use_srgr and is_full_swath:
        enhanced = apply_srgr_rectification(enhanced, altitude_m=altitude_m, max_range_m=max_range_m, is_full_swath=is_full_swath)
        intensity_preserving = apply_srgr_rectification(intensity_preserving, altitude_m=altitude_m, max_range_m=max_range_m, is_full_swath=is_full_swath)
        
    metrics = compute_speckle_metrics(gray, enhanced)
    if return_intensity_preserving:
        return enhanced, metrics, intensity_preserving
    return enhanced, metrics

