import os
import sys
import io
import base64
import time
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.detection.model import SSSObjectDetector
from pipeline.modules.filtering.speckle_filter import preprocess_sonar_swath, compute_speckle_metrics
from pipeline.modules.filtering.shadow_verifier import AcousticShadowVerifier
from pipeline.modules.filtering.calibration import ConfidenceCalibrator
from pipeline.modules.geotagging.xtf_parser import XTFParser
from pipeline.modules.geotagging.georeference import AcousticGeoreferencer
from pipeline.modules.geotagging.report_export import AnomalyReportEngine

# Cloud sync — strictly optional, never blocks report generation
try:
    from pipeline.modules.cloud_sync import CloudSyncEngine
    HAS_CLOUD_SYNC = True
except ImportError:
    HAS_CLOUD_SYNC = False
    CloudSyncEngine = None

app = FastAPI(
    title="Side-Scan Sonar (SSS) Marine Anomaly & Ghost Net Detection API",
    description="End-to-end 4-Module CV Pipeline for underwater debris detection, confidence calibration, and GIS reporting.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline singletons
print("[Backend] Initializing SSS Detection Pipeline...")
detector = SSSObjectDetector()
verifier = AcousticShadowVerifier()
# Load dynamically fitted calibration temperature if available
calib_config_path = PROJECT_ROOT / "pipeline" / "models" / "calibration_config.json"
fitted_temp = 0.934
if calib_config_path.exists():
    try:
        import json
        with open(calib_config_path, "r") as f:
            cfg = json.load(f)
            fitted_temp = float(cfg.get("temperature", 0.934))
    except Exception:
        pass

calibrator = ConfidenceCalibrator(temperature=fitted_temp, physics_weight=0.35)
georeferencer = AcousticGeoreferencer(max_range_m=75.0, pixel_resolution_m=0.075)
report_engine = AnomalyReportEngine(survey_id="SIH2026-SURVEY-SSS-01")

# Cloud sync engine — optional, non-blocking
sync_engine = None
if HAS_CLOUD_SYNC:
    try:
        sync_engine = CloudSyncEngine()
        print("[Backend] Cloud Sync Engine initialised (optional layer).")
    except Exception as e:
        print(f"[Backend] Cloud Sync Engine not available ({e}) — local-only mode.")

# In-memory store for current survey records
current_survey_records = []

# Report output directory
REPORT_OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "test_outputs"

def numpy_to_base64_jpg(img_np):
    """Converts a numpy image (BGR or grayscale) to base64 JPEG string."""
    success, encoded = cv2.imencode(".jpg", img_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("utf-8")

# ------------------------------------------------------------------
# Lifecycle events — start/stop background sync
# ------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    if sync_engine:
        sync_engine.start_background_sync()

@app.on_event("shutdown")
def shutdown_event():
    if sync_engine:
        sync_engine.stop_background_sync()


@app.get("/api/health")
def get_health():
    import torch
    return {
        "status": "online",
        "pipeline": "active",
        "model_architecture": "YOLOv8n",
        "weights_path": detector.model_path,
        "device": str(detector.device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "modules": {
            "module_1_detector": "ready",
            "module_2_filter_calibrator": "ready",
            "module_3_geotagger": "ready",
            "module_4_dashboard": "ready"
        }
    }

@app.get("/api/samples")
def get_sample_files():
    """Lists pre-loaded benchmark samples available for quick testing."""
    test_img_dir = DATA_DIR / "common_sss_dataset" / "test" / "images"
    if not test_img_dir.exists():
        test_img_dir = DATA_DIR / "merged_sss_debris_dataset" / "test" / "images"
        
    samples = []
    
    if test_img_dir.exists():
        # 1. Synthetic Ghost Net samples
        for f in sorted(test_img_dir.glob("sodd_*.jpg")):
            samples.append({
                "id": f.name,
                "name": f"Ghost Net Swath ({f.stem})",
                "category": "ghost_net",
                "type": "image"
            })
            if len(samples) >= 3:
                break
                
        # 2. Real KLSG Sonar Debris samples
        for f in sorted(test_img_dir.glob("klsg_*.jpg")):
            category = "shipwreck" if "ship" in f.name else "aircraft" if "air" in f.name else "seafloor_debris"
            samples.append({
                "id": f.name,
                "name": f"Sonar Debris ({f.stem[:18]}...)",
                "category": category,
                "type": "image"
            })
            if len(samples) >= 6:
                break

        # 3. Marine_PULSE Subsea Infrastructure samples
        for f in sorted(test_img_dir.glob("pulse_*.jpg")):
            category = "pipeline_cable" if "pipe" in f.name or "cable" in f.name else "engineering_platform"
            samples.append({
                "id": f.name,
                "name": f"Subsea Asset ({f.stem[:18]}...)",
                "category": category,
                "type": "image"
            })
            if len(samples) >= 9:
                break

    # Add raw XTF sample if exists
    xtf_files = [
        DATA_DIR / "raw_xtf" / "DATA0000117.H-PU.xtf"
    ]
    for xf in xtf_files:
        if xf.exists():
            samples.append({
                "id": xf.name,
                "name": f"Raw XTF Binary Sonar Log ({xf.stat().st_size // (1024*1024)} MB)",
                "category": "xtf_raw",
                "type": "xtf"
            })
            break

    return {"samples": samples}

@app.post("/api/process_sonar")
async def process_sonar(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Query(None),
    use_wavelet: bool = Query(True),
    use_lee: bool = Query(True),
    conf_threshold: float = Query(0.20)
):
    """
    End-to-End 4-Module Sonar Processing Pipeline:
    1. Ingestion
    2. Wavelet & Lee Speckle Filtering + CLAHE
    3. YOLOv8n Object Detection
    4. Physics-Consistent Acoustic Shadow Verification (HSCR)
    5. Temperature Scaling & Confidence Calibration
    6. Slant-to-Ground Georeferencing (WGS84 Lat/Lon) & Height Estimation
    7. Structured Report Generation
    """
    global current_survey_records
    t_start = time.perf_counter()

    img_bgr = None
    telemetry_pings = []
    source_filename = "upload"

    if sample_id:
        source_filename = sample_id
        if sample_id.endswith(".xtf"):
            xtf_path = DATA_DIR / "raw_xtf" / sample_id
            parser = XTFParser(xtf_path)
            telemetry_pings = parser.parse_pings(max_pings=200)
            img_bgr = parser.extract_waterfall_image(max_pings=200)
            if img_bgr is None:
                sample_img = list((DATA_DIR / "common_sss_dataset" / "test" / "images").glob("sodd_*.jpg"))[0]
                img_bgr = cv2.imread(str(sample_img))
        else:
            img_path = DATA_DIR / "common_sss_dataset" / "test" / "images" / sample_id
            if not img_path.exists():
                img_path = DATA_DIR / "merged_sss_debris_dataset" / "test" / "images" / sample_id
            if img_path.exists():
                img_bgr = cv2.imread(str(img_path))
            else:
                raise HTTPException(status_code=404, detail="Sample not found.")
    elif file:
        source_filename = file.filename
        contents = await file.read()
        if file.filename.lower().endswith(".xtf"):
            temp_xtf = PROJECT_ROOT / "pipeline" / "test_outputs" / "temp_uploaded.xtf"
            temp_xtf.write_bytes(contents)
            parser = XTFParser(temp_xtf)
            telemetry_pings = parser.parse_pings(max_pings=200)
            img_bgr = parser.extract_waterfall_image(max_pings=200)
            if img_bgr is None:
                raise HTTPException(status_code=400, detail="Could not reconstruct acoustic swath from XTF.")
        else:
            nparr = np.frombuffer(contents, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded file or sample_id.")

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode sonar image.")

    h_orig, w_orig = img_bgr.shape[:2]

    # Telemetry generation if not already loaded from XTF
    if not telemetry_pings:
        telemetry_pings = XTFParser.generate_synthetic_telemetry(num_pings=max(30, h_orig // 10))

    # STAGE 1: Raw Sonar Base64
    raw_b64 = numpy_to_base64_jpg(img_bgr)

    # STAGE 2: Speckle Denoising & Physics Preprocessing
    # Produces:
    # 1. denoised_for_display (CLAHE-enhanced for visual human inspection)
    # 2. intensity_preserving_gray (radiometric denoising without CLAHE, for physics contrast)
    t_denoise_start = time.perf_counter()
    denoised_gray, speckle_metrics, intensity_preserving_gray = preprocess_sonar_swath(
        img_bgr,
        use_wavelet=use_wavelet,
        use_lee=use_lee,
        use_clahe=True,
        return_intensity_preserving=True
    )
    denoise_time_ms = (time.perf_counter() - t_denoise_start) * 1000.0
    denoised_b64 = numpy_to_base64_jpg(denoised_gray)

    # STAGE 3: YOLOv8n Object Detection
    # Passes raw image to YOLO to match its training distribution, preserving spatial alignment
    t_detect_start = time.perf_counter()
    raw_results = detector.predict(img_bgr, conf=conf_threshold)
    detect_time_ms = (time.perf_counter() - t_detect_start) * 1000.0

    # STAGE 4: Physics-Consistent Acoustic Shadow Verification
    # Uses intensity_preserving_gray (NO CLAHE) to measure genuine physical highlight-to-shadow contrast (HSCR)
    t_verify_start = time.perf_counter()
    verified_detections = verifier.filter_detections(intensity_preserving_gray, raw_results["detections"])
    verify_time_ms = (time.perf_counter() - t_verify_start) * 1000.0

    # STAGE 5: Uncalibrated Composite Ranking Score (Detector + Physics Fusion)
    # Explicitly labeled as ranking aid, NOT empirical probability
    t_calib_start = time.perf_counter()
    calibrated_detections = calibrator.calibrate_all_detections(verified_detections)
    calib_time_ms = (time.perf_counter() - t_calib_start) * 1000.0

    # STAGE 6: Acoustic Georeferencing with 3D Attitude Correction (Pitch & Roll)
    t_geo_start = time.perf_counter()
    final_detections = []
    for det in calibrated_detections:
        # Match detection to corresponding along-track telemetry ping
        cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
        ping_idx = int((cy / h_orig) * (len(telemetry_pings) - 1))
        ping_idx = max(0, min(len(telemetry_pings) - 1, ping_idx))
        current_ping = telemetry_pings[ping_idx]

        geo_data = georeferencer.georeference_detection(det, (h_orig, w_orig), current_ping)
        det["georeference"] = geo_data
        det["timestamp_utc"] = current_ping["timestamp"]
        final_detections.append(det)
    geo_time_ms = (time.perf_counter() - t_geo_start) * 1000.0

    # STAGE 7: Draw Overlays & Build Report Records
    annotated_img = detector.draw_detections(img_bgr, final_detections, show_confidence=True)
    overlay_b64 = numpy_to_base64_jpg(annotated_img)

    # Build standardized report records with metadata hierarchy
    records = report_engine.build_report_records(final_detections, telemetry_pings)
    current_survey_records = records

    total_pipeline_ms = (time.perf_counter() - t_start) * 1000.0
    fps = 1000.0 / total_pipeline_ms if total_pipeline_ms > 0 else 0.0

    # Build GPS Survey Track for Map Overlay (filtering null coords for unnavigated pings)
    track_coords = [
        [p["latitude"], p["longitude"]]
        for p in telemetry_pings
        if p.get("latitude") is not None and p.get("longitude") is not None
    ]

    first_ping = telemetry_pings[0] if telemetry_pings else {}

    return {
        "source_file": source_filename,
        "image_shape": [w_orig, h_orig],
        "stages": {
            "raw_sonar_b64": raw_b64,
            "denoised_sonar_b64": denoised_b64,
            "detections_overlay_b64": overlay_b64
        },
        "speckle_metrics": speckle_metrics,
        "detections": records,
        "summary": {
            "total_detected": len(records),
            "high_priority_reviews": sum(1 for r in records if r.get("review_status") == "high_priority_review"),
            "critical_entanglement_nets": sum(1 for r in records if r["classification"] == "ghost_net" and r.get("composite_score_pct", 0) >= 60.0),
            "average_composite_score": float(np.mean([r.get("composite_score_pct", 0) for r in records])) if records else 0.0,
            "calibration_status": "pending_ground_truth",
            "score_disclaimer": "Score is an uncalibrated ranking aid; probability calibration is pending ground-truth validation."
        },
        "telemetry_track": {
            "start_coords": track_coords[0] if track_coords else None,
            "track_coords": track_coords,
            "pings_count": len(telemetry_pings),
            "gps_source": first_ping.get("gps_source", "unnavigated"),
            "georeference_quality": first_ping.get("georeference_quality", "unnavigated_relative"),
            "attitude_quality": first_ping.get("attitude_quality", "valid_imu"),
        },
        "latency_benchmark": {
            "denoising_ms": round(denoise_time_ms, 2),
            "inference_ms": round(detect_time_ms, 2),
            "physics_verification_ms": round(verify_time_ms, 2),
            "calibration_ms": round(calib_time_ms, 2),
            "georeferencing_ms": round(geo_time_ms, 2),
            "total_pipeline_ms": round(total_pipeline_ms, 2),
            "realtime_fps": round(fps, 1)
        }
    }

@app.get("/api/export/csv")
def export_csv():
    """Download the current survey anomaly report as CSV."""
    if not current_survey_records:
        raise HTTPException(status_code=400, detail="No survey records available. Please run sonar processing first.")
    
    out_path = REPORT_OUTPUT_DIR / "current_survey_report.csv"
    report_engine.export_csv(current_survey_records, out_path)
    _auto_queue_upload(out_path, "csv")
    return FileResponse(
        str(out_path),
        media_type="text/csv",
        filename="SSS_Marine_Debris_Survey_Report.csv"
    )

@app.get("/api/export/json")
def export_json():
    """Download the current survey anomaly report as JSON."""
    if not current_survey_records:
        raise HTTPException(status_code=400, detail="No survey records available. Please run sonar processing first.")
        
    out_path = REPORT_OUTPUT_DIR / "current_survey_report.json"
    report_engine.export_json(current_survey_records, out_path)
    _auto_queue_upload(out_path, "json")
    return FileResponse(
        str(out_path),
        media_type="application/json",
        filename="SSS_Marine_Debris_Survey_Report.json"
    )

@app.get("/api/export/geojson")
def export_geojson():
    """Download the current survey anomaly report as standard GeoJSON."""
    if not current_survey_records:
        raise HTTPException(status_code=400, detail="No survey records available. Please run sonar processing first.")
        
    out_path = REPORT_OUTPUT_DIR / "current_survey_report.geojson"
    report_engine.export_geojson(current_survey_records, out_path)
    _auto_queue_upload(out_path, "geojson")
    return FileResponse(
        str(out_path),
        media_type="application/geo+json",
        filename="SSS_Marine_Debris_Survey_Report.geojson"
    )

@app.get("/api/export/pdf")
def export_pdf():
    """Download the current survey anomaly report as a professional PDF audit document."""
    if not current_survey_records:
        raise HTTPException(status_code=400, detail="No survey records available. Please run sonar processing first.")
        
    out_path = REPORT_OUTPUT_DIR / "current_survey_report.pdf"
    report_engine.export_pdf(current_survey_records, out_path)
    _auto_queue_upload(out_path, "pdf")
    return FileResponse(
        str(out_path),
        media_type="application/pdf",
        filename="SSS_Marine_Debris_Survey_Report.pdf"
    )


# ------------------------------------------------------------------
# Auto-queue helper (never blocks, never raises)
# ------------------------------------------------------------------
def _auto_queue_upload(file_path, report_format):
    """Queues a report for cloud upload. Completely non-blocking and optional."""
    if sync_engine:
        try:
            sync_engine.queue_report(report_engine.survey_id, str(file_path), report_format)
        except Exception:
            pass  # Cloud sync is optional — never block local export


# ==================================================================
# Cloud Sync API — strictly additive, no impact on detection pipeline
# ==================================================================

@app.get("/api/cloud/status")
def cloud_status():
    """Returns cloud sync engine status, manifest entries, and connectivity state."""
    if not sync_engine:
        return {
            "enabled": False,
            "message": "Cloud sync not available (module not loaded)",
            "entries": [],
            "engine_status": {},
        }
    return {
        "enabled": True,
        "engine_status": sync_engine.status,
        "entries": sync_engine.get_all_entries(limit=200),
    }


@app.post("/api/cloud/upload")
def cloud_upload_single(report_format: str = Query(..., description="csv|json|geojson|pdf")):
    """
    Immediately uploads a single report file to Azure Blob Storage.
    The file must have been previously exported via /api/export/{format}.
    """
    if not sync_engine:
        raise HTTPException(status_code=503, detail="Cloud sync not available.")

    format_map = {
        "csv": "current_survey_report.csv",
        "json": "current_survey_report.json",
        "geojson": "current_survey_report.geojson",
        "pdf": "current_survey_report.pdf",
    }
    filename = format_map.get(report_format)
    if not filename:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {report_format}")

    file_path = REPORT_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report file not found. Please export {report_format.upper()} first.",
        )

    result = sync_engine.queue_report(report_engine.survey_id, str(file_path), report_format)
    entry = result.get("entry")
    if entry and entry.get("upload_status") == "pending":
        upload_result = sync_engine.upload_now(entry["id"])
        return {"result": upload_result, "entry": sync_engine.manifest.get_entry(entry["id"])}
    return {"result": result, "entry": entry}


@app.post("/api/cloud/upload-all")
def cloud_upload_all():
    """
    Creates a ZIP of all pending/failed report files and uploads it to Azure.
    Individual entries are marked as uploaded on success.
    """
    if not sync_engine:
        raise HTTPException(status_code=503, detail="Cloud sync not available.")
    result = sync_engine.upload_all_as_zip(report_engine.survey_id)
    return result


@app.post("/api/cloud/retry/{entry_id}")
def cloud_retry(entry_id: int):
    """Manual retry for a specific failed manifest entry."""
    if not sync_engine:
        raise HTTPException(status_code=503, detail="Cloud sync not available.")
    result = sync_engine.retry_entry(entry_id)
    return {"result": result, "entry": sync_engine.manifest.get_entry(entry_id)}


@app.post("/api/cloud/retry-all")
def cloud_retry_all():
    """Retries all actionable (pending/failed) entries."""
    if not sync_engine:
        raise HTTPException(status_code=503, detail="Cloud sync not available.")
    results = sync_engine.retry_all_actionable()
    return {"results": results, "retried_count": len(results)}


@app.get("/api/cloud/connectivity")
def cloud_connectivity():
    """Checks and returns current Azure reachability status."""
    if not sync_engine:
        return {"is_online": False, "message": "Cloud sync not available."}
    is_online = sync_engine.check_connectivity()
    return {
        "is_online": is_online,
        "last_check": sync_engine._last_connectivity_check,
    }


@app.post("/api/cloud/prune")
def cloud_prune():
    """Prunes uploaded manifest entries older than retention period (files untouched)."""
    if not sync_engine:
        raise HTTPException(status_code=503, detail="Cloud sync not available.")
    return sync_engine.prune_completed()

