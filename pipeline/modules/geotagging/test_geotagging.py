import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.geotagging.xtf_parser import XTFParser
from pipeline.modules.geotagging.georeference import AcousticGeoreferencer
from pipeline.modules.geotagging.report_export import AnomalyReportEngine

def test_module_3():
    print("=" * 65)
    print("  TESTING MODULE 3: ANOMALOUS REPORTING & GEOTAGGING ENGINE")
    print("=" * 65)

    # 1. Test XTF Parser on Real Raw Sonar File
    xtf_path = PROJECT_ROOT / "data" / "raw_xtf" / "DATA0000117.H-PU.xtf"
    print(f"\n[Stage 1] Ingesting real raw XTF sonar log: {xtf_path.name}")
    if xtf_path.exists():
        parser = XTFParser(xtf_path)
        pings = parser.parse_pings(max_pings=15)
        print(f"  Successfully parsed {len(pings)} pings from XTF file!")
        print(f"  Header metadata:")
        print(f"    - Sonar Name   : {parser.file_header.get('sonar_name')}")
        print(f"    - Channels     : {parser.file_header.get('num_channels')}")
        print(f"    - Range Source : {parser.file_header.get('range_source')}")
        print(f"  Sample Ping #1 Telemetry:")
        p0 = pings[0]
        print(f"    - Timestamp    : {p0['timestamp']}")
        print(f"    - Coordinates  : ({p0['latitude']} N, {p0['longitude']} E) [{p0['gps_source']}]")
        print(f"    - Altitude     : {p0['altitude_m']:.2f} m [{p0['altitude_source']}]")
        print(f"    - Heading      : {p0['heading_deg']:.1f} deg | Pitch: {p0['pitch_deg']:.2f}° | Roll: {p0['roll_deg']:.2f}°")
        print(f"    - Quality      : {p0['georeference_quality']} | IMU: {p0['attitude_quality']}")
    else:
        print("  Real XTF not found, testing synthetic telemetry generation...")
        pings = XTFParser.generate_synthetic_telemetry(num_pings=10)

    # 2. Test Georeferencing & Slant-Range Correction with Attitude (Pitch & Roll)
    print("\n[Stage 2] Testing Slant-Range Correction & Attitude-Aware Georeferencing...")
    georef = AcousticGeoreferencer(max_range_m=75.0, pixel_resolution_m=0.075)

    # Simulated candidate detection with physics verification
    sample_detection = {
        "class_name": "ghost_net",
        "bbox": [650, 40, 920, 310],  # across-track width = 270px, height = 270px
        "raw_confidence": 0.945,
        "detector_score": 0.945,
        "composite_score_pct": 88.5,
        "score_type": "uncalibrated_composite",
        "calibration_status": "pending_ground_truth",
        "review_status": "high_priority_review",
        "physics": {
            "is_verified": True,
            "hscr": 1.42,
            "shadow_area_ratio": 0.32,
            "physics_score": 0.78
        }
    }

    image_shape = (500, 1000)  # 500 pings x 1000 across-track pixels
    telemetry = pings[0]
    geo_result = georef.georeference_detection(sample_detection, image_shape, telemetry)
    sample_detection["georeference"] = geo_result

    print(f"  Georeferencing Results for {sample_detection['class_name'].upper()}:")
    print(f"    - Slant Range         : {geo_result['slant_range_m']:.2f} m")
    print(f"    - Across-Track Ground : {geo_result['across_track_distance_m']:.2f} m ({geo_result['channel'].upper()} channel)")
    print(f"    - Along-Track Pitch   : {geo_result['along_track_pitch_offset_m']:.2f} m offset")
    print(f"    - Target Coordinates  : ({geo_result['latitude']} N, {geo_result['longitude']} E)")
    print(f"    - Georef Quality      : {geo_result['georeference_quality']} (GPS: {geo_result['gps_source']})")
    print(f"    - Physical Dimensions : {geo_result['estimated_length_m']:.2f}m x {geo_result['estimated_width_m']:.2f}m")
    print(f"    - Estimated Height    : {geo_result['estimated_height_m']:.2f}m off seabed (Eff. Altitude: {geo_result['effective_altitude_m']}m)")
    print(f"    - Estimated Footprint : {geo_result['estimated_area_m2']:.2f} m^2")

    # 3. Test Report Engine (JSON & CSV Export)
    print("\n[Stage 3] Generating Structured Anomaly Reports...")
    reporter = AnomalyReportEngine(survey_id="SIH2026-ARIS-EXP-01")
    records = reporter.build_report_records([sample_detection], pings)

    out_dir = PROJECT_ROOT / "pipeline" / "test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "survey_anomaly_report.csv"
    json_file = out_dir / "survey_anomaly_report.json"

    reporter.export_csv(records, csv_file)
    reporter.export_json(records, json_file)

    print(f"  Generated Record Preview:")
    print(f"    - Anomaly ID      : {records[0]['anomaly_id']}")
    print(f"    - Review Status   : {records[0]['review_status']}")
    print(f"    - Composite Score : {records[0]['composite_score_pct']}/100 ({records[0]['score_type']})")
    print(f"    - Calibration     : {records[0]['calibration_status']}")
    print(f"    - Description     : {records[0]['hazard_description']}")
    print(f"    - Action Plan     : {records[0]['recommended_action']}")

    print("\n" + "=" * 65)
    print("  MODULE 3 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_module_3()
