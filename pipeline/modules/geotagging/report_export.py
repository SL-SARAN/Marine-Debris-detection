import json
import csv
from datetime import datetime
from pathlib import Path

PRIORITY_RULES = {
    "ghost_net": {
        "tier_high": "CRITICAL - Immediate Diver / ROV Retrieval Required (Ghost Gear Entanglement Risk)",
        "tier_mid": "HIGH - Priority Marine Life Entanglement Risk",
        "action": "Deploy acoustic transponder beacon and ROV with line cutter / grappling hook."
    },
    "shipwreck": {
        "tier_high": "HIGH - Significant Underwater Navigation Obstruction",
        "tier_mid": "MODERATE - Marine Habitat / Cultural Heritage Inspection",
        "action": "Log nautical chart obstruction notice; dispatch bathymetric survey."
    },
    "aircraft": {
        "tier_high": "CRITICAL - Anthropogenic Submerged Wreckage",
        "tier_mid": "HIGH - Submerged Metal Debris",
        "action": "Coordinate with maritime salvage team."
    },
    "seafloor_debris": {
        "tier_high": "MODERATE - Artificial Seafloor Clutter",
        "tier_mid": "LOW - Minor Seabed Anomaly",
        "action": "Monitor during scheduled survey passes."
    },
    "pipeline_cable": {
        "tier_high": "CRITICAL - Subsea Infrastructure Integrity Risk / Anchor Drag Threat",
        "tier_mid": "HIGH - Pipeline / Cable Crossing Inspection",
        "action": "Notify offshore asset operator; schedule cathodic and visual ROV inspection."
    },
    "engineering_platform": {
        "tier_high": "HIGH - Wellhead / Subsea Template Obstruction",
        "tier_mid": "MODERATE - Subsea Facility Footprint",
        "action": "Verify against bathymetric navigational charts and infrastructure database."
    }
}

class AnomalyReportEngine:
    """
    Standardized Anomaly Reporting Engine for Marine Sonar Surveys.
    Generates structured, field-ready reports in JSON and CSV formats.
    """
    def __init__(self, survey_id="SIH2026-SURVEY-SSS-01"):
        self.survey_id = survey_id

    def build_report_records(self, detections, telemetry_pings=None):
        """
        Builds standardized anomaly report records from detection and georeferencing payloads.
        Includes explicit score breakdown (detector, physics, composite), calibration status,
        and georeference quality flags.
        """
        records = []
        for i, det in enumerate(detections):
            anom_id = f"ANOM-SSS-{i+1:03d}"
            cname = det.get("class_name", "debris")
            
            # Score hierarchy
            detector_score = float(det.get("detector_score", det.get("raw_confidence", 0.5)))
            physics = det.get("physics", {})
            physics_score = float(det.get("physics_score", physics.get("physics_score", 0.5)))
            composite_pct = float(det.get("composite_score_pct", det.get("calibrated_confidence_pct", 50.0)))
            score_type = det.get("score_type", "uncalibrated_composite")
            calibration_status = det.get("calibration_status", "pending_ground_truth")
            review_status = det.get("review_status", "standard_review")
            
            geo = det.get("georeference", {})

            # Priority assignment based on operational composite ranking
            rules = PRIORITY_RULES.get(cname, PRIORITY_RULES["seafloor_debris"])
            if composite_pct >= 75.0 and physics.get("is_verified", True):
                priority = "CRITICAL" if cname == "ghost_net" else "HIGH"
                desc = rules["tier_high"]
            elif composite_pct >= 45.0:
                priority = "HIGH" if cname == "ghost_net" else "MODERATE"
                desc = rules["tier_mid"]
            else:
                priority = "LOW"
                desc = "LOW - Tentative Acoustic Candidate"

            rec = {
                "anomaly_id": anom_id,
                "survey_id": self.survey_id,
                "timestamp_utc": det.get("timestamp_utc", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")),
                "classification": cname,
                "detector_score": round(detector_score, 3),
                "physics_score": round(physics_score, 3),
                "composite_score_pct": composite_pct,
                "score_type": score_type,
                "calibration_status": calibration_status,
                "review_status": review_status,
                "score_disclaimer": "Score is an uncalibrated ranking aid, not a probability of correctness. Ground-truth calibration pending.",
                # Backward compatibility aliases
                "calibrated_confidence_pct": composite_pct,
                "confidence_tier": review_status.upper(),
                "priority_level": priority,
                "hazard_description": desc,
                "recommended_action": rules["action"],
                "latitude": geo.get("latitude"),
                "longitude": geo.get("longitude"),
                "relative_east_m": geo.get("relative_east_m", 0.0),
                "relative_north_m": geo.get("relative_north_m", 0.0),
                "channel": geo.get("channel", "starboard"),
                "across_track_m": geo.get("across_track_distance_m", 15.0),
                "slant_range_m": geo.get("slant_range_m", 15.0),
                "along_track_pitch_offset_m": geo.get("along_track_pitch_offset_m", 0.0),
                "estimated_length_m": geo.get("estimated_length_m", 2.5),
                "estimated_width_m": geo.get("estimated_width_m", 1.8),
                "estimated_height_m": geo.get("estimated_height_m", 1.2),
                "estimated_area_m2": geo.get("estimated_area_m2", 4.5),
                "georeference_quality": geo.get("georeference_quality", "unnavigated_relative"),
                "gps_source": geo.get("gps_source", "unnavigated"),
                "range_source": geo.get("range_source", "operator_survey_calibration"),
                "altitude_source": geo.get("altitude_source", "estimated_default"),
                "attitude_quality": geo.get("attitude_quality", "valid_imu"),
                "towfish_altitude_m": geo.get("towfish_altitude_m", 12.0),
                "towfish_heading_deg": geo.get("towfish_heading_deg", 0.0),
                "towfish_pitch_deg": geo.get("towfish_pitch_deg", 0.0),
                "towfish_roll_deg": geo.get("towfish_roll_deg", 0.0),
                "hscr_contrast": physics.get("hscr", 0.0),
                "physics_verified": physics.get("is_verified", False),
                "raw_confidence": round(detector_score, 3),
                "bbox": det.get("bbox", [0, 0, 0, 0])
            }
            records.append(rec)
        return records

    def export_csv(self, records, output_path):
        """Exports anomaly records to a standard survey CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "anomaly_id", "timestamp_utc", "classification", "composite_score_pct",
            "detector_score", "physics_score", "score_type", "calibration_status",
            "review_status", "priority_level", "latitude", "longitude",
            "georeference_quality", "gps_source", "range_source", "channel",
            "across_track_m", "estimated_length_m", "estimated_width_m", "estimated_height_m",
            "estimated_area_m2", "hscr_contrast", "physics_verified", "hazard_description",
            "recommended_action", "score_disclaimer"
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        print(f"[AnomalyReportEngine] Exported CSV report to: {path}")
        return str(path)

    def export_json(self, records, output_path, metadata=None):
        """Exports anomaly records and full survey metadata to a structured JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "survey_metadata": {
                "survey_id": self.survey_id,
                "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_anomalies_detected": len(records),
                "high_priority_reviews_count": sum(1 for r in records if r.get("review_status") == "high_priority_review"),
                "critical_hazards_count": sum(1 for r in records if r["priority_level"] == "CRITICAL"),
                "sensor_platform": "Side-Scan Sonar (SSS) Survey System",
                "calibration_notice": "Score is an uncalibrated composite ranking aid. TODO: Once calibration is supported by labelled, representative detections, evaluate it on a separate held-out set. Then you can decide whether to report a calibrated probability and retain meaningful probability-based tiers.",
                "custom_metadata": metadata or {}
            },
            "anomalies": records
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[AnomalyReportEngine] Exported JSON report to: {path}")
        return str(path)

    def export_geojson(self, records, output_path):
        """Exports anomaly records as a standard GeoJSON FeatureCollection."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        features = []
        for r in records:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(r.get("longitude", 0.0)), float(r.get("latitude", 0.0))]
                },
                "properties": {k: v for k, v in r.items() if k not in ("latitude", "longitude")}
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "survey_id": self.survey_id,
                "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_features": len(features)
            },
            "features": features
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        print(f"[AnomalyReportEngine] Exported GeoJSON report to: {path}")
        return str(path)

    def export_pdf(self, records, output_path, metadata=None):
        """
        Generates a professional, print-ready Maritime Sonar Hazard Audit PDF report
        using ReportLab. Completely pure-Python, zero browser dependencies.
        """
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor('#475569'),
            spaceAfter=10
        )
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=colors.HexColor('#1E293B'),
            spaceBefore=8,
            spaceAfter=4
        )
        cell_style = ParagraphStyle(
            'CellText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10
        )
        cell_bold = ParagraphStyle(
            'CellBold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10
        )

        story = []

        # Title Banner
        story.append(Paragraph("AUV SIDE-SCAN SONAR MARITIME DEBRIS SURVEY AUDIT", title_style))
        story.append(Paragraph(f"Survey Reference: {self.survey_id} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Platform: Autonomous Underwater Vehicle (AUV)", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284C7'), spaceBefore=2, spaceAfter=8))

        # Executive Summary Stats Table
        crit_count = sum(1 for r in records if r.get("priority_level") == "CRITICAL")
        high_count = sum(1 for r in records if r.get("priority_level") == "HIGH")
        nets_count = sum(1 for r in records if r.get("classification") == "ghost_net")

        summary_data = [
            [Paragraph("<b>Total Anomalies Detected</b>", cell_bold), Paragraph(str(len(records)), cell_style),
             Paragraph("<b>Critical Debris Hazards</b>", cell_bold), Paragraph(str(crit_count), cell_style)],
            [Paragraph("<b>Confirmed Ghost Nets</b>", cell_bold), Paragraph(str(nets_count), cell_style),
             Paragraph("<b>High Priority Targets</b>", cell_bold), Paragraph(str(high_count), cell_style)]
        ]
        summary_table = Table(summary_data, colWidths=[1.8*inch, 1.3*inch, 1.8*inch, 1.3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # Hazard Table
        story.append(Paragraph("DETECTED SUBSEA ANOMALIES & BATHYMETRIC MENSURATION", section_style))
        table_headers = [
            Paragraph("<font color='white'><b>ID</b></font>", cell_bold),
            Paragraph("<font color='white'><b>Classification</b></font>", cell_bold),
            Paragraph("<font color='white'><b>Conf. %</b></font>", cell_bold),
            Paragraph("<font color='white'><b>Priority</b></font>", cell_bold),
            Paragraph("<font color='white'><b>Coordinates (WGS84)</b></font>", cell_bold),
            Paragraph("<font color='white'><b>Dimensions (L×W×H)</b></font>", cell_bold),
            Paragraph("<font color='white'><b>Action Recommendation</b></font>", cell_bold)
        ]

        table_rows = [table_headers]
        for r in records[:30]:  # Up to 30 records in primary table
            pri = r.get("priority_level", "LOW")
            pri_color = "#DC2626" if pri == "CRITICAL" else ("#D97706" if pri == "HIGH" else "#475569")
            pri_html = f"<font color='{pri_color}'><b>{pri}</b></font>"

            coords = f"{float(r.get('latitude', 0.0)):.6f}°N<br/>{float(r.get('longitude', 0.0)):.6f}°E"
            dims = f"{float(r.get('estimated_length_m', 0.0)):.1f}m × {float(r.get('estimated_width_m', 0.0)):.1f}m<br/>H: {float(r.get('estimated_height_m', 0.0)):.1f}m"

            row = [
                Paragraph(str(r.get("anomaly_id", "N/A")), cell_style),
                Paragraph(f"<b>{str(r.get('classification', 'debris')).upper()}</b>", cell_style),
                Paragraph(f"{float(r.get('calibrated_confidence_pct', 0.0)):.1f}%", cell_style),
                Paragraph(pri_html, cell_style),
                Paragraph(coords, cell_style),
                Paragraph(dims, cell_style),
                Paragraph(str(r.get("recommended_action", "Inspect")), cell_style)
            ]
            table_rows.append(row)

        hazard_table = Table(table_rows, colWidths=[0.8*inch, 1.1*inch, 0.65*inch, 0.75*inch, 1.3*inch, 1.1*inch, 1.8*inch])
        hazard_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('TOPPADDING', (0, 0), (-1, 0), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ]))
        story.append(hazard_table)

        doc.build(story)
        print(f"[AnomalyReportEngine] Exported PDF report to: {path}")
        return str(path)
