"""
Side-Scan Sonar Acoustic Georeferencing & Dimensional Analysis Engine.

Implements:
1. Metadata Hierarchy with Explicit Quality Flags (Measured vs Estimated vs Unnavigated).
2. Pitch & Roll 3D Towfish Geometric Transformation:
   - Roll corrects effective altitude and slant-to-ground range across-track.
   - Pitch corrects along-track ray displacement.
   - Roll-corrected altitude feeds into physical target height estimation.
3. Acoustic shadow height estimation from similar-triangle geometry:
   H_target = (L_shadow * H_eff) / (R_ground + L_shadow)
4. Geodetic projection (WGS84) with dead-reckoning uncertainty or relative local coordinates.
"""

import numpy as np


class AcousticGeoreferencer:
    """
    Side-Scan Sonar Acoustic Georeferencer with Attitude (Pitch/Roll) Correction
    and Metadata Hierarchy Quality Flags.
    """

    def __init__(self, max_range_m=75.0, pixel_resolution_m=0.075, range_source="survey_calibration"):
        """
        Parameters:
            max_range_m: Maximum one-sided swath range in meters (e.g. 75m)
            pixel_resolution_m: Ground sampling resolution in meters per pixel (~7.5 cm/px)
            range_source: Source of range calibration ('xtf_channel_header', 'survey_calibration', 'uncalibrated')
        """
        self.max_range_m = float(max_range_m) if max_range_m else 75.0
        self.pixel_resolution_m = float(pixel_resolution_m) if pixel_resolution_m else 0.075
        self.range_source = range_source

    def slant_to_ground_range_with_roll(self, slant_range_m, altitude_m, roll_deg=0.0, channel="starboard"):
        """
        Converts acoustic slant range to horizontal ground range with roll attitude correction:
        
        Towfish roll (theta_r) tilts the transducer array:
        - On Starboard: depression angle decreases, effective clearance shifts:
          H_eff = H_alt * cos(theta_r) + y_approx * sin(theta_r)
        - On Port: depression angle increases, effective clearance shifts:
          H_eff = H_alt * cos(theta_r) - y_approx * sin(theta_r)
        
        Returns:
            ground_range_m: float
            effective_altitude_m: float
        """
        roll_rad = np.radians(roll_deg)
        cos_r = np.cos(roll_rad)
        sin_r = np.sin(roll_rad)

        # Baseline effective altitude under roll
        base_h_eff = max(0.5, altitude_m * cos_r)

        if slant_range_m <= base_h_eff:
            return 0.0, float(round(base_h_eff, 2))

        # Iterative refinement of across-track ground range and roll-projected clearance
        y_approx = np.sqrt(max(0.0, slant_range_m ** 2 - base_h_eff ** 2))
        roll_sign = 1.0 if channel == "starboard" else -1.0
        h_eff = max(0.5, (altitude_m * cos_r) + (roll_sign * y_approx * sin_r))

        ground_range = np.sqrt(max(0.0, slant_range_m ** 2 - h_eff ** 2))
        return float(round(ground_range, 2)), float(round(h_eff, 2))

    def estimate_target_height(self, shadow_length_m, range_m, effective_altitude_m):
        """
        Estimates target protrusion height off the seabed using acoustic shadow geometry
        with roll-corrected effective altitude:

            H_target = (L_shadow * H_eff) / (R_ground + L_shadow)

        Derived via similar triangles from the acoustic ray passing over target summit:
            H_target / L_shadow = (H_eff - H_target) / R_ground
            => H_target = (L_shadow * H_eff) / (R_ground + L_shadow)
        """
        if range_m <= 0.1 or effective_altitude_m <= 0.1 or shadow_length_m <= 0.05:
            return 0.3
        h_est = (shadow_length_m * effective_altitude_m) / (range_m + shadow_length_m)
        return float(round(np.clip(h_est, 0.15, 15.0), 2))

    def georeference_detection(self, detection, image_shape, telemetry_ping, calibration=None):
        """
        Calculates geographic coordinates and physical dimensions for a detected anomaly
        with explicit quality flags and pitch/roll attitude corrections.

        Parameters:
            detection: dict with 'bbox': [x1, y1, x2, y2]
            image_shape: (height, width) of the waterfall swath
            telemetry_ping: dict with telemetry metadata
            calibration: optional dict overriding max_range_m, pixel_resolution_m, range_source
        """
        h_img, w_img = image_shape[:2]
        x1, y1, x2, y2 = detection["bbox"]

        # Center of detection in pixels
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_w_px = x2 - x1
        box_h_px = y2 - y1

        # Towfish track nadir column
        nadir_x = w_img / 2.0
        pixel_offset = cx - nadir_x  # positive = starboard, negative = port
        channel = "starboard" if pixel_offset >= 0 else "port"

        # Apply calibration overrides if supplied
        calib = calibration or {}
        max_range_m = float(calib.get("max_range_m", self.max_range_m))
        pixel_res_m = float(calib.get("pixel_resolution_m", self.pixel_resolution_m))
        range_source = calib.get("range_source", self.range_source)

        # Telemetry variables & Quality Flags
        raw_lat = telemetry_ping.get("latitude")
        raw_lon = telemetry_ping.get("longitude")
        heading = float(telemetry_ping.get("heading_deg", 0.0))
        altitude = float(telemetry_ping.get("altitude_m", 12.0))
        pitch_deg = float(telemetry_ping.get("pitch_deg", 0.0))
        roll_deg = float(telemetry_ping.get("roll_deg", 0.0))

        gps_source = telemetry_ping.get("gps_source", "unnavigated")
        altitude_source = telemetry_ping.get("altitude_source", "measured_altimeter" if altitude > 0.5 else "estimated_default")
        position_uncertainty_m = float(telemetry_ping.get("position_uncertainty_m", 0.0))

        # Check attitude validity
        is_excessive_attitude = abs(pitch_deg) > 15.0 or abs(roll_deg) > 15.0
        attitude_quality = "excessive_motion" if is_excessive_attitude else "valid_imu"

        # 1. Physical slant range from nadir offset
        half_width_px = w_img / 2.0
        slant_range_m = (abs(pixel_offset) / half_width_px) * max_range_m

        # 2. Roll-corrected ground range and effective towfish clearance
        ground_range_m, h_eff_m = self.slant_to_ground_range_with_roll(
            slant_range_m, altitude, roll_deg=roll_deg, channel=channel
        )

        # 3. Pitch-corrected along-track ray displacement
        # Pitch tilts the acoustic beam fan forward or backward
        pitch_rad = np.radians(pitch_deg)
        along_track_offset_m = float(round(h_eff_m * np.tan(pitch_rad), 2))

        # 4. Real-world target dimensions
        width_m = float(round(box_w_px * pixel_res_m, 2))
        length_m = float(round(box_h_px * pixel_res_m, 2))
        area_m2 = float(round(width_m * length_m, 2))

        # 5. Shadow length & Target height using roll-corrected clearance
        shadow_px = box_w_px * detection.get("physics", {}).get("shadow_area_ratio", 0.30)
        shadow_length_m = float(round(shadow_px * pixel_res_m, 2))
        height_m = self.estimate_target_height(shadow_length_m, ground_range_m, h_eff_m)

        # 6. Coordinate calculation & Quality Assessment
        # Across-track azimuth: starboard = heading + 90 deg, port = heading - 90 deg
        across_azimuth_deg = (heading + 90.0) if channel == "starboard" else (heading - 90.0)
        across_azimuth_rad = np.radians(across_azimuth_deg % 360.0)
        heading_rad = np.radians(heading % 360.0)

        # Combined 3D offset: across-track ground range + along-track pitch displacement
        dx_east = (ground_range_m * np.sin(across_azimuth_rad)) + (along_track_offset_m * np.sin(heading_rad))
        dy_north = (ground_range_m * np.cos(across_azimuth_rad)) + (along_track_offset_m * np.cos(heading_rad))

        # Geodetic projection if valid navigation is present
        target_lat = None
        target_lon = None
        if raw_lat is not None and raw_lon is not None and gps_source != "unnavigated":
            d_lat = dy_north / 111139.0
            d_lon = dx_east / (111139.0 * np.cos(np.radians(raw_lat)))
            target_lat = float(round(raw_lat + d_lat, 7))
            target_lon = float(round(raw_lon + d_lon, 7))

            if gps_source == "ping_telemetry":
                georeference_quality = "measured" if not is_excessive_attitude else "degraded_attitude"
            elif gps_source == "dead_reckoned":
                georeference_quality = "estimated_dead_reckoning"
            else:
                georeference_quality = "estimated"
        else:
            georeference_quality = "unnavigated_relative"
            gps_source = "unnavigated"

        return {
            "latitude": target_lat,
            "longitude": target_lon,
            "relative_east_m": float(round(dx_east, 2)),
            "relative_north_m": float(round(dy_north, 2)),
            "channel": channel,
            "across_track_distance_m": ground_range_m,
            "slant_range_m": float(round(slant_range_m, 2)),
            "along_track_pitch_offset_m": along_track_offset_m,
            "estimated_length_m": max(0.5, length_m),
            "estimated_width_m": max(0.5, width_m),
            "estimated_height_m": max(0.2, height_m),
            "estimated_area_m2": max(0.25, area_m2),
            "towfish_altitude_m": altitude,
            "effective_altitude_m": h_eff_m,
            "towfish_heading_deg": heading,
            "towfish_pitch_deg": pitch_deg,
            "towfish_roll_deg": roll_deg,
            "attitude_quality": attitude_quality,
            "georeference_quality": georeference_quality,
            "gps_source": gps_source,
            "range_source": range_source,
            "altitude_source": altitude_source,
            "position_uncertainty_m": position_uncertainty_m,
            "attitude_corrected": True,
        }
