import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.modules.geotagging.georeference import AcousticGeoreferencer

class TestAcousticGeoreferenceMath(unittest.TestCase):
    """
    Rigorous unit verification of acoustic slant-to-ground range transformation
    and shadow-geometry target height estimation based on exact mathematical derivations.
    """
    def setUp(self):
        self.geo = AcousticGeoreferencer(max_range_m=75.0, pixel_resolution_m=0.075)

    def test_slant_to_ground_range_hand_computed_cases(self):
        """
        Verify R_ground = sqrt(max(0, R_slant^2 - H_alt^2))
        """
        # Case A: 3-4-5 triangle scaled (H_alt=10, R_slant=26 => R_g = sqrt(676 - 100) = sqrt(576) = 24.0m)
        r_g1 = self.geo.slant_to_ground_range(slant_range_m=26.0, altitude_m=10.0)
        self.assertAlmostEqual(r_g1, 24.0, places=4, msg="Failed Case A (26, 10 -> 24.0)")

        # Case B: H_alt=12, R_slant=37 => R_g = sqrt(1369 - 144) = sqrt(1225) = 35.0m
        r_g2 = self.geo.slant_to_ground_range(slant_range_m=37.0, altitude_m=12.0)
        self.assertAlmostEqual(r_g2, 35.0, places=4, msg="Failed Case B (37, 12 -> 35.0)")

        # Case C: At nadir boundary (R_slant == H_alt => R_g = 0.0)
        r_g3 = self.geo.slant_to_ground_range(slant_range_m=15.0, altitude_m=15.0)
        self.assertAlmostEqual(r_g3, 0.0, places=4, msg="Failed Case C: Nadir boundary must be 0.0")

        # Case D: Inside water column (R_slant < H_alt => 0.0)
        r_g4 = self.geo.slant_to_ground_range(slant_range_m=8.0, altitude_m=12.0)
        self.assertEqual(r_g4, 0.0, msg="Failed Case D: Inside water column must be 0.0")

    def test_target_height_hand_computed_cases(self):
        """
        Verify H_t = (L_g * H_alt) / (R_g + L_g) for 5 distinct geometric scenarios.
        
        Geometric Derivation:
        Similar triangles formed by acoustic ray from towfish (0, H_alt)
        grazing top of debris (R_g, H_t) and terminating at shadow tip (R_g + L_g, 0):
        H_t / L_g = (H_alt - H_t) / R_g
        => H_t * (R_g + L_g) = L_g * H_alt
        => H_t = (L_g * H_alt) / (R_g + L_g)
        """
        cases = [
            # (H_alt, R_g, L_g, Expected H_t, Description)
            (10.0, 20.0, 5.0, 2.00, "Case 1: Standard mid-range debris (L=5, R=20, H_alt=10 => 50/25 = 2.0m)"),
            (15.0, 45.0, 15.0, 3.75, "Case 2: Moderate range hazard (L=15, R=45, H_alt=15 => 225/60 = 3.75m)"),
            (12.0, 6.0, 2.0, 3.00, "Case 3: Steep near-nadir angle (L=2, R=6, H_alt=12 => 24/8 = 3.00m)"),
            (8.0, 72.0, 8.0, 0.80, "Case 4: Shallow far-range low-grazing (L=8, R=72, H_alt=8 => 64/80 = 0.80m)"),
            (20.0, 30.0, 30.0, 10.00, "Case 5: Substantial obstacle casting 1:1 shadow (L=30, R=30, H_alt=20 => 600/60 = 10.00m)")
        ]

        print("\n" + "=" * 70)
        print("  TARGET HEIGHT MATHEMATICAL VERIFICATION (5 KNOWN CASES)")
        print("=" * 70)
        print(f"{'Case':<8} | {'H_alt':<7} | {'R_g':<7} | {'L_g':<7} | {'Expected H_t':<13} | {'Calculated':<10} | {'Status'}")
        print("-" * 70)

        for idx, (h_alt, r_g, l_g, expected_ht, desc) in enumerate(cases, 1):
            calculated_ht = self.geo.estimate_target_height(
                shadow_length_m=l_g,
                range_m=r_g,
                altitude_m=h_alt,
                is_ground_range=True
            )
            is_match = abs(calculated_ht - expected_ht) < 1e-2
            status = "PASS" if is_match else "FAIL"
            print(f"Case {idx:<3} | {h_alt:<7.1f} | {r_g:<7.1f} | {l_g:<7.1f} | {expected_ht:<13.2f} | {calculated_ht:<10.2f} | {status}")
            self.assertAlmostEqual(
                calculated_ht,
                expected_ht,
                places=2,
                msg=f"Failed {desc}: expected {expected_ht}m, got {calculated_ht}m"
            )

    def test_target_height_boundary_conditions(self):
        """Verify handling of degenerate/zero/negative values."""
        # Zero shadow length
        h_zero_shadow = self.geo.estimate_target_height(0.0, 30.0, 10.0)
        self.assertGreaterEqual(h_zero_shadow, 0.15, "Zero shadow should return baseline minimum (0.15-0.3m)")

        # Negative/Zero range
        h_zero_range = self.geo.estimate_target_height(5.0, 0.0, 10.0)
        self.assertGreaterEqual(h_zero_range, 0.15)

if __name__ == "__main__":
    unittest.main()
