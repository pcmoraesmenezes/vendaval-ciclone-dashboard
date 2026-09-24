"""Contract checks for the supplied scientific grids and their displayed coordinates."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import excursion_sets as e


class ThetaContractTest(unittest.TestCase):
    def test_all_fields_and_display_orientation(self):
        east, north = np.meshgrid(e.AXIS_KM, e.AXIS_KM)
        domain = east**2 + north**2 <= 1100**2
        for referencial in e.REFERENCE_FRAMES:
            for band in e.BANDS:
                for coefficient in e.COEFFICIENTS:
                    with self.subTest(referencial=referencial, band=band, coefficient=coefficient):
                        frame = e.REFERENCE_FRAMES[referencial]
                        fields = {p: e.load_theta(p, band, coefficient, referencial) for p in e.PHASES}
                        fig = e.theta_figure(fields, coefficient, frame)
                        global_min = min(np.nanmin(value[0]) for value in fields.values())
                        global_max = max(np.nanmax(value[0]) for value in fields.values())
                        for trace, (estimate, lower, upper) in zip(fig.data[::2], fields.values()):
                            np.testing.assert_array_equal(np.isfinite(estimate), domain)
                            boundaries = np.linspace(global_min, global_max, len(e.HEAT_COLORS) + 1)
                            expected_bands = np.digitize(estimate, boundaries[1:-1], right=False).astype(float)
                            expected_bands[~np.isfinite(estimate)] = np.nan
                            self.assertTrue(np.allclose(trace.z, expected_bands, equal_nan=True))
                            self.assertTrue(np.allclose(trace.customdata[:, :, 0], estimate, equal_nan=True))
                            self.assertTrue(np.allclose(trace.customdata[:, :, 1], lower, equal_nan=True))
                            self.assertTrue(np.allclose(trace.customdata[:, :, 2], upper, equal_nan=True))
                            self.assertEqual((trace.x[0], trace.x[44], trace.x[-1]), (-1100, 0, 1100))
                            self.assertEqual((trace.y[0], trace.y[44], trace.y[-1]), (-1100, 0, 1100))
                        self.assertEqual(fig.layout.coloraxis.cmin, 0)
                        self.assertEqual(fig.layout.coloraxis.cmax, len(e.HEAT_COLORS))

    def test_fixed_and_rotated_share_the_same_mask(self):
        for phase in e.PHASES:
            for band in e.BANDS:
                fixed = e.load_theta(phase, band, 'theta5', 'fixed')[0]
                rotated = e.load_theta(phase, band, 'theta5', 'rotated')[0]
                np.testing.assert_array_equal(np.isfinite(fixed), np.isfinite(rotated))

    def test_invalid_referencial_rejected(self):
        with self.assertRaises(ValueError):
            e.load_theta('decay', 'p95', 'theta5', 'northwest')

    def test_independent_reference_profile(self):
        # The supplied README reports 229.0 / 175.8 km in the 0–100 km ring.
        east, north = np.meshgrid(e.AXIS_KM, e.AXIS_KM)
        core = np.hypot(east, north) < 100
        for phase, expected in [('incipient', 229.0), ('mature', 175.8)]:
            estimate = e.load_theta(phase, 'p95', 'theta5')[0]
            self.assertAlmostEqual(float(np.nanmean(estimate[core])), expected, delta=0.05)


if __name__ == '__main__':
    unittest.main()
