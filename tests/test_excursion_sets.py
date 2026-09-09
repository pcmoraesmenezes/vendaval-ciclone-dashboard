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
        for band in e.BANDS:
            for coefficient in e.COEFFICIENTS:
                with self.subTest(band=band, coefficient=coefficient):
                    fields = {p: e.load_theta(p, band, coefficient) for p in e.PHASES}
                    fig = e.theta_figure(fields, coefficient)
                    for trace, (estimate, lower, upper) in zip(fig.data[::2], fields.values()):
                        np.testing.assert_array_equal(np.isfinite(estimate), domain)
                        np.testing.assert_array_equal(trace.z, estimate)
                        np.testing.assert_array_equal(trace.customdata[:, :, 0], lower)
                        np.testing.assert_array_equal(trace.customdata[:, :, 1], upper)
                        self.assertEqual((trace.x[0], trace.x[44], trace.x[-1]), (-1100, 0, 1100))
                        self.assertEqual((trace.y[0], trace.y[44], trace.y[-1]), (-1100, 0, 1100))
                    self.assertEqual(fig.layout.coloraxis.cmin, min(np.nanmin(v[0]) for v in fields.values()))
                    self.assertEqual(fig.layout.coloraxis.cmax, max(np.nanmax(v[0]) for v in fields.values()))

    def test_independent_reference_profile(self):
        # The supplied README reports 229.0 / 175.8 km in the 0–100 km ring.
        east, north = np.meshgrid(e.AXIS_KM, e.AXIS_KM)
        core = np.hypot(east, north) < 100
        for phase, expected in [('incipient', 229.0), ('mature', 175.8)]:
            estimate = e.load_theta(phase, 'p95', 'theta5')[0]
            self.assertAlmostEqual(float(np.nanmean(estimate[core])), expected, delta=0.05)


if __name__ == '__main__':
    unittest.main()
