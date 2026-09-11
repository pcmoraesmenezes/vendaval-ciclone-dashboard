"""Contract checks for the composite 0-1 index built on top of the supplied fields."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import composite_index as c
import excursion_sets as e


class CompositeContractTest(unittest.TestCase):
    def test_frequency_is_replicated_from_its_native_100km_cells(self):
        """Every 25km pixel must carry the value of the 100km cell it falls inside."""
        for band, nivel in c.BAND_TO_NIVEL.items():
            with self.subTest(band=band):
                fields = c.load_frequency_grid(band)
                source = pd.read_csv(c.GRID_CSV)
                source = source[(source['quad_type'] == 'fixed') & (source['nivel'] == nivel)]
                east, north = np.meshgrid(e.AXIS_KM, e.AXIS_KM)
                for phase, grid in fields.items():
                    sub = source[source['phase'] == phase]
                    for _, row in sub.iterrows():
                        inside = ((np.floor(east / c.FREQ_BIN_KM) == row['cell_x'])
                                  & (np.floor(north / c.FREQ_BIN_KM) == row['cell_y']))
                        if not inside.any():
                            continue
                        values = np.unique(grid[inside])
                        self.assertEqual(values.size, 1)
                        self.assertAlmostEqual(float(values[0]), row['taxa_contagem_media'])
                    # A full interior cell covers exactly 16 pixels of 25km.
                    counts = pd.Series(
                        np.floor(east / c.FREQ_BIN_KM).ravel().astype(int)).astype(str) + '_' + pd.Series(
                        np.floor(north / c.FREQ_BIN_KM).ravel().astype(int)).astype(str)
                    self.assertEqual(int(counts.value_counts().max()), 16)

    def test_composite_is_the_plain_mean_of_the_three_normalised_fields(self):
        for band in c.BANDS:
            with self.subTest(band=band):
                raw, normed, composite, ranges, mask = c.load_composite(band)
                for phase in e.PHASES:
                    expected = sum(normed[k][phase] for k in c.COMPONENTS) / 3.0
                    np.testing.assert_allclose(composite[phase], expected, equal_nan=True)

    def test_normalisation_is_global_across_phases_and_bounded(self):
        """One min/max per component per band, swept over the four phases together."""
        for band in c.BANDS:
            with self.subTest(band=band):
                raw, normed, composite, ranges, mask = c.load_composite(band)
                for component in c.COMPONENTS:
                    stack = np.concatenate([normed[component][p][np.isfinite(normed[component][p])]
                                            for p in e.PHASES])
                    self.assertAlmostEqual(float(stack.min()), 0.0)
                    self.assertAlmostEqual(float(stack.max()), 1.0)
                    # Exactly one phase may own the global extreme; per-phase normalisation
                    # would instead give every phase both a 0 and a 1.
                    owners_max = sum(np.isclose(np.nanmax(normed[component][p]), 1.0) for p in e.PHASES)
                    self.assertEqual(owners_max, 1)
                    vmin, vmax = ranges[component]
                    for phase in e.PHASES:
                        finite = np.isfinite(raw[component][phase])
                        np.testing.assert_allclose(
                            normed[component][phase][finite],
                            (raw[component][phase][finite] - vmin) / (vmax - vmin))
                for phase in e.PHASES:
                    finite = composite[phase][np.isfinite(composite[phase])]
                    self.assertGreaterEqual(float(finite.min()), 0.0)
                    self.assertLessEqual(float(finite.max()), 1.0)

    def test_index_exists_only_where_all_three_components_exist(self):
        for band in c.BANDS:
            with self.subTest(band=band):
                raw, normed, composite, ranges, mask = c.load_composite(band)
                for phase in e.PHASES:
                    present = [np.isfinite(raw[k][phase]) for k in c.COMPONENTS]
                    np.testing.assert_array_equal(np.isfinite(composite[phase]),
                                                  np.logical_and.reduce(present))
                    np.testing.assert_array_equal(np.isfinite(composite[phase]), mask[phase])
                    # Never wider than the theta domain, which is the tighter of the two masks.
                    theta = np.isfinite(e.load_theta(phase, band, 'theta5')[0])
                    self.assertTrue(np.all(np.isfinite(composite[phase]) <= theta))

    def test_figure_pins_the_scale_to_zero_one_and_carries_every_component(self):
        for band in c.BANDS:
            with self.subTest(band=band):
                raw, normed, composite, ranges, mask = c.load_composite(band)
                fig = c.composite_figure(raw, normed, composite)
                self.assertEqual(fig.layout.coloraxis.cmin, 0)
                self.assertEqual(fig.layout.coloraxis.cmax, len(e.HEAT_COLORS))
                boundaries = np.linspace(0.0, 1.0, len(e.HEAT_COLORS) + 1)
                for trace, phase in zip(fig.data[::2], e.PHASES):
                    expected = np.digitize(composite[phase], boundaries[1:-1], right=False).astype(float)
                    expected[~np.isfinite(composite[phase])] = np.nan
                    self.assertTrue(np.allclose(trace.z, expected, equal_nan=True))
                    self.assertTrue(np.allclose(trace.customdata[:, :, 0], composite[phase], equal_nan=True))
                    for i, component in enumerate(c.COMPONENTS, start=1):
                        self.assertTrue(np.allclose(trace.customdata[:, :, i],
                                                    normed[component][phase], equal_nan=True))
                    for i, component in enumerate(c.COMPONENTS, start=4):
                        self.assertTrue(np.allclose(trace.customdata[:, :, i],
                                                    raw[component][phase], equal_nan=True))
                    self.assertEqual((trace.x[0], trace.x[44], trace.x[-1]), (-1100, 0, 1100))
                    self.assertEqual((trace.y[0], trace.y[44], trace.y[-1]), (-1100, 0, 1100))

    def test_frequency_band_sits_between_the_single_levels_that_bracket_it(self):
        """The band rate is the mean over a range of thresholds, so it must fall between the
        rates of the levels that bracket that range — a single-threshold field would not.

        b95 averages q89..q95: q90 is its second-lowest threshold (so the highest rate still
        inside the band, bar q89) and q95 its highest (the lowest rate). b99 averages q95..q99
        the same way. Catches the whole band pipeline silently falling back to one threshold.
        """
        source = pd.read_csv(c.GRID_CSV)
        source = source[source['quad_type'] == 'fixed']
        for band, low, high in [('b95', 'q90', 'q95'), ('b99', 'q95', 'q99')]:
            with self.subTest(band=band):
                key = ['phase', 'cell_x', 'cell_y']
                merged = (source[source['nivel'] == band].set_index(key)['taxa_contagem_media']
                          .to_frame('banda')
                          .join(source[source['nivel'] == low].set_index(key)['taxa_contagem_media']
                                .rename('limiar_baixo'), how='inner')
                          .join(source[source['nivel'] == high].set_index(key)['taxa_contagem_media']
                                .rename('limiar_alto'), how='inner'))
                self.assertGreater(len(merged), 0)
                self.assertTrue((merged['banda'] >= merged['limiar_alto'] - 1e-12).all())
                self.assertTrue((merged['banda'] <= merged['limiar_baixo'] + 1e-12).all())
                # And it must not simply BE one of them, which is what a silent fallback looks like.
                self.assertFalse(np.allclose(merged['banda'], merged['limiar_alto']))

    def test_figures_stretch_in_width_and_declare_a_height_that_fits_the_aspect_lock(self):
        """Width must come from the container so the plot reaches the edges, and height must be
        declared — the 1:1 aspect lock draws a square of side min(panel width, panel height),
        so a height much smaller than the panel width leaves dead space at the sides.

        Checks the height is at least ~70% of the panel width expected on a wide monitor. Pins
        no upper bound: a tall 2x2 is a legitimate choice, a squashed one is not.
        """
        raw, normed, composite, ranges, mask = c.load_composite('p95')
        largura_tipica = 1600
        casos = [(c.composite_figure(raw, normed, composite), 4)]
        casos += [(c.component_figure(normed, k), 4) for k in c.COMPONENTS]
        casos.append((e.theta_figure({p: e.load_theta(p, 'p95', 'theta5') for p in e.PHASES}, 'theta5'), 2))
        for fig, colunas in casos:
            self.assertIsNone(fig.layout.width, 'largura fixa impede o gráfico de encostar nas bordas')
            self.assertIsNot(fig.layout.autosize, False)
            self.assertGreater(fig.layout.height, 0)
            linhas = 1 if colunas == 4 else 2
            largura_painel = largura_tipica / colunas
            altura_painel = fig.layout.height / linhas
            self.assertGreater(altura_painel, 0.7 * largura_painel,
                               f'painel achatado: {altura_painel:.0f}px de altura para '
                               f'{largura_painel:.0f}px de largura')

    def test_only_the_fixed_referential_feeds_the_index(self):
        """Rotated cells must never reach the composite: theta is fixed-frame only."""
        source = pd.read_csv(c.GRID_CSV)
        self.assertIn('rotated', set(source['quad_type']))
        for band, nivel in c.BAND_TO_NIVEL.items():
            fields = c.load_frequency_grid(band)
            fixed = source[(source['quad_type'] == 'fixed') & (source['nivel'] == nivel)]
            rotated = source[(source['quad_type'] == 'rotated') & (source['nivel'] == nivel)]
            for phase, grid in fields.items():
                observed = set(np.round(grid[np.isfinite(grid)].ravel(), 12))
                allowed = set(np.round(fixed[fixed['phase'] == phase]['taxa_contagem_media'], 12))
                self.assertTrue(observed <= allowed)
                only_rotated = set(np.round(
                    rotated[rotated['phase'] == phase]['taxa_contagem_media'], 12)) - allowed
                self.assertFalse(observed & only_rotated)


if __name__ == '__main__':
    unittest.main()
