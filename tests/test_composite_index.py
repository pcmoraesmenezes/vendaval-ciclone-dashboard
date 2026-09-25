"""Contract checks for the composite 0-1 index built on top of the supplied fields."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import composite_index as c
import excursion_sets as e
import heat_scale as h


REFERENCIAIS = list(e.REFERENCE_FRAMES)


class CompositeContractTest(unittest.TestCase):
    def test_frequency_is_replicated_from_its_native_100km_cells(self):
        """Every 25km pixel must carry the value of the 100km cell it falls inside."""
        for referencial in REFERENCIAIS:
            for band, nivel in c.BAND_TO_NIVEL.items():
                with self.subTest(referencial=referencial, band=band):
                    fields = c.load_frequency_grid(band, referencial)
                    source = pd.read_csv(c.GRID_CSV)
                    source = source[(source['quad_type'] == referencial) & (source['nivel'] == nivel)]
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
        for referencial in REFERENCIAIS:
            for band in c.BANDS:
                with self.subTest(referencial=referencial, band=band):
                    raw, normed, composite, ranges, mask = c.load_composite(band, referencial)
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
        for referencial in REFERENCIAIS:
            for band in c.BANDS:
                with self.subTest(referencial=referencial, band=band):
                    raw, normed, composite, ranges, mask = c.load_composite(band, referencial)
                    for phase in e.PHASES:
                        present = [np.isfinite(raw[k][phase]) for k in c.COMPONENTS]
                        np.testing.assert_array_equal(np.isfinite(composite[phase]),
                                                      np.logical_and.reduce(present))
                        np.testing.assert_array_equal(np.isfinite(composite[phase]), mask[phase])
                        # Never wider than the theta domain, which is the tighter of the two masks.
                        theta = np.isfinite(e.load_theta(phase, band, 'theta5', referencial)[0])
                        self.assertTrue(np.all(np.isfinite(composite[phase]) <= theta))

    def test_figure_pins_the_scale_to_zero_one_and_carries_every_component(self):
        for band in c.BANDS:
            with self.subTest(band=band):
                raw, normed, composite, ranges, mask = c.load_composite(band)
                fig = c.composite_figure(raw, normed, composite)
                vmin, vmax = c.observed_range(composite)
                self.assertEqual(fig.layout.coloraxis.cmin, vmin)
                self.assertEqual(fig.layout.coloraxis.cmax, vmax)
                for trace, phase in zip(fig.data[::2], e.PHASES):
                    # z é o índice em si, não um número de faixa: mover o limiar muda a cor,
                    # nunca o dado por trás do pixel nem o que o cursor lê.
                    self.assertTrue(np.allclose(trace.z, composite[phase], equal_nan=True))
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

        b90 averages q80..q90 (q90 its highest threshold, lowest rate in the band). b95 averages
        q89..q95: q90 is its second-lowest threshold (so the highest rate still inside the band,
        bar q89) and q95 its highest (the lowest rate). b99 averages q95..q99 the same way.
        Catches the whole band pipeline silently falling back to one threshold, in either
        referential.
        """
        source = pd.read_csv(c.GRID_CSV)
        for referencial in REFERENCIAIS:
            ref_source = source[source['quad_type'] == referencial]
            for band, low, high in [('b90', None, 'q90'), ('b95', 'q90', 'q95'), ('b99', 'q95', 'q99')]:
                with self.subTest(referencial=referencial, band=band):
                    key = ['phase', 'cell_x', 'cell_y']
                    banda = ref_source[ref_source['nivel'] == band].set_index(key)['taxa_contagem_media']
                    alto = ref_source[ref_source['nivel'] == high].set_index(key)['taxa_contagem_media']
                    merged = banda.to_frame('banda').join(alto.rename('limiar_alto'), how='inner')
                    self.assertGreater(len(merged), 0)
                    self.assertTrue((merged['banda'] >= merged['limiar_alto'] - 1e-12).all())
                    if low:
                        baixo = ref_source[ref_source['nivel'] == low].set_index(key)['taxa_contagem_media']
                        merged = merged.join(baixo.rename('limiar_baixo'), how='inner')
                        self.assertTrue((merged['banda'] <= merged['limiar_baixo'] + 1e-12).all())
                    # And it must not simply BE the bracketing level, which is what a silent
                    # fallback to a single threshold looks like.
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
        casos.append((e.theta_figure({p: e.load_theta(p, 'p95', 'theta5') for p in e.PHASES}, 'theta5',
                                     e.REFERENCE_FRAMES['fixed']), 2))
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

    def test_the_floor_is_dynamic_and_only_grey_sits_below_it(self):
        """Pedido de 12/09/2026: o piso do cinza é arrastável, como no heatmap de extremos.

        Fixa o contrato que a tela depende: (a) abaixo do limiar a cor é o cinza e nada mais;
        (b) a rampa quente inteira sobrevive acima dele — subir o piso redistribui as nove
        cores em vez de descartar níveis, que é o ponto de arrastar; (c) a fronteira do cinza
        cai exatamente no limiar pedido, senão a legenda mente sobre o que está escondido.
        """
        for floor in (0.0, 0.11, 0.4, 0.75):
            with self.subTest(floor=floor):
                scale = h.heat_colorscale(0.0, 1.0, floor)
                cores = [cor for _, cor in scale]
                quentes = [cor for cor in cores if cor != h.HEAT_UNDER_COLOR]
                self.assertEqual(sorted(set(quentes)), sorted(set(h.HEAT_COLORS)),
                                 'subir o limiar não pode custar um nível da rampa')
                if floor == 0.0:
                    self.assertNotIn(h.HEAT_UNDER_COLOR, cores)
                    continue
                cinzas = [pos for pos, cor in scale if cor == h.HEAT_UNDER_COLOR]
                self.assertEqual(min(cinzas), 0.0)
                self.assertAlmostEqual(max(cinzas), floor, places=6,
                                       msg='a divisa do cinza tem que ser o limiar pedido')
                # nenhuma cor quente pode aparecer abaixo da divisa
                for pos, cor in scale:
                    if cor != h.HEAT_UNDER_COLOR:
                        self.assertGreaterEqual(pos, floor - 1e-9)

    def test_the_colour_covers_the_observed_range_not_zero_to_one(self):
        """O ponto do pedido de 12/09/2026: a cor tem que cobrir o que existe.

        O índice é média de três campos normalizados, então nunca encosta em 0 nem em 1 — preso
        a [0, 1] ele usaria uma fatia estreita da rampa e os valores baixos sairiam todos com a
        mesma cor. Este teste falha se alguém reancorar a escala no 0-1.
        """
        for band in c.BANDS:
            with self.subTest(band=band):
                raw, normed, composite, ranges, mask = c.load_composite(band)
                vmin, vmax = c.observed_range(composite)
                self.assertGreater(vmin, 0.0)
                self.assertLess(vmax, 1.0)
                fig = c.composite_figure(raw, normed, composite)
                self.assertAlmostEqual(fig.layout.coloraxis.cmin, vmin)
                self.assertAlmostEqual(fig.layout.coloraxis.cmax, vmax)
                # a cor mais quente tem que ser efetivamente alcançada pelo maior valor
                self.assertEqual(fig.layout.coloraxis.colorscale[-1][1], h.HEAT_COLORS[-1])
                # e o valor máximo do dado bate no topo da escala, não em 80% dela
                topo = max(float(t.z[np.isfinite(t.z)].max()) for t in fig.data[::2])
                self.assertAlmostEqual(topo, vmax)

    def test_the_floor_moves_inside_the_observed_range_and_greys_the_lowest(self):
        """Arrastar o limiar apaga os menores e reorganiza as cores no que sobra."""
        raw, normed, composite, ranges, mask = c.load_composite('p95')
        vmin, vmax = c.observed_range(composite)
        meio = vmin + (vmax - vmin) / 2
        for floor in (vmin, meio):
            fig = c.composite_figure(raw, normed, composite, floor, (vmin, vmax))
            escala = [list(par) for par in fig.layout.coloraxis.colorscale]
            self.assertEqual(escala, h.heat_colorscale(vmin, vmax, floor))
            cores = [cor for _, cor in escala]
            if floor == vmin:
                self.assertNotIn(h.HEAT_UNDER_COLOR, cores, 'no mínimo observado nada fica cinza')
            else:
                self.assertEqual(cores[0], h.HEAT_UNDER_COLOR)
                # a divisa do cinza cai exatamente no limiar, em unidades do dado
                divisa = max(pos for pos, cor in escala if cor == h.HEAT_UNDER_COLOR)
                self.assertAlmostEqual(vmin + divisa * (vmax - vmin), floor, places=6)
                # e sobra rampa inteira acima dele
                self.assertEqual(sorted(set(cor for cor in cores if cor != h.HEAT_UNDER_COLOR)),
                                 sorted(set(h.HEAT_COLORS)))
        # os mapas de componente recebem o mesmo corte, em fração do próprio intervalo
        fracao = (meio - vmin) / (vmax - vmin)
        comp_fig = c.component_figure(normed, 'freq', fracao)
        self.assertEqual([list(par) for par in comp_fig.layout.coloraxis.colorscale],
                         h.heat_colorscale(0.0, 1.0, fracao))

    def test_each_referencial_feeds_the_index_from_its_own_quad_type_only(self):
        """Fixed and rotated must never cross: each referencial's frequency grid only carries
        values from its own quad_type, never from the other one."""
        source = pd.read_csv(c.GRID_CSV)
        self.assertEqual(set(source['quad_type']), set(REFERENCIAIS))
        for referencial in REFERENCIAIS:
            other = [r for r in REFERENCIAIS if r != referencial][0]
            for band, nivel in c.BAND_TO_NIVEL.items():
                fields = c.load_frequency_grid(band, referencial)
                own = source[(source['quad_type'] == referencial) & (source['nivel'] == nivel)]
                other_rows = source[(source['quad_type'] == other) & (source['nivel'] == nivel)]
                for phase, grid in fields.items():
                    observed = set(np.round(grid[np.isfinite(grid)].ravel(), 12))
                    allowed = set(np.round(own[own['phase'] == phase]['taxa_contagem_media'], 12))
                    self.assertTrue(observed <= allowed)
                    only_other = set(np.round(
                        other_rows[other_rows['phase'] == phase]['taxa_contagem_media'], 12)) - allowed
                    self.assertFalse(observed & only_other)

    def test_composite_covers_both_referenciais(self):
        """The index is buildable end to end for fixed and rotated alike, not just fixed."""
        for referencial in REFERENCIAIS:
            with self.subTest(referencial=referencial):
                raw, normed, composite, ranges, mask = c.load_composite('p95', referencial)
                for phase in e.PHASES:
                    self.assertTrue(np.isfinite(composite[phase]).any())


if __name__ == '__main__':
    unittest.main()
