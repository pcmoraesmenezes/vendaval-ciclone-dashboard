"""Pareamento dos limiares absolutos na persistencia (--abs-poly) e regra temporal inalterada.

Dado sintetico com pontos em que in_poly_b90 e verdadeiro e in_poly_b99 e falso: so o pareamento
b90 deve contar eventos absolutos nesses pontos; os percentis devem sair identicos qualquer que
seja a opcao. A regra temporal (passo de 6 h, um passo ausente tolerado) e verificada com
sequencias conhecidas.
"""
import datetime as dt
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import persistence_grid as pg

T0 = dt.datetime(2000, 1, 1)
ORIGINAL = {  # pareamento padrao da linha de comando
    "q90": ("exceeded_q90", "in_poly_b90"),
    "q95": ("exceeded_q95", "in_poly_b95"),
    "q99": ("exceeded_q99", "in_poly_b99"),
    "m156": ("exceeded_15_6", "in_poly_b99"),
    "m20": ("exceeded_20_0", "in_poly_b99"),
    "m25": ("exceeded_25_0", "in_poly_b99"),
}


def _row(track, step, lat, lon, b90, b95, b99, abs_exc=True, pct_exc=True):
    return {
        "track_id": track, "time": T0 + dt.timedelta(hours=6 * step), "lat": lat, "lon": lon,
        "exceeded_q90": pct_exc, "exceeded_q95": pct_exc, "exceeded_q99": pct_exc,
        "exceeded_15_6": abs_exc, "exceeded_20_0": abs_exc, "exceeded_25_0": abs_exc,
        "in_poly_b90": b90, "in_poly_b95": b95, "in_poly_b99": b99,
    }


def _write(rows, path):
    df = pd.DataFrame(rows)
    tbl = pa.Table.from_pandas(df, preserve_index=False).cast(pa.schema([
        ("track_id", pa.int64()), ("time", pa.timestamp("ms")), ("lat", pa.float64()),
        ("lon", pa.float64()), *[(c, pa.bool_()) for c in df.columns[4:]]]))
    pq.write_table(tbl, path)


class TestPareamento(unittest.TestCase):
    def test_padrao_reproduz_original(self):
        self.assertEqual(pg.thresholds(), ORIGINAL)
        self.assertEqual(pg.THRESHOLDS, ORIGINAL)
        self.assertEqual(pg.thresholds("b99"), ORIGINAL)

    def test_absolutos_seguem_abs_poly(self):
        for band in ("b90", "b95", "b99"):
            m = pg.thresholds(band)
            for k, flag in (("m156", "exceeded_15_6"), ("m20", "exceeded_20_0"),
                            ("m25", "exceeded_25_0")):
                self.assertEqual(m[k], (flag, f"in_poly_{band}"), (band, k))

    def test_percentis_nao_mudam(self):
        for band in ("b90", "b95", "b99"):
            m = pg.thresholds(band)
            self.assertEqual(m["q90"], ("exceeded_q90", "in_poly_b90"))
            self.assertEqual(m["q95"], ("exceeded_q95", "in_poly_b95"))
            self.assertEqual(m["q99"], ("exceeded_q99", "in_poly_b99"))

    def test_banda_invalida(self):
        with self.assertRaises(ValueError):
            pg.thresholds("b80")


class TestRegraTemporal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "src.parquet"
        rows = []
        # Celula A (-20, -45): dentro de b90 e b99. Track 1: passos 0,1,3 (1 buraco) -> 1 evento 24 h.
        for s in (0, 1, 3):
            rows.append(_row(1, s, -20.0, -45.0, True, True, True))
        # Track 1: passos 10 e 13 (2 buracos) -> 2 eventos de 6 h.
        for s in (10, 13):
            rows.append(_row(1, s, -20.0, -45.0, True, True, True))
        # Track 2 no mesmo ponto e passos 0..1: ciclone diferente -> evento proprio 12 h.
        for s in (0, 1):
            rows.append(_row(2, s, -20.0, -45.0, True, True, True))
        # Celula B (-21, -46): so b90 (fora de b95/b99); absolutos e percentis excedidos.
        for s in (0, 1, 2):
            rows.append(_row(3, s, -21.0, -46.0, True, False, False))
        _write(rows, self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_eventos_celula_a(self):
        lat, lon, dur = pg._runs(str(self.src), "exceeded_15_6", "in_poly_b99", None)
        got = sorted(zip(lat.tolist(), lon.tolist(), dur.tolist()))
        self.assertEqual(got, [(-20.0, -45.0, 6), (-20.0, -45.0, 6),
                               (-20.0, -45.0, 12), (-20.0, -45.0, 24)])

    def test_b90_inclui_celula_b_e_b99_nao(self):
        _, _, d99 = pg._runs(str(self.src), "exceeded_15_6", "in_poly_b99", None)
        lat, lon, d90 = pg._runs(str(self.src), "exceeded_15_6", "in_poly_b90", None)
        self.assertEqual(len(d99), 4)
        self.assertEqual(len(d90), 5)
        b = [d for la, lo, d in zip(lat, lon, d90) if la == -21.0 and lo == -46.0]
        self.assertEqual(b, [18])

    def test_build_percentis_identicos_e_absolutos_mudam(self):
        out99 = Path(self.tmp.name) / "p99.parquet"
        out90 = Path(self.tmp.name) / "p90.parquet"
        pg.build(str(self.src), str(out99), None, "b99")
        pg.build(str(self.src), str(out90), None, "b90")
        self.assertEqual(pd.read_parquet(out90).attrs["scientific_parameters"]["absolute_polygon"], "b90")
        self.assertEqual(pd.read_parquet(out99).attrs["scientific_parameters"]["absolute_polygon"], "b99")
        g99 = pd.read_parquet(out99).set_index(["lat", "lon"]).sort_index()
        g90 = pd.read_parquet(out90).set_index(["lat", "lon"]).sort_index()
        # colunas cujo sufixo de limiar e percentil (ex.: dur_mean_q95, n_runs_q99)
        pct = sorted(set(c for c in g99.columns if c.rsplit("_", 1)[-1] in ("q90", "q95", "q99")))
        pd.testing.assert_frame_equal(g99[pct], g90[pct])
        for thr in ("m156", "m20", "m25"):
            self.assertEqual(int(g99[f"n_runs_{thr}"].sum()), 4)
            self.assertEqual(int(g90[f"n_runs_{thr}"].sum()), 5)
            self.assertEqual(int(g90.loc[(-21.0, -46.0), f"n_runs_{thr}"]), 1)
            self.assertEqual(int(g99.loc[(-21.0, -46.0), f"n_runs_{thr}"]), 0)


if __name__ == "__main__":
    unittest.main()
