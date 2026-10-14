"""Corretude do filtro, da resolução de caminho e do carregamento da página de export.

O dado sintético inclui fase nula e fase secundária ('decay 2'), que existem no arquivo
real: sem isso o teste não exercita o caminho que quebrava (`sorted` sobre None).
"""
import datetime as dt
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import exceedances_export as x

ROWS = [
    (dt.datetime(1979, 6, 1), 'mature', 100.0, True, False),
    (dt.datetime(1995, 1, 1), 'decay', 500.0, True, True),
    (dt.datetime(2020, 12, 31), 'mature', 1050.0, True, False),
    (dt.datetime(2005, 1, 1), 'mature', 900.0, False, False),
    (dt.datetime(2010, 1, 1), 'incipient', 200.0, True, True),
    (dt.datetime(2012, 1, 1), None, 150.0, True, False),
    (dt.datetime(2015, 1, 1), 'decay 2', 300.0, True, True),
]


def _tiny_parquet(path):
    data = {c: [] for c in x.COLUMNS}
    for i, (time, phase, dist, e156, eq99) in enumerate(ROWS):
        data['track_id'].append(i)
        data['time'].append(time)
        data['phase'].append(phase)
        for c in ('lat_center', 'lon_center', 'lat', 'lon', 'wind_speed'):
            data[c].append(float(i))
        data['distance_km'].append(dist)
        data['fixed_quadrant'].append(1)
        data['rotated_quadrant'].append(1)
        data['exceeded_15_6'].append(e156)
        for c in ('exceeded_20_0', 'exceeded_25_0', 'exceeded_q90', 'exceeded_q95'):
            data[c].append(False)
        data['exceeded_q99'].append(eq99)
    schema = pa.schema([(c, pa.string() if c == 'phase' else
                         pa.timestamp('ms') if c == 'time' else
                         pa.bool_() if c.startswith('exceeded_') else
                         pa.int64() if c in ('track_id', 'fixed_quadrant', 'rotated_quadrant')
                         else pa.float32()) for c in x.COLUMNS])
    pq.write_table(pa.table(data, schema=schema), path)


ALL_NAMED = ['decay', 'decay 2', 'incipient', 'mature']


class ExportContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = str(Path(cls.tmp.name) / 'exc.parquet')
        _tiny_parquet(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _count(self, threshold, phases, include_null, years, dmax):
        return x._dataset(self.path).count_rows(
            filter=x._filter(threshold, phases, include_null, years, dmax))

    def test_phase_options_drops_null_and_sorts(self):
        phases, has_null = x._phase_options(self.path)
        self.assertEqual(phases, ALL_NAMED)
        self.assertTrue(has_null)

    def test_resolve_path_env_override(self):
        os.environ[x.ENV_VAR] = self.path
        try:
            self.assertEqual(str(x.resolve_path()), self.path)
        finally:
            del os.environ[x.ENV_VAR]

    def test_resolve_path_absent(self):
        os.environ.pop(x.ENV_VAR, None)
        if any(p.exists() for p in x.DEFAULT_PATHS):
            self.skipTest('parquet real presente na máquina')
        self.assertIsNone(x.resolve_path())

    def test_all_phases_with_null_keeps_the_null_phase_row(self):
        self.assertEqual(self._count('exceeded_15_6', ALL_NAMED, True, (1979, 2020), 1100), 6)

    def test_excluding_null_drops_only_the_unclassified_row(self):
        self.assertEqual(self._count('exceeded_15_6', ALL_NAMED, False, (1979, 2020), 1100), 5)

    def test_only_null_phase(self):
        self.assertEqual(self._count('exceeded_15_6', [], True, (1979, 2020), 1100), 1)

    def test_empty_selection_and_no_null_is_nothing(self):
        self.assertEqual(self._count('exceeded_15_6', [], False, (1979, 2020), 1100), 0)

    def test_distance_cap_excludes_far_points(self):
        self.assertEqual(self._count('exceeded_15_6', ALL_NAMED, True, (1979, 2020), 1000), 5)

    def test_year_window_is_inclusive_on_both_ends(self):
        self.assertEqual(self._count('exceeded_15_6', ALL_NAMED, True, (1979, 1979), 1100), 1)
        self.assertEqual(self._count('exceeded_15_6', ALL_NAMED, True, (2020, 2020), 1100), 1)

    def test_secondary_phase_is_selectable(self):
        self.assertEqual(self._count('exceeded_15_6', ['decay 2'], False, (1979, 2020), 1100), 1)

    def test_q99_threshold(self):
        self.assertEqual(self._count('exceeded_q99', ALL_NAMED, True, (1979, 2020), 1100), 3)

    def test_write_slice_streams_filtered_rows_to_disk(self):
        for fmt, reader in (('Parquet', pq.read_table),
                            ('CSV', lambda p: pacsv.read_csv(p))):
            with self.subTest(fmt=fmt):
                out = Path(self.tmp.name) / f'slice.{fmt.lower()}'
                x._write_slice(self.path, 'exceeded_q99', tuple(ALL_NAMED), True,
                               (1979, 2020), 1100, fmt, str(out))
                back = reader(str(out))
                self.assertEqual(back.num_rows, 3)
                self.assertEqual(set(back.column_names), set(x.COLUMNS))
                self.assertTrue(all(back.column('exceeded_q99').to_pylist()))


class ExportPageAppTest(unittest.TestCase):
    def test_page_renders_without_exception_on_data_with_null_phase(self):
        from streamlit.testing.v1 import AppTest
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = str(Path(tmp.name) / 'exc.parquet')
        _tiny_parquet(path)
        os.environ[x.ENV_VAR] = path
        self.addCleanup(lambda: os.environ.pop(x.ENV_VAR, None))
        app = Path(__file__).resolve().parents[1] / 'app' / 'streamlit_app.py'
        at = AppTest.from_file(str(app), default_timeout=60)
        at.session_state['nav_choice'] = '📤 Exportar excedentes'
        at.run()
        self.assertFalse(at.exception, msg=str(at.exception))


if __name__ == '__main__':
    unittest.main()
