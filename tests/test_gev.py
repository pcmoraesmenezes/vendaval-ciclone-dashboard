"""Grade do ajuste GEV: contrato dos dados, fórmula, corte, travas do gerador e a página."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))
import gev
import gev_grid
from irb_data import PRODUCTS


@pytest.fixture(scope="module")
def grid():
    return gev.load_grid()


def synthetic(rows):
    """Tabela de origem com os pontos dados como (lat, lon, location, scale, shape)."""
    data = []
    for lat, lon, location, scale, shape in rows:
        levels = gev_grid.expected_level(location, scale, shape, np.array(gev_grid.PERIODS))
        data.append([lat, lon, location, scale, shape, *levels])
    return pd.DataFrame(data, columns=gev_grid.SOURCE_COLUMNS)


def test_every_level_follows_the_gev_formula(grid):
    for period in gev_grid.PERIODS:
        calc = gev_grid.expected_level(grid.location, grid.scale, grid["shape"], period)
        assert (np.abs(calc - grid[f"NR_{period}_estimado"]) <= gev_grid.FORMULA_TOLERANCE).all()


def test_cut_point_is_absent_and_the_rest_is_kept(grid):
    assert len(grid) == 9834
    for lat, lon in gev_grid.CUT_POINTS:
        assert not ((grid.lat == lat) & (grid.lon == lon)).any()
    assert list(PRODUCTS["gev"]["cut_points"]) == [list(p) for p in gev_grid.CUT_POINTS]


def test_flags_and_coverage(grid):
    assert int(grid.forma_alta.sum()) == 21
    assert (grid.forma_alta == (grid["shape"] > gev_grid.HIGH_SHAPE)).all()
    assert not grid.duplicated(["lat", "lon"]).any()
    levels = grid[[f"NR_{p}_estimado" for p in gev_grid.PERIODS]].to_numpy()
    assert (np.diff(levels, axis=1) > 0).all()


def test_prepare_cuts_the_listed_point_and_marks_high_shape(monkeypatch):
    table = synthetic([(0.0, 0.0, 8.0, 1.0, -0.1), (0.25, 0.0, 8.0, 1.0, 0.7), (1.0, 1.0, 9.0, 1.0, 0.2)])
    monkeypatch.setattr(gev_grid, "CUT_POINTS", ((1.0, 1.0),))
    out = gev_grid.prepare(table)
    assert out[["lat", "lon"]].values.tolist() == [[0.0, 0.0], [0.25, 0.0]]
    assert out.forma_alta.tolist() == [False, True]


def test_prepare_refuses_a_level_off_the_formula_that_was_not_cut(monkeypatch):
    table = synthetic([(0.0, 0.0, 8.0, 1.0, 0.2), (1.0, 1.0, 9.0, 1.0, 0.2)])
    table.loc[0, "NR_100_estimado"] += 5
    monkeypatch.setattr(gev_grid, "CUT_POINTS", ((1.0, 1.0),))
    with pytest.raises(ValueError, match="fora da fórmula depois do corte"):
        gev_grid.prepare(table)


def test_prepare_refuses_when_the_cut_point_is_missing():
    with pytest.raises(ValueError, match="não está na tabela"):
        gev_grid.prepare(synthetic([(0.0, 0.0, 8.0, 1.0, 0.2)]))


@pytest.mark.parametrize("shape", [-0.3, 0.0, 1e-12, 0.4])
def test_expected_level_is_continuous_at_zero_shape(shape):
    gumbel = gev_grid.expected_level(8.0, 1.0, 0.0, 50)
    near = gev_grid.expected_level(8.0, 1.0, shape, 50)
    assert np.isfinite(near)
    if abs(shape) < 1e-9:
        assert near == pytest.approx(gumbel)


def test_gumbel_matches_known_value():
    # nível de 100 anos da Gumbel padrão: -ln(-ln(1 - 1/100)) = 4,6001
    assert gev_grid.expected_level(0.0, 1.0, 0.0, 100) == pytest.approx(4.60015, abs=1e-4)


def test_generator_refuses_existing_output_and_wrong_source(tmp_path):
    src = tmp_path / "entrada.parquet"
    pd.DataFrame({"lat": [0.0]}).to_parquet(src)
    out = tmp_path / "grade.parquet"
    out.write_bytes(b"existente")
    with pytest.raises(FileExistsError):
        gev_grid.build(str(src), str(out))
    assert out.read_bytes() == b"existente"
    with pytest.raises(ValueError, match="SHA256 da tabela do ajuste GEV"):
        gev_grid.build(str(src), str(tmp_path / "nova.parquet"))
    with pytest.raises(ValueError, match="não recebe gravações"):
        gev_grid.build(str(src), str(tmp_path / "irb_corrigido" / "nova.parquet"))


def test_generator_requires_out():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gev_grid.py"), "--src", "x"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--out" in r.stderr


def app():
    return AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(ROOT / 'app')!r})\n"
        "from gev import render_gev\nrender_gev()"
    ).run(timeout=30)


def test_every_period_and_parameter_renders():
    at = app()
    assert not at.exception and not at.error and not at.warning
    assert [int(o.split()[0]) for o in at.selectbox[0].options] == gev.PERIODS
    for period in gev.PERIODS:
        at.selectbox[0].set_value(period).run(timeout=30)
        assert not at.exception and not at.error
    for param in gev.PARAMS:
        at.selectbox[1].set_value(param).run(timeout=30)
        assert not at.exception and not at.error
    assert any("21 pontos" in c.value for c in at.caption)
    assert any("9.834 pontos" in c.value for c in at.caption)


def test_flagged_points_are_marked_not_removed(grid):
    for fig in (gev.level_map(grid, 100), gev.param_map(grid, "shape"), gev.param_map(grid, "scale")):
        main, flagged = fig.data
        assert len(main.lon) == len(grid) == 9834
        assert len(flagged.lon) == 21 and flagged.marker.symbol == "x"


def test_level_scale_is_shared_across_periods(grid):
    ticks = {tuple(gev.level_map(grid, p).data[0].marker.colorbar.ticktext) for p in gev.PERIODS}
    assert len(ticks) == 1


def test_missing_data_never_reads_anything_else(monkeypatch, tmp_path):
    monkeypatch.setattr(gev, "GRID_PATH", tmp_path / PRODUCTS["gev"]["filename"])
    at = app()
    assert not at.exception
    assert len(at.error) == 1 and "Nenhum produto histórico" in at.error[0].value
    assert not at.selectbox


def test_page_text_has_no_names_dates_or_dashes():
    at = app()
    text = " ".join(str(e.value) for e in (*at.caption, *at.markdown, *at.header, *at.subheader, *at.warning))
    text += gev._method_text()
    assert "—" not in text and "–" not in text.replace("1979–2020", "")
    for forbidden in ("Tavares", "Danilo", "Paulo", "Carol", "Emanuel", "novembro", "2026"):
        assert forbidden not in text


def test_method_text_tells_that_a_point_was_removed():
    assert "Ponto retirado" in gev._method_text()


def test_main_navigation_has_the_page():
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py")).run(timeout=30)
    at.sidebar.radio[0].set_value("Nível de retorno (GEV)").run(timeout=30)
    assert not at.exception and not at.error
    assert len(at.selectbox[0].options) == len(gev.PERIODS)


def test_page_renders_without_flagged_points(monkeypatch, grid):
    clean = grid.assign(forma_alta=False)
    monkeypatch.setattr(gev, "load_grid", lambda: clean)
    for fig in (gev.level_map(clean, 50), gev.param_map(clean, "shape")):
        assert len(fig.data[1].lon) == 0
    at = app()
    assert not at.exception and not at.error
    assert any("(0 pontos)" in c.value for c in at.caption)
