"""Integração com as grades reais adotadas; não executa geradores científicos."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))
import footprint
import persistence
from irb_data import DATA_DIR, PRODUCTS, POLY, THRESHOLDS, IRBDataError, load_product
import polygon_config

TOTALS = {
    "q90": (4330793, 3580048, 4221559), "q95": (2828203, 2248521, 2416846),
    "q99": (856089, 571392, 556980), "m156": (15385, 11750, 10904),
    "m20": (189, 96, 91), "m25": (2, 2, 2),
}


@pytest.mark.parametrize("kind", PRODUCTS)
def test_real_hash_metadata_shape(kind):
    spec = PRODUCTS[kind]
    path = DATA_DIR / spec["filename"]
    assert sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    assert json.loads(path.with_suffix(".metadata.json").read_text()) == spec
    assert list(load_product(kind).shape) == spec["shape"]


@pytest.mark.parametrize("thr", TOTALS)
def test_real_totals_and_no_occurrence(thr):
    f, p = footprint.load_grid(), persistence.load_grid()
    hc, hd, nr = TOTALS[thr]
    assert f[f"hits_{thr}_contam"].sum() == hc
    assert f[f"hits_{thr}_desc"].sum() == hd
    assert p[f"n_runs_{thr}"].sum() == nr
    assert (f[f"hits_{thr}_contam"] >= f[f"hits_{thr}_desc"]).all()
    for phase in ("contam", "desc"):
        zero = f[f"hits_{thr}_{phase}"] == 0
        assert f.loc[zero, f"media_{thr}_{phase}"].isna().all()
    zero = p[f"n_runs_{thr}"] == 0
    assert p.loc[zero, f"dur_mean_{thr}"].isna().all()
    if thr in ("m20", "m25"):
        for h in (24, 48, 72, 144):
            assert p[f"ge{h}_{thr}"].sum() == 0


def test_malvinas_real_cells():
    expected = {(-52., -59.5), (-51.75, -59.25)}
    for d, col in ((footprint.load_grid(), "hits_m25_desc"),
                   (persistence.load_grid(), "n_runs_m25")):
        sub = d[d[col] > 0]
        assert set(zip(sub.lat, sub.lon)) == expected
        assert sub[col].tolist() == [1, 1]
    d = persistence.load_grid()
    assert (d.loc[d.n_runs_m25 > 0, "dur_max_m25"] == 6).all()


@pytest.mark.parametrize("kind", PRODUCTS)
@pytest.mark.parametrize("failure", ["missing", "metadata", "hash", "missing_metadata"])
def test_fail_closed(tmp_path, kind, failure):
    spec = PRODUCTS[kind]
    path = tmp_path / spec["filename"]
    shutil.copy(DATA_DIR / path.name, path)
    meta = path.with_suffix(".metadata.json")
    shutil.copy((DATA_DIR / path.name).with_suffix(".metadata.json"), meta)
    if failure == "missing":
        path.unlink()
    elif failure == "metadata":
        value = json.loads(meta.read_text()); value["absolute_polygon"] = "b99"
        meta.write_text(json.dumps(value))
    elif failure == "hash":
        path.write_bytes(path.read_bytes() + b"invalid")
    else:
        meta.unlink()
    with pytest.raises(IRBDataError, match="Nenhum produto histórico"):
        load_product(kind, path)


def app(page):
    return AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(ROOT / 'app')!r})\n"
        f"from {page} import render_{page}\nrender_{page}()"
    ).run(timeout=30)


@pytest.mark.parametrize("page", ["footprint", "persistence"])
def test_render_all_six_thresholds(page):
    at = app(page)
    assert not at.exception
    assert len(at.selectbox[0].options) == 6
    for thr in THRESHOLDS:
        at.selectbox[0].set_value(thr).run(timeout=30)
        assert not at.exception
        assert not at.error
        assert any(POLY[thr] in c.value for c in at.caption)
        if thr in ("m20", "m25"):
            assert any("≥24 h" in i.value for i in at.info)
        if thr == "m25":
            assert any("Malvinas" in i.value for i in at.info)
        if page == "footprint":
            g = footprint.load_grid()
            removed = int((g[f"hits_{thr}_contam"] - g[f"hits_{thr}_desc"] > 0).sum())
            assert at.dataframe[0].value.iloc[2]["Células"] == footprint._br(removed)
        # Render every secondary metric, including sparse/constant 25 m/s.
        for option in (footprint.WIND_METRICS if page == "footprint" else persistence.DUR_METRICS):
            at.selectbox[1].set_value(option).run(timeout=30)
            assert not at.exception
        if page == "persistence":
            for key in persistence.BUCKETS:
                at.selectbox[2].set_value(key).run(timeout=30)
                assert not at.exception


@pytest.mark.parametrize("page", ["footprint", "persistence"])
def test_ui_missing_official_data_never_reads_historical(monkeypatch, tmp_path, page):
    module = {"footprint": footprint, "persistence": persistence}[page]
    monkeypatch.setattr(module, "GRID_PATH", tmp_path / PRODUCTS[page]["filename"])
    at = app(page)
    assert not at.exception
    assert len(at.error) == 1
    assert "Nenhum produto histórico" in at.error[0].value
    assert not at.selectbox


def test_shared_polygon_defaults():
    import footprint_grid as fg
    import persistence_grid as pg
    assert fg.ABS_POLY == pg.ABS_POLY_DEFAULT == "b99"
    assert POLY == polygon_config.polygons("b90")
    assert pg.thresholds is polygon_config.thresholds


@pytest.mark.parametrize("values", [[], [1], [1, 1], [25.08, 25.12], [1, 1, 1, 2, 3, 4, 5, 6, 7]])
def test_sparse_bands(values):
    edges, labels = footprint._auto_bands(values)
    assert len(edges) == len(labels)
    assert np.isfinite(edges).all()
    assert len(set(edges)) == len(edges)
    for page in (footprint, persistence):
        sub = pd.DataFrame({"lat": [-52.] * len(values), "lon": [-59.5] * len(values), "v": values})
        if page is footprint:
            fig = page._band_map(sub, "v", edges, page.HITS_COLORS, labels, "hits", ".0f")
        else:
            fig = page._band_map(sub, "v", str, "h", ".0f")
        assert fig.data[0].marker.cmax > fig.data[0].marker.cmin


@pytest.mark.parametrize("page", ["footprint", "persistence"])
def test_render_no_occurrence(monkeypatch, page):
    module = {"footprint": footprint, "persistence": persistence}[page]
    grid = module.load_grid().copy()
    for col in grid:
        if col not in ("lat", "lon"):
            grid[col] = 0 if col.startswith(("hits_", "n_runs_", "ge")) else np.nan
    monkeypatch.setattr(module, "load_grid", lambda: grid)
    at = app(page)
    assert not at.exception
    if page == "persistence":
        assert at.warning[0].value == "Sem eventos para este limiar."
    else:
        assert at.dataframe[0].value["Hits"].tolist() == ["0", "0", "0"]


@pytest.mark.parametrize("kind", ["footprint", "persistence"])
def test_corrected_output_requires_explicit_b90(kind):
    import footprint_grid as fg
    import persistence_grid as pg
    with pytest.raises(ValueError, match="--abs-poly b90 explicitamente"):
        if kind == "footprint":
            fg.build("unused", "fp_C_perc1979_b90_gt.parquet", "unused")
        else:
            pg.build("unused", "persistence_v2_b90.parquet", None)


def test_main_navigation_initialization():
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py")).run(timeout=30)
    assert not at.exception
    for label in ("Footprint", "Persistência do extremo"):
        at.sidebar.radio[0].set_value(label).run(timeout=30)
        assert not at.exception
        assert not at.error
        assert len(at.selectbox[0].options) == 6
