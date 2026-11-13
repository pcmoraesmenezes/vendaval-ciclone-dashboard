"""Persistência sazonal: estação pelo passo de início e reconciliação com a grade anual."""
import datetime as dt
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import persistence_grid as pg
import persistencia_sazonal as ps
import polygon_config as pc

FLAGS = ["exceeded_q90", "exceeded_q95", "exceeded_q99", "exceeded_15_6", "exceeded_20_0", "exceeded_25_0",
         "in_poly_b90", "in_poly_b95", "in_poly_b99"]


def _rows(track, start, steps, lat=-20.0, lon=-45.0):
    t0 = dt.datetime.fromisoformat(start)
    return [dict(track_id=track, time=t0 + dt.timedelta(hours=6 * s), lat=lat, lon=lon,
                 **{f: True for f in FLAGS}) for s in steps]


@pytest.fixture
def data(tmp_path, monkeypatch):
    rows = (_rows(1, "2000-02-28 00:00", range(0, 12))      # começa em DJF, termina em MAM: 72 h
            + _rows(2, "2000-07-10 00:00", range(0, 3))      # JJA: 18 h
            + _rows(3, "2000-12-31 18:00", range(0, 2))      # DJF: 12 h
            + _rows(4, "2000-04-01 00:00", range(0, 2), lat=-30.0, lon=-50.0))
    src = tmp_path / "serie.parquet"
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype("datetime64[ms]")
    df.to_parquet(src)
    mask = tmp_path / "terra.csv"
    pd.DataFrame({"lat": [-20.0, -30.0], "lon": [-45.0, -50.0]}).to_csv(mask, index=False)
    ref = tmp_path / "anual.parquet"
    pg.build(str(src), str(ref), str(tmp_path / "terra.parquet") if _mask_parquet(tmp_path) else None, "b90")
    monkeypatch.setattr(pc, "PERSISTENCE_SRC_SHA256", pc.hashlib.sha256(src.read_bytes()).hexdigest()[:16])
    return src, mask, ref


def _mask_parquet(tmp_path):
    pd.DataFrame({"lat": [-20.0, -30.0], "lon": [-45.0, -50.0]}).to_parquet(tmp_path / "terra.parquet")
    return True


def test_season_of_uses_meteorological_seasons():
    t = np.array(["2000-01-15", "2000-02-29", "2000-03-01", "2000-06-30", "2000-09-01", "2000-11-30", "2000-12-01"],
                 dtype="datetime64[s]")
    t6 = (t.astype("int64") // (6 * 3600)).astype("int32")
    assert ps.season_of(t6).tolist() == ["DJF", "DJF", "MAM", "JJA", "SON", "SON", "DJF"]


def test_event_belongs_to_the_season_it_starts_in(tmp_path, data):
    src, mask, ref = data
    out = tmp_path / "sazonal.parquet"
    ps.build(str(src), str(out), str(mask), str(ref))
    g = pd.read_parquet(out)
    cell = g[(g.lat == -20.0) & (g.lon == -45.0)].set_index("season")
    assert cell.loc["DJF", "n_runs_q90"] == 2 and cell.loc["DJF", "dur_max_q90"] == 72
    assert cell.loc["JJA", "n_runs_q90"] == 1 and cell.loc["JJA", "dur_max_q90"] == 18
    assert "MAM" not in cell.index
    assert g.attrs["scientific_parameters"]["event_season"] == "start step"
    assert list(g.columns[:3]) == ["lat", "lon", "season"]


def test_nothing_is_written_when_reconciliation_fails(tmp_path, data):
    src, mask, ref = data
    wrong = pd.read_parquet(ref)
    wrong.loc[0, "n_runs_q90"] += 1
    bad_ref = tmp_path / "errada.parquet"
    wrong.to_parquet(bad_ref)
    out = tmp_path / "sazonal.parquet"
    with pytest.raises(ValueError, match="RECONCILIACAO FALHOU"):
        ps.build(str(src), str(out), str(mask), str(bad_ref))
    assert not out.exists() and not Path(str(out) + ".tmp").exists()


def test_series_must_be_the_official_one(tmp_path, data, monkeypatch):
    src, mask, ref = data
    monkeypatch.setattr(pc, "PERSISTENCE_SRC_SHA256", "0" * 16)
    with pytest.raises(ValueError, match="descontaminada v2"):
        ps.build(str(src), str(tmp_path / "sazonal.parquet"), str(mask), str(ref))


def test_existing_output_is_refused(tmp_path, data):
    src, mask, ref = data
    out = tmp_path / "sazonal.parquet"
    out.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        ps.build(str(src), str(out), str(mask), str(ref))
