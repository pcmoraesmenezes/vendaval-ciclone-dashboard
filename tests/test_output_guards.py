"""Travas dos geradores: caminho de saída e parâmetros das saídas corrigidas."""
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import types

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import footprint_grid as fg
import persistence_grid as pg
import polygon_config as pc


@pytest.fixture
def parquet(tmp_path):
    path = tmp_path / "entrada.parquet"
    pd.DataFrame({"lat": [-20.0, -21.0, -22.0], "lon": [-45.0, -46.0, -47.0]}).to_parquet(path)
    return path


def test_existing_output_is_refused_and_left_intact(tmp_path, parquet):
    out = tmp_path / "grade.parquet"
    out.write_bytes(b"historico")
    for build in (lambda: pg.build(str(parquet), str(out), None),
                  lambda: fg.build(str(parquet), str(out), "unused")):
        with pytest.raises(FileExistsError):
            build()
    assert out.read_bytes() == b"historico"


def test_adopted_products_folder_is_never_written(tmp_path):
    out = tmp_path / "irb_corrigido" / "novo.parquet"
    with pytest.raises(ValueError, match="não recebe gravações"):
        pc.validate_output_path(out)


def test_output_taken_during_a_run_is_not_replaced(tmp_path):
    tmp, out = tmp_path / "x.tmp", tmp_path / "x.parquet"
    tmp.write_bytes(b"novo")
    out.write_bytes(b"existente")
    with pytest.raises(FileExistsError):
        pc.publish_output(tmp, out)
    assert out.read_bytes() == b"existente" and not tmp.exists()


def test_new_output_is_published(tmp_path):
    tmp, out = tmp_path / "x.tmp", tmp_path / "x.parquet"
    tmp.write_bytes(b"novo")
    pc.publish_output(tmp, out)
    assert out.read_bytes() == b"novo" and not tmp.exists()


@pytest.mark.parametrize("script", ["persistence_grid.py", "footprint_grid.py"])
def test_out_is_required(script):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], capture_output=True, text=True)
    assert r.returncode == 2 and "--out" in r.stderr


def test_corrected_footprint_requires_strict_comparison(tmp_path):
    with pytest.raises(ValueError, match="--estrito"):
        fg.build("unused", str(tmp_path / "fp_b90.parquet"), "unused", estrito=False, abs_poly="b90")


def test_corrected_footprint_requires_original_raw_rows(tmp_path, parquet):
    with pytest.raises(ValueError, match="linhas"):
        pc.validate_corrected_footprint(parquet, "unused", True)


@pytest.mark.parametrize("description, ok", [
    ("Percentis locais de vento, período histórico 2010-2020. Níveis: [50, 99]", False),
    ("Percentis locais de vento, período histórico 1979-2020. Níveis: [50, 99]", True),
    ("", False),
])
def test_corrected_footprint_requires_percentiles_1979_2020(monkeypatch, parquet, description, ok):
    monkeypatch.setattr(pc, "FOOTPRINT_RAW_ROWS", 3)
    import xarray
    ds = types.SimpleNamespace(attrs={"description": description})
    monkeypatch.setattr(xarray, "open_dataset", lambda p: _Ctx(ds))
    if ok:
        pc.validate_corrected_footprint(parquet, "perc.nc", True)
    else:
        with pytest.raises(ValueError, match="1979-2020"):
            pc.validate_corrected_footprint(parquet, "perc.nc", True)


class _Ctx:
    def __init__(self, ds):
        self.ds = ds

    def __enter__(self):
        return self.ds

    def __exit__(self, *a):
        return False


def test_corrected_persistence_requires_official_inputs(monkeypatch, tmp_path, parquet):
    mask = tmp_path / "mask.parquet"
    mask.write_bytes(b"mascara")
    with pytest.raises(ValueError, match="máscara de terra"):
        pc.validate_corrected_persistence(parquet, None)
    with pytest.raises(ValueError, match="descontaminada v2"):
        pc.validate_corrected_persistence(parquet, mask)
    monkeypatch.setattr(pc, "PERSISTENCE_SRC_SHA256", sha256(parquet.read_bytes()).hexdigest()[:16])
    with pytest.raises(ValueError, match="Máscara"):
        pc.validate_corrected_persistence(parquet, mask)
    monkeypatch.setattr(pc, "CONTINENTAL_MASK_SHA256", sha256(b"mascara").hexdigest()[:16])
    pc.validate_corrected_persistence(parquet, mask)


def test_corrected_persistence_build_stops_before_reading_data(tmp_path, parquet):
    with pytest.raises(ValueError, match="descontaminada v2"):
        pg.build(str(parquet), str(tmp_path / "persistence_b90.parquet"), str(parquet), "b90")


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_events_do_not_depend_on_timestamp_unit(tmp_path, unit):
    import datetime as dt
    flags = {c: True for c in ("exceeded_q90", "in_poly_b90")}
    rows = [dict(track_id=1, time=dt.datetime(2000, 1, 1) + dt.timedelta(hours=6 * s),
                 lat=-20.0, lon=-45.0, **flags) for s in (0, 1, 2)]
    df = pd.DataFrame(rows)
    df["time"] = df["time"].astype(f"datetime64[{unit}]")
    src = tmp_path / "src.parquet"
    df.to_parquet(src)
    _, _, dur = pg._runs(str(src), "exceeded_q90", "in_poly_b90", None)
    assert dur.tolist() == [18]
