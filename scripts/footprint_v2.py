"""Produtos do footprint a partir do by_track e dos percentis locais do mesmo periodo.

  python scripts/footprint_v2.py --src <by_track.parquet> --perc <local_percentiles.nc> \
      --terra <celulas_terra.csv> --out <pasta> [--tracks <catalogo.csv> --sufixo v3]

Escreve em --out, sem sobrescrever arquivo existente (sufixo padrao v2):
  cyclone_exceedances_by_track_1979_2020_completo_<sufixo>.parquet   by_track + in_poly_b* + frac_b*
  cyclone_exceedances_descontaminado_1979_2020_<sufixo>.parquet      linhas com in_poly em ao menos uma banda
  footprint_hits_continente_1979_2020_<sufixo>.parquet               linhas em celula de terra, com in_poly_b*

Com --tracks, o deslocamento do ciclone vem da trajetoria completa (regra do gerador da M9); sem ele,
das linhas presentes no by_track.

in_poly_b*: pertinencia ao poligono do indice composto (rotacionado, nivel 0,5), calculada por
filtro_poligono_composto. frac_b*: fracao dos niveis da banda cujo percentil local o vento excede
(b90 = q80..q90, b95 = q89..q95, b99 = q95..q99), comparacao estrita.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filtro_poligono_composto as F  # noqa: E402

BAND_LEVELS = {"b90": range(80, 91), "b95": range(89, 96), "b99": range(95, 100)}
KEYS = ["track_id", "time", "phase", "lat_center", "lon_center", "lat", "lon", "wind_speed"]
HIT_COLS = ["track_id", "time", "phase", "lat_center", "lon_center", "lat", "lon", "wind_speed",
            "distance_km"]


class Percentis:
    def __init__(self, path):
        ds = xr.open_dataset(path, engine="netcdf4")
        self.lat, self.lon = ds["latitude"].values, ds["longitude"].values
        self.q = {k: ds[k].values for k in ds.data_vars}
        ds.close()

    def indices(self, lat, lon):
        yi = np.rint((self.lat[0] - lat.astype("float64")) / 0.25).astype(int)
        xi = np.rint((lon.astype("float64") - self.lon[0]) / 0.25).astype(int)
        assert np.abs(self.lat[yi] - lat).max() < 1e-3 and np.abs(self.lon[xi] - lon).max() < 1e-3
        return yi, xi


def ler_ano(src, ano, colunas=None):
    lo, hi = pd.Timestamp(f"{ano}-01-01"), pd.Timestamp(f"{ano + 1}-01-01")
    return pq.read_table(src, columns=colunas, filters=[("time", ">=", lo), ("time", "<", hi)])


def vetores_de_deslocamento(src):
    """(track_id, time) -> vetor unitario leste/norte pela diferenca central dos instantes do arquivo.
    Movimento nulo vira (0, 0): o ponto cai na celula central da matriz."""
    partes = []
    for b in pq.ParquetFile(src).iter_batches(batch_size=2_000_000,
                                              columns=["track_id", "time", "lat_center", "lon_center"]):
        partes.append(b.to_pandas().drop_duplicates(["track_id", "time"]))
    cen = pd.concat(partes, ignore_index=True).drop_duplicates(["track_id", "time"])
    cen = cen.sort_values(["track_id", "time"], kind="stable").reset_index(drop=True)
    ux_all, uy_all = np.empty(len(cen)), np.empty(len(cen))
    la, lo = cen["lat_center"].values.astype("float64"), cen["lon_center"].values.astype("float64")
    ids = cen["track_id"].values
    limites = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    for a, b in zip(limites[:-1], limites[1:]):
        lats, lons = la[a:b], lo[a:b]
        dlat = np.gradient(lats) if len(lats) > 1 else np.zeros_like(lats)
        dlon = np.gradient(lons) if len(lons) > 1 else np.zeros_like(lons)
        ux, uy = dlon * np.cos(np.radians(lats)), dlat
        norma = np.hypot(ux, uy)
        norma[norma == 0] = 1.0
        ux_all[a:b], uy_all[a:b] = ux / norma, uy / norma
    return pd.DataFrame({"track_id": ids, "time": cen["time"].values, "ux": ux_all, "uy": uy_all})


def vetores_da_trajetoria(tracks_csv):
    """Mesma regra do gerador da M9: trajetoria completa em passos de 6 h, por ano, diferenca
    central com recuo para diferenca unilateral nas pontas e vetor (1, 0) quando o movimento e nulo."""
    tr_all = pd.read_csv(tracks_csv, usecols=["track_id", "date", "lon vor", "lat vor"])
    tr_all["date"] = pd.to_datetime(tr_all["date"])
    tr_all["lon vor"] = ((tr_all["lon vor"] + 180) % 360) - 180
    partes = []
    for ano, tr in tr_all.groupby(tr_all["date"].dt.year):
        tempos = pd.date_range(f"{ano}-01-01", f"{ano}-12-31 18:00", freq="6h").values
        idx = np.clip(np.searchsorted(tempos, tr["date"].values), 1, len(tempos) - 1)
        esq, dir_ = tempos[idx - 1], tempos[idx]
        idx = np.where((tr["date"].values - esq) <= (dir_ - tr["date"].values), idx - 1, idx)
        tr = tr.assign(t_idx=idx, dt=np.abs(tr["date"].values - tempos[idx]))
        tr = tr.sort_values("dt").drop_duplicates(["track_id", "t_idx"]).sort_values(["track_id", "t_idx"])
        g = tr.groupby("track_id")
        dlon = g["lon vor"].shift(-1) - g["lon vor"].shift(1)
        dlat = g["lat vor"].shift(-1) - g["lat vor"].shift(1)
        dlon = dlon.fillna(g["lon vor"].diff()).fillna(g["lon vor"].diff(-1).mul(-1))
        dlat = dlat.fillna(g["lat vor"].diff()).fillna(g["lat vor"].diff(-1).mul(-1))
        dx = (((dlon + 180) % 360) - 180) * np.cos(np.radians(tr["lat vor"]))
        mag = np.sqrt(dx ** 2 + dlat ** 2).replace(0, np.nan)
        partes.append(pd.DataFrame({"track_id": tr["track_id"].values,
                                    "time": tempos[tr["t_idx"].values].astype("datetime64[s]"),
                                    "ux": (dx / mag).fillna(1.0).values,
                                    "uy": (dlat / mag).fillna(0.0).values}))
    return pd.concat(partes, ignore_index=True)


def pertinencia(df, uv, polys, axis):
    d = df[["track_id", "time"]].merge(uv, on=["track_id", "time"], how="left")
    lat, lon, latc, lonc = (df[c].values for c in ("lat", "lon", "lat_center", "lon_center"))
    dlon = (lon - lonc + 180) % 360 - 180
    leste = dlon * F.KM_PER_DEG * np.cos(np.radians((lat + latc) / 2))
    norte = (lat - latc) * F.KM_PER_DEG
    ux, uy = d["ux"].values, d["uy"].values
    frente, direita = leste * ux + norte * uy, leste * uy - norte * ux
    ok = np.isfinite(frente)
    lin, col = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)
    lin[ok], col[ok] = F._cell(frente[ok], axis), F._cell(direita[ok], axis)
    unicas = df["phase"].unique().tolist()
    codigos, nomes = F._base_phase_codes(unicas)
    base = df["phase"].map(dict(zip(unicas, codigos.tolist()))).values.astype(int)
    valido = (base >= 0) & ok
    saida = {}
    for banda in polys:
        dentro = np.zeros(len(df), dtype=bool)
        for cod, fase in enumerate(nomes):
            sel = valido & (base == cod)
            if sel.any():
                dentro[sel] = polys[banda][fase][lin[sel], col[sel]]
        saida[banda] = dentro
    return saida


def fracoes(ws, perc, yi, xi):
    out = {}
    for banda, niveis in BAND_LEVELS.items():
        acima = sum((ws > perc.q[f"q{n}"][yi, xi]).astype("int8") for n in niveis)
        out[banda] = (acima / len(niveis)).astype("float32")
    return out


def gerar(src, perc_path, terra_path, out, tracks=None, sufixo="v2"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destinos = {
        "completo": out / f"cyclone_exceedances_by_track_1979_2020_completo_{sufixo}.parquet",
        "desc": out / f"cyclone_exceedances_descontaminado_1979_2020_{sufixo}.parquet",
        "hits": out / f"footprint_hits_continente_1979_2020_{sufixo}.parquet",
    }
    for p in destinos.values():
        if p.exists():
            raise SystemExit(f"ja existe, nao sobrescrevo: {p}")

    perc = Percentis(perc_path)
    terra = pd.read_csv(terra_path)
    ty, tx = perc.indices(terra["lat"].values.astype("float32"), terra["lon"].values.astype("float32"))
    mascara = np.zeros((len(perc.lat), len(perc.lon)), dtype=bool)
    mascara[ty, tx] = True
    polys, axis = F.load_polygons()
    uv = vetores_da_trajetoria(tracks) if tracks else vetores_de_deslocamento(src)

    escritores, total, contagem = {}, 0, {k: 0 for k in ("completo", "desc", "hits")}
    anos = range(1979, 2021)
    for ano in anos:
        tbl = ler_ano(src, ano)
        df = tbl.select(KEYS).to_pandas()
        dentro = pertinencia(df, uv, polys, axis)
        yi, xi = perc.indices(df["lat"].values, df["lon"].values)
        fr = fracoes(df["wind_speed"].values, perc, yi, xi)
        for b, m in dentro.items():
            tbl = tbl.append_column(f"in_poly_{b}", pa.array(m))
        for b, v in fr.items():
            tbl = tbl.append_column(f"frac_{b}", pa.array(v))
        qualquer = np.logical_or.reduce(list(dentro.values()))
        em_terra = mascara[yi, xi]
        pedacos = {
            "completo": tbl,
            "desc": tbl.filter(pa.array(qualquer)),
            "hits": tbl.filter(pa.array(em_terra)).select(
                HIT_COLS + [f"in_poly_{b}" for b in polys]),
        }
        for nome, t in pedacos.items():
            if nome not in escritores:
                escritores[nome] = pq.ParquetWriter(destinos[nome], t.schema, compression="zstd",
                                                    compression_level=9)
            escritores[nome].write_table(t)
            contagem[nome] += t.num_rows
        total += tbl.num_rows
        print(ano, f"{tbl.num_rows:>10,}", flush=True)
    for w in escritores.values():
        w.close()
    print("linhas:", f"{total:,}", {k: f"{v:,}" for k, v in contagem.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--perc", required=True)
    ap.add_argument("--terra", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracks", help="catalogo de trajetorias; sem ele o deslocamento vem das linhas do by_track")
    ap.add_argument("--sufixo", default="v2")
    a = ap.parse_args()
    gerar(a.src, a.perc, a.terra, a.out, a.tracks, a.sufixo)
