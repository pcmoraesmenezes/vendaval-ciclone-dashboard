"""Persistencia do extremo por celula de grade (1979-2020).

Para cada limiar de intensidade (percentis locais q90/q95/q99 e absolutos 15,6/20/25 m/s), mede
quanto tempo seguido o vento extremo de ciclone dura em cada ponto. Um evento e uma sequencia de
timesteps de 6 h do mesmo ciclone (track_id) em que o ponto e extremo dentro do poligono de
pertinencia, tolerando um buraco de um passo (6 h). A duracao e o intervalo coberto pela
sequencia, em horas (4 passos = 24 h). Por ponto e limiar: media, maximo e percentis q90/q95/q99
da duracao, numero de eventos, e quantos eventos passaram de 24/48/72/144 h.

Le a serie descontaminada por track (cyclone_exceedances_descontaminado_1979_2020.parquet). O
corte unico e o poligono pareado ja vem como colunas (exceeded_* e in_poly_*), um limiar por
passada para manter a memoria baixa.

Os percentis mantem q90->b90, q95->b95 e q99->b99. Os absolutos nao tem banda propria:
--abs-poly escolhe seu poligono (b90|b95|b99). A regra temporal independe dessa opcao.

  python scripts/persistence_grid.py
  python scripts/persistence_grid.py --src <parquet> --out <parquet>
  python scripts/persistence_grid.py --src <parquet> --out <parquet> --abs-poly b90
"""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_SRC = Path.home() / "Área de Trabalho" / "cyclone_exceedances_descontaminado_1979_2020.parquet"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "persistence_grid.parquet"
# Mascara continental: celulas de terra do footprint (mesma definicao da pagina Footprint).
DEFAULT_CONT = Path.home() / "Área de Trabalho" / "footprint_hits_continente_1979_2020.parquet"

STEP_H = 6
BUCKETS = (24, 48, 72, 144)
from polygon_config import (ABS_POLY_DEFAULT, ABS_POLY_CHOICES, thresholds,
                            validate_output_polygon)

THRESHOLDS = thresholds()


def _code(lat, lon):
    return np.round(lat * 100).astype("int64") * 1000000 + np.round(lon * 100).astype("int64")


def _runs(src, exc, poly, cont):
    """Duracoes (h) de cada evento continuo, com a celula onde ocorreu."""
    tbl = pq.read_table(src, columns=["track_id", "time", "lat", "lon"],
                        filters=[(exc, "==", True), (poly, "==", True)])
    lat = np.round(tbl.column("lat").to_numpy(zero_copy_only=False), 2).astype("float32")
    lon = np.round(tbl.column("lon").to_numpy(zero_copy_only=False), 2).astype("float32")
    tr = tbl.column("track_id").to_numpy(zero_copy_only=False).astype("int32")
    # timestamp[ms] -> indice de passo de 6 h (cabe em int32: ~74 mil em 2020)
    t6 = (tbl.column("time").cast("int64").to_numpy(zero_copy_only=False)
          // (STEP_H * 3600 * 1000)).astype("int32")
    del tbl

    if cont is not None:
        m = np.isin(_code(lat, lon), cont)
        lat, lon, tr, t6 = lat[m], lon[m], tr[m], t6[m]

    order = np.lexsort((t6, tr, lon, lat))
    lat, lon, tr, t6 = lat[order], lon[order], tr[order], t6[order]
    del order

    n = len(t6)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = ((lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1]) | (tr[1:] != tr[:-1])
               | ((t6[1:] - t6[:-1]) > 2))  # gap > 1 passo quebra o evento
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1
    dur = (t6[ends] - t6[starts] + 1) * STEP_H
    return lat[starts], lon[starts], dur


def _stats(lat, lon, dur, thr):
    r = pd.DataFrame({"lat": lat, "lon": lon, "dur": dur})
    g = r.groupby(["lat", "lon"])["dur"]
    out = g.agg(**{
        f"dur_mean_{thr}": "mean", f"dur_max_{thr}": "max", f"n_runs_{thr}": "size",
        f"dur_q90_{thr}": lambda s: s.quantile(0.90),
        f"dur_q95_{thr}": lambda s: s.quantile(0.95),
        f"dur_q99_{thr}": lambda s: s.quantile(0.99),
    })
    for h in BUCKETS:
        out[f"ge{h}_{thr}"] = r.assign(hit=r["dur"] >= h).groupby(["lat", "lon"])["hit"].sum()
    return out


def build(src, out, continental, abs_poly=ABS_POLY_DEFAULT):
    validate_output_polygon(out, abs_poly)
    cont = None
    if continental:
        fp = pq.read_table(continental, columns=["lat", "lon"]).to_pandas()
        cont = np.unique(_code(np.round(fp["lat"].to_numpy(), 2).astype("float32"),
                               np.round(fp["lon"].to_numpy(), 2).astype("float32")))
        print(f"mascara continental: {len(cont):,} celulas de terra")

    thr_map = thresholds(abs_poly)
    print(f"poligono dos absolutos: {abs_poly}")
    grid = None
    for thr, (exc, poly) in thr_map.items():
        lat, lon, dur = _runs(src, exc, poly, cont)
        s = _stats(lat, lon, dur, thr)
        print(f"  {thr:5s}: {len(dur):>10,} eventos | {len(s):>5,} celulas | "
              f"dur media {dur.mean():.1f}h max {int(dur.max())}h" if len(dur) else
              f"  {thr:5s}: sem eventos")
        grid = s if grid is None else grid.join(s, how="outer")
        del lat, lon, dur, s
        gc.collect()

    for thr in thr_map:
        for h in BUCKETS:
            c = f"ge{h}_{thr}"
            grid[c] = grid[c].fillna(0).astype("int64")
        grid[f"n_runs_{thr}"] = grid[f"n_runs_{thr}"].fillna(0).astype("int64")

    grid = grid.reset_index()
    grid.attrs["scientific_parameters"] = dict(
        absolute_polygon=abs_poly,
        polygons={k: v[1].removeprefix("in_poly_") for k, v in thr_map.items()},
        source=str(src), step_hours=STEP_H, allowed_missing_steps=1,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    grid.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, out)
    print(f"celulas: {len(grid):,} | colunas: {len(grid.columns)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--continental", default=str(DEFAULT_CONT),
                    help="parquet com os pontos de terra; vazio desativa a mascara")
    ap.add_argument("--abs-poly", default=ABS_POLY_DEFAULT, choices=ABS_POLY_CHOICES,
                    help="poligono pareado aos limiares absolutos; padrao b99; dados corrigidos IRB requerem --abs-poly b90")
    a = ap.parse_args()
    build(a.src, a.out, a.continental or None, a.abs_poly)
