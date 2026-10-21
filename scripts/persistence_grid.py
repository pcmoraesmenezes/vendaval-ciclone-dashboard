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

  python scripts/persistence_grid.py
  python scripts/persistence_grid.py --src <parquet> --out <parquet>
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

STEP_H = 6
BUCKETS = (24, 48, 72, 144)
# Limiar -> (flag de excedencia, poligono pareado). Absolutos pareados ao b99 (mais apertado).
THRESHOLDS = {
    "q90": ("exceeded_q90", "in_poly_b90"),
    "q95": ("exceeded_q95", "in_poly_b95"),
    "q99": ("exceeded_q99", "in_poly_b99"),
    "m156": ("exceeded_15_6", "in_poly_b99"),
    "m20": ("exceeded_20_0", "in_poly_b99"),
    "m25": ("exceeded_25_0", "in_poly_b99"),
}


def _runs(src, exc, poly):
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


def build(src, out):
    grid = None
    for thr, (exc, poly) in THRESHOLDS.items():
        lat, lon, dur = _runs(src, exc, poly)
        s = _stats(lat, lon, dur, thr)
        print(f"  {thr:5s}: {len(dur):>10,} eventos | {len(s):>5,} celulas | "
              f"dur media {dur.mean():.1f}h max {int(dur.max())}h" if len(dur) else
              f"  {thr:5s}: sem eventos")
        grid = s if grid is None else grid.join(s, how="outer")
        del lat, lon, dur, s
        gc.collect()

    for thr in THRESHOLDS:
        for h in BUCKETS:
            c = f"ge{h}_{thr}"
            grid[c] = grid[c].fillna(0).astype("int64")
        grid[f"n_runs_{thr}"] = grid[f"n_runs_{thr}"].fillna(0).astype("int64")

    grid = grid.reset_index()
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
    a = ap.parse_args()
    build(a.src, a.out)
