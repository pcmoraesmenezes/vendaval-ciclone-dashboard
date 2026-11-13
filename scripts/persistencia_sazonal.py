"""Persistencia do extremo por estacao do ano (DJF, MAM, JJA, SON), mesma regra de eventos de persistence_grid.

Cada evento entra na estacao meteorologica do seu passo de inicio. As colunas por limiar sao as da
grade anual (dur_mean_, dur_max_, n_runs_, dur_q90_..q99_, ge24..ge144_), uma linha por celula e
estacao. Os seis limiares usam os poligonos do produto adotado (percentis pareados, absolutos em b90).

A saida so e gravada se, somando as estacoes, cada celula reproduz a grade anual de referencia
(eventos, maximo e faixas de duracao). Exige a serie descontaminada v2 oficial (SHA256).

  python scripts/persistencia_sazonal.py --src <parquet> --continental <csv ou parquet> \\
      --ref outputs/irb_corrigido/persistence_v2_b90.parquet --out <parquet novo>
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from persistence_grid import BUCKETS, STEP_H, _code, _events, _stats
from polygon_config import (publish_output, thresholds, validate_output_path, validate_series)

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
THRESHOLDS = thresholds("b90")


def season_of(start_t6):
    """Nome da estacao de cada passo de 6 h de inicio (indice desde 1970-01-01)."""
    seconds = start_t6.astype("int64") * STEP_H * 3600
    month = seconds.astype("datetime64[s]").astype("datetime64[M]").astype("int64") % 12 + 1
    names = np.empty(13, dtype="U3")
    for name, months in SEASONS.items():
        names[list(months)] = name
    return names[month]


def load_mask(path):
    cols = pd.read_csv(path) if str(path).endswith(".csv") else pq.read_table(path, columns=["lat", "lon"]).to_pandas()
    return np.unique(_code(np.round(cols["lat"].to_numpy(), 2).astype("float32"),
                           np.round(cols["lon"].to_numpy(), 2).astype("float32")))


def seasonal_grid(src, mask):
    parts = {s: None for s in SEASONS}
    for thr, (exc, poly) in THRESHOLDS.items():
        lat, lon, dur, start = _events(src, exc, poly, mask)
        season = season_of(start) if len(dur) else np.array([], dtype="U3")
        for s in SEASONS:
            m = season == s
            stats = _stats(lat[m], lon[m], dur[m], thr) if m.any() else None
            if stats is not None:
                parts[s] = stats if parts[s] is None else parts[s].join(stats, how="outer")
        print(f"  {thr:5s}: {len(dur):>10,} eventos | " +
              " ".join(f"{s} {int((season == s).sum()):,}" for s in SEASONS))
        del lat, lon, dur, start, season
        gc.collect()
    frames = []
    for s, grid in parts.items():
        if grid is not None:
            frames.append(grid.reset_index().assign(season=s))
    grid = pd.concat(frames, ignore_index=True)
    grid.insert(2, "season", grid.pop("season"))
    for thr in THRESHOLDS:
        for col in [f"n_runs_{thr}"] + [f"ge{h}_{thr}" for h in BUCKETS]:
            grid[col] = grid[col].fillna(0).astype("int64")
    order = grid["season"].map({s: i for i, s in enumerate(SEASONS)})
    return grid.assign(_o=order).sort_values(["_o", "lat", "lon"]).drop(columns="_o").reset_index(drop=True)


def reconcile(grid, ref):
    """Lista de divergencias entre a soma das estacoes e a grade anual de referencia."""
    ref = ref.set_index(["lat", "lon"])
    problems = []
    for thr in THRESHOLDS:
        by_cell = grid.groupby(["lat", "lon"])
        for col, agg in ([(f"n_runs_{thr}", "sum"), (f"dur_max_{thr}", "max")] +
                         [(f"ge{h}_{thr}", "sum") for h in BUCKETS]):
            got = by_cell[col].agg(agg)
            cells = got.index.union(ref.index)
            bad = int((got.reindex(cells).fillna(0) != ref[col].reindex(cells).fillna(0)).sum())
            if bad:
                problems.append(f"{col}: {bad} celulas diferentes da referencia")
        n = grid[f"n_runs_{thr}"]
        mean = (grid[f"dur_mean_{thr}"].fillna(0) * n).groupby([grid["lat"], grid["lon"]]).sum() / \
            n.groupby([grid["lat"], grid["lon"]]).sum().replace(0, np.nan)
        want = ref[f"dur_mean_{thr}"].reindex(mean.index)
        if not np.allclose(mean.fillna(-1), want.fillna(-1), rtol=0, atol=1e-6):
            problems.append(f"dur_mean_{thr}: media ponderada das estacoes difere da anual")
    return problems


def build(src, out, continental, ref):
    validate_output_path(out)
    validate_series(src)
    mask = load_mask(continental)
    print(f"mascara continental: {len(mask):,} celulas de terra")
    grid = seasonal_grid(src, mask)
    problems = reconcile(grid, pd.read_parquet(ref))
    if problems:
        raise ValueError("RECONCILIACAO FALHOU, nada foi gravado:\n  " + "\n  ".join(problems))
    print("reconciliacao com a grade anual: ok (eventos, maximo, faixas e media)")
    grid.attrs["scientific_parameters"] = dict(
        seasons={s: list(m) for s, m in SEASONS.items()}, event_season="start step",
        polygons={k: v[1].removeprefix("in_poly_") for k, v in THRESHOLDS.items()},
        source=Path(src).name, step_hours=STEP_H, allowed_missing_steps=1,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    grid.to_parquet(tmp, compression="zstd", index=False)
    publish_output(tmp, out)
    print(f"linhas: {len(grid):,} | colunas: {len(grid.columns)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--continental", required=True, help="celulas de terra (csv ou parquet com lat e lon)")
    ap.add_argument("--ref", required=True, help="grade anual adotada, contra a qual a soma das estacoes e conferida")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.src, a.out, a.continental, a.ref)
