"""Grade por celula do footprint continental, fonte da pagina Footprint do painel.

Le o parquet de hits (4,3M linhas, 1979-2020) e agrega por celula de 0,25 graus, para o
contaminado (todo hit) e para cada banda descontaminada (in_poly_b90/b95/b99):
contagem de hits e, do vento, media, maximo e percentis q90/q95/q99. O parquet bruto nao
acompanha o deploy; esta grade (~9 mil celulas) sim.

  python scripts/footprint_grid.py
  python scripts/footprint_grid.py --src <parquet> --out <parquet>
"""
import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_SRC = Path.home() / "Área de Trabalho" / "footprint_hits_continente_1979_2020.parquet"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "footprint_continente_grid.parquet"
BANDS = ["b90", "b95", "b99"]


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["lat", "lon"], sort=True)["wind_speed"]
    base = g.agg(hits="size", media="mean", max="max")
    q = g.quantile([0.90, 0.95, 0.99]).unstack()
    q.columns = ["q90", "q95", "q99"]
    return base.join(q)


def build(src, out):
    cols = ["lat", "lon", "wind_speed"] + [f"in_poly_{b}" for b in BANDS]
    df = pd.read_parquet(src, columns=cols)
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)

    grid = _agg(df).add_suffix("_contam")
    for b in BANDS:
        grid = grid.join(_agg(df[df[f"in_poly_{b}"]]).add_suffix(f"_{b}"))
    for b in BANDS:
        grid[f"hits_{b}"] = grid[f"hits_{b}"].fillna(0).astype("int64")

    grid = grid.reset_index()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    grid.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, out)

    print(f"celulas: {len(grid):,} | colunas: {len(grid.columns)} -> {out}")
    print(f"  contaminado: {int(grid['hits_contam'].sum()):,} hits, {len(grid):,} celulas, "
          f"media {grid['media_contam'].mean():.2f} max {grid['max_contam'].max():.1f}")
    for b in BANDS:
        sel = grid[f"hits_{b}"] > 0
        print(f"  {b}: {int(grid[f'hits_{b}'].sum()):,} hits, {int(sel.sum()):,} celulas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    build(a.src, a.out)
