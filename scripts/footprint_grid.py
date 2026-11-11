"""Grade por celula do footprint continental, fonte da pagina Footprint do painel.

Le o parquet de hits (4,3M linhas, 1979-2020) e agrega por celula de 0,25 graus para cada
limiar de intensidade, no contaminado (todo hit do limiar dentro do raio de busca) e no
descontaminado (hit dentro do poligono de pertinencia pareado): contagem de hits e, do vento,
media, maximo e percentis q90/q95/q99. O parquet bruto nao acompanha o deploy; esta grade
(~9 mil celulas) sim.

O corte e unico: os percentis locais (q90/q95/q99, do mesmo local_percentiles.nc que alimenta
o indice composto) e os absolutos (15,6/20/25 m/s). Como o bruto inclui todo ponto-instante
acima do limiar mais fraco, ele e superset valido para re-cortar em qualquer um dos seis.

  python scripts/footprint_grid.py
  python scripts/footprint_grid.py --src <parquet> --out <parquet> --perc <local_percentiles.nc>
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import xarray as xr

DEFAULT_SRC = Path.home() / "Área de Trabalho" / "footprint_hits_continente_1979_2020.parquet"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "outputs" / "footprint_continente_grid.parquet"
DEFAULT_PERC = (Path.home() / "Área de Trabalho" / "IRB" / "vendaval-ciclone"
                / "data" / "local_percentiles.nc")

# Limiar de intensidade -> banda do poligono de pertinencia pareada.
PCT_POLY = {"q90": "b90", "q95": "b95", "q99": "b99"}
ABS_MS = {"m156": 15.6, "m20": 20.0, "m25": 25.0}
ABS_POLY = "b99"  # absoluto nao tem banda propria; usa o poligono mais apertado


def _agg(sub: pd.DataFrame) -> pd.DataFrame:
    g = sub.groupby(["lat", "lon"], sort=True)["wind_speed"]
    return g.agg(
        hits="size", media="mean", max="max",
        q90=lambda s: s.quantile(0.90),
        q95=lambda s: s.quantile(0.95),
        q99=lambda s: s.quantile(0.99),
    )


def build(src, out, perc, estrito=False, abs_poly=ABS_POLY):
    corte = (lambda a, b: a > b) if estrito else (lambda a, b: a >= b)
    poly_cols = [f"in_poly_{b}" for b in ("b90", "b95", "b99")]
    df = pd.read_parquet(src, columns=["lat", "lon", "wind_speed"] + poly_cols)
    df["lat"] = df["lat"].round(2)
    df["lon"] = df["lon"].round(2)

    pc = xr.open_dataset(perc)[["q90", "q95", "q99"]].to_dataframe().reset_index()
    pc = pc.rename(columns={"latitude": "lat", "longitude": "lon",
                            "q90": "loc_q90", "q95": "loc_q95", "q99": "loc_q99"})
    # local_percentiles pode vir em longitude 0-360; o bruto esta em -180..180.
    if pc["lon"].max() > 180:
        pc["lon"] = ((pc["lon"] + 180) % 360) - 180
    pc["lat"] = pc["lat"].round(2)
    pc["lon"] = pc["lon"].round(2)
    df = df.merge(pc, on=["lat", "lon"], how="left")
    miss = float(df["loc_q90"].isna().mean())

    masks = {
        "q90": corte(df["wind_speed"], df["loc_q90"]),
        "q95": corte(df["wind_speed"], df["loc_q95"]),
        "q99": corte(df["wind_speed"], df["loc_q99"]),
        "m156": corte(df["wind_speed"], ABS_MS["m156"]),
        "m20": corte(df["wind_speed"], ABS_MS["m20"]),
        "m25": corte(df["wind_speed"], ABS_MS["m25"]),
    }
    poly = {**{k: f"in_poly_{v}" for k, v in PCT_POLY.items()},
            **{k: f"in_poly_{abs_poly}" for k in ABS_MS}}

    grid = df[["lat", "lon"]].drop_duplicates().sort_values(["lat", "lon"]).set_index(["lat", "lon"])
    for thr, m in masks.items():
        sub = df[m]
        grid = grid.join(_agg(sub).add_suffix(f"_{thr}_contam"))
        grid = grid.join(_agg(sub[sub[poly[thr]]]).add_suffix(f"_{thr}_desc"))
    for thr in masks:
        for v in ("contam", "desc"):
            c = f"hits_{thr}_{v}"
            grid[c] = grid[c].fillna(0).astype("int64")

    grid = grid.reset_index()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    grid.to_parquet(tmp, compression="zstd", index=False)
    os.replace(tmp, out)

    print(f"celulas: {len(grid):,} | colunas: {len(grid.columns)} -> {out}")
    print(f"pontos sem percentil local no merge: {miss:.4%}")
    for thr in masks:
        hc = int(grid[f"hits_{thr}_contam"].sum())
        hd = int(grid[f"hits_{thr}_desc"].sum())
        cc = int((grid[f"hits_{thr}_contam"] > 0).sum())
        print(f"  {thr:5s}: contam {hc:>10,} hits / {cc:>5,} celulas | desc {hd:>10,} hits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--perc", default=str(DEFAULT_PERC))
    ap.add_argument("--estrito", action="store_true", help="vento > limiar, como o gerador; padrao >=")
    ap.add_argument("--abs-poly", default=ABS_POLY, choices=("b90", "b95", "b99"),
                    help="poligono pareado aos limiares absolutos")
    a = ap.parse_args()
    build(a.src, a.out, a.perc, a.estrito, a.abs_poly)
