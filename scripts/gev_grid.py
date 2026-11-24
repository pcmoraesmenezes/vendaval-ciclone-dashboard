"""Grade do ajuste de valores extremos (GEV) do vento de ciclone, por ponto.

Lê a tabela de resultados do ajuste (conferida por SHA256), retira o ponto cortado, confere que
cada nível de retorno dos demais bate com os parâmetros pela fórmula da GEV e marca os pontos com
forma acima de 0,5.

  python scripts/gev_grid.py --src <gev_america_do_sul.parquet> --out <parquet novo>

A saída é sempre um caminho novo: arquivo existente e a pasta outputs/irb_corrigido são recusados.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from polygon_config import publish_output, validate_gev_source, validate_output_path

PERIODS = (2, 5, 10, 20, 25, 30, 50, 100)
HIGH_SHAPE = 0.5
FORMULA_TOLERANCE = 1e-6
PARAMS = ("location", "scale", "shape")
SOURCE_COLUMNS = ["lat", "lon", *PARAMS, *[f"NR_{t}_estimado" for t in PERIODS]]
# Níveis de 50 e 100 anos fora da fórmula do ajuste, com poucos anos de dado.
CUT_POINTS = ((-23.75, -68.25),)


def expected_level(location, scale, shape, period) -> np.ndarray:
    """Nível de retorno da GEV para um período de retorno em blocos anuais."""
    y = -np.log1p(-1.0 / period)
    location, scale, shape = (np.asarray(v, dtype=float) for v in (location, scale, shape))
    small = np.abs(shape) < 1e-9
    safe = np.where(small, 1.0, shape)
    return np.where(small, location - scale * np.log(y), location - scale / safe * (1 - y ** -safe))


def prepare(table: pd.DataFrame) -> pd.DataFrame:
    """Confere a tabela, retira os pontos cortados e marca os de forma alta."""
    if list(table.columns) != SOURCE_COLUMNS:
        raise ValueError(f"colunas inesperadas: {list(table.columns)}")
    if table.isna().any().any() or table.duplicated(["lat", "lon"]).any():
        raise ValueError("a tabela tem nulos ou pontos repetidos")
    levels = table[[f"NR_{t}_estimado" for t in PERIODS]].to_numpy()
    if not (np.diff(levels, axis=1) > 0).all():
        raise ValueError("nível de retorno não cresce com o período em todas as linhas")

    cut = pd.Series(False, index=table.index)
    for lat, lon in CUT_POINTS:
        found = (table.lat == lat) & (table.lon == lon)
        if not found.any():
            raise ValueError(f"ponto cortado ({lat}, {lon}) não está na tabela")
        cut |= found
    kept = table[~cut]

    for period in PERIODS:
        calc = expected_level(kept.location, kept.scale, kept["shape"], period)
        if (np.abs(calc - kept[f"NR_{period}_estimado"].to_numpy()) > FORMULA_TOLERANCE).any():
            raise ValueError(f"há nível de retorno de {period} anos fora da fórmula depois do corte")
    return kept.assign(forma_alta=kept["shape"] > HIGH_SHAPE).reset_index(drop=True)


def build(src: str, out: str) -> pd.DataFrame:
    validate_output_path(out)
    validate_gev_source(src)
    grid = prepare(pd.read_parquet(src))
    tmp = Path(out).with_suffix(".tmp")
    grid.to_parquet(tmp, index=False)
    publish_output(tmp, out)
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    grid = build(args.src, args.out)
    print(f"{args.out}: {len(grid)} pontos, forma alta {int(grid.forma_alta.sum())}")


if __name__ == "__main__":
    main()
