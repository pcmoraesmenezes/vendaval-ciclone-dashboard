"""Grade do ajuste de valores extremos (GEV) do vento de ciclone, por ponto.

Lê a tabela de resultados do ajuste (conferida por SHA256), confere que cada nível de retorno
bate com os parâmetros pela fórmula da GEV e marca dois grupos de pontos: forma acima de 0,5 e
nível de retorno fora da fórmula. Nenhum ponto é removido nem alterado.

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


def expected_level(location, scale, shape, period) -> np.ndarray:
    """Nível de retorno da GEV para um período de retorno em blocos anuais."""
    y = -np.log1p(-1.0 / period)
    location, scale, shape = (np.asarray(v, dtype=float) for v in (location, scale, shape))
    small = np.abs(shape) < 1e-9
    safe = np.where(small, 1.0, shape)
    return np.where(small, location - scale * np.log(y), location - scale / safe * (1 - y ** -safe))


def build(src: str, out: str) -> pd.DataFrame:
    validate_output_path(out)
    validate_gev_source(src)
    table = pd.read_parquet(src)
    if list(table.columns) != SOURCE_COLUMNS:
        raise ValueError(f"colunas inesperadas: {list(table.columns)}")
    if table.isna().any().any() or table.duplicated(["lat", "lon"]).any():
        raise ValueError("a tabela tem nulos ou pontos repetidos")
    levels = table[[f"NR_{t}_estimado" for t in PERIODS]].to_numpy()
    if not (np.diff(levels, axis=1) > 0).all():
        raise ValueError("nível de retorno não cresce com o período em todas as linhas")

    off = np.zeros(len(table), dtype=bool)
    for period in PERIODS:
        calc = expected_level(table.location, table.scale, table["shape"], period)
        off |= np.abs(calc - table[f"NR_{period}_estimado"].to_numpy()) > FORMULA_TOLERANCE
    grid = table.assign(forma_alta=table["shape"] > HIGH_SHAPE, nr_fora_da_formula=off)

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
    print(f"{args.out}: {len(grid)} pontos, forma alta {int(grid.forma_alta.sum())}, "
          f"fora da fórmula {int(grid.nr_fora_da_formula.sum())}")


if __name__ == "__main__":
    main()
