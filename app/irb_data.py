"""Contrato dos produtos adotados pelo IRB; leitura sem fallback."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "outputs" / "irb_corrigido"
THRESHOLDS = {
    "q90": "q90 (percentil local 0,90)",
    "q95": "q95 (percentil local 0,95)",
    "q99": "q99 (percentil local 0,99)",
    "m156": "15,6 m/s", "m20": "20 m/s", "m25": "25 m/s",
}
POLY = {"q90": "b90", "q95": "b95", "q99": "b99",
        "m156": "b90", "m20": "b90", "m25": "b90"}
COMMON = {
    "period": [1979, 2020], "percentile_period": [1979, 2020],
    "variable": "ERA5 mean wind at 10 m (not gust)", "grid_degrees": 0.25,
    "step_hours": 6, "polygons": POLY, "absolute_polygon": "b90",
    "index_cutoff": 0.5, "index_weights": [1/3, 1/3, 1/3],
    "polygon_role": "descriptive; not causal attribution; same-sample mask",
    "land_mask": "original footprint land cells, including islands",
    "direction": "original polygon flags, consistent with v2",
}
PRODUCTS = {
    "footprint": {
        **COMMON, "filename": "fp_C_perc1979_b90_gt.parquet",
        "sha256": "3f198b2c47cb0ce60c1767b6f60081099d7e8c07fcae2b961c3dfbf2d744a0a8",
        "variant": "C", "comparison": ">", "shape": [9237, 74],
        "source": "footprint_hits_continente_1979_2020.parquet; local_percentiles.nc 1979-2020",
        "source_rows": 4330793, "search_radius_km": 1100,
        "count_unit": "cyclone x time x cell hit; not independent ERA5 observations",
    },
    "persistence": {
        **COMMON, "filename": "persistence_v2_b90.parquet",
        "sha256": "d1c2e0e60682efbc4ff346053b850f9c3ca9483072efbb0dde042a5694263efc",
        "variant": "v2+b90", "comparison": "precomputed exceeded_* flags from full v2 series",
        "shape": [9229, 62], "source": "cyclone_exceedances_descontaminado_1979_2020_v2.parquet",
        "count_unit": "event per cyclone and cell", "allowed_missing_steps": 1,
        "duration": "last time - first time + 6 hours, including tolerated gap",
        "duration_buckets_hours": [24, 48, 72, 144],
    },
}

PRODUCTS["persistence_season"] = {
    **PRODUCTS["persistence"], "filename": "persistence_sazonal_v2_b90.parquet",
    "sha256": "0efe82ccab5623634c2b4ce46a057ee47f736c1aa8fa43735360859d6ea8573c", "variant": "v2+b90 por estacao", "shape": [36748, 63],
    "seasons": {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]},
    "event_season": "start step",
}


class IRBDataError(ValueError):
    """Produto oficial ausente ou incompatível com o contrato científico."""


def load_product(kind, path=None):
    expected = PRODUCTS[kind]
    path = Path(path) if path is not None else DATA_DIR / expected["filename"]
    try:
        metadata = json.loads(path.with_suffix(".metadata.json").read_text())
        if metadata != expected or path.name != expected["filename"]:
            raise ValueError("metadados científicos divergentes")
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError("SHA256 divergente do registro IRB")
        grid = pd.read_parquet(BytesIO(payload))
        if list(grid.shape) != expected["shape"]:
            raise ValueError("dimensões inesperadas")
        return grid
    except (OSError, ValueError) as exc:
        raise IRBDataError(
            f"Não foi possível carregar o produto corrigido IRB {path.name}: {exc}. "
            "Restaure o Parquet oficial e seu .metadata.json em outputs/irb_corrigido. "
            "Nenhum produto histórico será usado."
        ) from exc


METHOD = """
- **Fonte e domínio**: ERA5, vento médio a 10 m (não rajada), 1979–2020,
  resolução de 0,25° e passo de 6 h; máscara de terra da América do Sul, **incluindo ilhas**.
- **Limiar**: percentil local q90/q95/q99 calculado para **1979–2020**, ou 15,6/20/25 m/s.
  O percentil local é um corte do vento em cada célula; não é a banda do polígono.
- **Pareamento**: q90→b90, q95→b95, q99→b99; os três absolutos usam **b90**.
  As bandas do índice são b90: q80–q90, b95: q89–q95, b99: q95–q99.
- **Máscara descritiva**: índice composto ≥0,5, pesos iguais (1/3), no referencial
  rotacionado do ciclone. Corte e pareamento são escolhas pragmáticas, sem validação
  independente; a máscara usa a mesma amostra e não demonstra atribuição causal.
- **Unidades distintas**: hits são pares ciclone × instante × célula; eventos são sequências
  temporais por ciclone e célula. Um instante ERA5 pode ser atribuído a vários ciclones:
  essas contagens **não são observações ERA5 independentes**.
"""
MALVINAS = (
    "25 m/s: os dois hits do footprint (2 contaminados e 2 descontaminados) e os dois "
    "eventos de persistência estão nas Ilhas Malvinas/Falklands: "
    "(−52,00°, −59,50°) e (−51,75°, −59,25°). Cada célula tem 1 hit e 1 evento de 6 h; "
    "ventos máximos de 25,08 e 25,12 m/s, respectivamente. A grade agregada não permite "
    "afirmar se os dois hits pertencem ao mesmo ciclone e instante."
)
