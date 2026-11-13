"""Pareamento entre limiares de vento e bandas dos polígonos, e travas das saídas."""
import hashlib
import os
import re
from pathlib import Path

ABS_POLY_DEFAULT = "b99"
ABS_POLY_CHOICES = ("b90", "b95", "b99")
PCT_POLY = {"q90": "b90", "q95": "b95", "q99": "b99"}
ABS_MS = {"m156": 15.6, "m20": 20.0, "m25": 25.0}
ABS_FLAGS = {"m156": "exceeded_15_6", "m20": "exceeded_20_0", "m25": "exceeded_25_0"}

CORRECTED_DIR = "irb_corrigido"
FOOTPRINT_RAW_ROWS = 4330793
PERCENTILES_PERIOD = re.compile(r"1979\D{1,3}2020")
# Prefixos de SHA256 das entradas da persistência corrigida: série descontaminada v2 e máscara de terra.
PERSISTENCE_SRC_SHA256 = "72d3637f6838c616"
CONTINENTAL_MASK_SHA256 = "eddc9ce576396f51"


def polygons(abs_poly=ABS_POLY_DEFAULT):
    if abs_poly not in ABS_POLY_CHOICES:
        raise ValueError(f"abs_poly invalido: {abs_poly!r} (use {ABS_POLY_CHOICES})")
    return {**PCT_POLY, **{k: abs_poly for k in ABS_MS}}


def thresholds(abs_poly=ABS_POLY_DEFAULT):
    return {k: (ABS_FLAGS.get(k, f"exceeded_{k}"), f"in_poly_{b}")
            for k, b in polygons(abs_poly).items()}


def is_corrected(out):
    path = Path(out)
    return "_b90" in path.stem or CORRECTED_DIR in path.parts


def validate_output_polygon(out, abs_poly):
    """Não permita rotular como b90 uma execução com o padrão b99."""
    if is_corrected(out) and abs_poly != "b90":
        raise ValueError("Saída corrigida b90 requer --abs-poly b90 explicitamente")


def validate_output_path(out):
    """Nenhuma execução grava sobre arquivo existente nem dentro da pasta dos produtos adotados."""
    path = Path(out).resolve()
    if CORRECTED_DIR in path.parts:
        raise ValueError(f"{CORRECTED_DIR}/ guarda os produtos adotados e não recebe gravações")
    if path.exists():
        raise FileExistsError(f"{path} já existe; use um caminho novo")


def publish_output(tmp, out):
    """Move a saída temporária para o destino, conferindo de novo que ele continua livre."""
    try:
        validate_output_path(out)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    os.replace(tmp, out)


def _sha256_starts_with(path, prefix):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().startswith(prefix)


def validate_corrected_footprint(src, perc, estrito):
    """Saída corrigida: vento > limiar, percentis 1979-2020 e o bruto continental original."""
    import pyarrow.parquet as pq
    import xarray as xr
    if not estrito:
        raise ValueError("Saída corrigida requer --estrito (vento > limiar)")
    rows = pq.ParquetFile(src).metadata.num_rows
    if rows != FOOTPRINT_RAW_ROWS:
        raise ValueError(f"Bruto com {rows:,} linhas; o original tem {FOOTPRINT_RAW_ROWS:,}")
    with xr.open_dataset(perc) as ds:
        description = str(ds.attrs.get("description", ""))
    if not PERCENTILES_PERIOD.search(description):
        raise ValueError(f"Percentis sem período 1979-2020 na descrição: {description!r}")


def validate_series(src):
    """A série de eventos precisa ser a descontaminada v2 oficial, conferida por SHA256."""
    if not _sha256_starts_with(src, PERSISTENCE_SRC_SHA256):
        raise ValueError(f"Série {src} não tem o SHA256 da descontaminada v2 ({PERSISTENCE_SRC_SHA256}...)")


def validate_corrected_persistence(src, continental):
    """Saída corrigida: série descontaminada v2 e máscara de terra oficiais, conferidas por SHA256."""
    if not continental:
        raise ValueError("Saída corrigida requer a máscara de terra (--continental)")
    validate_series(src)
    if not _sha256_starts_with(continental, CONTINENTAL_MASK_SHA256):
        raise ValueError(f"Máscara {continental} não tem o SHA256 oficial ({CONTINENTAL_MASK_SHA256}...)")
