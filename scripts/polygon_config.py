"""Pareamento entre limiares de vento e bandas dos polígonos."""
ABS_POLY_DEFAULT = "b99"
ABS_POLY_CHOICES = ("b90", "b95", "b99")
PCT_POLY = {"q90": "b90", "q95": "b95", "q99": "b99"}
ABS_MS = {"m156": 15.6, "m20": 20.0, "m25": 25.0}
ABS_FLAGS = {"m156": "exceeded_15_6", "m20": "exceeded_20_0", "m25": "exceeded_25_0"}


def polygons(abs_poly=ABS_POLY_DEFAULT):
    if abs_poly not in ABS_POLY_CHOICES:
        raise ValueError(f"abs_poly invalido: {abs_poly!r} (use {ABS_POLY_CHOICES})")
    return {**PCT_POLY, **{k: abs_poly for k in ABS_MS}}


def thresholds(abs_poly=ABS_POLY_DEFAULT):
    return {k: (ABS_FLAGS.get(k, f"exceeded_{k}"), f"in_poly_{b}")
            for k, b in polygons(abs_poly).items()}


def validate_output_polygon(out, abs_poly):
    """Não permita rotular como b90 uma execução com o padrão b99."""
    from pathlib import Path
    path = Path(out)
    if ("_b90" in path.stem or "irb_corrigido" in path.parts) and abs_poly != "b90":
        raise ValueError("Saída corrigida b90 requer --abs-poly b90 explicitamente")
