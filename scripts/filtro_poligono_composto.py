"""Pertinencia de extremo ao ciclone pelo poligono do indice composto (>= 0,5).

Criterio de pertinencia: o vento extremo associado ao ciclone e assimetrico, entao o raio de
1.100 km inclui pontos que nao pertencem ao sistema. O poligono do indice composto (campo 0 a 1
por fase, no referencial do ciclone) corta em 0,5 e define a regiao que pertence ao ciclone: o
ponto so conta se cair dentro desse poligono da fase vigente.

Decisoes (defaults, mudam se o Danilo pedir, e a metodologia registra):
  referencial: rotacionado (a assimetria e relativa ao sentido de deslocamento)
  um poligono por banda: b90, b95, b99 -> uma coluna de pertinencia por banda
  um poligono por fase: incipient/intensification/mature/decay (reocorrencias caem na fase-base)
  extremo = banda: a pertinencia (in_poly_b*) combina-se com frac_b* do parquet para atribuir
  fase vazia/residual: fica fora do poligono (sem fase-base, sem atribuicao)

Reusa o campo do indice composto de app/composite_index.py (nao recalcula frequencia nem theta).

  Amostra:  python scripts/filtro_poligono_composto.py
  Completo: python scripts/filtro_poligono_composto.py --full --src <parquet> --out <parquet>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from composite_index import load_composite          # noqa: E402
from excursion_sets import AXIS_KM, PHASES           # noqa: E402

REFERENCIAL = "rotated"
POLY_LEVEL = 0.5
KM_PER_DEG = 111.32
BANDS = {"b90": "p90", "b95": "p95", "b99": "p99"}
REOCC_TO_BASE = {"intensification 2": "intensification", "decay 2": "decay",
                 "mature 2": "mature", "incipient 2": "incipient"}


def load_polygons(referencial=REFERENCIAL, level=POLY_LEVEL):
    """{banda: {fase: mask 89x89 booleana onde composite >= level}}.
    Eixo (metodologia 6.2/6.3): linha e coluna em AXIS_KM; no rotacionado linha e frente no
    movimento e coluna e direita do movimento."""
    polys = {}
    for col, banda in BANDS.items():
        _, _, composite, _, _ = load_composite(banda, referencial)
        polys[col] = {p: (np.isfinite(composite[p]) & (composite[p] >= level)) for p in PHASES}
    return polys, np.asarray(AXIS_KM, dtype=float)


def movement_unit_vectors(src):
    """(ux_leste, uy_norte) normalizado por (track_id, time), diferenca central da trajetoria."""
    centers = {}
    pf = pq.ParquetFile(src)
    for b in pf.iter_batches(batch_size=2_000_000,
                             columns=["track_id", "time", "lat_center", "lon_center"]):
        tid = b.column("track_id").to_numpy()
        t = b.column("time").to_numpy()
        la = b.column("lat_center").to_numpy()
        lo = b.column("lon_center").to_numpy()
        for k, tt, a, o in zip(tid.tolist(), t.tolist(), la.tolist(), lo.tolist()):
            centers.setdefault(k, {}).setdefault(tt, (a, o))
    uv = {}
    for k, d in centers.items():
        times = sorted(d)
        lats = np.array([d[tt][0] for tt in times])
        lons = np.array([d[tt][1] for tt in times])
        dlat = np.gradient(lats) if len(lats) > 1 else np.zeros_like(lats)
        dlon = np.gradient(lons) if len(lons) > 1 else np.zeros_like(lons)
        ux = dlon * np.cos(np.radians(lats))
        uy = dlat
        norm = np.hypot(ux, uy)
        norm[norm == 0] = 1.0
        for i, tt in enumerate(times):
            uv[(k, tt)] = (ux[i] / norm[i], uy[i] / norm[i])
    return uv


def _cell(coord_km, axis):
    return np.clip(np.round((coord_km - axis[0]) / (axis[1] - axis[0])).astype(int), 0, len(axis) - 1)


def _base_phase_codes(phases):
    base = np.full(len(phases), -1, dtype=np.int8)
    order = {p: i for i, p in enumerate(PHASES)}
    for i, ph in enumerate(phases):
        p = ph if ph in order else REOCC_TO_BASE.get(ph)
        if p is not None:
            base[i] = order[p]
    return base, list(PHASES)


def membership(batch, polys, axis, uv):
    """{banda: bool array} de pertinencia ao poligono da fase, no referencial rotacionado."""
    lat = batch.column("lat").to_numpy(); lon = batch.column("lon").to_numpy()
    latc = batch.column("lat_center").to_numpy(); lonc = batch.column("lon_center").to_numpy()
    phases = batch.column("phase").to_pylist()
    keys = list(zip(batch.column("track_id").to_numpy().tolist(),
                    batch.column("time").to_numpy().tolist()))
    dlon = (lon - lonc + 180) % 360 - 180
    east = dlon * KM_PER_DEG * np.cos(np.radians((lat + latc) / 2))
    north = (lat - latc) * KM_PER_DEG
    ux = np.array([uv.get(k, (np.nan, np.nan))[0] for k in keys])
    uy = np.array([uv.get(k, (np.nan, np.nan))[1] for k in keys])
    fwd = east * ux + north * uy
    rgt = east * uy - north * ux
    rows, cols = _cell(fwd, axis), _cell(rgt, axis)
    base, phase_names = _base_phase_codes(phases)
    valid = (base >= 0) & np.isfinite(fwd)
    out = {}
    for col in polys:
        res = np.zeros(len(lat), dtype=bool)
        for code, pname in enumerate(phase_names):
            sel = valid & (base == code)
            if sel.any():
                res[sel] = polys[col][pname][rows[sel], cols[sel]]
        out[col] = res
    return out


def dry_run(src, n_batches=3):
    polys, axis = load_polygons()
    uv = movement_unit_vectors(src)
    pf = pq.ParquetFile(src)
    tot = 0; inside = {c: 0 for c in polys}
    for i, b in enumerate(pf.iter_batches(batch_size=1_000_000)):
        if i >= n_batches:
            break
        tot += b.num_rows
        for c, m in membership(b, polys, axis, uv).items():
            inside[c] += int(m.sum())
    print(f"referencial={REFERENCIAL} nivel>={POLY_LEVEL} | amostra {tot:,} pontos")
    for c in polys:
        print(f"  in_poly_{c}: {inside[c]:,} ({100*inside[c]/tot:.1f}%)")


def full_run(src, out):
    polys, axis = load_polygons()
    uv = movement_unit_vectors(src)
    pf = pq.ParquetFile(src)
    schema = pf.schema_arrow
    for c in polys:
        schema = schema.append(pa.field(f"in_poly_{c}", pa.bool_()))
    writer = pq.ParquetWriter(out, schema, compression="zstd", compression_level=9)
    done = 0; inside = {c: 0 for c in polys}
    for b in pf.iter_batches(batch_size=1_000_000):
        t = pa.Table.from_batches([b])
        for c, m in membership(b, polys, axis, uv).items():
            t = t.append_column(f"in_poly_{c}", pa.array(m))
            inside[c] += int(m.sum())
        writer.write_table(t)
        done += t.num_rows
    writer.close()
    print(f"linhas: {done:,} -> {out}")
    for c in polys:
        print(f"  in_poly_{c}: {inside[c]:,} ({100*inside[c]/done:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--src", default="/home/paulo/Área de Trabalho/cyclone_exceedances_by_track_1979_2020.parquet")
    ap.add_argument("--out", default="/home/paulo/Área de Trabalho/cyclone_exceedances_by_track_1979_2020.parquet")
    a = ap.parse_args()
    if a.full:
        tmp = a.out + ".tmp"
        full_run(a.src, tmp)
        os.replace(tmp, a.out)
    else:
        dry_run(a.src)
