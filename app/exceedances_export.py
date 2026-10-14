"""Exportação local dos excedentes de vento por ponto de grade (1979-2020).

A fonte é um parquet pesado que não acompanha o deploy público; a página só existe
onde o arquivo está presente na máquina. O caminho vem de VENDAVAL_EXCEEDANCES_PARQUET
ou dos candidatos padrão.
"""
import datetime as dt
import os
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import streamlit as st

ENV_VAR = 'VENDAVAL_EXCEEDANCES_PARQUET'
DEFAULT_PATHS = [
    Path.home() / 'Área de Trabalho' / 'IRB' / 'entrega_danilo'
    / 'cyclone_exceedances_by_track_1979_2020.parquet',
    Path(__file__).resolve().parents[1] / 'outputs' / 'parquet'
    / 'cyclone_exceedances_by_track_1979_2020.parquet',
]
THRESHOLDS = {
    'exceeded_15_6': '15,6 m/s (fixo)', 'exceeded_20_0': '20,0 m/s (fixo)',
    'exceeded_25_0': '25,0 m/s (fixo)', 'exceeded_q90': 'q90 local',
    'exceeded_q95': 'q95 local', 'exceeded_q99': 'q99 local',
}
COLUMNS = ['track_id', 'time', 'phase', 'lat_center', 'lon_center', 'lat', 'lon',
           'wind_speed', 'distance_km', 'fixed_quadrant', 'rotated_quadrant',
           'exceeded_15_6', 'exceeded_20_0', 'exceeded_25_0',
           'exceeded_q90', 'exceeded_q95', 'exceeded_q99']
YEARS = (1979, 2020)
EXPORT_DIR_ENV = 'VENDAVAL_EXPORT_DIR'


def _export_dir():
    env = os.environ.get(EXPORT_DIR_ENV)
    return Path(env) if env else Path.home() / 'Downloads'


def resolve_path():
    env = os.environ.get(ENV_VAR)
    for candidate in ([Path(env)] if env else []) + DEFAULT_PATHS:
        if candidate.exists():
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def _dataset(path_str):
    return ds.dataset(path_str, format='parquet')


@st.cache_data(show_spinner=False)
def _phase_options(path_str):
    """Fases distintas (sem o nulo) e se existe ponto sem fase classificada. O nulo é
    tratado à parte porque entra no `sorted` e porque excluí-lo silenciosamente apagaria
    os pontos sem fase."""
    values = pc.unique(_dataset(path_str).to_table(columns=['phase'])['phase']).to_pylist()
    return sorted(v for v in values if v is not None), (None in values)


def _filter(threshold, phases, include_null, year_range, distance_max):
    clauses = []
    if phases:
        clauses.append(ds.field('phase').isin(phases))  # isin([]) erra no pyarrow: não passar vazio
    if include_null:
        clauses.append(ds.field('phase').is_null())
    phase_clause = clauses[0] if clauses else pc.scalar(False)
    for extra in clauses[1:]:
        phase_clause = phase_clause | extra
    y0, y1 = year_range
    expr = (ds.field(threshold) & phase_clause
            & (ds.field('time') >= dt.datetime(y0, 1, 1))
            & (ds.field('time') < dt.datetime(y1 + 1, 1, 1))
            & (ds.field('distance_km') <= distance_max))
    return expr


def _write_slice(path_str, threshold, phases, include_null, year_range, distance_max, fmt, out_path):
    """Grava a fatia em streaming, um lote por vez, com memória limitada em qualquer tamanho."""
    import pyarrow as pa
    scanner = _dataset(path_str).scanner(
        filter=_filter(threshold, list(phases), include_null, year_range, distance_max),
        columns=COLUMNS)
    if fmt == 'CSV':
        import pyarrow.csv as pacsv
        writer = None
        try:
            for batch in scanner.to_reader():
                if writer is None:
                    writer = pacsv.CSVWriter(out_path, batch.schema)
                writer.write(batch)
        finally:
            if writer is not None:
                writer.close()
        return
    import pyarrow.parquet as pq
    writer = None
    try:
        for batch in scanner.to_reader():
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema, compression='zstd')
            writer.write_table(pa.Table.from_batches([batch]))
    finally:
        if writer is not None:
            writer.close()


def render_exceedances_export():
    st.header('📤 Exportar excedentes (1979-2020)')
    path = resolve_path()
    if path is None:
        st.info('Disponível só na execução local, com o parquet de excedentes na máquina. '
                f'Aponte a variável {ENV_VAR} para o arquivo.')
        return
    st.caption('Cada linha é um ponto de grade ERA5 a até 1.100 km de um centro de ciclone que '
               'passou do limiar de entrada, de 6 em 6 h, 1979-2020. Os percentis locais foram '
               'calculados sobre a série inteira. Escolha o recorte e baixe para análise.')
    path_str = str(path)
    phases_all, has_null = _phase_options(path_str)

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.selectbox('Limiar', list(THRESHOLDS), format_func=THRESHOLDS.get)
    with c2:
        year_range = st.select_slider('Período', options=list(range(YEARS[0], YEARS[1] + 1)),
                                       value=YEARS)
    with c3:
        distance_max = st.slider('Distância máxima ao centro (km)', 0, 1100, 1100, step=50)
    phases = st.multiselect('Fases do ciclo de vida', phases_all, default=phases_all)
    include_null = st.checkbox('Incluir pontos sem fase classificada', value=True) if has_null \
        else False

    n = _dataset(path_str).count_rows(
        filter=_filter(threshold, phases, include_null, year_range, distance_max))
    st.metric('Linhas no recorte', f'{n:,}'.replace(',', '.'))
    if n == 0:
        st.warning('Nenhuma linha no recorte. Afrouxe os filtros.')
        return

    fmt = st.radio('Formato', ['Parquet', 'CSV'], horizontal=True,
                   help='Parquet é mais leve e preserva os tipos; CSV abre em qualquer lugar.')
    ext = 'csv' if fmt == 'CSV' else 'parquet'
    y0, y1 = year_range
    out_path = _export_dir() / f'excedentes_{threshold.replace("exceeded_", "")}_{y0}_{y1}.{ext}'
    st.caption(f'A fatia é gravada em `{out_path}` (streaming, sem limite de tamanho).')
    if fmt == 'CSV' and n > 20_000_000:
        st.warning('CSV com mais de 20 milhões de linhas fica muito grande em disco; '
                   'prefira Parquet ou aperte o filtro.')
    if st.button(f'Salvar {n:,} linhas em disco'.replace(',', '.')):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner('Gravando...'):
            _write_slice(path_str, threshold, tuple(phases), include_null, year_range,
                         distance_max, fmt, str(out_path))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        st.success(f'Salvo em {out_path} ({size_mb:.1f} MB).')
