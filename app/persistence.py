"""Persistencia do extremo de ciclone por celula de grade continental (1979-2020).

Dois mapas por limiar (percentis locais q90/q95/q99): a duracao dos eventos (media, maximo ou
percentis q90/q95/q99, em horas) e a contagem de eventos persistentes (>= 24/48/72/144 h). Um
evento e uma sequencia de passos de 6 h do mesmo ciclone (track_id) acima do limiar, dentro do
poligono de pertinencia, tolerando um buraco de um passo. Restrito aos pontos de terra (mesma
mascara continental da pagina Footprint). Fonte: scripts/persistence_grid.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from footprint import _auto_bands, _count_fmt, _discrete_colorscale, HITS_COLORS

GRID_PATH = Path(__file__).resolve().parents[1] / "outputs" / "persistence_grid.parquet"

# Corte unico por percentil local -> banda do poligono pareada.
THRESHOLDS = {
    "q90": "q90 (percentil local 0,90)",
    "q95": "q95 (percentil local 0,95)",
    "q99": "q99 (percentil local 0,99)",
}
DUR_METRICS = {"dur_mean": "Duração média", "dur_max": "Duração máxima",
               "dur_q90": "Duração q90", "dur_q95": "Duração q95", "dur_q99": "Duração q99"}
BUCKETS = {"ge24": "pelo menos 24 h", "ge48": "pelo menos 48 h",
           "ge72": "pelo menos 72 h", "ge144": "pelo menos 144 h"}
LON_RANGE = [-85, -33]
LAT_RANGE = [-56, -5]


@st.cache_data(show_spinner=False)
def _read_grid(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(GRID_PATH)


def load_grid() -> pd.DataFrame | None:
    if not GRID_PATH.exists():
        return None
    return _read_grid(GRID_PATH.stat().st_mtime)


def _hours_fmt(v: float) -> str:
    return f"{int(round(v))}"


def _geo(fig: go.Figure) -> None:
    fig.update_geos(
        scope="south america",
        lataxis_range=LAT_RANGE, lonaxis_range=LON_RANGE,
        showland=True, landcolor="#d9d9d9",
        showocean=False, showlakes=False,
        showcountries=True, countrycolor="#9a9a9a", countrywidth=0.5,
        showsubunits=True, subunitcolor="#bcbcbc", subunitwidth=0.3,
        showcoastlines=True, coastlinecolor="#6e6e6e", coastlinewidth=0.6,
        showframe=False, bgcolor="rgba(0,0,0,0)", resolution=50,
    )
    fig.update_layout(height=620, margin=dict(t=10, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")


def _band_map(sub: pd.DataFrame, col: str, fmt_band, title: str, hover_fmt: str) -> go.Figure:
    edges, labels = _auto_bands(sub[col].dropna(), fmt=fmt_band)
    n = len(HITS_COLORS)
    band = np.clip(np.digitize(sub[col], edges) - 1, 0, n - 1)
    fig = go.Figure(go.Scattergeo(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker=dict(
            color=band + 0.5, colorscale=_discrete_colorscale(HITS_COLORS), cmin=0, cmax=n,
            size=4, symbol="square", line=dict(width=0),
            colorbar={"title": title, "tickmode": "array",
                      "tickvals": [i + 0.5 for i in range(n)], "ticktext": labels,
                      "thickness": 12, "len": 0.9},
        ),
        customdata=sub[col],
        hovertemplate=f"%{{customdata:{hover_fmt}}} {title}<extra></extra>",
    ))
    _geo(fig)
    return fig


def _br(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def render_persistence() -> None:
    st.header("Persistência do extremo de ciclone")
    grid = load_grid()
    if grid is None:
        st.info("Grade não encontrada. Rode `python scripts/persistence_grid.py` para gerar "
                "`outputs/persistence_grid.parquet`.")
        return

    st.caption(
        "Quanto tempo seguido o vento extremo de ciclone dura em cada ponto de terra da América do "
        "Sul, 1979-2020. Um evento é uma sequência de passos de 6 h do mesmo ciclone acima do "
        "limiar, dentro do polígono de pertinência, tolerando um buraco de um passo. 4 passos = 24 h."
    )

    with st.expander("Como é calculado"):
        st.markdown(
            "- **Fonte**: série por ciclone de 1979 a 2020 (ERA5, passo de 6 h), restrita aos "
            "pontos de terra da América do Sul (mesma máscara continental da página Footprint) "
            "dentro do polígono de pertinência do ciclone (índice composto no nível 0,5, "
            "referencial rotacionado).\n"
            "- **Extremo**: num ponto e instante, o vento acima do percentil local escolhido "
            "(q90/q95/q99: vento acima do percentil daquele ponto). Cada percentil usa o polígono "
            "da sua banda (q90 com b90, q95 com b95, q99 com b99).\n"
            "- **Evento**: uma sequência de passos de 6 h do mesmo ciclone em que o ponto fica "
            "extremo sem parar. Um buraco de um passo (6 h) é tolerado antes de encerrar o evento; "
            "se o ciclone vai embora e outro chega, é um evento novo.\n"
            "- **Duração**: o intervalo coberto pelo evento, em horas. 1 passo = 6 h, 4 passos = 24 h.\n"
            "- **Por ponto**: duração média, máxima e percentis q90/q95/q99 das durações, número de "
            "eventos, e quantos passaram de 24/48/72/144 h.\n"
            "- **Cores**: 6 faixas por quantil da própria distribuição, ancoradas no máximo, para "
            "exaltar onde o extremo é mais persistente.\n"
            "- **Limites**: a amostragem é de 6 h, então as durações saem em múltiplos de 6 h; o "
            "extremo é por corte único, não pela banda do índice composto. Os limiares absolutos "
            "(15,6/20/25 m/s) ficam de fora em terra porque o vento quase não os atinge."
        )

    thr = st.selectbox("Limiar do extremo", list(THRESHOLDS), format_func=THRESHOLDS.get)
    nruns = f"n_runs_{thr}"
    tot = int(grid[nruns].sum())
    if tot == 0:
        st.warning("Sem eventos para este limiar.")
        return

    st.subheader("Duração dos eventos")
    dmetric = st.selectbox("Métrica da duração", list(DUR_METRICS), format_func=DUR_METRICS.get)
    sub = grid[grid[nruns] > 0]
    st.plotly_chart(_band_map(sub, f"{dmetric}_{thr}", _hours_fmt, "h", ".0f"), width="stretch")

    st.subheader("Eventos persistentes")
    bucket = st.selectbox("Faixa de duração", list(BUCKETS), format_func=BUCKETS.get)
    subb = grid[grid[f"{bucket}_{thr}"] > 0]
    if len(subb):
        st.plotly_chart(_band_map(subb, f"{bucket}_{thr}", _count_fmt, "eventos", ",.0f"),
                        width="stretch")
    else:
        st.info(f"Nenhum ponto de terra teve evento {BUCKETS[bucket]} neste limiar.")

    st.caption(
        f"Limiar {THRESHOLDS[thr]}: {_br(tot)} eventos no total, "
        f"{_br((grid[nruns] > 0).sum())} células com ao menos um evento. A duração escala pelo "
        "próprio máximo para exaltar as regiões onde o extremo se arrasta; a contagem por faixa "
        "mostra onde o extremo, quando acontece, é persistente."
    )
