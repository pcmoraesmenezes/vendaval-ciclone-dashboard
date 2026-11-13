"""Visualização dos resultados corrigidos IRB, 1979–2020, em terra incluindo ilhas."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from footprint import _auto_bands, _count_fmt, _discrete_colorscale, HITS_COLORS

from irb_data import (DATA_DIR, PRODUCTS, THRESHOLDS, POLY, IRBDataError,
                      load_product, METHOD, MALVINAS)

GRID_PATH = DATA_DIR / PRODUCTS["persistence"]["filename"]

DUR_METRICS = {"dur_mean": "Duração média", "dur_max": "Duração máxima",
               "dur_q90": "Duração q90", "dur_q95": "Duração q95", "dur_q99": "Duração q99"}
BUCKETS = {"ge24": "pelo menos 24 h", "ge48": "pelo menos 48 h",
           "ge72": "pelo menos 72 h", "ge144": "pelo menos 144 h"}
LON_RANGE = [-85, -33]
LAT_RANGE = [-56, -5]


def load_grid() -> pd.DataFrame:
    return load_product("persistence", GRID_PATH)


def _hours_fmt(v: float) -> str:
    return f"{int(round(v))}"


def _geo(fig: go.Figure) -> None:
    # Fundo do tema escuro: geo transparente renderiza branco no Streamlit.
    fig.update_geos(
        scope="south america",
        lataxis_range=LAT_RANGE, lonaxis_range=LON_RANGE,
        showland=True, landcolor="#d9d9d9",
        showocean=False, showlakes=False,
        showcountries=True, countrycolor="#9a9a9a", countrywidth=0.5,
        showsubunits=True, subunitcolor="#bcbcbc", subunitwidth=0.3,
        showcoastlines=True, coastlinecolor="#6e6e6e", coastlinewidth=0.6,
        showframe=False, bgcolor="#0d0d0d", resolution=50,
    )
    fig.update_layout(height=620, margin=dict(t=10, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)")


def _band_map(sub: pd.DataFrame, col: str, fmt_band, title: str, hover_fmt: str) -> go.Figure:
    edges, labels = _auto_bands(sub[col].dropna(), fmt=fmt_band)
    colors = [HITS_COLORS[i] for i in np.linspace(0, len(HITS_COLORS) - 1, len(edges)).astype(int)]
    n = len(colors)
    band = np.clip(np.digitize(sub[col], edges) - 1, 0, n - 1)
    fig = go.Figure(go.Scattergeo(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker=dict(
            color=band + 0.5, colorscale=_discrete_colorscale(colors), cmin=0, cmax=n,
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
    try:
        grid = load_grid()
    except IRBDataError as exc:
        st.error(str(exc))
        return

    st.caption("Resultados IRB, 1979–2020 · ERA5, vento médio a 10 m · "
               "Terra da América do Sul, incluindo ilhas. Células sem ocorrência ficam cinza.")
    with st.expander("Como é calculado"):
        st.markdown(METHOD)
        st.markdown(
            "Persistência v2+b90: série v2 completa, usando os flags de excedência e a máscara "
            "pareada. Um evento reúne passos de 6 h do mesmo ciclone e célula, tolerando "
            "um passo ausente. Duração = último instante − primeiro instante + 6 h, "
            "incluindo a lacuna tolerada. Dois passos ausentes separam eventos; outro "
            "ciclone inicia outro evento. As durações são múltiplos de 6 h."
        )

    thr = st.selectbox("Limiar do extremo", list(THRESHOLDS), format_func=THRESHOLDS.get)
    st.caption(f"Máscara selecionada: {POLY[thr]} (filtro descritivo).")
    if thr in ("m20", "m25"):
        st.info("Não há eventos com duração ≥24 h neste limiar.")
    if thr == "m25":
        st.info(MALVINAS)
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
