"""Footprint continental de vento extremo de ciclone (1979-2020).

Contaminado: todo hit no footprint. Descontaminado: so os hits dentro do poligono de
pertinencia do ciclone (indice composto >= 0,5, referencial rotacionado), pela banda escolhida.

A grade por celula vem de scripts/footprint_grid.py (outputs/footprint_continente_grid.parquet);
o parquet bruto de 4,3M hits nao acompanha o deploy.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from heat_scale import HEAT_COLORS, heat_colorbar, heat_colorscale

GRID_PATH = Path(__file__).resolve().parents[1] / "outputs" / "footprint_continente_grid.parquet"

BANDS = {
    "b90": "b90 (percentis locais 0,80 a 0,90)",
    "b95": "b95 (0,89 a 0,95)",
    "b99": "b99 (0,95 a 0,99)",
}
HITS_TICKS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
# Mesma extensao da figura do relatorio (footprint_continente_1979-2020.png).
LON_RANGE = [-85, -33]
LAT_RANGE = [-56, -8]
CELL_SIZE_PX = 5


@st.cache_data(show_spinner=False)
def load_grid() -> pd.DataFrame | None:
    if not GRID_PATH.exists():
        return None
    return pd.read_parquet(GRID_PATH)


def _apply_geo(fig: go.Figure) -> None:
    fig.update_geos(
        projection_type="equirectangular",
        lonaxis_range=LON_RANGE, lataxis_range=LAT_RANGE,
        showland=True, landcolor="#dcdcdc",
        showocean=True, oceancolor="#ffffff",
        showcountries=True, countrycolor="#9a9a9a", countrywidth=0.5,
        showcoastlines=True, coastlinecolor="#6e6e6e", coastlinewidth=0.6,
        showframe=True, framecolor="#9a9a9a",
        resolution=50,
    )
    fig.update_layout(height=680, margin=dict(t=30, b=0, l=0, r=0))


def _hits_map(df: pd.DataFrame, hits_col: str) -> go.Figure:
    # Escala log: a contagem vai de 1 a milhares e se concentra na Patagonia; no linear
    # quase toda celula cairia na cor mais baixa e o padrao sumiria.
    vmin, vmax = float(np.log10(df[hits_col].min())), float(np.log10(df[hits_col].max()))
    ticks = [t for t in HITS_TICKS if t <= 10 ** vmax]
    fig = go.Figure(go.Scattergeo(
        lon=df["lon"], lat=df["lat"], mode="markers",
        marker=dict(
            color=np.log10(df[hits_col]), colorscale=HEAT_COLORS, cmin=vmin, cmax=vmax,
            size=CELL_SIZE_PX, symbol="square", line=dict(width=0),
            colorbar={"title": "Hits", "tickmode": "array",
                      "tickvals": [np.log10(t) for t in ticks],
                      "ticktext": [f"{t:,}".replace(",", ".") for t in ticks]},
        ),
        customdata=df[hits_col],
        hovertemplate="Lon %{lon:.2f}<br>Lat %{lat:.2f}<br>%{customdata:,.0f} hits<extra></extra>",
    ))
    _apply_geo(fig)
    return fig


def _wind_map(df: pd.DataFrame, wind_col: str) -> go.Figure:
    vmin, vmax = float(df[wind_col].min()), float(df[wind_col].max())
    fig = go.Figure(go.Scattergeo(
        lon=df["lon"], lat=df["lat"], mode="markers",
        marker=dict(
            color=df[wind_col], colorscale=heat_colorscale(vmin, vmax, vmin),
            cmin=vmin, cmax=vmax, size=CELL_SIZE_PX, symbol="square", line=dict(width=0),
            colorbar=heat_colorbar(vmin, vmax, "m/s", vmin, fmt=".1f"),
        ),
        customdata=df[wind_col],
        hovertemplate="Lon %{lon:.2f}<br>Lat %{lat:.2f}<br>%{customdata:.1f} m/s<extra></extra>",
    ))
    _apply_geo(fig)
    return fig


def render_footprint() -> None:
    st.header("Footprint continental de vento extremo")
    grid = load_grid()
    if grid is None:
        st.info("Grade não encontrada. Rode `python scripts/footprint_grid.py` para gerar "
                "`outputs/footprint_continente_grid.parquet`.")
        return

    st.caption(
        "Incidência de vento extremo de ciclone sobre o continente sul-americano, 1979-2020, "
        "em células de 0,25°. Dois mapas: quantas vezes cada célula de terra registrou um "
        "extremo, e o vento médio dessas ocorrências."
    )

    modo = st.radio(
        "Pertinência", ["contaminado", "descontaminado"],
        format_func=lambda k: {"contaminado": "Contaminado (todo o footprint)",
                               "descontaminado": "Descontaminado (dentro do polígono do ciclone)"}[k],
        horizontal=True,
    )
    st.caption(
        "Contaminado conta todo ponto extremo dentro do footprint do ciclone (até 1.100 km do "
        "centro). Descontaminado mantém só os pontos dentro do polígono de pertinência (índice "
        "composto maior ou igual a 0,5, referencial rotacionado) da banda escolhida."
    )

    if modo == "descontaminado":
        banda = st.selectbox("Banda", list(BANDS), index=1, format_func=BANDS.get)
        suffix, banda_label = banda, BANDS[banda]
    else:
        suffix, banda_label = "contam", "todo o footprint"

    hits_col, wind_col = f"hits_{suffix}", f"wind_{suffix}"
    sub = grid[grid[hits_col] > 0]
    total_hits = int(sub[hits_col].sum())
    n_cells = len(sub)
    mean_wind = float(sub[wind_col].mean())

    m1, m2, m3 = st.columns(3)
    m1.metric("Total de hits", f"{total_hits:,}".replace(",", "."))
    m2.metric("Células com hit", f"{n_cells:,}".replace(",", "."))
    m3.metric("Vento médio", f"{mean_wind:.1f} m/s")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Contagem de hits")
        st.plotly_chart(_hits_map(sub, hits_col), width="stretch")
    with c2:
        st.subheader("Vento médio (m/s)")
        st.plotly_chart(_wind_map(sub, wind_col), width="stretch")

    st.caption(
        f"Banda: {banda_label}. Célula de terra sem ocorrência fica em cinza (sem marcador); "
        "o contorno dos países e da costa é o mesmo da figura do relatório."
    )
