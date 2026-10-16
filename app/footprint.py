"""Footprint continental de vento extremo de ciclone (1979-2020).

Contaminado: todo hit no footprint. Descontaminado: so os hits dentro do poligono de
pertinencia do ciclone (indice composto >= 0,5, referencial rotacionado), pela banda escolhida.

A grade por celula vem de scripts/footprint_grid.py (outputs/footprint_continente_grid.parquet);
o parquet bruto de 4,3M hits nao acompanha o deploy. As bandas de cor e a extensao reproduzem a
figura do relatorio (footprint_continente_1979-2020*.png): celula sem ocorrencia fica cinza
(terra), com contorno de pais e de costa.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

GRID_PATH = Path(__file__).resolve().parents[1] / "outputs" / "footprint_continente_grid.parquet"

BANDS = {
    "b90": "b90 (percentis locais 0,80 a 0,90)",
    "b95": "b95 (0,89 a 0,95)",
    "b99": "b99 (0,95 a 0,99)",
}
# Extensao da figura do relatorio.
LON_RANGE = [-85, -33]
LAT_RANGE = [-56, -5]
HALF = 0.125  # meia celula de 0,25 graus

# Bandas e cores da figura. Hits: YlOrRd em 6 faixas de contagem. Vento: Reds em 6 faixas (m/s).
HITS_EDGES = [1, 16, 63, 192, 547, 1081]
HITS_COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"]
HITS_LABELS = ["1-15", "16-62", "63-191", "192-546", "547-1080", "≥1081"]
WIND_EDGES = [5.3, 6.1, 6.7, 7.3, 7.8, 9.0]
WIND_COLORS = ["#fee5d9", "#fcae91", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"]
WIND_LABELS = ["5,3-6,1", "6,1-6,7", "6,7-7,3", "7,3-7,8", "7,8-9,0", "≥9,0"]


@st.cache_data(show_spinner=False)
def load_grid() -> pd.DataFrame | None:
    if not GRID_PATH.exists():
        return None
    return pd.read_parquet(GRID_PATH)


@st.cache_data(show_spinner=False)
def _cells_geojson() -> dict:
    """Um quadrado de 0,25 graus por celula da grade; o id da feature e o indice da linha."""
    grid = load_grid()
    feats = []
    for i, lat, lon in zip(grid.index, grid["lat"], grid["lon"]):
        feats.append({
            "type": "Feature", "id": int(i),
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon - HALF, lat - HALF], [lon + HALF, lat - HALF],
                [lon + HALF, lat + HALF], [lon - HALF, lat + HALF], [lon - HALF, lat - HALF],
            ]]},
        })
    return {"type": "FeatureCollection", "features": feats}


def _discrete_colorscale(colors: list[str]) -> list[list]:
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    scale[-1][0] = 1.0
    return scale


def _apply_geo(fig: go.Figure) -> None:
    fig.update_geos(
        projection_type="equirectangular",
        lonaxis_range=LON_RANGE, lataxis_range=LAT_RANGE,
        showland=True, landcolor="#d9d9d9",
        showocean=True, oceancolor="#ffffff",
        showcountries=True, countrycolor="#9a9a9a", countrywidth=0.5,
        showcoastlines=True, coastlinecolor="#6e6e6e", coastlinewidth=0.6,
        showframe=True, framecolor="#6e6e6e",
        resolution=50,
    )
    fig.update_layout(height=700, margin=dict(t=30, b=0, l=0, r=0),
                      paper_bgcolor="rgba(0,0,0,0)")


def _band_map(sub: pd.DataFrame, value_col: str, edges: list[float], colors: list[str],
              labels: list[str], title: str, value_fmt: str) -> go.Figure:
    n = len(colors)
    band = np.clip(np.digitize(sub[value_col], edges) - 1, 0, n - 1)
    fig = go.Figure(go.Choropleth(
        geojson=_cells_geojson(), locations=sub.index, z=band + 0.5,
        zmin=0, zmax=n, colorscale=_discrete_colorscale(colors), marker_line_width=0,
        colorbar={"title": title, "tickmode": "array",
                  "tickvals": [i + 0.5 for i in range(n)], "ticktext": labels,
                  "thickness": 14, "len": 0.9},
        customdata=sub[value_col],
        hovertemplate=f"%{{customdata:{value_fmt}}} {title}<extra></extra>",
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
        "extremo, e o vento médio dessas ocorrências. Célula de terra sem ocorrência fica cinza."
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
        st.plotly_chart(
            _band_map(sub, hits_col, HITS_EDGES, HITS_COLORS, HITS_LABELS, "hits/célula", ",.0f"),
            width="stretch",
        )
    with c2:
        st.subheader("Vento médio (m/s)")
        st.plotly_chart(
            _band_map(sub, wind_col, WIND_EDGES, WIND_COLORS, WIND_LABELS, "m/s", ".1f"),
            width="stretch",
        )

    st.caption(f"Banda: {banda_label}. Bandas de cor e extensão iguais às da figura do relatório.")
