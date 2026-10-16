"""Footprint continental de vento extremo de ciclone (1979-2020).

Tres mapas lado a lado:
- Contaminado: todo extremo dentro do raio de busca de 1.100 km do centro.
- Descontaminado: so os extremos dentro do poligono de pertinencia (indice composto >= 0,5,
  referencial rotacionado) da banda escolhida.
- Diferenca: quantos hits o poligono remove em cada celula (contaminado menos descontaminado).

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

# Bandas e cores da figura. Hits: YlOrRd em 6 faixas de contagem. Vento: Reds em 6 faixas (m/s).
HITS_EDGES = [1, 16, 63, 192, 547, 1081]
HITS_COLORS = ["#ffffb2", "#fecc5c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"]
HITS_LABELS = ["1-15", "16-62", "63-191", "192-546", "547-1080", "≥1081"]
WIND_EDGES = [5.3, 6.1, 6.7, 7.3, 7.8, 9.0]
WIND_COLORS = ["#fee5d9", "#fcae91", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"]
WIND_LABELS = ["5,3-6,1", "6,1-6,7", "6,7-7,3", "7,3-7,8", "7,8-9,0", "≥9,0"]
# Diferenca em roxo, para nao se confundir com o footprint vermelho; mesmas faixas de contagem.
DIFF_COLORS = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#54278f"]


@st.cache_data(show_spinner=False)
def load_grid() -> pd.DataFrame | None:
    if not GRID_PATH.exists():
        return None
    return pd.read_parquet(GRID_PATH)


def _discrete_colorscale(colors: list[str]) -> list[list]:
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    scale[-1][0] = 1.0
    return scale


def _apply_geo(fig: go.Figure) -> None:
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


def _band_map(sub: pd.DataFrame, value_col: str, edges: list[float], colors: list[str],
              labels: list[str], title: str, value_fmt: str) -> go.Figure:
    n = len(colors)
    band = np.clip(np.digitize(sub[value_col], edges) - 1, 0, n - 1)
    fig = go.Figure(go.Scattergeo(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker=dict(
            color=band + 0.5, colorscale=_discrete_colorscale(colors), cmin=0, cmax=n,
            size=4, symbol="square", line=dict(width=0),
            colorbar={"title": title, "tickmode": "array",
                      "tickvals": [i + 0.5 for i in range(n)], "ticktext": labels,
                      "thickness": 12, "len": 0.9},
        ),
        customdata=sub[value_col],
        hovertemplate=f"%{{customdata:{value_fmt}}} {title}<extra></extra>",
    ))
    _apply_geo(fig)
    return fig


def _diverging_map(sub: pd.DataFrame, value_col: str, vmax: float, title: str) -> go.Figure:
    fig = go.Figure(go.Scattergeo(
        lon=sub["lon"], lat=sub["lat"], mode="markers",
        marker=dict(
            color=sub[value_col], colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
            cmin=-vmax, cmax=vmax, cmid=0, size=4, symbol="square", line=dict(width=0),
            colorbar={"title": title, "thickness": 12, "len": 0.9},
        ),
        customdata=sub[value_col],
        hovertemplate="%{customdata:+.2f} " + title + "<extra></extra>",
    ))
    _apply_geo(fig)
    return fig


def _br(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def render_footprint() -> None:
    st.header("Footprint continental de vento extremo")
    grid = load_grid()
    if grid is None:
        st.info("Grade não encontrada. Rode `python scripts/footprint_grid.py` para gerar "
                "`outputs/footprint_continente_grid.parquet`.")
        return

    st.caption(
        "Incidência de vento extremo de ciclone sobre o continente sul-americano, 1979-2020, "
        "em células de 0,25°. Célula de terra sem ocorrência fica cinza."
    )

    banda = st.selectbox(
        "Banda do descontaminado", list(BANDS), index=2, format_func=BANDS.get,
        help="A banda é a faixa de percentis locais que define o extremo. b99 é a mais apertada "
             "(corta mais a franja), b90 a mais larga.",
    )
    hits_b, wind_b = f"hits_{banda}", f"wind_{banda}"

    g = grid.copy()
    g["removido"] = g["hits_contam"] - g[hits_b]

    st.markdown(
        "- **Contaminado**: todo extremo dentro do raio de busca de 1.100 km do centro do ciclone.\n"
        f"- **Descontaminado ({banda})**: só os extremos dentro do polígono de pertinência do "
        "ciclone (contorno do índice composto no nível 0,5, referencial rotacionado) da banda.\n"
        "- **Diferença**: quantos hits o polígono remove em cada célula (contaminado menos "
        "descontaminado), ou seja, a franja de extremos que cai dentro do raio mas fora do ciclone."
    )

    cont_h, cont_c = int(g["hits_contam"].sum()), int((g["hits_contam"] > 0).sum())
    des_h, des_c = int(g[hits_b].sum()), int((g[hits_b] > 0).sum())
    st.dataframe(
        pd.DataFrame({
            "Mapa": ["Contaminado", f"Descontaminado {banda}", "Removido pelo polígono"],
            "Hits": [_br(cont_h), _br(des_h), _br(cont_h - des_h)],
            "Células": [_br(cont_c), _br(des_c), _br(cont_c - des_c)],
        }),
        hide_index=True, width="stretch",
    )

    st.subheader("Contagem de hits")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Contaminado (raio de 1.100 km)")
        st.plotly_chart(_band_map(g[g["hits_contam"] > 0], "hits_contam", HITS_EDGES, HITS_COLORS,
                                  HITS_LABELS, "hits", ",.0f"), width="stretch")
    with c2:
        st.caption(f"Descontaminado ({banda}, polígono)")
        st.plotly_chart(_band_map(g[g[hits_b] > 0], hits_b, HITS_EDGES, HITS_COLORS,
                                  HITS_LABELS, "hits", ",.0f"), width="stretch")
    with c3:
        st.caption("Diferença (hits que o polígono remove)")
        st.plotly_chart(_band_map(g[g["removido"] > 0], "removido", HITS_EDGES, DIFF_COLORS,
                                  HITS_LABELS, "removidos", ",.0f"), width="stretch")

    st.subheader("Vento médio (m/s)")
    both = g[(g["hits_contam"] > 0) & (g[hits_b] > 0)].copy()
    both["d_vento"] = both["wind_contam"] - both[wind_b]
    vmax_w = float(both["d_vento"].abs().max()) or 0.1
    w1, w2, w3 = st.columns(3)
    with w1:
        st.caption("Contaminado")
        st.plotly_chart(_band_map(g[g["hits_contam"] > 0], "wind_contam", WIND_EDGES, WIND_COLORS,
                                  WIND_LABELS, "m/s", ".1f"), width="stretch")
    with w2:
        st.caption(f"Descontaminado ({banda})")
        st.plotly_chart(_band_map(g[g[hits_b] > 0], wind_b, WIND_EDGES, WIND_COLORS,
                                  WIND_LABELS, "m/s", ".1f"), width="stretch")
    with w3:
        st.caption("Diferença (contaminado − descontaminado)")
        st.plotly_chart(_diverging_map(both, "d_vento", vmax_w, "Δ m/s"), width="stretch")

    st.caption("Hits: bandas de cor e extensão iguais às da figura do relatório; a diferença de "
               "contagem usa as mesmas faixas, em roxo. Vento: a diferença é contaminado menos "
               "descontaminado, azul onde a média sobe ao descontaminar (saíram pontos de vento "
               "mais fraco), vermelho onde cai, só nas células presentes nos dois mapas.")
