"""Footprint continental de vento extremo de ciclone (1979-2020).

Tres mapas lado a lado, por limiar de extremo (percentil local q90/q95/q99):
- Contaminado: todo extremo dentro do raio de busca de 1.100 km do centro.
- Descontaminado: so os extremos dentro do poligono de pertinencia (indice composto >= 0,5,
  referencial rotacionado) da banda pareada ao corte.
- Diferenca: quantos hits o poligono remove em cada celula (contaminado menos descontaminado).

A grade por celula vem de scripts/footprint_grid.py (outputs/footprint_continente_grid.parquet);
o parquet bruto de 4,3M hits nao acompanha o deploy. As bandas de cor dos hits e do vento medio
reproduzem a figura do relatorio; a diferenca e as demais metricas de vento escalam pelo proprio
dado para exaltar o contraste.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

GRID_PATH = Path(__file__).resolve().parents[1] / "outputs" / "footprint_continente_grid.parquet"

# Corte unico por percentil local -> banda do poligono de pertinencia pareada.
THRESHOLDS = {
    "q90": "q90 (percentil local 0,90)",
    "q95": "q95 (percentil local 0,95)",
    "q99": "q99 (percentil local 0,99)",
}
POLY = {"q90": "b90", "q95": "b95", "q99": "b99"}
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
# Diferenca em roxo, para nao se confundir com o footprint vermelho.
DIFF_COLORS = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#54278f"]
WIND_METRICS = {"media": "Vento médio", "max": "Vento máximo",
                "q90": "Vento q90", "q95": "Vento q95", "q99": "Vento q99"}


@st.cache_data(show_spinner=False)
def _read_grid(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(GRID_PATH)


def load_grid() -> pd.DataFrame | None:
    # mtime entra na chave do cache: regenerar a grade invalida o cache em vez de servir colunas velhas.
    if not GRID_PATH.exists():
        return None
    return _read_grid(GRID_PATH.stat().st_mtime)


def _count_fmt(v: float) -> str:
    return f"{int(round(v)):,}".replace(",", ".")


def _ms_fmt(v: float) -> str:
    return f"{v:.1f}".replace(".", ",")


def _discrete_colorscale(colors: list[str]) -> list[list]:
    n = len(colors)
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / n, c], [(i + 1) / n, c]]
    scale[-1][0] = 1.0
    return scale


def _auto_bands(values, fmt=_ms_fmt) -> tuple[list[float], list[str]]:
    """6 faixas por quantil dos dados, ancoradas no maximo observado."""
    qs = np.quantile(values, np.linspace(0, 1, 7))
    edges = [float(qs[i]) for i in range(6)]
    labels = [f"{fmt(qs[i])}-{fmt(qs[i + 1])}" for i in range(5)] + [f"≥{fmt(qs[5])}"]
    return edges, labels


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

    with st.expander("Como é calculado"):
        st.markdown(
            "- **Fonte**: série por ciclone de 1979 a 2020 (ERA5, passo de 6 h), agregada por "
            "célula de 0,25° sobre os pontos de terra da América do Sul.\n"
            "- **Extremo**: num ponto e instante, o vento acima do percentil local escolhido "
            "(q90/q95/q99: vento acima do percentil daquele ponto da grade).\n"
            "- **Contaminado**: todo extremo dentro do raio de busca de 1.100 km do centro do "
            "ciclone. **Descontaminado**: só os extremos dentro do polígono de pertinência "
            "(contorno do índice composto no nível 0,5, referencial rotacionado). **Diferença**: "
            "o que o polígono remove (contaminado menos descontaminado).\n"
            "- **Banda b**: faixa de percentis locais que monta o polígono (b90 cobre 0,80 a 0,90, "
            "b95 cobre 0,89 a 0,95, b99 cobre 0,95 a 0,99); cada corte usa a banda pareada "
            "(q90 com b90, q95 com b95, q99 com b99).\n"
            "- **Por célula**: contagem de hits e, do vento, média, máximo e percentis q90/q95/q99.\n"
            "- **Cores**: os hits usam as faixas da figura do relatório; a diferença e o vento (fora "
            "a média) escalam por quantil ancorado no máximo, para exaltar o contraste.\n"
            "- **Limites**: os limiares absolutos (15,6/20/25 m/s) ficam de fora deste footprint "
            "continental porque em terra o vento quase não os atinge; eles têm base na série por "
            "ponto que inclui o oceano."
        )

    thr = st.selectbox(
        "Limiar do extremo", list(THRESHOLDS), index=0, format_func=THRESHOLDS.get,
        help="O extremo é o vento do ponto acima do percentil local. q99 é o mais raro e "
             "intenso, q90 o mais frequente.",
    )
    band = POLY[thr]
    hc, hd = f"hits_{thr}_contam", f"hits_{thr}_desc"

    g = grid.copy()
    g["removido"] = g[hc] - g[hd]

    st.markdown(
        f"- **Contaminado**: todo extremo ({THRESHOLDS[thr]}) dentro do raio de busca de 1.100 km "
        "do centro do ciclone.\n"
        f"- **Descontaminado**: só os extremos dentro do polígono de pertinência do ciclone "
        f"(contorno do índice composto no nível 0,5, referencial rotacionado), banda {band}.\n"
        "- **Diferença**: quantos hits o polígono remove em cada célula (contaminado menos "
        "descontaminado), a franja de extremos que cai dentro do raio mas fora do ciclone."
    )
    st.caption(
        f"A banda {band} é a faixa de percentis locais que monta o polígono de pertinência: "
        "b90 cobre 0,80 a 0,90, b95 cobre 0,89 a 0,95, b99 cobre 0,95 a 0,99. Cada corte usa o "
        "polígono da banda correspondente (q90 com b90, q95 com b95, q99 com b99). Os limiares "
        "absolutos (15,6/20/25 m/s) ficam de fora deste footprint continental: sobre terra o "
        "vento quase não os atinge, então o extremo em terra é o percentil local; esses absolutos "
        "têm base na série por ponto que inclui o oceano."
    )

    cont_h, cont_c = int(g[hc].sum()), int((g[hc] > 0).sum())
    des_h, des_c = int(g[hd].sum()), int((g[hd] > 0).sum())
    st.dataframe(
        pd.DataFrame({
            "Mapa": ["Contaminado", f"Descontaminado ({band})", "Removido pelo polígono"],
            "Hits": [_br(cont_h), _br(des_h), _br(cont_h - des_h)],
            "Células": [_br(cont_c), _br(des_c), _br(cont_c - des_c)],
        }),
        hide_index=True, width="stretch",
    )

    st.subheader("Contagem de hits")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Contaminado (raio de 1.100 km)")
        st.plotly_chart(_band_map(g[g[hc] > 0], hc, HITS_EDGES, HITS_COLORS,
                                  HITS_LABELS, "hits", ",.0f"), width="stretch")
    with c2:
        st.caption(f"Descontaminado ({band}, polígono)")
        st.plotly_chart(_band_map(g[g[hd] > 0], hd, HITS_EDGES, HITS_COLORS,
                                  HITS_LABELS, "hits", ",.0f"), width="stretch")
    with c3:
        st.caption("Diferença (hits que o polígono remove)")
        rem = g[g["removido"] > 0]
        if len(rem):
            d_edges, d_labels = _auto_bands(rem["removido"], fmt=_count_fmt)
        else:
            d_edges, d_labels = HITS_EDGES, HITS_LABELS
        st.plotly_chart(_band_map(rem, "removido", d_edges, DIFF_COLORS,
                                  d_labels, "removidos", ",.0f"), width="stretch")

    metric = st.selectbox("Métrica do vento", list(WIND_METRICS), format_func=WIND_METRICS.get)
    mc, mb = f"{metric}_{thr}_contam", f"{metric}_{thr}_desc"
    if metric == "media":
        edges, labels = WIND_EDGES, WIND_LABELS
    else:
        edges, labels = _auto_bands(g.loc[g[hc] > 0, mc].dropna())

    st.subheader(f"{WIND_METRICS[metric]} (m/s)")
    both = g[(g[hc] > 0) & (g[hd] > 0)].copy()
    both["d_vento"] = both[mc] - both[mb]
    vmax_w = float(both["d_vento"].abs().max()) or 0.1
    w1, w2, w3 = st.columns(3)
    with w1:
        st.caption("Contaminado")
        st.plotly_chart(_band_map(g[g[hc] > 0], mc, edges, WIND_COLORS,
                                  labels, "m/s", ".1f"), width="stretch")
    with w2:
        st.caption(f"Descontaminado ({band})")
        st.plotly_chart(_band_map(g[g[hd] > 0], mb, edges, WIND_COLORS,
                                  labels, "m/s", ".1f"), width="stretch")
    with w3:
        st.caption("Diferença (contaminado − descontaminado)")
        st.plotly_chart(_diverging_map(both, "d_vento", vmax_w, "Δ m/s"), width="stretch")

    st.caption("Hits contaminado e descontaminado usam as faixas da figura do relatório; a "
               "diferença escala pelos próprios valores (quantis, ancorada no máximo) para exaltar "
               "o contraste. Vento: escolha a métrica (média, máximo ou q90/q95/q99); a média usa "
               "as faixas da figura, as outras seguem os quantis dos dados, iguais nos dois mapas "
               "para dar pra comparar. A diferença do vento é contaminado menos descontaminado, "
               "azul onde sobe ao descontaminar, vermelho onde cai.")
