"""Visualização dos resultados corrigidos IRB, 1979–2020, em terra incluindo ilhas."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from irb_data import (DATA_DIR, PRODUCTS, THRESHOLDS, POLY, IRBDataError,
                      load_product, METHOD, MALVINAS)

GRID_PATH = DATA_DIR / PRODUCTS["footprint"]["filename"]

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


def load_grid() -> pd.DataFrame:
    return load_product("footprint", GRID_PATH)


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
    """Até seis faixas distintas; aceita vazio e valores constantes."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [0.0], ["Sem ocorrência"]
    unique = np.unique(values)
    if len(unique) <= 6:
        labels = [fmt(v) for v in unique]
        if len(set(labels)) != len(labels):
            labels = [f"{v:.6g}".replace(".", ",") for v in unique]
        return unique.tolist(), labels
    bounds = np.unique(np.quantile(values, np.linspace(0, 1, 7)))
    edges = bounds[:-1].tolist()
    labels = [f"{fmt(a)}–{fmt(b)}" for a, b in zip(bounds[:-1], bounds[1:])]
    return edges, labels


def _apply_geo(fig: go.Figure) -> None:
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


def _band_map(sub: pd.DataFrame, value_col: str, edges: list[float], colors: list[str],
              labels: list[str], title: str, value_fmt: str) -> go.Figure:
    colors = [colors[i] for i in np.linspace(0, len(colors) - 1, len(edges)).astype(int)]
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
    st.header("Footprint de vento extremo")
    try:
        grid = load_grid()
    except IRBDataError as exc:
        st.error(str(exc))
        return

    st.caption("Resultados IRB — 1979–2020 · ERA5, vento médio a 10 m · "
               "Terra da América do Sul, incluindo ilhas. Células sem ocorrência ficam cinza.")
    with st.expander("Como é calculado"):
        st.markdown(METHOD)
        st.markdown(
            "Footprint C: comparação estrita **vento > limiar**. Contaminado: hits no raio de "
            "1.100 km; descontaminado: hits também dentro da máscara pareada. "
            "Por célula: contagem de hits, média, máximo e percentis do vento."
        )

    thr = st.selectbox(
        "Limiar do extremo", list(THRESHOLDS), index=0, format_func=THRESHOLDS.get,
        help="Vento acima do percentil local 1979–2020 ou do limiar absoluto selecionado.",
    )
    band = POLY[thr]
    hc, hd = f"hits_{thr}_contam", f"hits_{thr}_desc"

    g = grid.copy()
    g["removido"] = g[hc] - g[hd]

    st.caption(f"Máscara selecionada: {band}. Diferença = contaminado − descontaminado; "
               "hits removidos pelo filtro descritivo, sem interpretação causal.")
    if thr in ("m20", "m25"):
        st.info("Na persistência v2+b90, não há eventos com duração ≥24 h neste limiar.")
    if thr == "m25":
        st.info(MALVINAS)

    cont_h, cont_c = int(g[hc].sum()), int((g[hc] > 0).sum())
    des_h, des_c = int(g[hd].sum()), int((g[hd] > 0).sum())
    st.dataframe(
        pd.DataFrame({
            "Mapa": ["Contaminado", f"Descontaminado ({band})", "Removido pelo polígono"],
            "Hits": [_br(cont_h), _br(des_h), _br(cont_h - des_h)],
            "Células": [_br(cont_c), _br(des_c), _br(int((g["removido"] > 0).sum()))],
        }),
        hide_index=True, width="stretch",
    )

    st.subheader("Contagem de hits")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Contaminado (raio de 1.100 km)")
        st.plotly_chart(_band_map(g[g[hc] > 0], hc, HITS_EDGES, HITS_COLORS,
                                  HITS_LABELS, "hits", ",.0f"), width="stretch", key="footprint_map_1")
    with c2:
        st.caption(f"Descontaminado ({band}, polígono)")
        st.plotly_chart(_band_map(g[g[hd] > 0], hd, HITS_EDGES, HITS_COLORS,
                                  HITS_LABELS, "hits", ",.0f"), width="stretch", key="footprint_map_2")
    with c3:
        st.caption("Diferença (hits que o polígono remove)")
        rem = g[g["removido"] > 0]
        if len(rem):
            d_edges, d_labels = _auto_bands(rem["removido"], fmt=_count_fmt)
        else:
            d_edges, d_labels = HITS_EDGES, HITS_LABELS
        st.plotly_chart(_band_map(rem, "removido", d_edges, DIFF_COLORS,
                                  d_labels, "removidos", ",.0f"), width="stretch", key="footprint_map_3")

    metric = st.selectbox("Métrica do vento", list(WIND_METRICS), format_func=WIND_METRICS.get)
    mc, mb = f"{metric}_{thr}_contam", f"{metric}_{thr}_desc"
    if metric == "media" and thr.startswith("q"):
        edges, labels = WIND_EDGES, WIND_LABELS
    else:
        edges, labels = _auto_bands(g.loc[g[hc] > 0, mc].dropna())

    st.subheader(f"{WIND_METRICS[metric]} (m/s)")
    both = g[(g[hc] > 0) & (g[hd] > 0)].copy()
    both["d_vento"] = both[mc] - both[mb]
    vmax_w = float(both["d_vento"].abs().max())
    if not np.isfinite(vmax_w) or vmax_w <= 0:
        vmax_w = 0.1
    w1, w2, w3 = st.columns(3)
    with w1:
        st.caption("Contaminado")
        st.plotly_chart(_band_map(g[g[hc] > 0], mc, edges, WIND_COLORS,
                                  labels, "m/s", ".1f"), width="stretch", key="footprint_map_4")
    with w2:
        st.caption(f"Descontaminado ({band})")
        st.plotly_chart(_band_map(g[g[hd] > 0], mb, edges, WIND_COLORS,
                                  labels, "m/s", ".1f"), width="stretch", key="footprint_map_5")
    with w3:
        st.caption("Diferença (contaminado − descontaminado)")
        st.plotly_chart(_diverging_map(both, "d_vento", vmax_w, "Δ m/s"), width="stretch", key="footprint_map_6")

    st.caption("Hits usam faixas fixas; vento médio nos percentis usa a escala de referência. "
               "As demais métricas usam faixas adaptadas aos dados, compartilhadas pelos mapas "
               "contaminado e descontaminado. Diferença do vento: azul onde sobe ao filtrar, "
               "vermelho onde cai. Células sem ocorrência não recebem valor de vento.")
