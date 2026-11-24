"""Nível de retorno do vento extremo de ciclone, por ponto de grade, 1979–2020."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from footprint import WIND_COLORS, _apply_geo, _auto_bands, _discrete_colorscale
from irb_data import DATA_DIR, PRODUCTS, IRBDataError, load_product

GRID_PATH = DATA_DIR / PRODUCTS["gev"]["filename"]
PERIODS = PRODUCTS["gev"]["return_periods_years"]
HIGH_SHAPE = 0.5
PARAMS = {
    "location": ("Posição", "m/s", "Vento típico do máximo anual, em m/s."),
    "scale": ("Escala", "m/s", "Quanto o máximo anual varia de um ano para o outro, em m/s."),
    "shape": ("Forma", "", "Positiva: máximos muito altos são mais prováveis. Negativa: há um teto."),
}
FLAG_COLOR = "#00d4ff"  # legível no fundo escuro da legenda e sobre as células vermelhas

METHOD = """
- **Dado**: vento médio a 10 m (ERA5, não é rajada) nos instantes em que o ponto está dentro da
  área que o índice composto delimita ao redor do ciclone (banda b90), {first}–{last}. O máximo de
  cada ano em cada ponto é o dado do ajuste.
- **Ajuste**: distribuição de valores extremos (GEV) sobre os máximos anuais, feita por grupos
  de pontos. Só entram pontos com pelo menos {min_years} anos de máximo anual; por isso o mapa não
  cobre toda a América do Sul.
- **Nível de retorno de N anos**: vento que, em média, é igualado ou superado em um ano a cada N.
- **Limites**: cada ponto tem de {min_years} a {max_years} máximos anuais, então 50 e 100 anos são
  extrapolação. O máximo é o do vento de ciclone, não o do vento total. O ajuste serve a um
  relatório e precisa de revisão mais rigorosa para um artigo.
- **Pontos marcados com x**: forma acima de 0,5, o que indica cauda muito pesada, em geral por
  pouco dado. Estão no mapa como vieram do ajuste.
- **Ponto retirado**: um ponto andino ficou de fora do mapa porque seus níveis de 50 e 100 anos não
  seguiam o próprio ajuste e ele tem poucos anos de dado.
"""


def load_grid() -> pd.DataFrame:
    return load_product("gev", GRID_PATH)


def _br(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def _method_text() -> str:
    spec = PRODUCTS["gev"]
    first, last = spec["period"]
    return METHOD.format(first=first, last=last, min_years=spec["min_block_years"],
                         max_years=last - first + 1)


def _flag_trace(grid: pd.DataFrame, column: str, unit: str) -> go.Scattergeo:
    flagged = grid[grid["forma_alta"]]
    if column == "shape":
        hover = "forma %{customdata[0]:.2f}<extra>ajuste pouco confiável</extra>"
    else:
        hover = f"%{{customdata[1]:.2f}} {unit} (forma %{{customdata[0]:.2f}})<extra>ajuste pouco confiável</extra>"
    return go.Scattergeo(
        lon=flagged["lon"], lat=flagged["lat"], mode="markers", name="Ajuste pouco confiável",
        marker=dict(symbol="x", size=8, color=FLAG_COLOR, line=dict(width=1.5, color=FLAG_COLOR)),
        customdata=np.column_stack([flagged["shape"], flagged[column]]),
        hovertemplate=hover, showlegend=True,
    )


def _finish(fig: go.Figure, grid: pd.DataFrame, column: str, unit: str) -> go.Figure:
    fig.add_trace(_flag_trace(grid, column, unit))
    _apply_geo(fig)
    fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0))
    return fig


def _wind_band_marker(values: pd.Series, edges, labels, title: str) -> dict:
    colors = [WIND_COLORS[i] for i in np.linspace(0, len(WIND_COLORS) - 1, len(edges)).astype(int)]
    n = len(colors)
    band = np.clip(np.digitize(values, edges) - 1, 0, n - 1)
    return dict(
        color=band + 0.5, colorscale=_discrete_colorscale(colors), cmin=0, cmax=n,
        size=4, symbol="square", line=dict(width=0),
        colorbar={"title": title, "tickmode": "array",
                  "tickvals": [i + 0.5 for i in range(n)], "ticktext": labels,
                  "thickness": 12, "len": 0.9},
    )


def level_map(grid: pd.DataFrame, period: int) -> go.Figure:
    """Mapa do nível de retorno de um período, com a escala comum a todos os períodos."""
    column = f"NR_{period}_estimado"
    edges, labels = _auto_bands(grid[[f"NR_{p}_estimado" for p in PERIODS]].to_numpy().ravel())
    fig = go.Figure(go.Scattergeo(
        lon=grid["lon"], lat=grid["lat"], mode="markers", showlegend=False,
        marker=_wind_band_marker(grid[column], edges, labels, "m/s"),
        customdata=grid[column], hovertemplate="%{customdata:.1f} m/s<extra></extra>",
    ))
    return _finish(fig, grid, column, "m/s")


def param_map(grid: pd.DataFrame, param: str) -> go.Figure:
    """Mapa de um parâmetro do ajuste; a forma usa escala divergente centrada em zero."""
    unit = PARAMS[param][1]
    if param == "shape":
        marker = dict(
            color=grid[param], colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
            cmin=-HIGH_SHAPE, cmax=HIGH_SHAPE, size=4, symbol="square", line=dict(width=0),
            colorbar={"title": "forma", "thickness": 12, "len": 0.9},
        )
        hover = "%{customdata:.2f}<extra></extra>"
    else:
        marker = _wind_band_marker(grid[param], *_auto_bands(grid[param]), unit)
        hover = f"%{{customdata:.2f}} {unit}<extra></extra>"
    fig = go.Figure(go.Scattergeo(
        lon=grid["lon"], lat=grid["lat"], mode="markers", showlegend=False,
        marker=marker, customdata=grid[param], hovertemplate=hover,
    ))
    return _finish(fig, grid, param, unit)


def render_gev() -> None:
    st.header("Nível de retorno do vento extremo de ciclone")
    try:
        grid = load_grid()
    except IRBDataError as exc:
        st.error(str(exc))
        return

    spec = PRODUCTS["gev"]
    flagged = int(grid["forma_alta"].sum())
    st.caption(
        f"Ajuste de valores extremos (GEV) sobre o máximo anual do vento a 10 m de ciclones, "
        f"{spec['period'][0]}–{spec['period'][1]} · {_br(len(grid))} pontos de grade de 0,25° "
        f"na América do Sul com pelo menos {spec['min_block_years']} anos de máximo anual."
    )
    with st.expander("Como é calculado"):
        st.markdown(_method_text())

    st.subheader("Que vento se espera uma vez a cada N anos?")
    period = st.selectbox("Período de retorno", PERIODS, format_func=lambda p: f"{p} anos")
    st.plotly_chart(level_map(grid, period), width="stretch")
    st.caption(
        f"Vento, em m/s, que o ajuste espera ver igualado ou superado em média uma vez a cada "
        f"{period} anos. A escala de cores é a mesma para todos os períodos. "
        f"Pontos com x: ajuste pouco confiável ({_br(flagged)} pontos)."
    )

    st.subheader("Como o ajuste varia de ponto a ponto")
    param = st.selectbox("Parâmetro", list(PARAMS), format_func=lambda k: PARAMS[k][0])
    st.plotly_chart(param_map(grid, param), width="stretch")
    st.caption(PARAMS[param][2])
