"""Escala de cor quente com piso cinza — fonte única do painel.

Vivia dentro de `streamlit_app.py`, onde só as abas de vento a alcançavam: `composite_index`
não pode importar de lá (é `streamlit_app` quem importa `composite_index`, o caminho de volta
fecharia um ciclo), então a mesma lógica tinha sido reescrita em versão mais pobre nos módulos
de excursion sets e de índice composto — nove faixas fixas, sem piso móvel. Três definições da
mesma paleta é exatamente a forma que a Regra 18 (auditoria de parâmetro vivo) descreve: mudar
a cor num lugar e descobrir meses depois que outro consumidor ficou para trás.

A ideia da escala: o que estiver abaixo de `under_threshold` sai cinza, e as nove cores quentes
se redistribuem no que sobra do intervalo. Subir o piso, portanto, não apaga informação — dá
mais resolução de cor à faixa que interessa, que é o motivo de o piso ser arrastável.
"""

# Paleta enviada por Danilo Couto de Souza em 12/08/2026: cinza para o que é essencialmente
# nada, rampa amarelo->vermelho de nove tons acima disso. Substituiu a escala "Blues".
HEAT_UNDER_COLOR = "#b3b3b3"
HEAT_UNDER_THRESHOLD = 0.01
HEAT_COLORS = [
    "#ffff99", "#ffe64d", "#ffcc00", "#ffb300", "#ff9900",
    "#ff7300", "#ff4d00", "#e62600", "#cc0000",
]


def heat_bands(vmin: float, vmax: float,
               under_threshold: float = HEAT_UNDER_THRESHOLD) -> tuple[list[str], list[float]]:
    """Cores e fronteiras (fração 0-1) das bandas discretas: banda cinza opcional (valores
    < under_threshold) + len(HEAT_COLORS) bandas iguais no resto de [vmin, vmax]. Base
    compartilhada por heat_colorscale (visual) e heat_colorbar (ticks da legenda), pra nunca
    divergir uma da outra.

    `under_threshold` é parametrizável (não só o padrão 0.01) desde 21/08/2026: a aba de
    Frequência de extremos ganhou um slider pra arrastar esse piso pra cima (pedido do Paulo —
    limiares baixos, sobretudo o 0.01 original, deixam quase tudo colorido e escondem onde os
    extremos de verdade se concentram; ver render_wind_spatial_pattern/render_wind_spatial_field).
    """
    frac = max(0.0, min(0.999, (under_threshold - vmin) / (vmax - vmin)))
    n = len(HEAT_COLORS)
    if frac > 0:
        colors = [HEAT_UNDER_COLOR, *HEAT_COLORS]
        edges = [0.0, frac] + [frac + (1 - frac) * i / n for i in range(1, n + 1)]
    else:
        colors = list(HEAT_COLORS)
        edges = [i / n for i in range(n + 1)]
    return colors, edges


def heat_colorscale(vmin: float, vmax: float,
                    under_threshold: float = HEAT_UNDER_THRESHOLD) -> list[list]:
    """Colorscale Plotly discretizada em bandas sólidas (sem gradiente entre elas) — transição
    dura via posições duplicadas no colorscale (`[hi, corA], [hi, corB]`), sem inserir nenhuma
    cor "de fronteira" entre as bandas. Uma versão anterior inseria uma faixa preta bem fina
    (~0,6% do range) em cada fronteira pra marcar a divisa visualmente — bug real: em qualquer
    heatmap com poucas células (ex.: a matriz 2x2 de quadrante), um valor de dado real caindo
    por coincidência dentro dessa faixa fina pintava a célula inteira de preto sólido, sumindo
    com o dado (achado 19/08/2026, matriz de Padrão espacial). Removida — o corte já é nítido
    sem ela (confirmado renderizando a mesma definição isolada, pixel a pixel)."""
    if vmax <= vmin:
        return [[0.0, HEAT_COLORS[0]], [1.0, HEAT_COLORS[-1]]]
    colors, edges = heat_bands(vmin, vmax, under_threshold)
    scale = []
    for i, color in enumerate(colors):
        lo, hi = edges[i], edges[i + 1]
        scale.append([lo, color])
        scale.append([hi, color])
    scale[-1][0] = 1.0  # ponto de flutuação: força o último degrau a fechar exatamente em 1.0
    return scale


def heat_colorbar(vmin: float, vmax: float, title: str,
                  under_threshold: float = HEAT_UNDER_THRESHOLD,
                  fmt: str = ".3g") -> dict:
    """Config de colorbar Plotly com ticks travados nas fronteiras reais de heat_colorscale —
    sem isso, o Plotly desenha uma régua numérica contínua (ticks igualmente espaçados por
    cmin/cmax) por cima de uma escala que já é discreta (achado do Danilo, 12/08/2026)."""
    if vmax <= vmin:
        return {"title": title}
    _, edges = heat_bands(vmin, vmax, under_threshold)
    boundaries = sorted({round(vmin + f * (vmax - vmin), 6) for f in edges})
    return {
        "title": title, "tickmode": "array",
        "tickvals": boundaries, "ticktext": [f"{v:{fmt}}" for v in boundaries],
    }
