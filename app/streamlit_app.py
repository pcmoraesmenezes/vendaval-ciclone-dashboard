"""Dashboard interativo (Streamlit) das análises de ciclones/vento extremo do projeto.

Lê exclusivamente os artefatos já gerados pelo pipeline em outputs/ (CSVs,
resumos Markdown, PNGs e GIFs) — não recalcula nada e não depende de xarray/
netCDF4 (evita o conflito de ABI numpy 1.x/2.x desses pacotes neste ambiente).

Rodar a partir da raiz do repositório:
    streamlit run app/streamlit_app.py
"""
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Caminhos e constantes
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "outputs" / "csv"
SUMMARY_DIR = ROOT / "outputs" / "summaries"
PLOTS_DIR = ROOT / "outputs" / "plots"
STATIC_DIR = ROOT / "app" / "static"


def _ensure_static_gifs() -> None:
    """app/static/ é servido via server.enableStaticServing (ver .streamlit/config.toml) para
    contornar o bug do st.image(): ele reabre a imagem via PIL e a redimensiona quando a largura
    excede ~1460px, o que achata GIFs animados largos (nossos têm ~2150px) em um frame estático.

    Copiamos (não symlinkamos) só os *.gif para dentro de app/static/ por dois motivos:
    (1) o handler de static serving do Streamlit resolve o realpath de cada arquivo e rejeita
    (400) qualquer um cujo alvo real fique fora da raiz — um symlink apontando para
    outputs/plots/ (fora de app/static/) é bloqueado por essa checagem de path traversal;
    (2) outputs/plots/ inteiro passa de 2 GB (PNGs, CSVs, zips) — acima de 1 GB o Streamlit
    desativa o static serving. Só os GIFs somam ~550 MB, com folga sob o limite.
    """
    STATIC_DIR.mkdir(exist_ok=True)
    for gif in PLOTS_DIR.glob("*.gif"):
        dest = STATIC_DIR / gif.name
        if not dest.exists() or dest.stat().st_mtime < gif.stat().st_mtime:
            shutil.copy2(gif, dest)


_ensure_static_gifs()

# Eventos de ciclone identificados no repositório (ID usado no nome dos arquivos, rótulo, período)
EVENTS = [
    {"id": "20100113", "nome": "Ciclone Jan/2010", "periodo": "10–16 jan 2010"},
    {"id": "20100560", "nome": "Ciclone Mai/2010", "periodo": "16–22 mai 2010"},
    {"id": "2010", "nome": "Ciclone Nov/2010", "periodo": "13–27 nov 2010"},
    {"id": "20110424", "nome": "Ciclone Abr/2011", "periodo": "21–27 abr 2011"},
    {"id": "20120402", "nome": "Ciclone Abr/2012", "periodo": "30 mar–5 abr 2012"},
    {"id": "20130312", "nome": "Ciclone Mar/2013", "periodo": "9–15 mar 2013"},
    {"id": "20140413", "nome": "Ciclone Abr/2014", "periodo": "10–16 abr 2014"},
    {"id": "20150911", "nome": "Ciclone Set/2015", "periodo": "8–14 set 2015"},
    {"id": "2019", "nome": "Ciclone Jan/2019", "periodo": "2 fev–2 mar 2019 (dados)"},
]
EVENTS_BY_ID = {e["id"]: e for e in EVENTS}

QUADRANTS = [(1, "NW"), (2, "NE"), (3, "SE"), (4, "SW")]
# Cor fixa por quadrante (não por posição no legend) em todos os gráficos do app —
# paleta categórica validada para contraste/CVD (slots 1-4: azul/laranja/verde-água/amarelo).
QUADRANT_COLORS = {"NW": "#3987e5", "NE": "#d95926", "SE": "#199e70", "SW": "#c98500"}
THRESHOLDS = [
    ("15_6", "15.6 m/s (fixo)"),
    ("20_0", "20.0 m/s (fixo)"),
    ("25_0", "25.0 m/s (fixo)"),
    ("q90", "Q90 (percentil)"),
    ("q95", "Q95 (percentil)"),
    ("q99", "Q99 (percentil)"),
]
ORIENTATIONS = [("fixed", "Fixo (NW/NE/SE/SW geográfico)"), ("rotated", "Rotacionado (alinhado ao movimento)")]

# animate_cyclone_trajectory.py só roda para os 3 eventos "dedicados" (dados baixados
# especificamente para eles, não a série histórica 6/6h). Candidatos em ordem de
# preferência: nome atual do script primeiro, nome legado (regenerações antigas) por último.
TRAJECTORY_GIF_CANDIDATES = {
    "20100560": ["movimento_ciclone_20100560.gif", "movimento_ciclone.gif"],
    "2010": ["movimento_ciclone_20101125.gif", "movimento_ciclone_2010.gif"],
    "2019": ["movimento_ciclone_20190126.gif", "movimento_ciclone_2019.gif"],
}

# Fase de vida do ciclone (as janelas de tempo vêm do Zenodo/LEC, ver outputs/methodology_lec.md,
# mas nenhum termo de energia do LEC é exibido no app — só usado como fonte da fase).
LEC_PHASE_LABELS = {
    "incipient": "Incipiente",
    "intensification": "Intensificação",
    "mature": "Maduro",
    "decay": "Decaimento",
    "residual": "Residual",
}
# Fases analisadas em todo cálculo de "extremos por hora" — residual fica fora a pedido do Paulo (03/08/2026).
LEC_EXTREMES_PHASES = ["incipient", "intensification", "mature", "decay"]
# Fase é ORDINAL (incipiente -> decaimento é uma progressão real), não identidade —
# por isso usa uma rampa de 1 matiz (mais escuro = fase mais avançada), nunca a paleta
# categórica de quadrante (dataviz skill, references/color-formula.md: "ordinal... uma
# matiz, degraus monotônicos de luminosidade"). Degraus da rampa sequencial azul validada
# (references/palette.md), dentro da faixa segura para superfície escura (até o step 600).
PHASE_COLORS = {
    "incipient": "#9ec5f4", "intensification": "#5598e7", "mature": "#256abf", "decay": "#184f95",
}
PHASE_MUTED = "#898781"  # cinza "muted ink" da paleta — usado só para 'unclassified' (contexto, não série)

# Paleta sequencial "quente" para densidade/frequência de vento extremo na página Ciclo de
# Vida (abas Padrão espacial e Distribuição espacial) — enviada por Danilo Couto de Souza
# (12/08/2026): cinza fixo abaixo de 0.01 (deixa "apagado" o que é essencialmente zero),
# rampa amarelo->vermelho de 9 tons acima disso. Substitui a escala "Blues" anterior.
HEAT_UNDER_COLOR = "#b3b3b3"
HEAT_UNDER_THRESHOLD = 0.01
# Piso inicial do slider de limiar nos heatmaps de Frequência de extremos (taxa_contagem_media,
# unidade "extremos/hora") — pedido do Paulo, 21/08/2026: 0.01 deixa quase toda célula colorida
# (a maioria das horas tem pelo menos alguma exceedência baixa), escondendo onde os extremos de
# verdade se concentram. Só afeta essa métrica — "Vento acumulado" (m/s, escala diferente) segue
# usando HEAT_UNDER_THRESHOLD.
EXTREMES_RATE_FLOOR_DEFAULT = 0.2
HEAT_COLORS = [
    "#ffff99", "#ffe64d", "#ffcc00", "#ffb300", "#ff9900",
    "#ff7300", "#ff4d00", "#e62600", "#cc0000",
]
def _heat_bands(vmin: float, vmax: float, under_threshold: float = HEAT_UNDER_THRESHOLD) -> tuple[list[str], list[float]]:
    """Cores e fronteiras (fração 0-1) das bandas discretas: banda cinza opcional (valores
    < under_threshold) + len(HEAT_COLORS) bandas iguais no resto de [vmin, vmax]. Base
    compartilhada por heat_colorscale (visual) e heat_colorbar (ticks da legenda), pra nunca
    divergir uma da outra.

    `under_threshold` é parametrizável (não só o padrão 0.01) desde 21/08/2026: a aba de
    Frequência de extremos ganhou um slider pra arrastar esse piso pra cima (pedido do Paulo —
    limiares baixos, sobretudo o 0.01 original, deixam quase tudo colorido e escondem onde os
    extremos de verdade se concentram; ver render_wind_spatial_pattern/render_wind_spatial_field)."""
    frac = max(0.0, min(0.999, (under_threshold - vmin) / (vmax - vmin)))
    n = len(HEAT_COLORS)
    if frac > 0:
        colors = [HEAT_UNDER_COLOR, *HEAT_COLORS]
        edges = [0.0, frac] + [frac + (1 - frac) * i / n for i in range(1, n + 1)]
    else:
        colors = list(HEAT_COLORS)
        edges = [i / n for i in range(n + 1)]
    return colors, edges


def heat_colorscale(vmin: float, vmax: float, under_threshold: float = HEAT_UNDER_THRESHOLD) -> list[list]:
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
    colors, edges = _heat_bands(vmin, vmax, under_threshold)
    scale = []
    for i, color in enumerate(colors):
        lo, hi = edges[i], edges[i + 1]
        scale.append([lo, color])
        scale.append([hi, color])
    scale[-1][0] = 1.0  # ponto de flutuação: força o último degrau a fechar exatamente em 1.0
    return scale


def extremes_rate_floor(metric: str, key: str) -> float:
    """Piso (cinza abaixo dele) do heatmap de Frequência de extremos (taxa_contagem_media,
    "extremos/hora") — arrastável via slider, pedido do Paulo 21/08/2026, pra revelar onde os
    extremos de verdade se concentram sem o ruído de células com exceedência quase nula. "Vento
    acumulado" (taxa_acumulada_media) tem escala diferente (m/s, não 0-1) — mantém o piso padrão
    fixo (HEAT_UNDER_THRESHOLD), sem slider."""
    if metric != "taxa_contagem_media":
        return HEAT_UNDER_THRESHOLD
    return st.slider(
        "Limiar mínimo exibido (extremos/hora)", min_value=0.0, max_value=1.0,
        value=EXTREMES_RATE_FLOOR_DEFAULT, step=0.01, key=key,
        help="Células abaixo deste valor ficam cinza — dado real, só não extremo o bastante pra "
             "colorir. Arraste pra cima pra ver só onde os extremos de fato se concentram.",
    )


def heat_colorbar(vmin: float, vmax: float, title: str, under_threshold: float = HEAT_UNDER_THRESHOLD) -> dict:
    """Config de colorbar Plotly com ticks travados nas fronteiras reais de heat_colorscale —
    sem isso, o Plotly desenha uma régua numérica contínua (ticks igualmente espaçados por
    cmin/cmax) por cima de uma escala que já é discreta (achado do Danilo, 12/08/2026)."""
    if vmax <= vmin:
        return {"title": title}
    _, edges = _heat_bands(vmin, vmax, under_threshold)
    boundaries = sorted({round(vmin + f * (vmax - vmin), 6) for f in edges})
    return {
        "title": title, "tickmode": "array",
        "tickvals": boundaries, "ticktext": [f"{v:.3g}" for v in boundaries],
    }


# ---------------------------------------------------------------------------
# Helpers de caminho (não hardcodeiam suposições — checam o disco)
# ---------------------------------------------------------------------------

def csv_path(event_id: str, methodology: str) -> Path:
    suffix = f"quadrant_proportions_local_{event_id}.csv" if methodology == "local" else f"quadrant_proportions_{event_id}.csv"
    return CSV_DIR / suffix


def summary_path(event_id: str, methodology: str) -> Path:
    suffix = f"quadrant_summary_local_{event_id}.md" if methodology == "local" else f"quadrant_summary_{event_id}.md"
    return SUMMARY_DIR / suffix


def plot_dir(event_id: str, methodology: str) -> Path:
    name = f"quadrant_plots_local_{event_id}" if methodology == "local" else f"quadrant_plots_{event_id}"
    return PLOTS_DIR / name


def gif_paths(event_id: str, methodology: str) -> tuple[Path, Path]:
    prefix = f"quadrant_plots_local_{event_id}" if methodology == "local" else f"quadrant_plots_{event_id}"
    return PLOTS_DIR / f"{prefix}_fixed.gif", PLOTS_DIR / f"{prefix}_quantiles.gif"


def available_methodologies(event_id: str) -> list[str]:
    return [m for m in ("global", "local") if csv_path(event_id, m).exists()]


def trajectory_gif_path(event_id: str) -> Path | None:
    for name in TRAJECTORY_GIF_CANDIDATES.get(event_id, []):
        p = PLOTS_DIR / name
        if p.exists():
            return p
    return None


def gif_html(path: Path, alt: str = "") -> str:
    """<img> apontando para app/static/ (cópia de outputs/plots/, ver .streamlit/config.toml).

    st.image() reabre a imagem via PIL e a redimensiona quando a largura excede
    ~1460px — nossos GIFs de quadrante têm ~2150px, então esse redimensionamento
    reempacota só o frame atual e a animação se perde (vira imagem estática).
    Servir como arquivo bruto via <img src> evita esse reprocessamento.
    """
    return f'<img src="app/static/{path.name}" alt="{alt}" style="width:100%;height:auto;">'


def count_pngs(event_id: str, methodology: str) -> int:
    d = plot_dir(event_id, methodology)
    return len(list(d.glob("*.png"))) if d.exists() else 0


def col_name(qtype: str, qnum: int, thresh_key: str, suffix: str) -> str:
    return f"{qtype}_q{qnum}_{thresh_key}_{suffix}"


# ---------------------------------------------------------------------------
# Loaders com cache
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_event_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(show_spinner=False)
def load_text(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_grid_percentiles() -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / "grid_point_percentiles.csv")


@st.cache_data(show_spinner=False)
def load_legacy_percentiles() -> dict | None:
    """data/historical_percentiles.json — metodologia antiga: um único Q90/Q95/Q99
    para o subdomínio inteiro (nunca calculou mediana nem máximo)."""
    path = ROOT / "data" / "historical_percentiles.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_max_winds_5years() -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / "cyclone_max_winds_5years.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(show_spinner=False)
def load_quadrant_proportions_5years() -> pd.DataFrame | None:
    """Mesmo schema de `load_event_csv` (colunas `{tipo}_q{n}_{limiar}_pct` etc.) — dá pra
    reaproveitar `quadrant_metric_long`/`quadrant_proportion_table` direto, sem CSV próprio."""
    path = CSV_DIR / "quadrant_proportions_5years_continuous.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(show_spinner=False)
def load_wind_extremes_summary() -> pd.DataFrame | None:
    path = CSV_DIR / "wind_phase_extremes_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["phase"] = pd.Categorical(df["phase"], categories=LEC_EXTREMES_PHASES, ordered=True)
    return df.sort_values(["nivel", "phase"])


@st.cache_data(show_spinner=False)
def load_track_position() -> pd.DataFrame | None:
    path = CSV_DIR / "track_position_by_phase.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, usecols=["track_id", "time", "phase"], dtype={"track_id": str}, parse_dates=["time"])


@st.cache_data(show_spinner=False)
def load_track_wind_speed() -> pd.DataFrame | None:
    path = CSV_DIR / "track_wind_speed_by_phase.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"track_id": str})
    df["time"] = pd.to_datetime(df["time"])
    return df


@st.cache_data(show_spinner=False)
def load_wind_spatial_pattern() -> pd.DataFrame | None:
    path = CSV_DIR / "wind_spatial_pattern_by_phase.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["phase"] = pd.Categorical(df["phase"], categories=LEC_EXTREMES_PHASES, ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_wind_spatial_field_grid() -> pd.DataFrame | None:
    """Heatmap fino (sem quadrante) — ver scripts/analysis/wind_spatial_field_by_phase.py."""
    path = CSV_DIR / "wind_spatial_field_by_phase_grid.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["phase"] = pd.Categorical(df["phase"], categories=LEC_EXTREMES_PHASES, ordered=True)
    return df


@st.cache_data(show_spinner=False)
def load_wind_spatial_field_points() -> pd.DataFrame | None:
    """Scatter bruto (pico de vento por hora, sem agregação) — mesmo pipeline acima."""
    path = CSV_DIR / "wind_spatial_field_by_phase_points.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"track_id": str}, parse_dates=["time"])
    df["phase"] = pd.Categorical(df["phase"], categories=LEC_EXTREMES_PHASES, ordered=True)
    return df


# Posição de cada quadrante numa grade 2x2 (linha, coluna) — Norte em cima, Oeste à esquerda,
# igual à convenção geográfica usada nos mapas do pipeline de quadrantes por evento.
QUADRANT_GRID_POS = {"NW": (0, 0), "NE": (0, 1), "SW": (1, 0), "SE": (1, 1)}


_OCCURRENCE_SUFFIX = re.compile(r"\s+\d+$")


def normalize_phase(phase: str) -> str:
    """Remove o sufixo de reocorrência ('intensification 2' -> 'intensification') — mesma
    normalização de `lec_phase_energetics.py`/`vorticity_phase_extremes.py`/`wind_phase_extremes.py`."""
    return _OCCURRENCE_SUFFIX.sub("", phase)


def quadrant_metric_long(df: pd.DataFrame, qtype: str, thresh_key: str, suffix: str, value_name: str) -> pd.DataFrame:
    """Formato longo (time, quadrante, value_name) para um tipo de quadrante, limiar e coluna
    (`pct`, `max_val` = vento máximo do quadrante, `max_dist_km` = distância desse máximo ao centro)."""
    frames = []
    for qnum, qname in QUADRANTS:
        c = col_name(qtype, qnum, thresh_key, suffix)
        if c not in df.columns:
            continue
        frames.append(pd.DataFrame({"time": df["time"], "quadrante": qname, value_name: df[c]}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "quadrante", value_name])


def quadrant_proportion_table(df: pd.DataFrame, thresh_key: str) -> pd.DataFrame:
    """Tabela curta (1 linha por quadrante) da % média de pontos excedendo `thresh_key` —
    antes os relatórios .md despejavam as 24 linhas (6 limiares x 4 quadrantes) juntas
    numa tabela só; aqui já vem recortada pro limiar escolhido no seletor (pedido do
    Paulo, 18/08/2026 — mais fácil de ler que caçar linha por linha)."""
    order = [q[1] for q in QUADRANTS]
    fixed = quadrant_metric_long(df, "fixed", thresh_key, "pct", "pct").groupby("quadrante")["pct"].mean()
    rot = quadrant_metric_long(df, "rotated", thresh_key, "pct", "pct").groupby("quadrante")["pct"].mean()
    return pd.DataFrame({
        "Quadrante": order,
        "Fixo (%)": [fixed.get(q) for q in order],
        "Rotacionado (%)": [rot.get(q) for q in order],
    })


def quadrant_extreme_table(df: pd.DataFrame, thresh_key: str) -> pd.DataFrame:
    """Idem, para distância (km) e intensidade (m/s) do vento máximo por quadrante,
    já recortada pro limiar selecionado."""
    order = [q[1] for q in QUADRANTS]
    fixed_dist = quadrant_metric_long(df, "fixed", thresh_key, "max_dist_km", "v").groupby("quadrante")["v"].mean()
    rot_dist = quadrant_metric_long(df, "rotated", thresh_key, "max_dist_km", "v").groupby("quadrante")["v"].mean()
    fixed_val = quadrant_metric_long(df, "fixed", thresh_key, "max_val", "v").groupby("quadrante")["v"].mean()
    rot_val = quadrant_metric_long(df, "rotated", thresh_key, "max_val", "v").groupby("quadrante")["v"].mean()
    return pd.DataFrame({
        "Quadrante": order,
        "Fixo (km)": [fixed_dist.get(q) for q in order],
        "Fixo (m/s)": [fixed_val.get(q) for q in order],
        "Rotacionado (km)": [rot_dist.get(q) for q in order],
        "Rotacionado (m/s)": [rot_val.get(q) for q in order],
    })


# column_config em vez de `DataFrame.style.format` — o ambiente de deploy não tem jinja2
# (dependência do `.style`), e column_config é nativo do Streamlit, sem dependência extra.
PCT_COLUMN_CONFIG = {
    "Fixo (%)": st.column_config.NumberColumn("Fixo (%)", format="%.2f%%"),
    "Rotacionado (%)": st.column_config.NumberColumn("Rotacionado (%)", format="%.2f%%"),
}
EXTREME_COLUMN_CONFIG = {
    "Fixo (km)": st.column_config.NumberColumn("Fixo (km)", format="%.1f km"),
    "Fixo (m/s)": st.column_config.NumberColumn("Fixo (m/s)", format="%.1f m/s"),
    "Rotacionado (km)": st.column_config.NumberColumn("Rotacionado (km)", format="%.1f km"),
    "Rotacionado (m/s)": st.column_config.NumberColumn("Rotacionado (m/s)", format="%.1f m/s"),
}


def summary_intro_text(raw_md: str) -> str:
    """Cabeçalho do relatório .md (título + limiares de extremos: Q90/Q95/Q99 locais,
    médias do domínio inteiro) sem as tabelas de 24 linhas que ele mistura — essas viram
    `quadrant_proportion_table`/`quadrant_extreme_table` interativas por limiar. Os dois
    geradores (`ciclone_quadrantes.py` por evento, `continuous_5year_analysis.py` pros
    5 anos) usam títulos de seção levemente diferentes ("## 1. Proporção..." x
    "## Proporção..."), daí checar os dois."""
    for marker in ("## 1. Proporção Média", "## Proporção Média"):
        if marker in raw_md:
            return raw_md.split(marker)[0].rstrip()
    return raw_md


# "Extremos por hora, por fase" — só vento real (ERA5 na posição da trajetória Mendeley/EXWAV),
# nenhum termo de energia nem vorticidade. Ver page_lifecycle() / render_extremes_chart().
WIND_EXTREMES_CFG = {
    "limiar_col": "limiar_wind_speed",
    "count_label": "Extremos por hora (média entre ciclones)",
    "accum_label": "Vento acumulado por hora (m/s, média entre ciclones)",
}


PHASE_LABEL_COLORS = {LEC_PHASE_LABELS[p]: PHASE_COLORS[p] for p in LEC_EXTREMES_PHASES}
FASE_CATEGORY_ORDER = {"Fase": [LEC_PHASE_LABELS[p] for p in LEC_EXTREMES_PHASES]}


def render_extremes_chart(extremes_df: pd.DataFrame, cfg: dict, key_prefix: str) -> None:
    """Os dois gráficos de barra (contagem / acumulado) + tabela completa. Fase é ordinal —
    cor segue PHASE_COLORS (1 matiz, mais escuro = mais tarde no ciclo de vida), não a
    paleta categórica de quadrante."""
    level = st.radio(
        "Limiar de extremo (percentil calculado dentro de cada fase)", ["q90", "q95", "q99"],
        format_func=lambda k: {"q90": "Q90", "q95": "Q95", "q99": "Q99"}[k], horizontal=True,
        key=f"{key_prefix}_level",
    )
    level_df = extremes_df[extremes_df["nivel"] == level].sort_values("phase")
    level_df = level_df.assign(Fase=level_df["phase"].map(LEC_PHASE_LABELS))

    st.caption(
        "Nº de ciclones por fase: " + " · ".join(f"{r.Fase} {r.n_ciclones}" for r in level_df.itertuples())
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_count = px.bar(
            level_df, x="Fase", y="taxa_contagem_media", color="Fase",
            category_orders=FASE_CATEGORY_ORDER, color_discrete_map=PHASE_LABEL_COLORS,
            labels={"taxa_contagem_media": cfg["count_label"]},
        )
        fig_count.update_layout(showlegend=False)
        st.plotly_chart(fig_count, width="stretch")
    with c2:
        fig_accum = px.bar(
            level_df, x="Fase", y="taxa_acumulada_media", color="Fase",
            category_orders=FASE_CATEGORY_ORDER, color_discrete_map=PHASE_LABEL_COLORS,
            labels={"taxa_acumulada_media": cfg["accum_label"]},
        )
        fig_accum.update_layout(showlegend=False)
        st.plotly_chart(fig_accum, width="stretch")

    with st.expander("Tabela completa deste limiar"):
        st.dataframe(
            level_df.drop(columns=["phase"]).rename(columns={
                "nivel": "Limiar", "taxa_contagem_media": "Taxa contagem (média)",
                "taxa_acumulada_media": "Taxa acumulada (média)", cfg["limiar_col"]: "Valor do limiar (m/s)",
                "n_ciclones": "Nº ciclones",
            }),
            width="stretch", hide_index=True,
        )


SPATIAL_NIVEIS = [
    ("15_6", "15.6 m/s (fixo)"), ("20_0", "20.0 m/s (fixo)"), ("25_0", "25.0 m/s (fixo)"),
    ("q90", "Q90 (percentil local)"), ("q95", "Q95 (percentil local)"), ("q99", "Q99 (percentil local)"),
]


def render_wind_spatial_pattern(df: pd.DataFrame, n_horas_por_fase: pd.Series) -> None:
    """192 combinações (4 fases x fixo/rotacionado x 4 quadrantes x 6 limiares). Mostra as
    4 fases lado a lado (pequenos múltiplos, mesma escala de cor) em vez de esconder 3 atrás
    de um seletor — a magnitude por quadrante é a identidade (posição geográfica), a cor é
    só grandeza (1 matiz, sequencial), por isso não esbarra no teto categórico de 3 séries
    para "small multiples" (dataviz skill, references/palette.md). Ver
    scripts/analysis/wind_spatial_pattern_by_phase.py."""
    c1, c2 = st.columns(2)
    with c1:
        quad_type = st.radio(
            "Quadrantes", ["fixed", "rotated"],
            format_func=lambda k: "Fixo (geográfico)" if k == "fixed" else "Rotacionado (movimento)",
            key="spatial_quadtype", horizontal=True,
        )
        st.caption(
            "💡 **Fixo**: NW/NE/SE/SW são direções geográficas reais, fixas no mapa.\n\n"
            "**Rotacionado**: os quadrantes giram junto com o ciclone, alinhados à direção do "
            "seu movimento."
        )
    with c2:
        metric = st.radio(
            "Métrica", ["taxa_contagem_media", "taxa_acumulada_media"],
            format_func=lambda k: "Frequência de extremos" if k == "taxa_contagem_media" else "Vento acumulado",
            key="spatial_metric", horizontal=True,
        )
        st.caption(
            "💡 **Frequência de extremos**: quantas horas excederam o limiar, em proporção às "
            "horas da fase.\n\n**Vento acumulado**: soma do vento nessas horas, também em "
            "proporção — pesa intensidade, não só contagem."
        )
    nivel = st.selectbox(
        "Limiar", [n[0] for n in SPATIAL_NIVEIS], format_func=lambda k: dict(SPATIAL_NIVEIS)[k], key="spatial_nivel"
    )

    sub_all = df[(df["quad_type"] == quad_type) & (df["nivel"] == nivel)]
    if sub_all.empty:
        st.info("Sem dados para esta combinação.")
        return

    unit = "extremos/hora" if metric == "taxa_contagem_media" else "m/s acumulado/hora"
    vmin, vmax = float(sub_all[metric].min()), float(sub_all[metric].max())
    under_threshold = extremes_rate_floor(metric, key="spatial_rate_floor")

    if under_threshold >= vmax:
        st.info(
            f"Nenhuma célula chega a {under_threshold:.2f} {unit} nesta combinação de quadrante/"
            "limiar — todo o valor observado fica abaixo do piso selecionado. Abaixe o slider "
            "pra voltar a ver alguma célula colorida."
        )
        return

    fig = make_subplots(
        rows=1, cols=len(LEC_EXTREMES_PHASES),
        subplot_titles=[LEC_PHASE_LABELS[p] for p in LEC_EXTREMES_PHASES],
        horizontal_spacing=0.04,
    )
    for i, phase in enumerate(LEC_EXTREMES_PHASES, start=1):
        sub = sub_all[sub_all["phase"] == phase]
        grid = np.full((2, 2), np.nan)
        for row in sub.itertuples():
            r, c = QUADRANT_GRID_POS[row.quadrante]
            grid[r, c] = getattr(row, metric)
        fig.add_trace(
            go.Heatmap(
                # z/text invertidos em linha (grid[::-1]) para casar com y=["S","N"]: no eixo Y
                # padrão (cresce de baixo pra cima), o primeiro rótulo fica embaixo — "S" embaixo,
                # "N" em cima, como um mapa de verdade.
                z=grid[::-1], x=["O", "L"], y=["S", "N"], coloraxis="coloraxis",
                text=np.round(grid[::-1], 3), texttemplate="%{text}", textfont={"size": 12},
                hovertemplate="%{y}%{x}: %{z:.3f}<extra></extra>",
            ),
            row=1, col=i,
        )
    fig.update_layout(
        coloraxis={
            "colorscale": heat_colorscale(vmin, vmax, under_threshold), "cmin": vmin, "cmax": vmax,
            "colorbar": heat_colorbar(vmin, vmax, unit, under_threshold),
        },
        height=280, margin=dict(t=40, b=10, l=10, r=10),
    )
    fig.update_xaxes(side="top")
    n_visible_by_phase = {
        LEC_PHASE_LABELS[p]: int((sub_all[sub_all["phase"] == p][metric] >= under_threshold).sum())
        for p in LEC_EXTREMES_PHASES
    }
    if any(v == 0 for v in n_visible_by_phase.values()):
        zeradas = [p for p, v in n_visible_by_phase.items() if v == 0]
        st.caption(
            f"⚠️ Com este piso, {', '.join(zeradas)} fica{'m' if len(zeradas) > 1 else ''} sem "
            "nenhuma célula colorida (todo o quadrante abaixo do limiar nessa fase)."
        )
    st.plotly_chart(fig, width="stretch")

    n_by_phase = sub_all.drop_duplicates("phase").set_index("phase")["n_ciclones"]
    st.caption(
        "Horas classificadas na fase: "
        + " · ".join(
            f"{LEC_PHASE_LABELS[p]} {n_horas_por_fase.get(p, 0):,} ({n_by_phase.get(p, 0)} ciclones)"
            for p in LEC_EXTREMES_PHASES
        )
    )

    with st.expander("Tabela completa deste limiar"):
        st.dataframe(
            sub_all.rename(columns={
                "phase": "Fase", "quad_type": "Tipo", "quadrante": "Quadrante", "nivel": "Limiar",
                "taxa_contagem_media": "Freq. extremos (média)", "taxa_acumulada_media": "Acumulado (média)",
                "n_ciclones": "Nº ciclones",
            }),
            width="stretch", hide_index=True,
        )


# Mesma resolução de scripts/analysis/wind_spatial_field_by_phase.py::BIN_KM — só para rótulo.
FIELD_BIN_KM = 100

# Eixos por tipo de quadrante — mesma grandeza (km relativo ao centro), rótulo muda porque
# o referencial muda (geográfico fixo vs. alinhado ao movimento do ciclone).
FIELD_AXIS_LABELS = {
    "fixed": {"x": "Leste (km)", "y": "Norte (km)", "x_col": "east_km", "y_col": "north_km"},
    "rotated": {"x": "Perpendicular ao movimento, à esquerda (km)", "y": "Ao longo do movimento (km)",
                "x_col": "left_km", "y_col": "fwd_km"},
}


def render_wind_spatial_field(grid_df: pd.DataFrame, points_df: pd.DataFrame, n_horas_por_fase: pd.Series) -> None:
    """Mesma pergunta de render_wind_spatial_pattern (onde o vento extremo se concentra ao
    redor do centro, por fase), sem discretizar em 4 quadrantes — ver docstring de
    scripts/analysis/wind_spatial_field_by_phase.py para a metodologia completa e a
    justificativa da resolução de 100km. Duas visões complementares, não uma substituindo
    a outra (decisão com o Paulo, 11/08/2026): heatmap fino (mesma fórmula de taxa da
    versão por quadrante, só que numa grade contínua) e scatter bruto (posição real do
    pico de vento por hora, sem nenhuma agregação — nem entre ciclones, nem espacial)."""
    c1, c2 = st.columns(2)
    with c1:
        quad_type = st.radio(
            "Referencial", ["fixed", "rotated"],
            format_func=lambda k: "Fixo (geográfico)" if k == "fixed" else "Rotacionado (movimento)",
            key="field_quadtype", horizontal=True,
        )
        st.caption(
            "💡 **Fixo**: NW/NE/SE/SW são direções geográficas reais, fixas no mapa.\n\n"
            "**Rotacionado**: os quadrantes giram junto com o ciclone, alinhados à direção do "
            "seu movimento."
        )
    with c2:
        nivel = st.selectbox(
            "Limiar", [n[0] for n in SPATIAL_NIVEIS], format_func=lambda k: dict(SPATIAL_NIVEIS)[k], key="field_nivel"
        )
    axis = FIELD_AXIS_LABELS[quad_type]

    st.subheader("Heatmap fino — taxa por célula de 100km")
    sub_all = grid_df[(grid_df["quad_type"] == quad_type) & (grid_df["nivel"] == nivel)]
    if sub_all.empty:
        st.info("Sem dados para esta combinação.")
    else:
        metric = st.radio(
            "Métrica", ["taxa_contagem_media", "taxa_acumulada_media"],
            format_func=lambda k: "Frequência de extremos" if k == "taxa_contagem_media" else "Vento acumulado",
            key="field_metric", horizontal=True,
        )
        st.caption(
            "💡 **Frequência de extremos**: quantas horas excederam o limiar, em proporção às "
            "horas da fase.\n\n**Vento acumulado**: soma do vento nessas horas, também em "
            "proporção — pesa intensidade, não só contagem."
        )
        unit = "extremos/hora" if metric == "taxa_contagem_media" else "m/s acumulado/hora"
        vmin, vmax = float(sub_all[metric].min()), float(sub_all[metric].max())
        under_threshold = extremes_rate_floor(metric, key="field_rate_floor")

        if under_threshold >= vmax:
            st.info(
                f"Nenhuma célula chega a {under_threshold:.2f} {unit} nesta combinação de "
                "referencial/limiar — todo o valor observado fica abaixo do piso selecionado. "
                "Abaixe o slider pra voltar a ver alguma célula colorida."
            )
            return

        fig = make_subplots(
            rows=1, cols=len(LEC_EXTREMES_PHASES),
            subplot_titles=[LEC_PHASE_LABELS[p] for p in LEC_EXTREMES_PHASES],
            horizontal_spacing=0.06,
        )
        for i, phase in enumerate(LEC_EXTREMES_PHASES, start=1):
            sub = sub_all[sub_all["phase"] == phase]
            fig.add_trace(
                go.Heatmap(
                    x=sub["cell_center_x_km"], y=sub["cell_center_y_km"], z=sub[metric],
                    coloraxis="coloraxis", hovertemplate=f"{axis['x']}: %{{x}}<br>{axis['y']}: %{{y}}<br>%{{z:.3f}} {unit}<extra></extra>",
                ),
                row=1, col=i,
            )
            fig.update_xaxes(title_text=axis["x"] if i == 1 else None, range=[-1150, 1150], row=1, col=i)
            fig.update_yaxes(title_text=axis["y"] if i == 1 else None, range=[-1150, 1150], row=1, col=i)
        fig.update_layout(
            coloraxis={
                "colorscale": heat_colorscale(vmin, vmax, under_threshold), "cmin": vmin, "cmax": vmax,
                "colorbar": heat_colorbar(vmin, vmax, unit, under_threshold),
            },
            height=420, margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, width="stretch")

        n_visible_by_phase = {
            LEC_PHASE_LABELS[p]: int((sub_all[sub_all["phase"] == p][metric] >= under_threshold).sum())
            for p in LEC_EXTREMES_PHASES
        }
        if any(v == 0 for v in n_visible_by_phase.values()):
            zeradas = [p for p, v in n_visible_by_phase.items() if v == 0]
            st.caption(
                f"⚠️ Com este piso, {', '.join(zeradas)} fica{'m' if len(zeradas) > 1 else ''} "
                "sem nenhuma célula colorida (todo o campo abaixo do limiar nessa fase)."
            )

        by_phase = sub_all.groupby("phase", observed=True).agg(
            n_ciclones=("n_ciclones", "first"),
            contrib_min=("n_ciclones_contrib", "min"),
            contrib_max=("n_ciclones_contrib", "max"),
        )
        st.caption(
            "Horas classificadas na fase: "
            + " · ".join(
                f"{LEC_PHASE_LABELS[p]} {n_horas_por_fase.get(p, 0):,} "
                f"({by_phase.loc[p, 'n_ciclones'] if p in by_phase.index else 0} ciclones)"
                for p in LEC_EXTREMES_PHASES
            )
            + ". Ciclones que efetivamente contribuíram (excederam) em cada célula variam de "
            + " · ".join(f"{LEC_PHASE_LABELS[p]} {int(by_phase.loc[p,'contrib_min'])}–{int(by_phase.loc[p,'contrib_max'])}"
                         for p in LEC_EXTREMES_PHASES if p in by_phase.index)
            + ". Células sem nenhuma exceedência ficam em branco (não são zero fabricado)."
        )

    st.divider()
    st.subheader("Scatter bruto — pico de vento por hora, sem agregação")
    st.caption(
        "Cada ponto é uma hora real de um ciclone: a posição exata do vento máximo dentro do "
        "raio de 1100km naquela hora (não uma média entre ciclones, não uma célula). Filtrado "
        "para horas em que o pico excedeu o limiar selecionado acima."
    )
    exceed_col = f"exceed_{nivel}"
    pts = points_df[points_df[exceed_col]]
    if pts.empty:
        st.info("Nenhuma hora excedeu esse limiar no pico.")
    else:
        # render_mode="webgl" (Scattergl) foi tentado e removido (11/08/2026): 4 facetas
        # trocadas repetidamente pelos radios/selectbox acima criam vários contextos WebGL em
        # sequência, e o limite de contextos simultâneos do navegador faz o gráfico renderizar
        # em branco depois de algumas trocas — reportado pelo Paulo em produção. SVG padrão não
        # tem esse teto e aqui o volume é pequeno (≤1.560 pontos no total, bem abaixo do ponto
        # em que WebGL passa a valer a pena).
        ws_min, ws_max = float(pts["wind_speed_ms"].min()), float(pts["wind_speed_ms"].max())
        fig_pts = px.scatter(
            pts, x=axis["x_col"], y=axis["y_col"], color="wind_speed_ms", facet_col="phase",
            category_orders={"phase": LEC_EXTREMES_PHASES},
            labels={axis["x_col"]: axis["x"], axis["y_col"]: axis["y"], "wind_speed_ms": "Vento no pico (m/s)"},
            color_continuous_scale=heat_colorscale(ws_min, ws_max), range_color=(ws_min, ws_max),
        )
        fig_pts.update_coloraxes(colorbar=heat_colorbar(ws_min, ws_max, "Vento no pico (m/s)"))
        fig_pts.for_each_annotation(lambda a: a.update(text=LEC_PHASE_LABELS.get(a.text.split("=")[-1], a.text)))
        fig_pts.update_traces(marker=dict(size=6, opacity=0.55))
        fig_pts.update_xaxes(range=[-1150, 1150])
        fig_pts.update_yaxes(range=[-1150, 1150], matches=None)
        fig_pts.update_layout(height=420, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_pts, width="stretch")
        st.caption(f"{len(pts):,} horas com pico acima do limiar (de {len(points_df):,} horas classificadas no total).")


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------

def _render_case_study_overview():
    st.subheader("Eventos analisados")
    rows = [{"Evento": e["nome"], "Período": e["periodo"]} for e in EVENTS]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_case_study_explorer():
    with st.container(border=True):
        options = {f"{e['nome']} ({e['id']}) — {e['periodo']}": e["id"] for e in EVENTS}
        label = st.selectbox("Ciclone", list(options.keys()))
        event_id = options[label]

        methods = available_methodologies(event_id)
        if not methods:
            st.error("Nenhum CSV encontrado para este evento.")
            return

        if len(methods) > 1:
            methodology = st.radio(
                "Metodologia de percentil", methods,
                format_func=lambda m: "Local (por ponto de grade — atual)" if m == "local" else "Global (subdomínio inteiro — legado)",
                index=methods.index("local") if "local" in methods else 0,
                horizontal=True,
            )
        else:
            methodology = methods[0]
            tag = "Local (atual)" if methodology == "local" else "Global (legado)"
            st.caption(f"Este evento só tem análise pela metodologia **{tag}**.")
        st.caption(
            "💡 **Global**: um único limiar vale pra toda a região. **Local** (metodologia atual): "
            "cada célula de grade tem seu próprio limiar."
        )

        c1, c2 = st.columns(2)
        orientation = c1.selectbox("Quadrantes", [o[0] for o in ORIENTATIONS], format_func=lambda k: dict(ORIENTATIONS)[k])
        thresh_key = c2.selectbox("Limiar de vento", [t[0] for t in THRESHOLDS], format_func=lambda k: dict(THRESHOLDS)[k])
        st.caption(
            "💡 **Fixo**: NW/NE/SE/SW são direções geográficas reais, fixas no mapa.\n\n"
            "**Rotacionado**: os quadrantes giram junto com o ciclone, alinhados à direção do "
            "seu movimento."
        )

    df = load_event_csv(str(csv_path(event_id, methodology)))
    category_orders = {"quadrante": [q[1] for q in QUADRANTS]}

    tab_pct, tab_wind, tab_gifs, tab_summary = st.tabs(
        ["📈 Proporção de área", "💨 Vento máximo", "🎞️ Animações", "📄 Resumo"]
    )

    with tab_pct:
        long_df = quadrant_metric_long(df, orientation, thresh_key, "pct", "pct")

        st.subheader("Proporção de área excedente por quadrante ao longo do tempo")
        fig = px.line(
            long_df, x="time", y="pct", color="quadrante", markers=True,
            color_discrete_map=QUADRANT_COLORS, category_orders=category_orders,
            labels={"time": "Data/Hora (UTC)", "pct": "% da área do quadrante excedendo o limiar", "quadrante": "Quadrante"},
        )
        fig.update_layout(legend_title_text="Quadrante", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        avg = long_df.groupby("quadrante")["pct"].mean().reindex([q[1] for q in QUADRANTS])
        cols = st.columns(4)
        for col, (qname, val) in zip(cols, avg.items()):
            col.metric(f"Média {qname}", f"{val:.2f}%")

    with tab_wind:
        st.caption(
            "A biblioteca de análise (`ciclone_quadrantes.py`) identifica, a cada instante, o vento máximo "
            "dentro de cada quadrante e a distância desse ponto ao centro do ciclone — sinal chave de "
            "assimetria que o resumo em Markdown já tabula como médias, mas que aqui fica navegável no tempo."
        )
        max_val_df = quadrant_metric_long(df, orientation, thresh_key, "max_val", "vento_max")
        max_dist_df = quadrant_metric_long(df, orientation, thresh_key, "max_dist_km", "dist_km")

        wcol1, wcol2 = st.columns(2)
        with wcol1:
            fig_val = px.line(
                max_val_df, x="time", y="vento_max", color="quadrante", markers=True,
                color_discrete_map=QUADRANT_COLORS, category_orders=category_orders,
                labels={"time": "Data/Hora (UTC)", "vento_max": "Vento máximo no quadrante (m/s)", "quadrante": "Quadrante"},
            )
            fig_val.update_layout(legend_title_text="Quadrante", hovermode="x unified")
            st.plotly_chart(fig_val, width="stretch")
        with wcol2:
            fig_dist = px.line(
                max_dist_df, x="time", y="dist_km", color="quadrante", markers=True,
                color_discrete_map=QUADRANT_COLORS, category_orders=category_orders,
                labels={"time": "Data/Hora (UTC)", "dist_km": "Distância do vento máximo ao centro (km)", "quadrante": "Quadrante"},
            )
            fig_dist.update_layout(legend_title_text="Quadrante", hovermode="x unified")
            st.plotly_chart(fig_dist, width="stretch")

        peak_rows = []
        for qnum, qname in QUADRANTS:
            sub = max_val_df[max_val_df["quadrante"] == qname].dropna(subset=["vento_max"])
            if sub.empty:
                continue
            peak = sub.loc[sub["vento_max"].idxmax()]
            dist_at_peak = max_dist_df.loc[
                (max_dist_df["quadrante"] == qname) & (max_dist_df["time"] == peak["time"]), "dist_km"
            ]
            peak_rows.append({
                "Quadrante": qname,
                "Vento máximo (m/s)": round(peak["vento_max"], 1),
                "Quando": peak["time"].strftime("%Y-%m-%d %H:%M"),
                "Distância ao centro (km)": round(dist_at_peak.iloc[0], 1) if not dist_at_peak.empty else None,
            })
        if peak_rows:
            st.dataframe(pd.DataFrame(peak_rows), width="stretch", hide_index=True)
        else:
            st.info("Nenhum excedente registrado para esta combinação de limiar/orientação.")

    with tab_gifs:
        gif_fixed, gif_quant = gif_paths(event_id, methodology)
        st.subheader("Mapas de quadrante por hora")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.caption("Limiares fixos")
            if gif_fixed.exists():
                st.markdown(gif_html(gif_fixed, alt="Animação de quadrantes — limiares fixos"), unsafe_allow_html=True)
            else:
                st.info("Animação ainda não gerada para esta metodologia neste evento.")
        with gcol2:
            st.caption("Percentis (quantis)")
            if gif_quant.exists():
                st.markdown(gif_html(gif_quant, alt="Animação de quadrantes — percentis"), unsafe_allow_html=True)
            else:
                st.info("Animação ainda não gerada para esta metodologia neste evento.")

        traj_gif = trajectory_gif_path(event_id)
        if traj_gif is not None:
            st.divider()
            st.subheader("Trajetória e Vento")
            st.caption(
                "Vento (contourf) + trajetória do centro rastreada frame a frame "
                "(`animate_cyclone_trajectory.py`) — só disponível para os eventos dedicados."
            )
            st.markdown(gif_html(traj_gif, alt="Animação de trajetória e vento"), unsafe_allow_html=True)

        pdir = plot_dir(event_id, methodology)
        n_pngs = count_pngs(event_id, methodology)
        if n_pngs:
            st.divider()
            st.subheader("Inspecionar hora específica")
            idx = st.slider("Hora (índice do passo de tempo)", 0, n_pngs - 1, 0)
            # O sufixo do PNG depende da FAMÍLIA DE LIMIAR (fixo x quantil), não da
            # orientação do quadrante — cada gif/PNG já mostra os dois orientações lado a lado.
            suffix = "fixed" if thresh_key in ("15_6", "20_0", "25_0") else "quantiles"
            matches = list(pdir.glob(f"hour_{idx:03d}_{suffix}*.png"))
            if matches:
                st.image(str(matches[0]), width="stretch")
            else:
                st.info("Frame não encontrado para esta hora.")

    with tab_summary:
        sp = summary_path(event_id, methodology)
        if sp.exists():
            st.markdown(summary_intro_text(load_text(str(sp))))
            st.divider()

            st.subheader("Proporção de área excedente, por quadrante")
            st.caption(
                "% dos pontos de grade dentro do raio de 1100 km, por quadrante, que excederam o "
                "limiar selecionado acima."
            )
            st.dataframe(
                quadrant_proportion_table(df, thresh_key),
                width="stretch", hide_index=True, column_config=PCT_COLUMN_CONFIG,
            )

            st.divider()
            st.subheader("Vento máximo, por quadrante")
            st.caption(
                "Distância (km) e velocidade (m/s) do ponto de vento mais forte em cada quadrante, "
                "só nos instantes em que o limiar selecionado foi excedido."
            )
            st.dataframe(
                quadrant_extreme_table(df, thresh_key),
                width="stretch", hide_index=True, column_config=EXTREME_COLUMN_CONFIG,
            )
        else:
            st.info("Nenhum relatório resumo encontrado para este evento.")


def _render_case_study_track_explorer():
    st.caption(
        "Diferente das outras 2 abas (9 ciclones nomeados, só ERA5 nos eventos escolhidos): aqui o "
        "universo é o catálogo completo — milhares de ciclones (Mendeley/Zenodo), vento real (ERA5) "
        "só onde track e período se sobrepõem. Mesma fonte de dado da página *Ciclo de Vida*, só que "
        "olhando 1 ciclone específico do início ao fim em vez de agregar por fase."
    )
    raw = load_track_wind_speed()
    if raw is None:
        st.error(
            "`outputs/csv/track_wind_speed_by_phase.csv` não encontrado — rode "
            "`.venv/bin/python scripts/analysis/track_wind_speed.py` a partir da raiz do repositório."
        )
        return

    st.markdown(
        "**A vida de um ciclone específico, do início ao fim** — veja o vento subir e "
        "descer conforme ele passa por cada fase (cor = fase)."
    )
    track_ids = sorted(raw["track_id"].unique())
    selected = st.selectbox(f"Track ID ({len(track_ids):,} disponíveis, 2010-2019)", track_ids)

    track_df = raw[raw["track_id"] == selected].sort_values("time").copy()
    track_df["Fase"] = track_df["phase"].map(normalize_phase).apply(lambda p: LEC_PHASE_LABELS.get(p, "Não classificada"))
    track_colors = {**PHASE_LABEL_COLORS, "Residual": PHASE_MUTED, "Não classificada": PHASE_MUTED}

    fig_track = px.line(
        track_df, x="time", y="wind_speed_ms", color="Fase", markers=True,
        category_orders={"Fase": [*FASE_CATEGORY_ORDER["Fase"], "Residual", "Não classificada"]},
        color_discrete_map=track_colors,
        labels={"time": "Data/Hora (UTC)", "wind_speed_ms": "Velocidade do vento (m/s)"},
    )
    fig_track.update_layout(hovermode="x unified")
    st.plotly_chart(fig_track, width="stretch")

    peak = track_df.loc[track_df["wind_speed_ms"].idxmax()]
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Vento máximo", f"{peak['wind_speed_ms']:.1f} m/s")
    pcol2.metric("Quando", peak["time"].strftime("%Y-%m-%d %H:%M"))
    pcol3.metric("Fase no pico", peak["Fase"])


def page_case_study():
    st.header("🔬 Estudos de Caso")
    st.caption(
        "9 ciclones nomeados em detalhe (Visão Geral, Explorar Evento) + exploração livre do "
        "catálogo completo de milhares de ciclones (Explorar Ciclone) — diferente do *Ciclo de "
        "Vida* (outra página), que agrega o catálogo inteiro por fase em vez de olhar 1 ciclone "
        "por vez."
    )

    tab_overview, tab_explore, tab_track = st.tabs(["📊 Visão Geral", "🌀 Explorar Evento", "🧭 Explorar Ciclone"])

    with tab_overview:
        _render_case_study_overview()

    with tab_explore:
        _render_case_study_explorer()

    with tab_track:
        _render_case_study_track_explorer()


CLIMATOLOGY_STATS = [
    ("Mediana", "Limiar_Mediana_ms", None),
    ("Q90", "Limiar_Q90_ms", "q90"),
    ("Q95", "Limiar_Q95_ms", "q95"),
    ("Q99", "Limiar_Q99_ms", "q99"),
    ("Máximo", "Limiar_Maximo_ms", None),
]


def _render_lifecycle_climatology():
    st.caption(
        "Mediana, percentis Q90/Q95/Q99 e máximo de vento — metodologia **local** (atual): "
        "calculados **por célula de grade** (0.25°×0.25°), usando a série histórica 2010–2020 "
        "inteira, sem recorte por evento."
    )

    df = load_grid_percentiles()
    legacy = load_legacy_percentiles()

    stat_label = st.radio("Estatística", [s[0] for s in CLIMATOLOGY_STATS], horizontal=True)
    value_col, legacy_key = next((c, k) for label, c, k in CLIMATOLOGY_STATS if label == stat_label)

    can_diff = legacy is not None and legacy_key is not None
    show_diff = False
    if can_diff:
        show_diff = st.checkbox(
            f"Mostrar diferença ponto a ponto em relação ao percentil global antigo "
            f"({legacy[legacy_key]:.2f} m/s — {stat_label} único para todo o subdomínio)"
        )
    else:
        st.caption(
            "Comparação com a metodologia antiga (percentil global único, `data/historical_percentiles.json`) "
            "só está disponível para Q90/Q95/Q99 — a versão antiga nunca calculou mediana nem máximo."
        )

    pivot = df.pivot(index="Latitude", columns="Longitude", values=value_col).sort_index(ascending=True)

    if show_diff:
        diff = pivot - legacy[legacy_key]
        bound = float(diff.abs().max().max())
        fig = px.imshow(
            diff.values, x=pivot.columns, y=pivot.index, origin="lower",
            color_continuous_scale="RdBu_r", range_color=(-bound, bound), aspect="auto",
            labels={"x": "Longitude", "y": "Latitude", "color": "Δ m/s"},
            title=f"{stat_label} local − {stat_label} global antigo (m/s)",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Azul: o limiar local é **menor** que o antigo global naquele ponto (o método antigo superestimava "
            "o extremo ali). Vermelho: o limiar local é **maior** (o método antigo mascarava o extremo, "
            "diluído pela média com o oceano aberto) — a motivação original da migração para percentis locais "
            "(ver aba *Metodologia*)."
        )
    else:
        # px.imshow renderiza a grade como uma única imagem raster (leve para ~62 mil células),
        # em vez de um go.Heatmap interativo célula-a-célula (muito mais pesado no navegador).
        fig = px.imshow(
            pivot.values, x=pivot.columns, y=pivot.index, origin="lower",
            color_continuous_scale="Blues", aspect="auto",
            labels={"x": "Longitude", "y": "Latitude", "color": "m/s"},
            title=f"{stat_label} local de velocidade do vento (m/s)",
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Domínio: {len(df):,} pontos de grade · Latitude [{df['Latitude'].min()}, {df['Latitude'].max()}] · "
        f"Longitude [{df['Longitude'].min()}, {df['Longitude'].max()}]"
    )


def _render_lifecycle_continuous():
    st.caption(
        "Toda hora registrada entre 2010 e 2020 (passo de 6h), comparada contra os percentis locais "
        "da aba *Climatologia* — sem recortar por ciclone nomeado. Enquanto *Estudos de Caso → "
        "Explorar Evento* olha um ciclone de cada vez, esta aba olha o vento extremo contínuo no "
        "tempo, como ele aconteceu."
    )

    winds = load_max_winds_5years()

    fig = px.line(
        winds, x="time", y="max_wind_speed",
        labels={"time": "Data/Hora (UTC)", "max_wind_speed": "Vento máximo no raio de 1100 km (m/s)"},
    )
    fig.update_traces(line=dict(width=1, color=QUADRANT_COLORS["NW"]))
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Resumo consolidado")
    summary_file = SUMMARY_DIR / "quadrant_summary_5years_continuous.md"
    if summary_file.exists():
        st.markdown(summary_intro_text(load_text(str(summary_file))))

    prop_df_5y = load_quadrant_proportions_5years()
    if prop_df_5y is not None:
        thresh_key_5y = st.selectbox(
            "Limiar de vento", [t[0] for t in THRESHOLDS], format_func=lambda k: dict(THRESHOLDS)[k],
            key="continuous_thresh_key",
        )
        st.caption(
            "% de pontos da grade, por quadrante, que excederam o limiar acima — toda a série de "
            "6 em 6h, 2010–2020."
        )
        st.dataframe(
            quadrant_proportion_table(prop_df_5y, thresh_key_5y),
            width="stretch", hide_index=True, column_config=PCT_COLUMN_CONFIG,
        )
    elif summary_file.exists():
        st.info("CSV de proporções não encontrado — mostrando só o resumo textual.")


def page_lifecycle():
    st.header("🌪️ Ciclo de Vida do Ciclone")
    st.caption(
        "Diferente de *Estudos de Caso* (9 ciclones escolhidos, só ERA5): aqui o vento real (ERA5) é "
        "cruzado com milhares de ciclones catalogados — trajetória e fase de vida (incipiente → "
        "intensificação → maduro → decaimento) já vêm juntas na mesma fonte, ponto a ponto, sem "
        "cruzamento por ID entre bases diferentes. Mesma geometria de quadrantes da página anterior, "
        "agora quebrada por fase. Metodologia completa na aba *Metodologia*."
    )

    position = load_track_position()
    raw = load_track_wind_speed()
    if raw is None:
        st.error(
            "`outputs/csv/track_wind_speed_by_phase.csv` não encontrado — rode "
            "`.venv/bin/python scripts/analysis/track_wind_speed.py` a partir da raiz do repositório."
        )
        return

    raw_phase_base = raw["phase"].map(normalize_phase)
    n_horas_por_fase = raw_phase_base[raw_phase_base.isin(LEC_EXTREMES_PHASES)].value_counts()

    tab_overview, tab_spatial, tab_field, tab_extremes, tab_dist, tab_climate, tab_continuous = st.tabs(
        ["🔍 Visão Geral", "🗺️ Padrão espacial (quadrantes)", "🌐 Distribuição espacial (sem quadrante)",
         "⚡ Extremos por fase", "📊 Distribuição por fase",
         "🌡️ Climatologia", "📈 Análise Contínua"]
    )

    with tab_overview:
        st.caption(
            "As 4 primeiras abas partem do mesmo dado — vento real (ERA5) na trajetória do ciclone, "
            "com fase de vida já atribuída ponto a ponto na mesma fonte. Diferem só na forma de "
            "agregar. Climatologia e Análise Contínua não dependem de trajetória nem de fase — olham "
            "o vento (ERA5) no domínio inteiro. A série temporal de 1 ciclone específico (antiga aba "
            "'Explorar ciclone') mudou de casa — está em *Estudos de Caso → Explorar Ciclone*, junto "
            "das outras explorações de instância única."
        )
        st.markdown(
            "- **Padrão espacial** — vento extremo por quadrante (NW/NE/SE/SW), por fase.\n"
            "- **Distribuição espacial** — mesma pergunta, sem dividir em quadrantes.\n"
            "- **Extremos por fase** — 1 número por fase: quantas horas excederam o limiar e quanto.\n"
            "- **Distribuição por fase** — distribuição bruta do vento por fase, sem agregação.\n"
            "- **Climatologia** — percentis de vento por célula de grade, sem recorte por evento nem "
            "por fase.\n"
            "- **Análise Contínua** — toda a série 2010–2020 de 6 em 6h, sem recorte por evento nem "
            "por fase."
        )
        st.subheader("Limitações conhecidas")
        st.caption("Como os dados chegam até aqui, e onde cada etapa perde cobertura:")
        if position is not None:
            n_total = position["track_id"].nunique()
            n_em_era5 = raw["track_id"].nunique()
            n_horas_total = len(raw)
            n_horas_classificadas = int(n_horas_por_fase.sum())
            n_ciclones_com_fase = raw.loc[raw_phase_base.isin(LEC_EXTREMES_PHASES), "track_id"].nunique()
            n_unclass_pct = (raw["phase"] == "unclassified").mean()
            st.markdown(
                f"1. **Fonte**: trajetória (posição + vorticidade) e fase de vida (incipiente → "
                f"intensificação → maduro → decaimento) vêm juntas, ponto a ponto, num catálogo único. "
                f"**{n_total:,}** ciclones ao todo (1979–2021).\n"
                f"2. **Cobertura do ERA5**: só **{n_em_era5:,}** dos {n_total:,} caem dentro do período "
                f"que o ERA5 baixado cobre (2010–2020). Os demais não têm vento real disponível.\n"
                f"3. **Resolução temporal**: a trajetória é hora a hora, o ERA5 é 6-horário (00h/06h/12h/"
                f"18h) — cada hora usa o timestep mais próximo, desvio de até 3h. Numa fronteira de ano, "
                f"alguns pontos (de {n_horas_total:,}) chegam a 4–5h; volume irrisório.\n"
                f"4. **Pontos sem fase**: cerca de **{n_unclass_pct:.1%}** das horas ficam `unclassified` "
                f"— perto do início/fim de cada trajetória, antes da fase incipiente começar ou depois "
                f"do decaimento terminar. {n_horas_classificadas:,} de {n_horas_total:,} horas ficam com "
                f"fase válida, **{n_ciclones_com_fase} ciclones distintos** nas análises por fase."
            )

    with tab_spatial:
        st.markdown("**Vento extremo por quadrante (NW/NE/SE/SW), quebrado por fase de vida do ciclone.**")
        st.caption(
            "Uma hora conta como extrema pro quadrante se pelo menos 1 ponto de grade ali passou do "
            "limiar. Não importa se o quadrante tem 50 pontos de grade ou 200. A taxa é: quantas "
            "dessas horas aconteceram, dividido pelo total de horas da fase."
        )
        spatial_df = load_wind_spatial_pattern()
        if spatial_df is None:
            st.error(
                "`outputs/csv/wind_spatial_pattern_by_phase.csv` não encontrado — rode "
                "`.venv/bin/python scripts/analysis/wind_spatial_pattern_by_phase.py`."
            )
        else:
            render_wind_spatial_pattern(spatial_df, n_horas_por_fase)

    with tab_field:
        st.markdown(
            "**Onde exatamente ao redor do centro o vento extremo se concentra.** Em vez de 4 "
            "quadrantes largos, aqui a resolução é mais fina: células de 100km, ou a posição exata "
            "de cada pico de vento, sem agrupar nada."
        )
        st.caption(
            "A célula conta como extrema numa hora se pelo menos 1 ponto de grade nativo ali passou "
            "do limiar. A taxa é: quantas dessas horas aconteceram, dividido pelo total de horas da "
            "fase."
        )
        field_grid_df = load_wind_spatial_field_grid()
        field_points_df = load_wind_spatial_field_points()
        if field_grid_df is None or field_points_df is None:
            st.error(
                "`outputs/csv/wind_spatial_field_by_phase_grid.csv`/`_points.csv` não encontrados — rode "
                "`.venv/bin/python scripts/analysis/wind_spatial_field_by_phase.py`."
            )
        else:
            render_wind_spatial_field(field_grid_df, field_points_df, n_horas_por_fase)

    with tab_extremes:
        st.markdown(
            "**Em qual fase da vida do ciclone o vento extremo é mais frequente e mais "
            "intenso.** Duas barras por fase: quantas horas excederam o limiar, e quanto o vento "
            "passou dele nessas horas."
        )
        extremes_df = load_wind_extremes_summary()
        if extremes_df is None:
            st.error(
                "`outputs/csv/wind_phase_extremes_summary.csv` não encontrado — rode "
                "`.venv/bin/python scripts/analysis/wind_phase_extremes.py`."
            )
        else:
            render_extremes_chart(extremes_df, WIND_EXTREMES_CFG, key_prefix="vento")

    with tab_dist:
        st.markdown(
            "**Como o vento se distribui dentro de cada fase** — não só os extremos: todo o "
            "intervalo de valores observados, do mais fraco ao mais forte."
        )
        dist_df = raw.copy()
        dist_df["phase_base"] = dist_df["phase"].map(normalize_phase)
        dist_df = dist_df[dist_df["phase_base"].isin(LEC_EXTREMES_PHASES)]
        dist_df = dist_df.assign(Fase=dist_df["phase_base"].map(LEC_PHASE_LABELS))

        fig_box = px.box(
            dist_df, x="Fase", y="wind_speed_ms", color="Fase",
            category_orders=FASE_CATEGORY_ORDER, color_discrete_map=PHASE_LABEL_COLORS,
            labels={"wind_speed_ms": "Velocidade do vento (m/s)"},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, width="stretch")

        stats = dist_df.groupby("Fase", observed=True)["wind_speed_ms"].describe()[["count", "mean", "50%", "max"]]
        stats.columns = ["Nº observações", "Média (m/s)", "Mediana (m/s)", "Máximo (m/s)"]
        st.dataframe(stats.round(2), width="stretch")

    with tab_climate:
        _render_lifecycle_climatology()

    with tab_continuous:
        _render_lifecycle_continuous()


def page_about():
    st.header("ℹ️ Metodologia")
    st.caption(
        "Complemento dos textos que já aparecem em cada seção — não a única explicação. Se você só "
        "quer entender um número específico, procure primeiro na própria seção onde ele aparece."
    )
    methodology_file = ROOT / "outputs" / "methodology.md"
    if methodology_file.exists():
        st.markdown(load_text(str(methodology_file)))

    st.divider()
    lec_methodology_file = ROOT / "outputs" / "methodology_lec.md"
    if lec_methodology_file.exists():
        st.markdown(load_text(str(lec_methodology_file)))

    st.divider()
    st.caption("Documentação completa de cada script do pipeline: `docs/SCRIPTS.md` (na raiz do repositório).")


def _set_nav_choice(target: str) -> None:
    # Callbacks rodam ANTES do script reexecutar do topo — setar session_state aqui é
    # seguro. Fazer isso depois de st.sidebar.radio(key="nav_choice") já ter instanciado
    # o widget nesta mesma execução (ex.: dentro de `if st.button(...): st.session_state[...] = ...`)
    # dispara StreamlitAPIException, porque o valor de um widget com key não pode ser
    # sobrescrito depois de instanciado na mesma rodada.
    st.session_state["nav_choice"] = target


def _nav_card(title: str, caption: str, target: str, key: str) -> None:
    with st.container(border=True):
        st.button(title, key=key, use_container_width=True, on_click=_set_nav_choice, args=(target,))
        st.caption(caption)


def page_home():
    st.header("🌀 Vendaval/Ciclone — Guia rápido")
    st.markdown(
        "Análises de vento extremo (ERA5, Hemisfério Sul) ao redor de ciclones — onde ele se "
        "concentra e como isso muda ao longo da vida do ciclone."
    )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        _nav_card(
            "🔬 Estudos de Caso",
            "9 ciclones nomeados, analisados em detalhe — mapas por hora, animações — mais a "
            "exploração livre de qualquer ciclone do catálogo completo, 1 por vez.",
            "🔬 Estudos de Caso", "nav_case_study",
        )
        _nav_card(
            "🌪️ Ciclo de Vida",
            "Milhares de ciclones catalogados, vento real (ERA5) quebrado por fase de vida — "
            "incipiente, intensificação, maduro, decaimento.",
            "🌪️ Ciclo de Vida", "nav_lifecycle",
        )
    with c2:
        _nav_card(
            "ℹ️ Metodologia",
            "Como cada número é calculado — complemento das explicações que já aparecem em "
            "cada seção, não a única fonte.",
            "ℹ️ Metodologia", "nav_about",
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Análise de Ciclones — Vendaval/Ciclone", page_icon="🌀", layout="wide")

st.sidebar.title("🌀 Vendaval/Ciclone")
st.sidebar.caption("Vento extremo (ERA5) + ciclo de vida do ciclone (catálogo próprio) — Hemisfério Sul")

PAGES = {
    "🏠 Início": page_home,
    "🔬 Estudos de Caso": page_case_study,
    "🌪️ Ciclo de Vida": page_lifecycle,
    "ℹ️ Metodologia": page_about,
}
choice = st.sidebar.radio("Navegação", list(PAGES.keys()), key="nav_choice")
PAGES[choice]()
