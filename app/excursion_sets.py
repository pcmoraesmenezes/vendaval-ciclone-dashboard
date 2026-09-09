"""Display the supplied theta estimates without recomputing or rescaling them."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[1] / 'outputs' / 'excursion_sets'
PHASES = {'incipient': 'Incipiente', 'intensification': 'Intensificação',
          'mature': 'Maduro', 'decay': 'Decaimento'}
COEFFICIENTS = {'theta5': 'θ₅ — extensão espacial (área/perímetro)',
                'theta2': 'θ₂ — alcance extremal superior'}
BANDS = {'p95': 'p95 — quantis locais 0,89–0,95', 'p99': 'p99 — quantis locais 0,95–0,99'}
AXIS_KM = np.arange(-1100, 1101, 25)

# Keep the same nine solid yellow→red bands used by the existing wind heatmaps.
# Excursion-set values are all valid inside the domain, so no gray under-threshold
# band is needed here.
HEAT_COLORS = [
    "#ffff99", "#ffe64d", "#ffcc00", "#ffb300", "#ff9900",
    "#ff7300", "#ff4d00", "#e62600", "#cc0000",
]


def _discrete_heat_colorscale(vmin, vmax):
    if vmax <= vmin:
        return [[0.0, HEAT_COLORS[0]], [1.0, HEAT_COLORS[-1]]]
    n = len(HEAT_COLORS)
    scale = []
    for i, color in enumerate(HEAT_COLORS):
        lo, hi = i / n, (i + 1) / n
        scale.extend([[lo, color], [hi, color]])
    scale[-1][0] = 1.0
    return scale


def _discrete_heat_colorbar(vmin, vmax, title):
    n = len(HEAT_COLORS)
    boundaries = np.linspace(vmin, vmax, n + 1)
    return {
        "title": title,
        "tickmode": "array",
        "tickvals": np.arange(n) + 0.5,
        "ticktext": [f"{boundaries[i]:.3g}–{boundaries[i + 1]:.3g}" for i in range(n)],
        "ticks": "",
    }


@st.cache_data
def load_theta(phase, band, coefficient):
    if phase not in PHASES or band not in BANDS or coefficient not in COEFFICIENTS:
        raise ValueError('Combinação de theta inválida.')
    suffixes = (['estimates', 'estimates_lower', 'estimates_upper'] if coefficient == 'theta5'
                else ['estimates_theta_2', 'theta2_lower', 'theta2_upper'])
    matrices = []
    for suffix in suffixes:
        path = DATA_DIR / '02_estimativas_theta' / phase / band / f'{phase}_{suffix}.csv'
        matrix = pd.read_csv(path).to_numpy(dtype=float)
        if matrix.shape != (89, 89) or np.isinf(matrix).any() or not np.isfinite(matrix).any():
            raise ValueError(f'Matriz theta inválida: {path.name}')
        matrices.append(matrix)
    estimate, lower, upper = matrices
    if not all(np.array_equal(np.isfinite(estimate), np.isfinite(m)) for m in matrices[1:]):
        raise ValueError('Máscaras dos intervalos theta incompatíveis.')
    if np.any(lower > upper):
        raise ValueError('Limites theta invertidos.')
    return estimate, lower, upper


def theta_figure(fields, coefficient):
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(PHASES.values()),
                        horizontal_spacing=0.12, vertical_spacing=0.15)
    vmin = min(np.nanmin(v[0]) for v in fields.values())
    vmax = max(np.nanmax(v[0]) for v in fields.values())
    for i, (phase, (estimate, lower, upper)) in enumerate(fields.items()):
        row, col = divmod(i, 2)
        boundaries = np.linspace(vmin, vmax, len(HEAT_COLORS) + 1)
        bands = np.digitize(estimate, boundaries[1:-1], right=False).astype(float)
        bands[~np.isfinite(estimate)] = np.nan
        fig.add_trace(go.Heatmap(
            x=AXIS_KM, y=AXIS_KM, z=bands,
            customdata=np.stack([estimate, lower, upper], axis=-1),
            coloraxis='coloraxis', hoverongaps=False,
            hovertemplate=('Leste: %{x} km<br>Norte: %{y} km<br>Estimativa: %{customdata[0]:.1f} km'
                           '<br>Limite inferior: %{customdata[1]:.1f} km'
                           '<br>Limite superior: %{customdata[2]:.1f} km<extra></extra>')),
            row=row+1, col=col+1)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(symbol='cross',
                      size=10, color='white', line=dict(color='#333', width=1)),
                      showlegend=False, hovertemplate='Centro do ciclone<extra></extra>'), row=row+1, col=col+1)
        fig.update_xaxes(title_text='Leste do centro (km)', range=[-1125,1125], row=row+1, col=col+1)
        fig.update_yaxes(title_text='Norte do centro (km)', range=[-1125,1125],
                         scaleanchor='x' if i == 0 else f'x{i+1}', scaleratio=1, row=row+1, col=col+1)
    title = ('θ₅' if coefficient == 'theta5' else 'θ₂') + ' (km)'
    fig.update_layout(height=850, margin=dict(t=40,b=20,l=20,r=20),
        coloraxis=dict(colorscale=_discrete_heat_colorscale(0, len(HEAT_COLORS)), cmin=0,
                       cmax=len(HEAT_COLORS), colorbar=_discrete_heat_colorbar(vmin, vmax, title)))
    return fig


def render_excursion_sets(key_prefix):
    st.subheader('Excursion sets — geometria dos extremos')
    st.caption('Extremos definidos por quantis locais de vento a 10 m (ERA5, 2010–2020, '
               '6-horário). Grade de 25 km, raio de 1.100 km, centrada no ciclone. '
               'Referencial geográfico fixo: norte para cima e leste à direita.')
    c1, c2 = st.columns(2)
    with c1:
        coefficient = st.radio('Coeficiente', list(COEFFICIENTS), format_func=COEFFICIENTS.get,
                               key=f'{key_prefix}_theta')
    with c2:
        band = st.radio('Banda de quantis locais', list(BANDS), format_func=BANDS.get,
                        key=f'{key_prefix}_band')
    st.markdown('**θ₅** descreve a extensão espacial dos extremos pela razão área/perímetro '
                'reescalada. **θ₂** descreve o alcance extremal superior. Ambos são expressos '
                'em quilômetros; não representam velocidade ou frequência do vento.')
    try:
        fields = {phase: load_theta(phase, band, coefficient) for phase in PHASES}
    except (OSError, ValueError) as exc:
        st.error(f'Não foi possível carregar os dados de Excursion sets: {exc}')
        return
    st.plotly_chart(theta_figure(fields, coefficient), width='stretch', key=f'{key_prefix}_map')
    st.caption('Mesma escala de cor nas quatro fases. Passe o cursor para consultar a estimativa '
               'e os limites bootstrap por pixel. Áreas sem dados ficam em branco. '
               'As bandas p95/p99 são quantis locais, não níveis de confiança dos intervalos.')
    if coefficient == 'theta2':
        st.warning('θ₂ é limitado pela extensão do domínio. A interpretação fornecida pela autora '
                   'é que o alcance extremal não distingue as fases dentro deste recorte; '
                   'estimar seu alcance completo exige um domínio maior.')
    with st.expander('Figuras originais — mapas e perfis radiais'):
        for filename in [f'mapa_{coefficient}_{band}.png', f'perfil_{coefficient}.png']:
            path = DATA_DIR / '01_figuras_theta' / filename
            if path.exists():
                st.image(str(path), width='stretch')
            else:
                st.info('Figura original indisponível.')
    with st.expander('Como interpretar e consultar os dados'):
        st.markdown('Resultados fornecidos por Carol em setembro de 2026. Cada fase tem '
                    '11.382 realizações; os ciclones distintos são 1.212 na fase incipiente, '
                    '343 na intensificação, 981 na madura e 322 no decaimento. '
                    'Os intervalos são bootstrap por pixel (200 réplicas), não um teste '
                    'da diferença entre fases. O gradiente norte–sul ainda pode refletir '
                    'latitude ou amostragem. Os mapas mostram o que é extremo para cada '
                    'pixel, não uma zona de excedência de velocidade absoluta.')
        phase = st.selectbox('Fase para baixar', list(PHASES), format_func=PHASES.get,
                             key=f'{key_prefix}_download_phase')
        estimate, lower, upper = fields[phase]
        east, north = np.meshgrid(AXIS_KM, AXIS_KM)
        table = pd.DataFrame({'leste_km': east.ravel(), 'norte_km': north.ravel(),
                              'estimativa_km': estimate.ravel(), 'limite_inferior_km': lower.ravel(),
                              'limite_superior_km': upper.ravel()}).dropna(subset=['estimativa_km'])
        st.download_button('Baixar estimativas e limites (CSV)', table.to_csv(index=False),
                           file_name=f'{phase}_{band}_{coefficient}.csv', mime='text/csv',
                           key=f'{key_prefix}_download')
