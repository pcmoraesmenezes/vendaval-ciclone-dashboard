"""Composite 0-1 index combining extreme frequency with the supplied theta geometry.

The three source fields answer different questions in incompatible units (extremes per
hour vs. kilometres), so each one is min-max normalised before they are averaged. The
normalisation sweeps the four phases together within a band, which is the same choice the
rest of the app already makes for its colour scales (see excursion_sets.theta_figure and
streamlit_app.render_wind_spatial_field): it keeps "which phase is more extreme" readable
instead of giving every phase a pixel worth 1.0.

Referential (fixed or rotated) is a single choice applied to all three components: the
frequency grid and the theta estimates are both read for the same `quad_type`/`referencial`,
so pairing them never crosses two different referentials.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from excursion_sets import AXIS_KM, HEAT_COLORS, PHASES, REFERENCE_FRAMES, load_theta
from heat_scale import heat_colorbar, heat_colorscale

GRID_CSV = Path(__file__).resolve().parents[1] / 'outputs' / 'csv' / 'wind_spatial_field_by_phase_grid.csv'

# The frequency field is binned at 100km; the theta grids are native 25km. Confirmed from the
# CSV itself (cell_center_x_km == cell_x * 100 + 50), not assumed: the analysis scripts that
# produced it live in the upstream vendaval repo, not here.
FREQ_BIN_KM = 100

# Both sides of the join now use the same band definition (corrected 10/09/2026). The theta
# estimator sweeps a range of local-quantile thresholds — 0.89-0.95 for p95 and 0.95-0.99 for
# p99 — so the frequency field is now the mean exceedance rate over that same range, produced
# upstream as the `b95`/`b99` levels of wind_spatial_field_by_phase.py. Until this change the
# frequency used a single threshold (q95/q99) and the two halves of the map called "p95" two
# different things.
BANDS = {'p90': 'p90', 'p95': 'p95', 'p99': 'p99'}
BAND_TO_NIVEL = {'p90': 'b90', 'p95': 'b95', 'p99': 'b99'}
BAND_QUANTILE_RANGE = {'p90': '0,80–0,90', 'p95': '0,89–0,95', 'p99': '0,95–0,99'}

# Quatro painéis numa linha só, como a aba "Distribuição espacial" já faz — num monitor largo
# uma grade 2x2 de painéis quadrados ou fica com metade da largura vazia, ou precisa de uma
# altura que obriga a rolar.
#
# A LARGURA é do container (st.plotly_chart(width='stretch')), não fixa: largura fixa nunca
# encosta nas bordas e sempre sobra faixa vazia. Só a ALTURA é fixa, e escolhida para ser
# próxima da largura que cada painel recebe num monitor largo (~1600px / 4 colunas ≈ 400px
# por painel). Isso importa por causa da trava de proporção 1:1 (`scaleanchor`, necessária
# para o domínio circular não virar elipse): o mapa desenhado é sempre um quadrado de lado
# min(largura_do_painel, altura_do_painel). Se a altura for muito menor que a largura, os
# mapas encolhem e sobra espaço lateral — que foi exatamente o sintoma relatado.
FIG_HEIGHT = 500
COMPONENT_FIG_HEIGHT = 440

COMPONENTS = {'freq': 'Frequência de extremos', 'theta2': 'θ₂ — alcance extremal superior',
              'theta5': 'θ₅ — extensão espacial'}

# Short labels for the per-phase table, where the full names would not fit as columns.
COMPONENT_SHORT = {'freq': 'Frequência', 'theta2': 'θ₂ (área alcançada)',
                   'theta5': 'θ₅ (extensão)'}


@st.cache_data
def load_frequency_grid(band, referencial='fixed'):
    """Frequency field replicated from its native 100km cells onto the 25km theta grid.

    Every 25km pixel takes the value of the 100km cell it falls inside (cell = floor(km/100)),
    so one cell feeds up to 16 pixels. This adds no information: it only makes the two grids
    addressable by the same index. Cells absent from the CSV (no exceedance anywhere in the
    phase) stay NaN rather than becoming a fabricated zero.
    """
    nivel = BAND_TO_NIVEL[band]
    df = pd.read_csv(GRID_CSV)
    disponiveis = sorted(set(df[df['quad_type'] == referencial]['nivel']))
    df = df[(df['quad_type'] == referencial) & (df['nivel'] == nivel)]
    if df.empty:
        # Deliberadamente sem fallback para o limiar único (q95/q99): cair de volta nele
        # remontaria o índice com a definição de percentil que esta mudança veio corrigir,
        # e o mapa ficaria plausível e errado, sem ninguém notar. Melhor não desenhar nada.
        raise ValueError(
            f'O campo de frequência ainda não foi regerado com os níveis de banda: '
            f'{GRID_CSV.name} não tem "{nivel}" (níveis presentes: {", ".join(disponiveis)}). '
            f'Rode `.venv/bin/python scripts/analysis/wind_spatial_field_by_phase.py` no '
            f'repositório do pipeline e copie o CSV para cá. As outras abas não dependem '
            f'disso e seguem funcionando normalmente.')
    east, north = np.meshgrid(AXIS_KM, AXIS_KM)
    cell_x = np.floor(east / FREQ_BIN_KM).astype(int)
    cell_y = np.floor(north / FREQ_BIN_KM).astype(int)
    fields = {}
    for phase in PHASES:
        sub = df[df['phase'] == phase]
        if sub.empty:
            raise ValueError(f'Fase {phase} ausente no campo de frequência ({nivel}).')
        lookup = {(int(x), int(y)): float(v) for x, y, v
                  in zip(sub['cell_x'], sub['cell_y'], sub['taxa_contagem_media'])}
        grid = np.full(cell_x.shape, np.nan)
        for (x, y), value in lookup.items():
            grid[(cell_x == x) & (cell_y == y)] = value
        fields[phase] = grid
    return fields


def min_max_norm(fields):
    """Scale to 0-1 with a single min/max swept across the four phases."""
    stack = np.concatenate([f[np.isfinite(f)] for f in fields.values()])
    vmin, vmax = float(stack.min()), float(stack.max())
    if vmax <= vmin:
        raise ValueError('Campo constante: não há intervalo para normalizar.')
    return {phase: (f - vmin) / (vmax - vmin) for phase, f in fields.items()}, vmin, vmax


@st.cache_data
def load_composite(band, referencial='fixed'):
    """Return per-phase raw components, normalised components and their mean.

    The composite exists only where all three fields exist. Pixels carried by fewer than
    three components are dropped instead of averaged over a smaller denominator, so the
    "divide by 3" the index is named after is always literally true.
    """
    if band not in BANDS:
        raise ValueError('Banda inválida.')
    if referencial not in REFERENCE_FRAMES:
        raise ValueError('Referencial inválido.')
    raw = {'freq': load_frequency_grid(band, referencial),
           'theta2': {p: load_theta(p, band, 'theta2', referencial)[0] for p in PHASES},
           'theta5': {p: load_theta(p, band, 'theta5', referencial)[0] for p in PHASES}}

    mask = {p: np.logical_and.reduce([np.isfinite(raw[c][p]) for c in COMPONENTS]) for p in PHASES}
    raw = {c: {p: np.where(mask[p], raw[c][p], np.nan) for p in PHASES} for c in COMPONENTS}

    normed, ranges = {}, {}
    for component in COMPONENTS:
        normed[component], vmin, vmax = min_max_norm(raw[component])
        ranges[component] = (vmin, vmax)

    composite = {p: sum(normed[c][p] for c in COMPONENTS) / len(COMPONENTS) for p in PHASES}
    return raw, normed, composite, ranges, mask


# O limiar corre dentro do intervalo REALMENTE observado, não de [0, 1]. O índice é uma média
# de três campos normalizados, e média não chega nas pontas: na prática ele ocupa ~0,15 a ~0,80,
# então uma escala presa a [0, 1] gasta as cores do topo com valores que não existem e comprime
# todos os dados reais em dois ou três tons. Ancorar nos extremos observados é o que a aba de
# Frequência de extremos já faz (`vmin, vmax = sub_all[metric].min(), .max()`), e era a isso que
# o pedido de 12/09/2026 se referia: "deixar dinâmico esse lance do cinza, igual tem no dos
# extremos". O valor absoluto não se perde — a legenda mostra os números reais, e o cursor, a
# tabela e o download continuam lendo o índice em si.
#
# Isto revoga a escolha anterior de fixar a escala em [0, 1] (que existia para deixar visível
# "o quão longe de 1" uma fase estava): a leitura que o painel precisa entregar é o contraste
# espacial entre células, e a distância até 1 continua legível na legenda numérica.
FLOOR_STEPS = 100

SCALE_TITLE = 'Índice composto'


def observed_range(composite):
    """Menor e maior índice nas quatro fases juntas — o intervalo que a cor vai cobrir."""
    stack = np.concatenate([f[np.isfinite(f)] for f in composite.values()])
    return float(stack.min()), float(stack.max())


def index_floor(key, vmin, vmax):
    """Limiar arrastável dentro do intervalo observado: abaixo dele o pixel sai cinza e as nove
    cores se redistribuem no que sobra. Começa no próprio mínimo — nada cinza, campo inteiro
    visível — e subir a partir dali é o que separa os menores valores uns dos outros em vez de
    deixá-los todos no mesmo tom."""
    step = round((vmax - vmin) / FLOOR_STEPS, 4)
    return st.slider(
        'Limiar mínimo exibido (índice)', min_value=round(vmin, 3), max_value=round(vmax, 3),
        value=round(vmin, 3), step=step, key=key,
        help='Pixels abaixo deste valor ficam cinza, e as cores se reorganizam no intervalo que '
             'sobra. Serve para enxergar diferença entre os valores mais baixos, que numa escala '
             'de 0 a 1 sairiam todos com a mesma cor.')


def composite_figure(raw, normed, composite, floor=None, vrange=None, frame=None):
    """2x2 small multiples on a fixed 0-1 scale, one panel per phase.

    The scale is pinned to 0-1 rather than to the observed range: the index is already
    normalised, so stretching it again per figure would hide how far from 1 a phase is.

    `floor` is the grey cut: below it a pixel is drawn grey and the nine warm colours split
    what is left of [floor, 1]. The heatmap carries the raw index as `z` (not a band number),
    so moving the floor only changes the colourscale — the data behind every pixel stays the
    same and the hover keeps reading the true value.
    """
    frame = frame or REFERENCE_FRAMES['fixed']
    vmin, vmax = vrange if vrange else observed_range(composite)
    floor = vmin if floor is None else floor
    fig = make_subplots(rows=1, cols=len(PHASES), subplot_titles=list(PHASES.values()),
                        horizontal_spacing=0.04)
    for i, phase in enumerate(PHASES):
        row, col = 0, i
        value = composite[phase]
        customdata = np.stack(
            [value] + [normed[c][phase] for c in COMPONENTS] + [raw[c][phase] for c in COMPONENTS],
            axis=-1)
        fig.add_trace(go.Heatmap(
            x=AXIS_KM, y=AXIS_KM, z=value, customdata=customdata,
            coloraxis='coloraxis', hoverongaps=False,
            hovertemplate=(
                f"{frame['x_hover']}: " + '%{x} km<br>' + f"{frame['y_hover']}: " + '%{y} km'
                '<br><b>Índice: %{customdata[0]:.3f}</b>'
                '<br>Frequência: %{customdata[1]:.3f} (%{customdata[4]:.4f} extremos/hora)'
                '<br>θ₂: %{customdata[2]:.3f} (%{customdata[5]:.1f} km)'
                '<br>θ₅: %{customdata[3]:.3f} (%{customdata[6]:.1f} km)<extra></extra>')),
            row=row + 1, col=col + 1)
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode='markers',
            marker=dict(symbol='cross', size=10, color='white', line=dict(color='#333', width=1)),
            showlegend=False, hovertemplate='Centro do ciclone<extra></extra>'),
            row=row + 1, col=col + 1)
        # Rótulo de eixo só nas bordas: repetido nos quatro painéis, ele rouba a área que os
        # mapas deveriam ocupar, e a grandeza é a mesma nos quatro.
        fig.update_xaxes(title_text=frame['x_label'], range=[-1125, 1125],
                         row=row + 1, col=col + 1)
        fig.update_yaxes(title_text=frame['y_label'] if i == 0 else None, range=[-1125, 1125],
                         showticklabels=i == 0,
                         scaleanchor='x' if i == 0 else f'x{i + 1}', scaleratio=1,
                         row=row + 1, col=col + 1)
    fig.update_layout(
        height=FIG_HEIGHT, autosize=True, margin=dict(t=40, b=10, l=10, r=10),
        coloraxis=dict(colorscale=heat_colorscale(vmin, vmax, floor), cmin=vmin, cmax=vmax,
                       colorbar=heat_colorbar(vmin, vmax, SCALE_TITLE, floor, fmt='.2f')))
    return fig


def component_figure(normed, component, floor_fraction=0.0):
    """The same 2x2 layout for one normalised component, so the mean can be audited.

    Shares the index's scale and floor on purpose: two 0-1 legends side by side with different
    colours for the same number would be worse than the floor applying to both.
    """
    fig = make_subplots(rows=1, cols=len(PHASES), subplot_titles=list(PHASES.values()),
                        horizontal_spacing=0.04)
    for i, phase in enumerate(PHASES):
        row, col = 0, i
        value = normed[component][phase]
        fig.add_trace(go.Heatmap(
            x=AXIS_KM, y=AXIS_KM, z=value, customdata=value, coloraxis='coloraxis',
            hoverongaps=False,
            hovertemplate=('Leste: %{x} km<br>Norte: %{y} km'
                           '<br>Normalizado: %{customdata:.3f}<extra></extra>')),
            row=row + 1, col=col + 1)
        fig.update_xaxes(range=[-1125, 1125], row=row + 1, col=col + 1)
        fig.update_yaxes(range=[-1125, 1125], showticklabels=i == 0,
                         scaleanchor='x' if i == 0 else f'x{i + 1}',
                         scaleratio=1, row=row + 1, col=col + 1)
    fig.update_layout(
        height=COMPONENT_FIG_HEIGHT, autosize=True,
        margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
        coloraxis=dict(colorscale=heat_colorscale(0.0, 1.0, floor_fraction), cmin=0.0, cmax=1.0,
                       colorbar=heat_colorbar(0.0, 1.0, f'{COMPONENTS[component]} (0–1)',
                                              floor_fraction, fmt='.2f')))
    return fig


def phase_means_table(normed, composite):
    """Per-phase mean of each normalised component next to the resulting index.

    Built because the three fields do not move together: frequency peaks in the mature phase
    while theta5 bottoms out there, so an equal-weight mean cancels most of the between-phase
    contrast. Showing the components beside the index keeps that visible instead of buried.
    """
    rows = []
    for phase, label in PHASES.items():
        finite = np.isfinite(composite[phase])
        row = {'Fase': label}
        for component, name in COMPONENT_SHORT.items():
            row[name] = round(float(np.nanmean(normed[component][phase][finite])), 3)
        row['Índice composto'] = round(float(np.nanmean(composite[phase][finite])), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    amplitude = {'Fase': 'Diferença entre a maior e a menor'}
    for column in table.columns[1:]:
        amplitude[column] = round(float(table[column].max() - table[column].min()), 3)
    return pd.concat([table, pd.DataFrame([amplitude])], ignore_index=True)


def render_composite_index(key_prefix):
    st.subheader('Índice composto de extremos')
    st.markdown(
        'Um único mapa por fase, de **0 a 1**, resumindo três coisas que o painel mostra '
        'separadas: com que frequência o vento extremo acontece, e os dois números que '
        'descrevem a forma da região extrema (θ₂ e θ₅). **0 é o ponto mais fraco da amostra, '
        '1 o mais forte.** Os três medem a mesma faixa de percentis.')

    c1, c2 = st.columns(2)
    with c1:
        band = st.radio('Percentil', list(BANDS), format_func=BANDS.get,
                        key=f'{key_prefix}_band', horizontal=True)
    with c2:
        referencial = st.radio('Referencial', list(REFERENCE_FRAMES),
                               format_func=lambda k: REFERENCE_FRAMES[k]['label'],
                               key=f'{key_prefix}_referencial', horizontal=True)
    frame = REFERENCE_FRAMES[referencial]
    faixa = BAND_QUANTILE_RANGE[band]
    st.caption(frame['caption'])

    try:
        raw, normed, composite, ranges, mask = load_composite(band, referencial)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Não foi possível montar o índice composto: {exc}')
        return

    # O slider vem depois de carregar os dados porque seus limites SÃO os dados: o intervalo
    # muda entre bandas e referenciais, e um limiar fixo em [0, 1] cairia fora do que existe.
    vmin, vmax = observed_range(composite)
    floor = index_floor(f'{key_prefix}_floor_{band}_{referencial}', vmin, vmax)
    st.caption(f'O índice observado vai de **{vmin:.3f}** a **{vmax:.3f}** nas quatro fases — '
               f'é esse intervalo que as cores cobrem, não o 0–1 inteiro, que deixaria tudo '
               f'no mesmo tom.')

    # responsive: o Plotly refaz o layout quando o container muda de tamanho, em vez de
    # congelar a largura do primeiro desenho — que é o que fazia o mapa nascer pequeno e só
    # crescer no rerun seguinte.
    st.plotly_chart(composite_figure(raw, normed, composite, floor, (vmin, vmax), frame),
                    width='stretch', config={'responsive': True}, key=f'{key_prefix}_map')
    cinza = {p: int(np.sum(composite[p][np.isfinite(composite[p])] < floor)) for p in PHASES}
    total = int(np.isfinite(composite['mature']).sum())
    escondidas = sum(cinza.values())
    st.caption(
        'Passe o cursor para ver o índice e os três valores que o formaram naquele ponto — o '
        'cursor lê o valor real mesmo onde o mapa está cinza. '
        + (f'Com o limiar em {floor:.3f}, ficam cinza: '
           + ' · '.join(f'{PHASES[p]} {n:,}'.replace(',', '.') for p, n in cinza.items())
           + f' de {total:,}'.replace(',', '.') + ' células por fase.'
           if escondidas else 'Nada cinza: o limiar está no mínimo observado, todo o campo '
                              'está colorido.'))

    with st.expander('Como esse mapa é feito', expanded=True):
        st.markdown(
            'Os três campos medem coisas diferentes: frequência é *extremos por hora*, θ₂ e θ₅ '
            'são *quilômetros*.\n\n'
            '1. Cada campo é reescalado para uma nota de 0 a 1 — o menor valor vira 0, o maior '
            'vira 1, o resto fica proporcional no meio.\n'
            '2. As notas são somadas e divididas por 3.\n\n'
            f'Os três medem a **mesma faixa de percentis** ({faixa}): a frequência é a média da '
            'taxa ao longo dessa faixa, que é exatamente o intervalo de limiares sobre o qual '
            'θ₂ e θ₅ são estimados.\n\n'
            'O menor e o maior de cada campo são procurados nas quatro fases juntas, e não '
            'dentro de cada fase. É isso que permite dizer que uma fase é mais forte que outra.')

    st.markdown('**Como cada campo contribui, por fase**')
    st.dataframe(phase_means_table(normed, composite), width='stretch', hide_index=True)

    with st.expander('Ressalvas'):
        st.markdown(
            f'- **Os três campos não andam juntos, e a média esconde isso.** A fase madura é a '
            f'mais frequente e ao mesmo tempo a menos extensa (θ₅). Com peso igual para os três, '
            f'um efeito anula o outro: veja a última linha da tabela acima. O mapa responde bem '
            f'**onde** o extremo se organiza, e mal **em qual fase** ele é maior.\n'
            f'- **θ₂ quase não varia entre fases** — é limitado pelo tamanho da área analisada, '
            f'na origem dos dados. Mesmo assim entra na média com peso 1/3.\n'
            f'- **A frequência não tem detalhe de 25 km.** O mapa é desenhado em células de '
            f'25 km porque é essa a resolução de θ₂ e θ₅. A frequência é medida em células de '
            f'100 km, e o valor de cada uma foi repetido nas 16 células menores que ela cobre. '
            f'Nada foi inventado, mas dois terços do mapa têm detalhe fino e um terço não.\n'
            f'- **O índice não tem unidade.** 0,7 não é velocidade nem distância: é "mais forte '
            f'que 0,3 nesta amostra", e não se compara com nenhum outro estudo.\n'
            f'- **A cor cobre o intervalo observado ({vmin:.3f} a {vmax:.3f}), não 0–1.** Como o '
            f'índice é média de três campos, ele nunca encosta nas pontas; esticar a cor até 1 '
            f'gastaria metade da rampa com valores inexistentes. Comparar cor entre p95 e p99 '
            f'exige olhar a legenda, porque o intervalo muda.\n'
            f'- **O limiar só muda a cor, nunca o dado.** Subir o slider manda para o cinza o que '
            f'estiver abaixo dele e redistribui as nove cores no que sobra — é assim que se '
            f'enxergam diferenças dentro de uma faixa estreita de valores. Os números do cursor, '
            f'da tabela e do download não mudam.\n'
            f'- **Cobre {int(np.isfinite(composite["mature"]).sum()):,} células por fase**, onde '
            f'os três campos coincidem no referencial escolhido.')

    if st.toggle('Ver os três campos separados', key=f'{key_prefix}_show_components'):
        component = st.radio('Campo', list(COMPONENTS), format_func=COMPONENTS.get,
                             key=f'{key_prefix}_component', horizontal=True)
        vmin, vmax = ranges[component]
        unit = 'extremos/hora' if component == 'freq' else 'km'
        origem = (f'média da excedência nos percentis {faixa}' if component == 'freq'
                  else f'estimado sobre os percentis {faixa}')
        st.caption(f'Já convertido para a nota de 0 a 1. Valor original: de {vmin:.4g} a '
                   f'{vmax:.4g} {unit} — {origem}.')
        # mesmo corte relativo do slider, traduzido para o intervalo do componente (0-1 por
        # construção: a normalização min-max garante que ele ocupa o intervalo inteiro).
        st.plotly_chart(component_figure(normed, component, (floor - vmin) / (vmax - vmin)),
                        width='stretch',
                        config={'responsive': True}, key=f'{key_prefix}_component_map')

    with st.expander('Baixar os dados'):
        phase = st.selectbox('Fase', list(PHASES), format_func=PHASES.get,
                             key=f'{key_prefix}_download_phase')
        east, north = np.meshgrid(AXIS_KM, AXIS_KM)
        table = pd.DataFrame({
            frame['x_col']: east.ravel(), frame['y_col']: north.ravel(),
            'indice_composto': composite[phase].ravel(),
            'freq_norm': normed['freq'][phase].ravel(),
            'theta2_norm': normed['theta2'][phase].ravel(),
            'theta5_norm': normed['theta5'][phase].ravel(),
            'freq_extremos_por_hora': raw['freq'][phase].ravel(),
            'theta2_km': raw['theta2'][phase].ravel(),
            'theta5_km': raw['theta5'][phase].ravel(),
        }).dropna(subset=['indice_composto'])
        st.download_button('Baixar índice e campos (CSV)', table.to_csv(index=False),
                           file_name=f'indice_composto_{phase}_{band}_{referencial}.csv',
                           mime='text/csv', key=f'{key_prefix}_download')
