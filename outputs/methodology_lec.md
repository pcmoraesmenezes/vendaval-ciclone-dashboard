# Metodologia — Ciclo de Vida do Ciclone (ERA5 × Mendeley × Zenodo)

Este documento cobre a aba **🌪️ Ciclo de Vida do Ciclone — ERA5 × Mendeley × Zenodo** do dashboard: como o
vento real é calculado e agregado ao longo da vida do ciclone, como os quadrantes geográficos são
definidos e rotacionados a cada passo de tempo, e como as figuras/GIFs de quadrantes são reconstruídas.
Para a metodologia de percentis locais por ponto de grade (usada nas abas *Explorar Evento*/*Análise
Contínua*), ver [`methodology.md`](methodology.md).

**Reescrito em 18/08/2026** — a versão anterior deste documento era centrada nos termos do Ciclo de Energia
de Lorenz (LEC): reservatórios/conversões de energia (`Az`, `Ae`, `Kz`, `Ke`, `Ce`, ...) e uma proxy de
energia (`Ke`) para "intensidade do ciclone". Nada disso é usado pelo dashboard hoje — a página *Ciclo de
Vida* mostra só velocidade real de vento (ERA5). Ver Seção 6 para o que o dataset do Zenodo/LEC ainda
contribui (só rótulo de fase).

---

## 1. Visão Geral — o que vem de cada fonte

*   **ERA5 (Copernicus)**: componentes de vento a 10m `u10`/`v10`, grade 0.25°, Hemisfério Sul. Toda
    velocidade de vento do dashboard é `wind_speed = sqrt(u10² + v10²)` — nenhuma outra grandeza (energia,
    vorticidade) é exibida.
*   **Mendeley/EXWAV**: a trajetória (latitude/longitude do centro) de cada ciclone extratropical, hora a
    hora. É o "esqueleto" geométrico sobre o qual tudo mais é calculado (distância ao centro, quadrantes).
*   **Zenodo/LEC**: usado **só** para rotular cada instante da trajetória com uma fase do ciclo de vida
    (`incipient` → `intensification` → `mature` → `decay` → `residual`), a partir das janelas de tempo de
    `periods.csv` daquele dataset. Nenhum termo energético do LEC é lido pelo dashboard.

---

## 2. Como as Estatísticas de Vento São Calculadas

### 2.1 Vento na posição da trajetória (`track_wind_speed.py`)

Para cada ponto hora a hora da trajetória Mendeley, busca-se o valor de `wind_speed` no ponto de grade ERA5
mais próximo (`nearest`, `np.searchsorted` por eixo — reescrito em 06/08/2026 por performance, ver
docstring do script). Duas limitações reais, documentadas e não contornadas:

*   **Cobertura de ano**: ERA5 baixado só cobre 2010–2020. Dos 4.052 `track_id` com correspondência entre
    Mendeley e LEC, só **1.004** caem nesse intervalo — o resto (1979–2009) não tem velocidade de vento.
*   **Resolução temporal**: grade ERA5 é 6-horária (00/06/12/18 UTC), trajetória é horária → cada ponto usa
    o timestep de grade mais próximo, com desvio de até 3h registrado na coluna `wind_delta_h`. Caso
    conhecido de fronteira de ano (arquivos NetCDF por ano isolados): 6 de 60.291 linhas (2010–2015) com
    desvio de 4–5h — decisão tomada com o Paulo (08/08/2026) de manter como está (volume irrisório).

Saída: `outputs/csv/track_wind_speed_by_phase.csv` — 1 linha por (ciclone, hora), com `wind_speed_ms`,
`phase` e `wind_delta_h`. É a fonte crua das abas *Distribuição por fase* e *Explorar ciclone* do
dashboard.

### 2.2 Extremos por fase (`wind_phase_extremes.py`)

Responde "quanto tempo o ciclone passa em vento extremo, e quão intenso, em cada fase do ciclo de vida":

1.  Fases-base: `incipient`, `intensification`, `mature`, `decay` (`residual` e `unclassified` são
    descartados). Reocorrências (`intensification 2`, `decay 2`, ...) são dobradas na fase-base antes de
    qualquer cálculo.
2.  Limiar de "extremo" = percentil **por fase-base** (Q90/Q95/Q99), calculado sobre todos os
    ciclones/timesteps daquela fase (pooled) — não um limiar único global, porque o vento típico muda de
    fase para fase.
3.  Resolução: 1 linha de `track_wind_speed_by_phase.csv` = 1 hora (`REPORT_INTERVAL_HOURS = 1`).

Para cada (ciclone, fase-base, nível de percentil):

$$n\_horas = n\_timesteps \times 1$$
$$taxa\_contagem = \frac{n\_extremos\ (wind\_speed > limiar)}{n\_horas} \qquad
taxa\_acumulada = \frac{\sum wind\_speed\ (\text{nos timesteps} > limiar)}{n\_horas}$$

Agregado final (`wind_phase_extremes_summary.csv`, consumido pela aba *⚡ Extremos por fase*): média de
`taxa_contagem` e `taxa_acumulada` entre todos os ciclones que passaram por aquela fase.

---

## 3. Quadrantes: Fixos vs. Rotacionados, Recalculados a Cada Passo de Tempo

Base de código: `ciclone_quadrantes.py` (`process_and_plot_hour`, rodado para cada hora `idx` do evento).
É a mesma geometria usada nas abas *🗺️ Padrão espacial* e *🌐 Distribuição espacial* da página *Ciclo de
Vida*, e nos GIFs de quadrante (`outputs/*.gif`).

### 3.1 Centro do ciclone e raio de análise

O centro (`lat_c`, `lon_c`) em cada hora vem da trajetória Mendeley/EXWAV (não recalculado — a única
exceção é o rastreamento próprio por circulação usado nos runners de evento mais antigos, que busca o
ponto de menor `score = -circulação + 2×vento_no_ponto` numa vizinhança). A partir do centro, calcula-se a
distância geodésica (haversine) de cada ponto de grade ERA5 e mantém-se só os pontos dentro do **raio de
1100 km** (`mask_circle`).

### 3.2 Quadrantes fixos (geográficos)

Alinhados a paralelos/meridianos, não mudam de orientação — só de posição (seguem o centro):

```
dlon_deg = (lon_grid - lon_c + 180) % 360 - 180   # diferença de longitude, corrigida para [-180, 180]
dlat_deg = lat_grid - lat_c

Q1 = NW: dlat_deg ≥ 0 e dlon_deg < 0     Q2 = NE: dlat_deg ≥ 0 e dlon_deg ≥ 0
Q4 = SW: dlat_deg < 0 e dlon_deg < 0     Q3 = SE: dlat_deg < 0 e dlon_deg ≥ 0
```

### 3.3 Quadrantes rotacionados (alinhados ao movimento) — recalculados a cada hora

A ideia: em vez de Norte/Sul/Leste/Oeste geográficos, os 4 setores passam a ser "à frente-esquerda /
à frente-direita / atrás-direita / atrás-esquerda" **relativos para onde o ciclone está se movendo naquele
instante**. Isso muda a cada hora porque a direção de deslocamento do ciclone muda a cada hora.

**Passo 1 — vetor de movimento da hora `t`** (`dx_motion[t]`, `dy_motion[t]`): diferença finita centrada da
trajetória, `(lat[t+1] - lat[t-1], lon[t+1] - lon[t-1])` (diferença progressiva/regressiva nas pontas),
longitude corrigida para `[-180,180]` e escalada por `cos(lat média)` para compensar a convergência dos
meridianos, depois normalizado para vetor unitário `(ux, uy)`. `motion_bearing[t]` é esse vetor convertido
para rumo em graus, sentido horário a partir do Norte.

**Passo 2 — rotação do vetor centro→ponto de grade pelo vetor de movimento** (é uma rotação 2D de
coordenadas, não uma rotação da imagem):

```
dx_escalado = dlon_deg × cos(radianos((lat_grid + lat_c)/2))    # km-equivalente, corrige meridiano
dy_escalado = dlat_deg

y' = dx_escalado·ux + dy_escalado·uy   # projeção na direção do movimento ("à frente" se ≥ 0)
x' = dx_escalado·uy - dy_escalado·ux   # projeção perpendicular ao movimento ("à direita" se ≥ 0)

Q1: y'≥0, x'<0 (à frente-esquerda)     Q2: y'≥0, x'≥0 (à frente-direita)
Q4: y'<0, x'<0 (atrás-esquerda)        Q3: y'<0, x'≥0 (atrás-direita)
```

Os rótulos de exibição continuam sendo `NW/NE/SE/SW` (posição 1/2/3/4) por consistência visual com a linha
"Fixo" — mas na linha "Rotacionado" eles **não** são pontos cardeais reais, são só os 4 códigos de posição
frente-esquerda/frente-direita/atrás-direita/atrás-esquerda relativos ao movimento daquela hora específica.

**Por que a cada passo de tempo**: `(ux, uy)` e `motion_bearing` são recalculados a partir da trajetória
local em cada índice `idx` — o ciclone muda de direção ao longo da vida (curva, desacelera, muda de rumo),
então o referencial "rotacionado" gira junto, hora a hora, para continuar apontando "à frente" do
deslocamento real naquele momento. Não existe um único ângulo de rotação fixo aplicado ao evento inteiro.

### 3.4 Limiares e estatística por quadrante

Cada quadrante (fixo e rotacionado) é comparado contra os mesmos limiares da Seção 2: fixos (15.6/20.0/25.0
m/s) e percentis Q90/Q95/Q99 — que podem ser um valor único (percentil global do evento) ou uma grade 2D de
percentis locais por ponto de grade (`data/local_percentiles.nc`, ver `methodology.md`). Por
(quadrante, limiar): `% de pontos excedentes` (proporção da área do quadrante acima do limiar) e o ponto de
vento máximo dentro do quadrante (valor + distância ao centro).

---

## 4. Padrão Espacial dos Extremos de Vento por Fase (`wind_spatial_pattern_by_phase.py`)

Generaliza a Seção 2.1 (que extrai 1 valor de vento por hora, no ponto da trajetória) para um **campo
espacial**: dentro do raio de 1100 km, reusa a geometria de quadrantes fixos/rotacionados da Seção 3, sem a
etapa de plotagem (rodar hora a hora para dezenas de ciclones geraria milhares de PNGs desnecessários aqui).

Para cada (ciclone, fase-base, tipo de quadrante, quadrante, limiar):

$$taxa\_contagem = \frac{n\_horas\ com\ o\ quadrante\ excedendo\ o\ limiar}{n\_horas\ do\ ciclone\ naquela\ fase}
\qquad
taxa\_acumulada = \frac{\sum\ vento\ máximo\ do\ quadrante\ nas\ horas\ em\ que\ excedeu}{n\_horas}$$

Agregado final (`wind_spatial_pattern_by_phase.csv`, consumido pela aba *🗺️ Padrão espacial*): média entre
todos os ciclones que passaram por aquela fase. Amostra pequena: só **1.560 das 96.459 horas**
classificadas caem numa fase válida (o resto é `unclassified`), o que deixa **31 ciclones distintos** no
total (3 incipiente, 20 intensificação, 17 maduro, 29 decaimento) — reportado via `n_ciclones` em cada
linha, não escondido.

---

## 5. Distribuição Espacial sem Agregar por Quadrante (`wind_spatial_field_by_phase.py`)

A Seção 4 colapsa cada hora em só 4 números (um por quadrante) — qualquer estrutura espacial *dentro* de um
quadrante (extremo sempre perto da borda do círculo vs. sempre perto do centro, por exemplo) fica
invisível. Mesma fonte/limiares/trajetória/fase da Seção 4, só muda a unidade espacial de agregação. As
duas visões (por quadrante e sem quadrante) ficam lado a lado no dashboard — uma não substitui a outra.

**5.1 Heatmap fino**: mesma fórmula da Seção 4, por **célula de grade contínua de 100 km** (~380 células
dentro do círculo) em vez de por quadrante geográfico — resolução escolhida entre 3 opções: nativa do ERA5
(~28 km, ~4.900 células — ruído demais para só 31 ciclones), 4 quadrantes (é o que este pipeline substitui)
e 100 km (cada célula ainda agrega ~13 pontos nativos por hora, ~90× mais granular que quadrante).
Denominador é o total de ciclones da fase (mesmo `n_ciclones` da Seção 4), não só os que tiveram
exceedência ali — correção de viés aplicada em 11/08/2026 (a primeira versão inflava a taxa em células de
baixa amostra em até 3–17×, dividindo só pelos ciclones que "acertaram" aquela célula).

**5.2 Scatter bruto**: para cada (ciclone, hora) com fase classificada, a posição exata do ponto de grade
de vento máximo (em km, relativa ao centro) — sem agregação nenhuma entre ciclones nem discretização
espacial. ~1.560 pontos no total.

**5.3 Geometria**: mesma trigonometria da Seção 3, em km contínuos (`KM_PER_DEG = 111.32`, longitude
escalada por `cos(lat)`) em vez de discretizada em quadrante — dois referenciais, ambos derivados do mesmo
vetor de movimento unitário da Seção 3.3 (**Fixo**: eixos Leste/Norte; **Rotacionado**: eixos ao longo do
movimento / perpendicular). Validado em 11/08/2026: agrupar as células finas pelo sinal de cada eixo
reproduz exatamente a taxa por quadrante da Seção 4 (cota matemática, checada nas 152 linhas com
contrapartida não-vazia — 0 violações).

---

## 6. Como a Figura/GIF de Quadrantes é Reconstruída

Esta seção explica como um frame estático (o tipo de imagem que o Paulo já mandou, extraída deste
pipeline) é gerado — e por que ela precisa ser recalculada do zero a cada hora, nunca só "girada" a partir
de um frame anterior.

### 6.1 Um painel por hora, dois arquivos por hora

`process_and_plot_hour(idx)` (Seção 3) roda uma vez por hora `idx` do evento e, ao final, chama
`plot_panel()` **duas vezes** — uma para os 3 limiares fixos, outra para os 3 quantis — cada chamada
gerando uma figura de **2 linhas × 3 colunas** (linha 1 = quadrantes Fixos, linha 2 = Rotacionados; cada
coluna = um limiar). Arquivos: `hour_{idx:03d}_fixed[_trackid].png` e `hour_{idx:03d}_quantiles[_trackid].png`.

### 6.2 O que muda a cada frame (e por quê)

Cada um dos 6 subpainéis de uma hora desenha, sobre um mapa Cartopy (PlateCarree) com extensão **fixa**
(bounding box da trajetória inteira do evento, calculada uma vez — para os frames não "pularem" de zoom):

*   Trajetória percorrida **até aquela hora** (`lon_traj[:idx+1]`) — cresce frame a frame.
*   Centro do ciclone daquela hora (estrela vermelha) e círculo de 1100 km ao redor dele (linha tracejada).
*   Seta vermelha = vetor de movimento daquela hora (`dx_motion[idx]`, `dy_motion[idx]`, Seção 3.3).
*   **Linhas divisórias dos quadrantes**: para a linha "Fixo", 4 pontos a 1100 km do centro nos rumos
    0°/90°/180°/270° (geografia pura). Para a linha "Rotacionado", os mesmos 4 pontos mas nos rumos
    `(motion_bearing[idx] + 0/90/180/270) % 360` — **usando o rumo de movimento daquela hora específica**.
    É este recálculo, frame a frame, que faz a "aspa" dos quadrantes rotacionados girar ao longo do GIF.
*   Pontos que excedem o limiar (dispersão colorida por velocidade), linha+marcador verde até o ponto de
    vento máximo de cada quadrante, e um rótulo de texto por quadrante (`% excedente`, `valor m/s | km`)
    posicionado a 1250 km do centro, no rumo bissetor daquele quadrante — geográfico para "Fixo",
    `(motion_bearing[idx] + offset) % 360` para "Rotacionado" (mesmo mecanismo das linhas divisórias).

Ou seja: **para reconstruir uma figura específica (uma hora específica de um ciclone específico) não basta
re-plotar um template genérico** — é preciso reprocessar a trajetória daquele evento até aquele índice,
recalcular `motion_bearing[idx]` a partir dos pontos vizinhos da trajetória naquele instante, e só então
redesenhar os 4 setores rotacionados com aquele ângulo específico. Um frame de hora 50 e um frame de hora
80 do mesmo ciclone quase sempre têm o referencial rotacionado apontando para rumos diferentes.

### 6.3 De PNGs por hora a um GIF

`create_analysis_gif.py` (`generate_gif`) varre a pasta de saída do evento por `hour_*_{fixed|quantiles}*.png`,
ordena **numericamente** pelo índice de hora extraído do nome do arquivo (não ordem alfabética — evita
`hour_10` vir antes de `hour_2`), abre um frame por vez (não carrega tudo em memória — eventos longos geram
centenas de MB de PNG) e salva como GIF animado via Pillow (`save_all`, `append_images`), 2 fps por padrão.
Os GIFs pré-computados que o dashboard/README referenciam (`outputs/quadrant_plots_local_20100113_fixed.gif`
e `_quantiles.gif`) são exatamente essa sequência de 6.2, hora a hora, para o evento de 13/01/2010.

---

## 7. Sobre o Dataset Zenodo/LEC — Só Rótulo de Fase

Dataset externo: **"Lorenz Energy Cycle (LEC) Results for Cyclones in the Southwestern Atlantic"**, Zenodo
(DOI [10.5281/zenodo.18243447](https://zenodo.org/records/18243447)), ~7.400 ciclones extratropicais
rastreados no Atlântico Sudoeste via ERA5, 1979–2020. Contém, por ciclone, os termos energéticos do LEC
passo a passo **e** um arquivo `periods.csv` com as janelas de tempo (`start`/`end`) de cada fase do ciclo
de vida (`incipient`, `intensification`, `mature`, `decay`, `residual`).

O dashboard usa **só o `periods.csv`**: cada timestamp da trajetória/vento é comparado contra essas janelas
(`assign_phase`) para herdar o rótulo de fase — se não cair em nenhuma janela, vira `unclassified` e é
descartado das agregações por fase (mas permanece nos CSVs raw). Reocorrências de fase no mesmo ciclone
(`intensification 2`, `decay 2`, ...) são normalizadas para a fase-base antes de qualquer agregação
(dobrando a contagem na fase-base; achado em 03/08/2026 — a primeira versão do script descartava essas
linhas por não normalizar antes do `reindex`, um bug já corrigido).

**Os termos energéticos do LEC em si (`Az`, `Ae`, `Kz`, `Ke`, `Ce`, ...) não são lidos, calculados nem
exibidos em nenhuma aba do dashboard hoje** — só serviram, em versões anteriores do projeto, de proxy de
intensidade antes da migração para vento real (Seção 2). Ficam fora de escopo deste documento.

---

## 8. Limitações Conhecidas

*   **Correspondência por `track_id`, não validada por sobreposição temporal**: nem todo `track_id` igual
    entre LEC e Mendeley descreve o mesmo ciclone físico (parte é coincidência de numeração entre execuções
    de algoritmos de rastreamento diferentes). A análise usa os que batem por ID como estão; achado em
    04/08/2026.
*   **Amostra pequena nas Seções 4/5**: só 31 ciclones distintos têm hora classificada + vento disponível
    (interseção das limitações de cobertura da Seção 2.1 com a cobertura de fase do Zenodo) — células/
    quadrantes com poucos ciclones contribuindo (sobretudo `incipient`) são reportados com `n_ciclones`
    visível, nunca escondidos.
*   **Desvio temporal > 3h em 12 pontos** (fronteira de ano, Seção 2.1) — volume irrisório, mantido como
    está por decisão de 08/08/2026.

---

## 9. Como Reproduzir

```bash
# Vento na trajetória (Seção 2.1) — depende de ERA5 2010-2020 já baixado e do join Mendeley×LEC:
.venv/bin/python scripts/analysis/track_wind_speed.py

# Extremos por fase (Seção 2.2) — depende da saída acima:
.venv/bin/python scripts/analysis/wind_phase_extremes.py

# Padrão espacial por quadrante (Seção 4) — depende de track_wind_speed_by_phase.csv e
# data/local_percentiles.nc:
.venv/bin/python scripts/analysis/wind_spatial_pattern_by_phase.py

# Distribuição espacial sem quadrante — heatmap fino + scatter (Seção 5), mesmos pré-requisitos:
.venv/bin/python scripts/analysis/wind_spatial_field_by_phase.py

# Painéis de quadrante por evento + GIF (Seção 6), roda por evento (ex.: ciclone_quadrantes_2010.py):
.venv/bin/python scripts/analysis/ciclone_quadrantes_2010.py
.venv/bin/python scripts/visualization/create_analysis_gif.py <pasta_de_plots>
```
