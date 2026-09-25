# Metodologia — Ciclo de Vida do Ciclone

Esta aba cruza vento real (ERA5) com a trajetória e a fase de vida de milhares de ciclones. Este documento
explica como cada número é calculado.

---

## 1. Fontes de Dados

**ERA5 (Copernicus)**: componentes de vento a 10m `u10`/`v10`, grade 0,25°, Hemisfério Sul, 2010–2020.
Velocidade de vento = `sqrt(u10² + v10²)`.

**Catálogo de ciclones**: arquivo `data/tracks_danilo/tracks_SAt_filtered_with_periods.csv` — trajetória
(latitude, longitude, vorticidade T42) e fase de vida (`incipient` → `intensification` → `mature` →
`decay`, com reocorrências `intensification 2`/`decay 2`/etc.) já vêm juntas, ponto a ponto, hora a hora.
6.789 ciclones, 1979–2021, recorte ao Atlântico Sul (regiões ARG/LA-PLATA/SE-BR).

`scripts/analysis/track_position_by_phase.py` processa esse arquivo:

1. Lê `track_id, date, lon vor, lat vor, vor42, region, geometry, period`.
2. Renomeia pro padrão do pipeline: `date`→`time`, `lon vor`→`lon`, `lat vor`→`lat`, `vor42`→
   `vorticidade_t42`, `period`→`phase`.
3. Ponto sem fase atribuída (fora de qualquer janela de fase) → `phase = "unclassified"`.
4. Descarta `geometry` (redundante com lon/lat).
5. Salva `outputs/csv/track_position_by_phase.csv`.

---

## 2. Como as Estatísticas de Vento São Calculadas

### 2.1 Vento na posição da trajetória (`track_wind_speed.py`)

Pra cada ciclone:

1. Pega a trajetória (Seção 1) — 1 ponto (lat, lon) por hora.
2. Pra cada ponto, busca o ponto de grade ERA5 mais próximo (`nearest`) e lê a velocidade do vento ali.
3. Anota o rótulo de fase da hora e o desvio de tempo entre a hora real e o timestep de grade usado
   (`wind_delta_h`).
4. Salva 1 linha por (ciclone, hora): `wind_speed_ms`, `phase`, `wind_delta_h`.

Saída: `outputs/csv/track_wind_speed_by_phase.csv`.

Limitações de cobertura:

- **Ano**: ERA5 só cobre 2010–2020. Dos 6.789 ciclones, **1.785** caem nesse intervalo.
- **Resolução temporal**: grade ERA5 é 6-horária (00/06/12/18 UTC), trajetória é horária — cada ponto usa o
  timestep mais próximo, desvio de até 3h (coluna `wind_delta_h`). 50 de 168.153 linhas (fronteira de ano)
  chegam a 4–5h.

### 2.2 Extremos por fase (`wind_phase_extremes.py`)

1. Considera só as 4 fases-base (`residual`/`unclassified` ficam de fora); reocorrências contam pra
   fase-base.
2. Pra cada fase-base e percentil (Q90/Q95/Q99): junta as horas de todos os ciclones daquela fase e
   calcula o percentil — um limiar por fase.
3. Pra cada ciclone, dentro da fase, conta horas que excederam o limiar (frequência) e soma o vento
   dessas horas (acumulado).
4. Divide as duas contagens pelo total de horas do ciclone na fase — taxa de frequência e taxa acumulada.
5. Tira a média das duas taxas entre todos os ciclones da fase.

Saída: `wind_phase_extremes_summary.csv`.

---

## 3. Quadrantes: Fixos vs. Rotacionados, Recalculados a Cada Passo de Tempo

Base de código: `ciclone_quadrantes.py` (`process_and_plot_hour`, uma vez por hora `idx` do evento).

### 3.1 Centro e raio

1. Centro (`lat_c`, `lon_c`) vem pronto da trajetória, sem recálculo.
2. Distância até cada ponto de grade ERA5: haversine (considera curvatura da Terra).
3. Mantém só pontos até **1100 km** do centro (`mask_circle`); o resto é descartado naquela hora.

### 3.2 Quadrantes fixos (geográficos)

```
dlon_deg = (lon_grid - lon_c + 180) % 360 - 180
dlat_deg = lat_grid - lat_c

Q1 = NW: dlat_deg ≥ 0 e dlon_deg < 0     Q2 = NE: dlat_deg ≥ 0 e dlon_deg ≥ 0
Q4 = SW: dlat_deg < 0 e dlon_deg < 0     Q3 = SE: dlat_deg < 0 e dlon_deg ≥ 0
```

### 3.3 Quadrantes rotacionados (alinhados ao movimento)

Os 4 setores passam a ser relativos à direção de deslocamento do ciclone naquela hora, não a pontos
cardeais fixos.

**Passo 1 — vetor de movimento** `(dx_motion[t], dy_motion[t])`: diferença finita centrada da trajetória,
`(lat[t+1]-lat[t-1], lon[t+1]-lon[t-1])` (progressiva/regressiva nas pontas), longitude corrigida e
escalada por `cos(lat média)`, normalizado a vetor unitário `(ux, uy)`.

**Passo 2 — rotação do vetor centro→ponto de grade**:

```
dx_escalado = dlon_deg × cos(radianos((lat_grid + lat_c)/2))
dy_escalado = dlat_deg

y' = dx_escalado·ux + dy_escalado·uy   # à frente se ≥ 0
x' = dx_escalado·uy - dy_escalado·ux   # à direita se ≥ 0

Q1: y'≥0, x'<0 (frente-esquerda)     Q2: y'≥0, x'≥0 (frente-direita)
Q4: y'<0, x'<0 (atrás-esquerda)      Q3: y'<0, x'≥0 (atrás-direita)
```

Rótulos de exibição continuam `NW/NE/SE/SW` (posição 1/2/3/4) por consistência visual, mas na linha
"Rotacionado" não são pontos cardeais reais.

`(ux, uy)` é recalculado a cada hora — o referencial rotacionado gira junto com a direção real do
ciclone, não existe um ângulo fixo pro evento inteiro.

### 3.4 Limiares e estatística por quadrante

1. Cada quadrante é comparado contra os limiares do pipeline: 3 fixos (15,6/20,0/25,0 m/s) e percentis
   Q90/Q95/Q99 (globais ou locais por ponto de grade, `data/local_percentiles.nc`).
2. Por (quadrante, limiar): % de pontos excedentes na área do quadrante, e vento máximo + distância ao
   centro.

---

## 4. Padrão Espacial dos Extremos de Vento por Fase (`wind_spatial_pattern_by_phase.py`)

Generaliza a Seção 2.1 (1 valor de vento por hora) para um campo espacial dentro do raio de 1100 km,
reusando a geometria de quadrantes da Seção 3.

Pra cada (ciclone, fase-base, tipo de quadrante, quadrante, limiar):

1. Conta horas em que o quadrante passou do limiar, divide pelo total de horas do ciclone na fase — taxa
   de frequência.
2. Soma o vento máximo do quadrante nessas horas, divide pelo mesmo total — taxa acumulada.

Agregado final (`wind_spatial_pattern_by_phase.csv`): média entre os ciclones da fase. 151.255 das
168.153 horas com vento (90%) caem numa fase válida — 1.785 ciclones distintos (1.212 incipiente, 1.751
intensificação, 1.233 maduro, 1.699 decaimento), reportado via `n_ciclones`.

---

## 5. Distribuição Espacial sem Agregar por Quadrante (`wind_spatial_field_by_phase.py`)

A Seção 4 colapsa cada hora em 4 números (um por quadrante). Esta seção usa a mesma fonte/limiares/fase,
só muda a unidade espacial — as duas visões ficam lado a lado, uma não substitui a outra.

**5.1 Heatmap fino**: mesma fórmula da Seção 4, por célula de grade contínua de 100 km (~380 células
dentro do círculo) em vez de por quadrante. Denominador é o total de ciclones da fase (mesmo `n_ciclones`
da Seção 4), não só os que tiveram exceedência ali.

**5.2 Scatter bruto**: pra cada (ciclone, hora) com fase classificada, a posição exata do ponto de grade
de vento máximo (km, relativo ao centro), sem agregação nem discretização. ~151.255 pontos.

**5.3 Geometria**: mesma trigonometria da Seção 3, em km contínuos (`KM_PER_DEG = 111,32`, longitude
escalada por `cos(lat)`), dois referenciais (Fixo: Leste/Norte; Rotacionado: ao longo do movimento /
perpendicular).

---

## 6. Excursion Sets — θ₂ e θ₅

Campos de origem externa: chegam prontos em `outputs/excursion_sets/` e o painel só lê e desenha, sem
conversão nem reescalonamento. Nenhum script deste repositório os calcula.

**6.1 O que os dois números medem**: as seções 4 e 5 respondem *com que frequência* o vento passa do
limiar. Aqui a pergunta é a *forma* da região que passou do limiar. **θ₅** é a extensão espacial dessa
região (razão área/perímetro reescalada) e **θ₂** é o alcance extremal superior. Ambos em quilômetros —
não são velocidade nem frequência.

**6.2 Grade**: 89×89 células de 25 km, raio de 1.100 km, centro em [44,44], linha crescente de sul para
norte e coluna de oeste para leste. 6.077 células com dado dentro do círculo; o resto fica vazio.

**6.3 Referencial**: **fixo** (geográfico, norte para cima, leste à direita) e **rotacionado**
(alinhado ao movimento do ciclone, eixo vertical apontando para adiante), selecionável no painel.
Cada referencial usa sua própria máscara e seus próprios eixos; não são a mesma grade rotacionada
por conta, e não devem ser cruzados um com o outro.

**6.4 Bandas p90, p95 e p99**: os limiares são quantis locais, e a banda é a **faixa de limiares que
o estimador percorre**: 0,80–0,90 para p90, 0,89–0,95 para p95 e 0,95–0,99 para p99, nunca um limiar
único. Essa diferença motivou a mudança descrita na Seção 7.3: a frequência de extremos usava um
percentil só, e por isso "p95" queria dizer coisas diferentes em cada metade do painel.

**6.5 Amostra e incerteza**: 11.382 realizações por fase, com 1.212 ciclones distintos na incipiente, 343
na intensificação, 981 na madura e 322 no decaimento. Os limites que aparecem no hover são bootstrap por
célula (200 réplicas) — descrevem a incerteza daquela célula, não testam a diferença entre fases.

**6.6 Limitações declaradas na origem**: θ₂ é limitado pela extensão do domínio e, neste recorte, não
distingue as fases; estimar seu alcance completo exigiria um domínio maior. O gradiente norte–sul pode
refletir latitude ou amostragem, não só o ciclone.

---

## 7. Índice Composto de Extremos (`app/composite_index.py`)

Junta num campo único, de 0 a 1, três coisas que hoje aparecem separadas: a frequência de extremos da
Seção 5 e os dois coeficientes da Seção 6. Referencial (fixo ou rotacionado) selecionável, mesmo
referencial nos três campos; limiar mínimo de exibição ajustável, sem afetar o dado.

**7.1 Por que normalizar**: os três campos estão em unidades incompatíveis — frequência é *extremos por
hora*, θ₂ e θ₅ são *quilômetros*. Somar direto não teria significado. Cada campo é reescalado para 0–1
pela fórmula `(valor − mínimo) / (máximo − mínimo)`, o que preserva a forma espacial e descarta a
unidade.

**7.2 Onde o mínimo e o máximo são procurados**: nas quatro fases juntas, dentro da banda (p90, p95 ou
p99) e do referencial escolhidos, um único par mínimo/máximo por campo. É a mesma escolha que o resto
do painel já faz nas suas escalas de cor, e é o que mantém a comparação entre fases legível: se cada
fase fosse normalizada sozinha, toda fase teria uma célula valendo 1 e a comparação sumiria.

**7.3 Percentil com a mesma definição dos dois lados**: a frequência de extremos é a **média da taxa
de excedência ao longo da faixa inteira de percentis**, onze níveis (q80…q90) para a banda p90, sete
(q89…q95) para a p95 e cinco (q95…q99) para a p99, em vez da contagem num percentil único. É a mesma
faixa que o estimador de θ percorre (Seção 6.4). Implementação: `calculate_local_percentiles.py`
extrai os níveis intermediários na mesma passada pelos 11 anos de ERA5 (nenhuma leitura a mais), e
`wind_spatial_field_by_phase.py` emite os níveis `b90`/`b95`/`b99`, acumulando a soma sobre os níveis
numa única chave e dividindo por N no fim, o que dá exatamente a média sem multiplicar o tamanho da
estrutura de agregação pelo número de níveis da banda. Os níveis `q90`/`q95`/`q99` continuam no CSV,
inalterados, para as seções 4 e 5.

**7.4 Como as duas grades são conciliadas**: a frequência é medida em células de 100 km (Seção 5) e θ é
nativo em 25 km (Seção 6). Cada célula de 25 km recebe o valor da célula de 100 km em que ela cai
(`célula = floor(km / 100)`, regra confirmada no próprio CSV: `cell_center_x_km == cell_x * 100 + 50`),
o que replica um valor em até 16 células. **Isso não cria informação**: o mapa é desenhado em 25 km, mas
o detalhe da frequência abaixo de 100 km continua não existindo.

**7.5 A média**: o índice é `(freq_norm + θ₂_norm + θ₅_norm) / 3`, com peso igual para os três. Só
existe onde os três campos existem — 6.077 células por fase, limitadas pela máscara de θ, que é a mais
estreita. Célula sem um dos três fica vazia em vez de ser calculada sobre um denominador menor.

**7.6 O que o índice revela — e o que ele apaga** (valores de 10/09/2026, já com a correção de banda
da Seção 7.3 — a tabela abaixo precisa ser refeita a cada regeração do campo de frequência): os três
campos não andam juntos. A fase madura tem a
maior frequência de extremos e, ao mesmo tempo, a menor extensão espacial (θ₅) — extremo mais frequente
e mais concentrado. Como a média usa peso igual, os dois efeitos se cancelam:

| Campo (média por célula, p95, referencial fixo) | Incipiente | Intensificação | Maduro | Decaimento | Amplitude |
|---|---|---|---|---|---|
| Frequência de extremos | 0,248 | 0,282 | 0,410 | 0,242 | 0,169 |
| θ₂ | 0,642 | 0,660 | 0,661 | 0,671 | 0,029 |
| θ₅ | 0,680 | 0,675 | 0,472 | 0,685 | 0,213 |
| **Índice composto** | 0,524 | 0,539 | 0,514 | 0,533 | **0,024** |

Em p99 o padrão se repete (amplitude do índice 0,040 contra 0,170 da frequência e 0,202 de θ₅). Ou seja:
o índice descreve bem **onde**, em torno do centro, o extremo se organiza, e quase não separa **em qual
fase** — a diferença entre fases fica uma ordem de grandeza menor que a de cada campo isolado. É
resultado da definição, não erro de cálculo.

---

## 8. Como a Figura/GIF de Quadrantes é Reconstruída

Cada frame é recalculado do zero por hora — nunca "girado" a partir de um frame anterior.

### 8.1 Um painel por hora, dois arquivos por hora

`process_and_plot_hour(idx)` chama `plot_panel()` duas vezes (limiares fixos, quantis), cada uma gerando
uma figura 2×3 (linha 1 = Fixo, linha 2 = Rotacionado; coluna = limiar). Arquivos:
`hour_{idx:03d}_fixed[_trackid].png` e `hour_{idx:03d}_quantiles[_trackid].png`.

### 8.2 O que muda a cada frame

Sobre um mapa Cartopy (PlateCarree) com extensão fixa (bounding box da trajetória inteira):

- Trajetória até aquela hora (cresce frame a frame).
- Centro do ciclone (estrela) e círculo de 1100 km.
- Seta = vetor de movimento da hora.
- Linhas divisórias dos quadrantes: "Fixo" nos rumos 0°/90°/180°/270°; "Rotacionado" em
  `(motion_bearing[idx] + 0/90/180/270) % 360` — é isso que faz os quadrantes rotacionados girarem no GIF.
- Pontos que excedem o limiar, marcador até o vento máximo de cada quadrante, rótulo (`% excedente`,
  `valor m/s | km`) a 1250 km do centro no rumo bissetor do quadrante.

Reconstruir um frame específico exige reprocessar a trajetória até aquele índice e recalcular
`motion_bearing[idx]` — não dá pra reusar um template genérico.

### 8.3 De PNGs por hora a um GIF

`create_analysis_gif.py` varre `hour_*_{fixed|quantiles}*.png`, ordena numericamente pelo índice de hora,
abre um frame por vez e salva como GIF via Pillow (`save_all`, `append_images`), 2 fps.

---

## 9. Limitações Conhecidas

- **Cobertura do ERA5**: só 1.785 dos 6.789 ciclones (2010–2020) têm vento real disponível.
- **Pontos sem fase**: ~7,8% das horas com vento não têm fase atribuída — normal perto do início/fim de
  cada trajetória, antes da fase incipiente começar ou depois do decaimento terminar.
- **Desvio temporal**: 50 de 168.153 pontos (fronteira de ano) chegam a 4–5h de desvio (0,03%).
- **θ₂ não separa as fases** neste domínio (Seção 6.6), e por isso entra quase plano no índice composto.
- **Bandas não cobrem os limiares absolutos**: os níveis `b95`/`b99` só existem para percentis locais. Os
  limiares fixos (15,6/20/25 m/s) continuam sendo limiar único, e não devem ser cruzados com θ.
- **Índice composto sem unidade física**: 0 é o menor valor observado na amostra e 1 o maior. Não é taxa,
  não é distância, e não é comparável a nenhum índice de outra fonte.

---

## 10. Como Reproduzir

```bash
# Trajetória + fase (Seção 1):
.venv/bin/python scripts/analysis/track_position_by_phase.py

# Vento na trajetória (Seção 2.1):
.venv/bin/python scripts/analysis/track_wind_speed.py

# Extremos por fase (Seção 2.2):
.venv/bin/python scripts/analysis/wind_phase_extremes.py

# Padrão espacial por quadrante (Seção 4) — depende de track_wind_speed_by_phase.csv e
# data/local_percentiles.nc:
.venv/bin/python scripts/analysis/wind_spatial_pattern_by_phase.py

# Distribuição espacial sem quadrante (Seção 5), mesmos pré-requisitos:
.venv/bin/python scripts/analysis/wind_spatial_field_by_phase.py

# Painéis de quadrante por evento + GIF (Seção 8):
.venv/bin/python scripts/analysis/ciclone_quadrantes_2010.py
.venv/bin/python scripts/visualization/create_analysis_gif.py <pasta_de_plots>
```

As Seções 6 e 7 não têm passo de reprodução aqui: os campos de θ chegam prontos de fora, em
`outputs/excursion_sets/`, e o índice composto é montado em tempo de execução pelo próprio painel
(`app/composite_index.py`), a partir desses arquivos e de `outputs/csv/wind_spatial_field_by_phase_grid.csv`.
