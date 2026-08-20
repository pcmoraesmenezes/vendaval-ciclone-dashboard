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

## 6. Como a Figura/GIF de Quadrantes é Reconstruída

Cada frame é recalculado do zero por hora — nunca "girado" a partir de um frame anterior.

### 6.1 Um painel por hora, dois arquivos por hora

`process_and_plot_hour(idx)` chama `plot_panel()` duas vezes (limiares fixos, quantis), cada uma gerando
uma figura 2×3 (linha 1 = Fixo, linha 2 = Rotacionado; coluna = limiar). Arquivos:
`hour_{idx:03d}_fixed[_trackid].png` e `hour_{idx:03d}_quantiles[_trackid].png`.

### 6.2 O que muda a cada frame

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

### 6.3 De PNGs por hora a um GIF

`create_analysis_gif.py` varre `hour_*_{fixed|quantiles}*.png`, ordena numericamente pelo índice de hora,
abre um frame por vez e salva como GIF via Pillow (`save_all`, `append_images`), 2 fps.

---

## 7. Limitações Conhecidas

- **Cobertura do ERA5**: só 1.785 dos 6.789 ciclones (2010–2020) têm vento real disponível.
- **Pontos sem fase**: ~7,8% das horas com vento não têm fase atribuída — normal perto do início/fim de
  cada trajetória, antes da fase incipiente começar ou depois do decaimento terminar.
- **Desvio temporal**: 50 de 168.153 pontos (fronteira de ano) chegam a 4–5h de desvio (0,03%).

---

## 8. Como Reproduzir

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

# Painéis de quadrante por evento + GIF (Seção 6):
.venv/bin/python scripts/analysis/ciclone_quadrantes_2010.py
.venv/bin/python scripts/visualization/create_analysis_gif.py <pasta_de_plots>
```
