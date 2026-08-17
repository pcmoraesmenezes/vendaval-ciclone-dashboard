# Metodologia — Ciclo de Energia de Lorenz (LEC) por Fase de Vida do Ciclone

Este documento descreve a segunda frente de análise do projeto, distinta do pipeline próprio de rastreamento
de quadrantes de vento (ver [`methodology.md`](methodology.md)): aqui a unidade de análise não é o quadrante
geográfico em torno do centro do ciclone, e sim a **fase do ciclo de vida** do ciclone extratropical, medida
através dos termos energéticos do Ciclo de Energia de Lorenz (Lorenz Energy Cycle, LEC).

---

## 1. Fonte dos Dados

Dataset externo (não gerado por este projeto): **"Lorenz Energy Cycle (LEC) Results for Cyclones in the
Southwestern Atlantic"**, publicado no Zenodo (DOI [10.5281/zenodo.18243447](https://zenodo.org/records/18243447)).
Contém os resultados do algoritmo **CycloPhaser/LEC** aplicados a aproximadamente **7.400 ciclones
extratropicais** rastreados no Atlântico Sudoeste a partir do ERA5, cobrindo o período 1979–2020.

O arquivo baixado (`data/lec_zenodo/LEC_Results_energetic-patterns_csv_only.tar.gz`, ~605 MB) contém, para cada
ciclone (`<track_id>_ERA5_track/`), dois arquivos relevantes ao join feito por `lec_phase_energetics.py`:

*   **`<track_id>_ERA5_track_results.csv`**: série temporal (passo a passo) dos termos energéticos do LEC —
    reservatórios de energia, conversões, fluxos de contorno, taxas de variação e resíduos. Não tem nenhuma
    coluna indicando a fase de vida.
*   **`periods.csv`**: as janelas de tempo (`start`/`end`) de cada fase do ciclo de vida daquele ciclone
    específico — o índice do arquivo (`incipient`, `intensification`, `mature`, `decay`, `residual`) é a
    "coluna" de fase mencionada no card original desta task, só que representada como intervalo de tempo em
    vez de rótulo linha a linha.

Cada ciclone também tem 8 arquivos `<Termo>_level.csv` (perfil vertical por nível de pressão) — fora do
escopo desta análise, que trabalha apenas com os termos integrados verticalmente presentes em `_results.csv`.

---

## 2. Termos do Ciclo de Energia de Lorenz

O LEC clássico (Lorenz, 1955) particiona a energia atmosférica em quatro reservatórios e descreve as
conversões entre eles. Os 10 termos centrais usados nesta análise (`ENERGY_TERMS` em
`lec_phase_energetics.py`):

| Termo | Significado |
|---|---|
| `Az` | Energia potencial disponível — componente zonal (média zonal) |
| `Ae` | Energia potencial disponível — componente de perturbação (eddy) |
| `Kz` | Energia cinética — componente zonal |
| `Ke` | Energia cinética — componente de perturbação (eddy) |
| `Cz` | Conversão $A_z \to K_z$ |
| `Ca` | Conversão $A_z \to A_e$ |
| `Ck` | Conversão $K_z \to K_e$ |
| `Ce` | Conversão $A_e \to K_e$ — o termo baroclínico central: energia potencial de perturbação convertida em energia cinética de perturbação, a assinatura energética de um ciclone se intensificando |
| `Gz` | Geração de energia potencial zonal (fontes diabáticas) |
| `Ge` | Geração de energia potencial de perturbação (fontes diabáticas) |

**Omitidos deliberadamente** (presentes no CSV bruto mas fora de `ENERGY_TERMS`): fluxos de contorno
(`BAz`, `BAe`, `BKz`, `BKe`, `BΦZ`, `BΦE`), taxas de variação por diferença finita (`∂Az/∂t`, ...) e resíduos
(`RGz`, `RKz`, `RGe`, `RKe`). Esses são termos derivados/diagnósticos do fechamento do balanço energético, não
os reservatórios e conversões clássicos do ciclo — omitidos para manter o resumo por fase focado no núcleo
interpretável do LEC.

---

## 3. Atribuição de Fase e Agregação

Para cada ciclone, cada timestamp de `_results.csv` é comparado contra as janelas de `periods.csv`
(`assign_phase`): se cair dentro de exatamente uma janela, recebe o rótulo da fase correspondente; se não
cair em nenhuma, recebe `unclassified` (instantes de transição/borda não cobertos por `periods.csv`) e é
descartado da agregação por fase — permanece apenas no CSV raw, não no resumo.

**Reocorrências de fase**: um ciclone pode reintensificar mais de uma vez no seu ciclo de vida — o
CycloPhaser rotula essas janelas extras como `intensification 2`, `decay 2`, `mature 2`, `incipient 2`
(vistos em 1.256 dos 6.789 ciclones processados). `normalize_phase()` remove esse sufixo numérico antes de
agregar o resumo, dobrando a reocorrência na fase-base — o CSV raw preserva o rótulo original com sufixo
para quem precisar do detalhe por ocorrência. **Correção aplicada em 03/08/2026**: a primeira versão do
script fazia `reindex` do resumo só pelas 5 fases-base sem normalizar antes, descartando silenciosamente
essas 17.584 linhas (8,4% dos dados classificados) do agregado — bug encontrado e corrigido antes de este
documento existir na versão atual.

Dois artefatos são gerados em `outputs/csv/`, consumidos pelo dashboard:

*   **`lec_energetics_by_phase.csv`** — dado raw, uma linha por (ciclone, timestamp), com a fase já atribuída.
    Existe para quem quiser reanalisar offline sem reprocessar o tar.gz (605 MB) de novo.
*   **`lec_phase_summary.csv`** — estatística agregada por fase (`groupby("phase")`): média, mediana e desvio
    padrão de cada um dos 10 termos energéticos, mais contagem de observações (`n_observacoes`) e de ciclones
    distintos (`n_ciclones`) que contribuíram para aquela fase. É este arquivo que o dashboard lê diretamente
    — mesmo padrão do resto do pipeline (o app só lê `outputs/` pré-computado, nunca recalcula).

A ordem das fases (`PHASE_ORDER`) segue a progressão natural do ciclo de vida extratropical:
`incipient → intensification → mature → decay → residual`.

---

## 4. Leitura Esperada

O sinal clássico de um ciclone extratropical em desenvolvimento é `Ce` (conversão energia potencial de
perturbação → energia cinética de perturbação) crescendo da fase `incipient` para `intensification`,
atingindo o pico próximo de `mature`, e caindo em `decay`/`residual` — a assinatura energética do processo
baroclínico que alimenta o ciclone. `Ke` (energia cinética de perturbação, essencialmente "quanto o ciclone
gira") deve acompanhar essa mesma curva com uma defasagem, já que é o reservatório que `Ce` alimenta. O
resumo por fase (`lec_phase_summary.csv`) permite checar diretamente se os ~7.400 ciclones do dataset, em
agregado, seguem esse padrão teórico esperado.

---

## 5. Extremos de Ke por Hora, por Fase (`lec_phase_extremes.py`)

Segunda frente de análise sobre o mesmo dataset, a pedido do Paulo (03/08/2026): para cada fase de vida,
quantificar a frequência e a intensidade acumulada de eventos extremos, normalizadas pela duração da fase —
"quanto tempo o ciclone passa em regime extremo, e quão intenso é esse extremo".

### 5.1 Proxy de "velocidade"

Nenhum dos 10 termos energéticos do dataset é velocidade de vento (m/s) literal. Diante dessa ambiguidade,
a decisão (tomada em conjunto com o Paulo, que pediu para deixar documentado) foi usar **Ke** (energia
cinética de perturbação) como proxy — é o termo padrão da literatura do Ciclo de Lorenz associado à
intensidade do vento perturbado, e é diretamente alimentado por `Ce` (a conversão baroclínica central).
**Isso não é uma medida física de vento** — é uma proxy de energia; tratar o resultado como m/s seria
incorreto.

### 5.2 Limiar de extremo — percentil por fase

Sem um valor físico óbvio (como 15.6 m/s no pipeline de quadrantes), o limiar de "extremo" é definido como
**percentil calculado separadamente dentro de cada fase-base** (Q90/Q95/Q99), pooling todos os ciclones e
timesteps daquela fase — não um percentil único global aplicado a todas as fases. Isso reflete que o
patamar típico de Ke muda de fase para fase (`mature` tende a ter Ke maior que `incipient`); um limiar
global penalizaria sistematicamente as fases de menor energia.

### 5.3 Fórmula

Fase residual e `unclassified` são descartadas da análise inteira. Reocorrências são dobradas na fase-base
(mesma normalização da Seção 3) antes de qualquer cálculo. A resolução de report do dataset é uniforme —
**3 horas**, validado por amostragem de 30 ciclones distintos (1.046 intervalos consecutivos, 100% = 3h).

Para cada (ciclone, fase-base, nível de percentil):

$$n\_horas = n\_timesteps \times 3$$
$$taxa\_contagem = \frac{n\_extremos\ (Ke > limiar)}{n\_horas}$$
$$taxa\_acumulada = \frac{\sum Ke\ (\text{nos timesteps com } Ke > limiar)}{n\_horas}$$

Agregado final (`lec_phase_extremes_summary.csv`, consumido pelo dashboard): média de `taxa_contagem` e de
`taxa_acumulada` entre todos os ciclones que passaram por aquela fase, por nível de percentil.

---

## 6. Como Reproduzir

```bash
# 1. Baixar manualmente de https://zenodo.org/records/18243447 (não automatizado — sem API key configurada):
#    LEC_Results_energetic-patterns_csv_only.tar.gz → data/lec_zenodo/

# 2. Processar (lê o tar.gz sem extrair para o disco, ~605 MB de entrada):
.venv/bin/python scripts/analysis/lec_phase_energetics.py

# 3. Extremos de Ke por hora, por fase (depende da saída do passo 2, não relê o tar.gz):
.venv/bin/python scripts/analysis/lec_phase_extremes.py
```

---

## 7. Padrão Espacial dos Extremos de Vento por Fase (`wind_spatial_pattern_by_phase.py`)

Terceira frente, a pedido do Paulo (09/08/2026) — sem nenhum termo de energia, só velocidade real do vento
(`sqrt(u10**2 + v10**2)`, ERA5). Generaliza `track_wind_speed.py` (que extrai 1 valor de vento no ponto mais
próximo da trajetória) para um **campo espacial**: dentro do raio de 1100 km ao redor do centro (a própria
trajetória Mendeley/EXWAV, sem retracking), reusa a geometria de quadrantes de `ciclone_quadrantes.py` — fixos
(NW/NE/SE/SW geográfico) e rotacionados ao vetor de movimento do ciclone — mas sem a etapa de plotagem daquele
script, que não fazia sentido rodar hora a hora para dezenas de ciclones.

### 7.1 Limiares

Ambos os tipos já usados no pipeline de quadrantes por evento — não o percentil pooled-por-fase da Seção 5.2:
fixos (15.6/20.0/25.0 m/s) e percentis **locais por ponto de grade** (`data/local_percentiles.nc`, Q90/Q95/Q99).
Decisão explícita do Paulo: reusar a metodologia de limiar já validada, não inventar uma terceira.

### 7.2 Fórmula

Mesma fórmula da Seção 5.3/`vorticity_phase_extremes.py`/`wind_phase_extremes.py`, agora quebrada por
quadrante em vez de um valor escalar único. Para cada (ciclone, fase-base, tipo de quadrante, quadrante, limiar):

$$taxa\_contagem = \frac{n\_horas\ com\ o\ quadrante\ excedendo\ o\ limiar}{n\_horas\ do\ ciclone\ naquela\ fase}$$
$$taxa\_acumulada = \frac{\sum\ vento\ máximo\ do\ quadrante\ nas\ horas\ em\ que\ excedeu}{n\_horas}$$

Agregado final (`wind_spatial_pattern_by_phase.csv`): média entre todos os ciclones que passaram por aquela
fase. Fase residual e `unclassified` excluídos, reocorrências somadas à fase-base — mesma normalização do
resto do projeto.

### 7.3 Amostra pequena (herdada da Seção 3 de `outputs/methodology_lec.md` sobre correspondência por track_id)

Só as horas que caem numa fase classificada entram aqui: **1.560 das 96.459 horas** de
`track_wind_speed_by_phase.csv` (o resto é `unclassified`). Isso deixa **31 ciclones distintos** no total —
3 em incipiente, 20 em intensificação, 17 em maduro, 29 em decaimento. Quebrar isso em 4 quadrantes × 2
orientações × 6 limiares deixa várias células com poucos ciclones contribuindo (sobretudo incipiente) — não
filtrado nem escondido, reportado via `n_ciclones` em cada linha da saída e exposto no dashboard.

```bash
# Depende de track_wind_speed_by_phase.csv (scripts/analysis/track_wind_speed.py) e de
# data/local_percentiles.nc já calculados:
.venv/bin/python scripts/analysis/wind_spatial_pattern_by_phase.py
```

---

## 8. Distribuição Espacial sem Média por Quadrante (`wind_spatial_field_by_phase.py`)

Quarta frente, a pedido do Paulo (card `b851729a`, Hub, 11/08/2026). A Seção 7 colapsa cada hora em só 4
números (NW/NE/SE/SW) — qualquer estrutura espacial *dentro* de um quadrante (o extremo sempre perto da
borda do círculo vs. sempre perto do centro, por exemplo) fica invisível. Este pipeline usa a mesma fonte,
os mesmos limiares e a mesma trajetória/fase da Seção 7 — a única mudança é a unidade espacial da agregação.

Decisão tomada com o Paulo (11/08/2026): **duas visões complementares, não uma substituindo a outra**. A
versão por quadrante (Seção 7) não foi removida do dashboard — as duas ficam lado a lado para comparação.

### 8.1 Heatmap fino

Mesma fórmula de taxa da Seção 7.2, agora por **célula de uma grade contínua de 100 km** em vez de por
quadrante geográfico:

$$taxa\_contagem = \frac{n\_horas\ com\ a\ célula\ excedendo\ o\ limiar}{n\_horas\ do\ ciclone\ naquela\ fase}$$
$$taxa\_acumulada = \frac{\sum\ vento\ máximo\ da\ célula\ nas\ horas\ em\ que\ excedeu}{n\_horas}$$

Agregado final: **soma** das taxas dos ciclones que excederam naquela célula, dividida pelo **total de
ciclones da fase** (`n_ciclones` — mesma coluna e mesmo valor, constante por fase, da versão por
quadrante) — não só pelos ciclones que efetivamente contribuíram ali. Mesma normalização da Seção 7.2
(fase residual e `unclassified` excluídos, reocorrências somadas à fase-base).

**Correção de denominador (11/08/2026):** a primeira versão do script dividia pela contagem de ciclones
que *tiveram* exceedência em cada célula (`nunique` sobre as linhas geradas), não pelo total de ciclones
da fase — um viés de seleção que descartava do denominador exatamente os ciclones que "zeraram" ali,
inflando a média em células de baixa amostra (achado na validação contra `wind_spatial_pattern_by_phase.csv`:
células com 1-2 ciclones contribuintes chegavam a 3-17× a taxa da versão por quadrante equivalente).
Corrigido antes de qualquer uso em produção. `n_ciclones_contrib` (quantos ciclones de fato excederam
naquela célula) permanece como coluna à parte — sinal de confiabilidade que a versão por quadrante nunca
precisou reportar, porque com só 4 setores todo ciclone praticamente sempre toca todos eles.

**Escolha da resolução (100 km):** três opções consideradas. (a) Resolução nativa do ERA5 (~28 km, ~4.900
células dentro do raio de 1100 km) — descartada: com só 31 ciclones no total (3 em incipiente), a maioria
das células teria 0-1 observação por fase, ruído dominando qualquer sinal. (b) Manter 4 quadrantes — é
exatamente o que este pipeline existe para substituir. (c) **100 km** (~380 células dentro do círculo,
escolhida) — cada célula ainda agrega ~13 pontos nativos por hora, suavizando ruído o suficiente pra ser
interpretável com amostra pequena, permanecendo ~90× mais granular que quadrante. Células sem nenhuma
exceedência ficam em branco no heatmap (não é zero fabricado — é ausência de dado, distinção que a
versão por quadrante também preserva).

### 8.2 Scatter bruto

Não existe equivalente na Seção 7. Para cada (ciclone, hora) com fase classificada, localiza-se o ponto de
grade nativo de vento máximo dentro do raio de 1100 km e registra-se sua posição relativa ao centro (km) e
valor — **sem agregação entre ciclones, sem discretização espacial**. Uma linha por hora classificada
(≈1.560 no total); flags booleanas por limiar (`exceed_<nível>`) permitem ao dashboard filtrar sem
recalcular. Esta é a resposta mais literal ao pedido do card ("mostrar a distribuição dos dados
espacialmente... em vez de valores agregados por quadrante") — o heatmap da Seção 8.1 ainda agrega (por
célula e por ciclone), o scatter não agrega nada.

### 8.3 Geometria e referenciais

Mesma trigonometria da Seção 7 (`ciclone_quadrantes.py`/`wind_spatial_pattern_by_phase.py`), expressa em
km contínuos em vez de discretizada em sinal (+/-): conversão grau→km via `KM_PER_DEG = 111.32`, longitude
escalada por `cos(lat)` antes de converter. Dois referenciais, ambos derivados do mesmo vetor de movimento
unitário usado na Seção 7:

*   **Fixo**: eixos Leste/Norte (km), sem rotação — mesmo referencial geográfico do quadrante NW/NE/SE/SW.
*   **Rotacionado**: eixos ao longo do movimento / perpendicular ao movimento (à esquerda) — projeção do
    referencial fixo pelo vetor de movimento, mesma lógica da Seção 7, sem discretizar em quadrante.

**Validação de consistência (rodada em 11/08/2026, após a correção do denominador acima):** agrupar as
células finas pelo sinal de cada eixo (Leste≥0/Norte≥0 → NE, e assim por diante) reproduz uma cota
matemática rígida, não uma aproximação — como a taxa do quadrante é a união dos eventos de todas as
células que o compõem (mesmo denominador `n_ciclones` da fase, após a correção), ela tem que ser **maior
ou igual** à taxa de qualquer célula individual dentro dele, para todo ciclone e, por preservação da
média, também no agregado. Checado nas 152 linhas de `wind_spatial_pattern_by_phase.csv` com contrapartida
não-vazia no heatmap fino: **0 violações** da cota, tanto em `taxa_contagem_media` quanto em
`taxa_acumulada_media`. As 40 linhas restantes (de 192) já tinham taxa 0,0 na versão antiga — consistente
com nenhuma célula fina ter registrado exceedência ali.

### 8.4 Amostra pequena

Mesma ressalva da Seção 7.3: 1.560 das 96.459 horas classificadas, 31 ciclones distintos (3 incipiente, 20
intensificação, 17 maduro, 29 decaimento) — herdada sem alteração, o pipeline usa a mesma fonte.

```bash
# Mesmos pré-requisitos da Seção 7 (track_wind_speed_by_phase.csv e data/local_percentiles.nc):
.venv/bin/python scripts/analysis/wind_spatial_field_by_phase.py
```
