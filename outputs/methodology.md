# Metodologia - Percentis Locais por Ponto de Grade (Grid-Point Climatological Percentiles)

Este documento descreve a transição metodológica realizada na análise de extremos de vento associados a ciclones no Hemisfério Sul, comparando a abordagem de **limiares globais** com a de **limiares locais (por ponto de grade)**.

---

## 1. Percentil Global — Como é Calculado

Abordagem original, hoje considerada legada:

1. Para cada ano do período histórico (2010–2015), carrega os dados de vento do ERA5 (componentes `u10`/`v10`) e recorta o subdomínio de estudo (Latitude $[-45^\circ, -15^\circ]$, Longitude $[-65^\circ, -30^\circ]$).
2. Calcula a velocidade do vento em cada ponto de grade e cada instante: `vento = sqrt(u10² + v10²)`.
3. Junta **todos** os pontos de grade e **todos** os instantes, de todos os anos, numa única lista de valores.
4. Calcula o percentil (Q90/Q95/Q99) sobre essa lista única.

Resultado: um único valor por percentil, válido pra qualquer ponto do subdomínio inteiro.

*   **Q90**: $10.85 \text{ m/s}$
*   **Q95**: $12.50 \text{ m/s}$
*   **Q99**: $15.55 \text{ m/s}$

---

## 2. Percentil Local — Como é Calculado

Abordagem atual, usada pelas abas *Explorar Evento* e *Análise Contínua* do dashboard:

1. Para cada ano do período histórico (2010–2015), carrega os dados de vento do ERA5 e recorta o mesmo subdomínio de estudo.
2. Calcula a velocidade do vento em cada ponto de grade e cada instante, do mesmo jeito que no cálculo global.
3. Desta vez, os pontos de grade **não são misturados entre si** — só os anos são juntados, mantendo cada ponto de grade (latitude, longitude) com sua própria série temporal.
4. Para cada ponto de grade, individualmente, calcula o percentil (Q90/Q95/Q99) usando só a série temporal daquele ponto.

Resultado: um mapa de percentis — um valor de Q90, um de Q95 e um de Q99 por ponto de grade (62.101 pontos, resolução $0.25^\circ \times 0.25^\circ$, série de 8.764 passos de 6 em 6 horas).

---

## 3. Comparação: Global × Local

Mesma pergunta, duas formas diferentes de agrupar os dados antes de calcular o percentil:

| | Global | Local |
|---|---|---|
| O que entra em cada cálculo | todos os pontos de grade + todos os instantes, juntos | só os instantes daquele ponto de grade |
| Quantos valores o cálculo produz | 1 por percentil (Q90/Q95/Q99) | 1 por percentil, por ponto de grade |
| Período usado | 2010–2015 | 2010–2015 |
