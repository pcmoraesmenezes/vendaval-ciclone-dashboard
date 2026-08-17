# Metodologia - Percentis Locais por Ponto de Grade (Grid-Point Climatological Percentiles)

Este documento descreve a transição metodológica realizada na análise de extremos de vento associados a ciclones no Hemisfério Sul, comparando a abordagem de **limiares globais** com a de **limiares locais (por ponto de grade)**.

---

## 1. Abordagem Anterior: Percentis Globais (Subdomínio Completo)

Na abordagem inicial, as estatísticas de excedência de vento utilizavam limiares de velocidade fixos ou baseados em percentis históricos globais calculados para todo o subdomínio geográfico estudado (Latitude $[-45^\circ, -15^\circ]$, Longitude $[-65^\circ, -30^\circ]$).

### Limiares Globais Utilizados:
*   **Q90**: $10.84 \text{ m/s}$
*   **Q95**: $12.49 \text{ m/s}$
*   **Q99**: $15.56 \text{ m/s}$

### Limitações da Abordagem Global:
1.  **Heterogeneidade Espacial**: O vento tem regimes climatológicos muito diferentes sobre a terra firme (onde o atrito é maior e as velocidades médias são menores) em comparação com o oceano aberto. Da mesma forma, latitudes mais ao sul (zona de ventos de oeste) possuem ventos naturalmente mais intensos que latitudes subtropicais.
2.  **Viés de Detecção**: Um limiar global de $10.84 \text{ m/s}$ (Q90) pode ser atingido quase constantemente sobre o oceano aberto no sul do domínio, enquanto pontos sobre o continente ou mais ao norte quase nunca atingirão tal limiar, mascarando anomalias locais relevantes nessas áreas continentais/norte.

---

## 2. Nova Abordagem: Percentis Locais (Por Ponto de Grade)

Para isolar a variabilidade espacial e focar em **anomalias de vento locais** associadas à dinâmica do ciclone, a metodologia foi refinada para calcular limiares específicos para cada célula de grade $(i, j)$ de latitude e longitude.

### Formulação Matemática:
Seja $WS(t, y, x)$ a velocidade de vento a 10 metros no instante $t$, na latitude $y$ e longitude $x$.

Para cada coordenada espacial fixa $(y, x)$, a série histórica contínua de 5 anos (2010 a 2014) é extraída ao longo do tempo $t$. Os limiares de percentil local $Q_p(y, x)$ para $p \in \{90, 95, 99\}$ são calculados como:

$$Q_{90}(y, x) = \text{Percentil}_{90}\Big( \{ WS(t, y, x) \}_{t=1}^{T} \Big)$$
$$Q_{95}(y, x) = \text{Percentil}_{95}\Big( \{ WS(t, y, x) \}_{t=1}^{T} \Big)$$
$$Q_{99}(y, x) = \text{Percentil}_{99}\Big( \{ WS(t, y, x) \}_{t=1}^{T} \Big)$$

Onde $T$ é o número total de timesteps da série histórica (7.304 passos de 6 em 6 horas de 2010 a 2014).

### Vantagens Científicas:
1.  **Normalização Climatológica**: Cada ponto de grade atua como seu próprio referencial de vento. Um vento de $8 \text{ m/s}$ em uma célula continental pode representar um extremo local (excedendo seu Q90 local), ao passo que o mesmo valor sobre o oceano sul seria considerado uma brisa normal.
2.  **Foco em Anomalias de Ciclone**: Permite detectar com precisão quais regiões do ciclone estão provocando ventos que divergem significativamente do padrão local histórico daquele local geográfico.

---

## 3. Estrutura de Armazenamento e Implementação

As matrizes bidimensionais de quantis locais são armazenadas em formato NetCDF em `data/local_percentiles.nc`:

*   **Dimensões**: `latitude`, `longitude` (com resolução espacial de $0.25^\circ \times 0.25^\circ$).
*   **Variáveis**:
    *   `q90(latitude, longitude)`: Matriz 2D de limiares de 90%
    *   `q95(latitude, longitude)`: Matriz 2D de limiares de 95%
    *   `q99(latitude, longitude)`: Matriz 2D de limiares de 99%

### Comparação Matricial Vetorizada:
No script de análise contínua (`continuous_5year_analysis.py`), a comparação é efetuada de forma vetorizada com o Numpy/Xarray:

```python
# ws_hour possui shape (n_lat, n_lon)
# local_q90 possui shape (n_lat, n_lon)
mask_exceed = mask_q & (ws_hour > local_q90)
```
Isso garante a manutenção do alto desempenho da pipeline (conclusão em menos de 2 minutos para os 5 anos de dados).
