# Metodologia: Percentis Locais de Vento

Esta análise migrou o cálculo de extremos de vento de um limiar global estático para percentis locais calculados individualmente para cada coordenada geográfica. Isso remove o viés geográfico (como a diferença natural de vento entre terra e mar) e foca na detecção de anomalias severas específicas de cada ponto.

---

## Tabela de Percentis de Vento por Coordenada

Os limites de extremos (Q90, Q95, Q99) foram calculados por ponto de grade (resolução de 0.25°). A tabela completa com todos os 62.101 pontos está disponível no arquivo `grid_point_percentiles.csv`.

### Amostra da Grade de Percentis:

| Latitude | Longitude | Q90 (m/s) | Q95 (m/s) | Q99 (m/s) |
| :---: | :---: | :---: | :---: | :---: |
| -10.00 | -85.00 | 9.14 | 9.84 | 11.11 |
| -10.00 | -84.75 | 9.10 | 9.76 | 11.05 |
| -10.00 | -84.50 | 9.08 | 9.74 | 11.04 |
| -10.00 | -84.25 | 9.05 | 9.72 | 11.03 |
| -10.00 | -84.00 | 9.02 | 9.69 | 10.99 |
| -10.00 | -83.75 | 9.01 | 9.66 | 10.92 |
| -10.00 | -83.50 | 8.98 | 9.61 | 10.88 |
| -10.00 | -83.25 | 8.93 | 9.55 | 10.85 |
| -10.00 | -83.00 | 8.93 | 9.56 | 10.85 |
| -10.00 | -82.75 | 8.84 | 9.48 | 10.77 |

---

## Resumo do Processo

1. **Base Histórica**: Série temporal de ventos horários de 2010 a 2015.
2. **Cálculo Espacial**: Determinação do percentil de tempo para cada coordenada de forma independente.
3. **Avaliação**: Comparação direta ponto a ponto do vento do ciclone contra o mapa de limites calculado.
