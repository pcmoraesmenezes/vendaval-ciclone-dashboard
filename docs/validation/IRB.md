# Validação do dashboard IRB, 13/11/2026

O fluxo validado é navegação Streamlit → leitura do Parquet e metadados oficiais →
seleção do limiar → indicadores e mapas. As páginas exibem somente footprint C e
persistência v2+b90, sem seletor de versões e sem fallback.

## Referências científicas

- [Decisões do responsável, incluindo adoção complementar](https://github.com/daniloceano/relatorio_ciclones_irb/blob/main/docs/vento/decisoes_responsavel_2026-11-11.md).
- [Registro das figuras e hashes oficiais](https://github.com/daniloceano/relatorio_ciclones_irb/blob/main/docs/vento/figuras_relatorio_C_v2b90.md).
- [Dossiê dos controles anteriores](https://github.com/daniloceano/relatorio_ciclones_irb/blob/main/docs/vento/candidato_b90_percentis1979.md).

## Produtos adotados e metadados

Os arquivos em `outputs/irb_corrigido/` foram copiados dos produtos adotados, sem
regravação ou recálculo. Cada um tem um `.metadata.json` com período 1979–2020,
origem, critérios, banda por limiar, unidade de contagem, máscara incluindo ilhas,
resolução e parâmetros temporais quando pertinentes. O contrato em `app/irb_data.py`
confere todos os metadados, nome, SHA256 e dimensões **em toda leitura**.
Falhas mostram erro informativo; não há caminho alternativo para os históricos.

Os metadados são externos para preservar os hashes oficiais. Em futuras execuções
dos scripts, o polígono e os parâmetros efetivamente usados também são gravados
nos metadados pandas do Parquet (`scientific_parameters`). Os scripts compartilham
`scripts/polygon_config.py`; `b99` permanece o padrão da CLI.

Travas dos geradores (`footprint_grid.py`, `persistence_grid.py` e `filtro_poligono_composto.py`):
- `--out` é obrigatório. Arquivo existente e a pasta `outputs/irb_corrigido/` são recusados,
  antes de ler dados e de novo antes de gravar.
- Saída nomeada `_b90` exige `--abs-poly b90`. No footprint, exige também `--estrito`
  (vento > limiar), percentis com 1979–2020 na descrição e o bruto original (4.330.793
  linhas). Na persistência, exige a série descontaminada v2 e a máscara de terra oficiais,
  conferidas por SHA256; período dos percentis e operador `>` já vêm nos flags da série, então
  o hash dela é a garantia.

Esta atualização **não executou** essas gerações. Novas execuções devem usar caminhos próprios, preservando os adotados.

| Produto | Dimensões | SHA256 |
|---|---|---|
| `fp_C_perc1979_b90_gt.parquet` | 9.237 × 74 | `3f198b2c47cb0ce60c1767b6f60081099d7e8c07fcae2b961c3dfbf2d744a0a8` |
| `persistence_v2_b90.parquet` | 9.229 × 62 | `d1c2e0e60682efbc4ff346053b850f9c3ca9483072efbb0dde042a5694263efc` |
| `persistence_sazonal_v2_b90.parquet` | 36.748 × 63 | `0efe82ccab5623634c2b4ce46a057ee47f736c1aa8fa43735360859d6ea8573c` |

A persistência por estação foi gerada por `scripts/persistencia_sazonal.py` a partir da série descontaminada v2 (SHA256 conferido), com a máscara de 9.237 células de terra. O gerador só grava se a soma das estações reproduz `persistence_v2_b90.parquet` (eventos, máximo, faixas de duração e média); a execução reconciliou nos seis limiares, e os eventos por limiar somam os mesmos totais da tabela abaixo (q90: 4.221.559 em DJF 1.068.017, MAM 796.706, JJA 1.102.642 e SON 1.254.194). O teste `test_seasonal_reconciles_with_annual` repete a conferência sobre os arquivos versionados.

## Testes sintéticos: escopo separado

`tests/test_persistence_abs_poly.py` verifica pareamento b90/b95/b99, padrão b99,
separação por ciclone, tolerância de um passo ausente, quebra após dois passos,
agregação de pequenos eventos construídos e metadados do arquivo gerado no diretório
temporário. Esses testes **não provam reprodução de uma grade científica real**.
Testes auxiliares verificam escalas vazias/constantes, ausência de ocorrência e
rejeição de saídas rotuladas b90 sem a opção explícita.

## Integração com dados reais

`tests/test_irb_integration.py` lê os dois Parquets incluídos no PR, sem executar
os geradores. Confere bytes por SHA256, metadados, dimensões, todos os totais abaixo,
células sem ocorrência, as duas posições nas Malvinas e eventos ≥24 h inexistentes.
AppTest renderiza cada página nos seis limiares e todas as métricas secundárias;
verifica inicialização pelo `streamlit_app.py`, mensagens e máscara automática.
Casos de arquivo ausente, hash inválido e metadados divergentes confirmam rejeição;
com o arquivo corrigido ausente, a interface mostra erro mesmo havendo históricos.

| Limiar | Hits contaminados | Hits descontaminados | Células com diferença positiva | Eventos | Eventos ≥24 h |
|---|---:|---:|---:|---:|---:|
| q90 | 4.330.793 | 3.580.048 | 8.790 | 4.221.559 | 270.646 |
| q95 | 2.828.203 | 2.248.521 | 8.882 | 2.416.846 | 109.582 |
| q99 | 856.089 | 571.392 | 8.814 | 556.980 | 9.378 |
| 15,6 m/s | 15.385 | 11.750 | 431 | 10.904 | 77 |
| 20 m/s | 189 | 96 | 23 | 91 | 0 |
| 25 m/s | 2 | 2 | 0 | 2 | 0 |

O indicador de remoção conta `(hits_contam - hits_desc > 0).sum()`, não a diferença
entre as quantidades de células ocupadas. Os dois hits e dois eventos em 25 m/s
estão em (−52,00°, −59,50°) e (−51,75°, −59,25°), cada evento com 6 h.
A grade não permite identificar se os hits pertencem ao mesmo ciclone/instante.

Execução: `python -m pytest -q`. Resultado: **68 passed**; saída em
[`pytest.txt`](pytest.txt), ambiente e resultados reais em [`real_data.json`](real_data.json).
Uma segunda execução em venv isolado, com Plotly **7.1.0** (atendendo
`requirements.txt`), também passou **68/68**: [`pytest-requirements.txt`](pytest-requirements.txt).
As capturas do navegador usam o ambiente local registrado (Plotly 5.22.0);
a suíte AppTest foi validada também com a versão 7.1.0.

## Comparação real b99: o que foi e não foi verificado

Nesta atualização, foi feita uma comparação somente de leitura entre
`outputs/persistence_grid.parquet` e a reprodução **já existente**
`../irb_vento_candidato_b90_20261111/out/persistencia/persistence_v2_b99_repro.parquet`.
Após indexar/ordenar por `lat,lon`, `pd.testing.assert_frame_equal(a,b,check_exact=True)`
confirmou os valores e tipos de **todas as 9.229 linhas e 60 colunas de dados**
(mais as duas coordenadas). Não foi apenas comparação do dicionário de limiares,
mas também **não foi uma nova execução do gerador**. Essa reprodução arquivada é
um insumo local externo e não é dependência dos testes portáveis do PR.

A afirmação anterior sobre reproduzir `_v2` e `_v3` provém do dossiê científico
referenciado acima. Não foi reexecutada nesta atualização e não é apresentada
como resultado da suíte sintética. A interface não lê v3 nem esses controles.

Os dois Parquets históricos versionados foram comparados byte a byte com a base
`7594690399b1ba3f09f5e7a9d50104418b90c984`: ambos preservados, conforme hashes em
`real_data.json`. Nenhuma figura canônica do relatório foi modificada.

## Verificação no navegador

Servidor local: `python -m streamlit run app/streamlit_app.py --server.headless true
--server.port 8511 --browser.gatherUsageStats false`. Navegação real pelo menu,
renderização dos mapas q90, seleção dos seis limiares disponíveis e caso 25 m/s.
O console inspecionado não apresentou erros ou avisos. AppTest cobre sistematicamente
as combinações; a inspeção visual confirma mapas e mensagens em q90 e 25 m/s.

A integração encontrou e corrigiu uma falha real: os mapas contaminado/descontaminado
idênticos em 25 m/s geravam IDs duplicados no Streamlit. Agora cada mapa tem `key`
explícita. Faixas de cores constantes/vazias e rótulos arredondados coincidentes
receberam tratamento próprio, sem alteração dos valores científicos.

| Evidência | Captura |
|---|---|
| Footprint q90 e indicadores | [captura](screenshots/footprint-q90.jpg) |
| Footprint q90, três mapas | [captura](screenshots/footprint-q90-mapas.jpg) |
| Footprint 25 m/s, aviso Malvinas | [captura](screenshots/footprint-m25.jpg) |
| Persistência q90 | [captura](screenshots/persistence-q90.jpg) |
| Persistência q90, duração | [captura](screenshots/persistence-q90-mapa.jpg) |
| Persistência 25 m/s, sem eventos ≥24 h | [captura](screenshots/persistence-m25.jpg) |

Sem deploy, merge ou alteração da coleção canônica. A limpeza da mensagem/autoria
é limitada ao commit do autor do PR; os commits de terceiros permanecem intactos.
