# Dados adotados pelo IRB

As páginas leem exclusivamente os Parquets desta pasta. Não regenerar nem
regravar estes arquivos: seus bytes correspondem aos SHA256 oficiais.

- `fp_C_perc1979_b90_gt.parquet`: footprint C, percentis 1979–2020, vento > limiar.
- `persistence_v2_b90.parquet`: persistência da série completa v2, passo de 6 h,
  tolerância de um passo ausente.
- `persistence_sazonal_v2_b90.parquet`: a mesma persistência por estação do ano (DJF, MAM,
  JJA, SON), com o evento contado na estação em que começa. Somando as estações, cada célula
  reproduz `persistence_v2_b90.parquet`.
- `gev_america_do_sul.parquet`: parâmetros e níveis de retorno (2 a 100 anos) do ajuste GEV sobre o
  máximo anual do vento, por ponto de grade, mais as marcas `forma_alta` e `nr_fora_da_formula`.
  Não tem a máscara de terra nem os limiares dos outros produtos.
- q90→b90, q95→b95, q99→b99; 15,6/20/25 m/s→b90.
- Domínio: terra incluindo ilhas. Máscara descritiva, índice ≥0,5 e pesos iguais.

Os `.metadata.json` registram o contrato científico sem alterar os bytes dos
Parquets. Nome, metadados, hash e dimensões são validados em cada leitura.
Arquivos antigos permanecem em seus caminhos e não são fallback da interface.

Ver [validação, hashes e referências](../../docs/validation/IRB.md).
