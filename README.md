# Balanceamento de carga e resiliência em redes integradas terrestres-satelitais

Projeto da disciplina **MO438 – Redes Complexas** (Unicamp) · Eduardo Mosca

A rede combina 108.453 estações-base do OpenCellID (todas tratadas como 5G NR) com os 4.425 satélites
de uma constelação Starlink simulada no ns-3-leo. Com ela, estudamos como o tráfego entre as estações se
distribui pelos satélites e como se redistribui quando partes da rede falham: falhas regionais de
tamanhos diferentes, falha de *hubs* e ataques direcionados, falhas na camada satelital (satélites
isolados e planos orbitais inteiros) e cascatas de sobrecarga.

- **Relatório final:** [`entregas/relatorio_final.pdf`](entregas/relatorio_final.pdf)
- Relatório da entrega parcial: [`entregas/relatorio_parcial.pdf`](entregas/relatorio_parcial.pdf)

## Principais resultados

- A carga dos satélites é muito concentrada (índice de Gini 0,90): 2,4% dos satélites levam metade dela,
  e o mais carregado, que não vê nenhuma estação, é atravessado por 12% dos pares de estações.
- A rede é robusta a falhas aleatórias e regionais e frágil ao ataque aos satélites de maior grau:
  remover 0,5% dos vértices desconecta 73% dos pares de estações.
- Falhas de satélites ao acaso, de um plano orbital inteiro ou do satélite de maior carga não
  desconectam estações, mas deslocam corredores inteiros de tráfego para outros satélites.
- Com capacidade proporcional à carga inicial (modelo de Motter–Lai), a falha de um único satélite
  derruba metade da constelação para α ≤ 2; com satélites idênticos, 16% de folga bastam.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `rede.py` | monta a rede em qualquer instante do traço: enlaces terrestres, entre satélites (ISL) e estação–satélite (GSL) |
| `analise_parcial.py` | análise estrutural da entrega parcial: graus, agrupamento, componentes, distâncias, mapa |
| `falhas.py` | entrega final: carga, falhas regionais, *hubs*, camada satelital, cascatas e evolução no tempo |
| `base_stations.csv` | estações-base do OpenCellID usadas no projeto |
| `relatorio/` | saída do `analise_parcial.py`: figuras e métricas da entrega parcial |
| `resultados_falhas/` | saída do `falhas.py`: um CSV por experimento, `resumo.txt`, configuração usada e figuras |
| `entregas/` | relatórios em PDF |

## Dados

| Arquivo | Origem | No repositório |
|---|---|---|
| `base_stations.csv` | [OpenCelliD](https://opencellid.org), licença CC BY-SA 4.0 | sim |
| `os.log` | traço de mobilidade gerado com o [ns-3-leo](https://github.com/dadada/ns-3-leo) (`Time:Satellite:x:y:z:Speed`, 4.425 satélites, 999 instantes de 1 s) | não, pelo tamanho: **[LINK PARA O os.log.xz]** |

O `rede.py` corrige o referencial do traço (o ns-3-leo gira os planos com o sinal da rotação da Terra
invertido) e lê arquivos compactados diretamente, como `--log os.log.xz` ou `--bs base_stations.csv.gz`.

## Como reproduzir

Python 3.10 ou mais novo (usado: 3.12).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# coloque o traço na raiz do repositório (os.log, ou use --log os.log.xz)

python rede.py diagnostico --threads 8       # confere unidades, referencial, camadas e tamanhos
python -u analise_parcial.py --threads 8     # entrega parcial -> relatorio/
python -u falhas.py todos --threads 28       # entrega final -> resultados_falhas/ (~22 min com 28 processos)
python falhas.py figuras                     # refaz só as figuras e o resumo a partir dos CSVs (não precisa do traço)
```

- `--threads` define quantos processos são usados; num servidor compartilhado, rode com `nice -n 10`.
- Os parâmetros da rede e da simulação (vizinhos terrestres, elevação mínima, raios das falhas regionais,
  tolerâncias α etc.) estão em `python falhas.py todos --help`.
- A configuração da execução que gerou os resultados está em `resultados_falhas/config_todos.json`,
  e os números principais em `resultados_falhas/resumo.txt`.
