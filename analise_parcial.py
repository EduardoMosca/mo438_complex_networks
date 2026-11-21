#!/usr/bin/env python3
"""
analise_parcial.py - análise inicial da rede (Entrega Parcial, MO438).

Calcula os 10 itens obrigatórios para a rede de um instante, gera as figuras
e escreve relatorio/metricas.tex, lido por relatorio/relatorio_parcial.tex.

Uso (no servidor):
    nohup nice -n 10 python -u analise_parcial.py --threads 8 > analise.log 2>&1 &
    cd relatorio && pdflatex relatorio_parcial.tex && pdflatex relatorio_parcial.tex
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')   # nada aqui usa BLAS multithread
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
import networkit as nk
import numpy as np

from rede import GSL, ISL, NOMES_TIPO, OMEGA_TERRA, R_TERRA, TERRESTRE, Rede, adicionar_argumentos, config_de_args

plt.style.use('default')
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'figure.dpi': 150})
CORES_CAMADA = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


# ------------------------------------------------------------------ formatação pt-BR
def fi(n):
    return f"{int(n):,}".replace(',', '.')


def fd(x, d=2):
    return f"{x:,.{d}f}".replace(',', 'X').replace('.', ',').replace('X', '.')


def fc(x, d=2):
    """Notação científica para LaTeX: $1{,}94\\times 10^{-3}$."""
    if x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    return f"${fd(x / 10**e, d).replace(',', '{,}')}\\times 10^{{{e}}}$"


# ------------------------------------------------------------------ métricas
def distancia_media(L, amostras, threads, semente):
    """Distância média na maior componente: exata (todas as fontes) ou estimada por BFS amostrado."""
    n = L.numberOfNodes()
    if n < 2:
        return 0.0, 0.0, True, n
    exato = amostras >= n
    fontes = np.arange(n) if exato else np.random.default_rng(semente).choice(n, amostras, replace=False)

    def media(s):  # o BFS do NetworKit libera o GIL: as threads rodam em paralelo
        d = np.asarray(nk.distance.BFS(L, int(s), storePaths=False).run().getDistances(asarray=True))
        return d.sum() / (n - 1)

    with ThreadPoolExecutor(max_workers=threads) as ex:
        medias = np.fromiter(ex.map(media, fontes), dtype=np.float64, count=len(fontes))
    ic = 0.0 if exato else 1.96 * medias.std(ddof=1) / np.sqrt(len(medias))
    return float(medias.mean()), float(ic), exato, len(fontes)


def hist_log(valores, n_total, bins_por_decada=8):
    """P(k) com bins logarítmicos, normalizado pelo total de vértices da rede."""
    v = valores[valores > 0]
    if len(v) == 0:
        return np.array([]), np.array([])
    bordas = np.unique(np.floor(np.logspace(0, np.log10(v.max() + 1),
                                            int(np.log10(v.max() + 1) * bins_por_decada) + 2)).astype(int))
    cont, _ = np.histogram(v, bins=bordas)
    larg = np.diff(bordas)
    centro = np.sqrt(bordas[:-1] * np.maximum(bordas[1:] - 1, bordas[:-1]))
    ok = cont > 0
    return centro[ok], (cont / larg / n_total)[ok]


# ------------------------------------------------------------------ figuras
def fig_graus(graus, eh_sat, caminho):
    n = len(graus)
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    x, y = hist_log(graus, n)
    ax.loglog(x, y, '-', color='0.6', lw=1.0, label='todos os vértices')
    for sel, rot, mk, cor in ((~eh_sat, 'estações-base', 's', '#1f77b4'), (eh_sat, 'satélites', '^', '#ff7f0e')):
        x, y = hist_log(graus[sel], n)
        ax.loglog(x, y, mk, ms=3.2, color=cor, label=rot, lw=0)
    ax.set_xlabel('grau $k$')
    ax.set_ylabel('$P(k)$')
    ax.legend(frameon=False)
    ax.set_title('(a) Distribuição de graus')
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def fig_clustering(local, eh_sat, caminho):
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    bins = np.linspace(0, 1, 41)
    ax.hist(local, bins=bins, weights=np.full(len(local), 1 / len(local)), color='0.75', label='todos os vértices')
    ax.hist(local[eh_sat], bins=bins, weights=np.full(eh_sat.sum(), 1 / max(eh_sat.sum(), 1)), histtype='step',
            color='#ff7f0e', lw=1.2, label='satélites (fração da camada)')
    ax.set_yscale('log')
    ax.set_xlabel('coeficiente de clusterização local $C_i$')
    ax.set_ylabel('fração de vértices')
    ax.legend(frameon=False, loc='upper right')
    ax.set_title('(b) Clusterização local')
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def fig_componentes(tamanhos, caminho):
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    tam, qtd = np.unique(tamanhos, return_counts=True)     # distribuição exata, sem bins
    ax.loglog(tam, qtd, 'o', ms=3.5, color='#2ca02c')
    ax.set_xlabel('tamanho da componente (vértices)')
    ax.set_ylabel('nº de componentes')
    ax.set_title('(c) Tamanhos das componentes')
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)


def _segmentos(lon1, lat1, lon2, lat2):
    ok = np.abs(lon1 - lon2) < 180          # não desenha quem cruza o antimeridiano
    return np.stack((np.column_stack((lon1[ok], lat1[ok])), np.column_stack((lon2[ok], lat2[ok]))), axis=1)


def fig_mapa(rede, a, caminho, semente=0):
    rng = np.random.default_rng(semente)
    P = a['pos_sat']
    slat = np.degrees(np.arcsin(P[:, 2] / np.linalg.norm(P, axis=1)))
    slon = np.degrees(np.arctan2(P[:, 1], P[:, 0]))
    blat, blon = rede.bs['lat'].to_numpy(), rede.bs['lon'].to_numpy()
    nb = rede.n_bs
    camada = rede.camada if rede.camada is not None else np.zeros(rede.n_sat, int)
    u, v, tipo = a['u'], a['v'], a['tipo']
    isl = tipo == ISL
    gsl = np.flatnonzero(tipo == GSL)
    terr = tipo == TERRESTRE

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={'width_ratios': [1.65, 1]})
    for ax, zoom in zip(axs, (False, True)):
        if zoom:   # janela na região de maior densidade de estações
            h, xe, ye = np.histogram2d(blon, blat, bins=(180, 90), range=((-180, 180), (-90, 90)))
            i, j = np.unravel_index(np.argmax(h), h.shape)
            c_lon, c_lat = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
            dlat = 9.0
            dlon = dlat / max(np.cos(np.radians(c_lat)), 0.3)
            ax.set_xlim(c_lon - dlon, c_lon + dlon)
            ax.set_ylim(c_lat - dlat, c_lat + dlat)
            dentro = (np.abs(blat - c_lat) < dlat) & (np.abs(blon - c_lon) < dlon)
            sel_bs = np.flatnonzero(dentro)
        else:
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            sel_bs = np.arange(nb)
        amostra = sel_bs if len(sel_bs) <= 300_000 else rng.choice(sel_bs, 300_000, replace=False)
        ax.scatter(blon[amostra], blat[amostra], s=1.5 if zoom else 0.8, c='0.1', lw=0, zorder=4, rasterized=True)
        if zoom:
            t = np.flatnonzero(terr)
            t = t[np.isin(u[t], sel_bs)]
            ax.add_collection(LineCollection(_segmentos(blon[u[t]], blat[u[t]], blon[v[t]], blat[v[t]]),
                                             colors='0.2', linewidths=0.3, alpha=0.6, rasterized=True))
            g = gsl[np.isin(u[gsl], sel_bs)]
        else:
            g = gsl
        if len(g):
            g = rng.choice(g, min(len(g), 400 if zoom else 1500), replace=False)
            s = v[g] - nb
            ax.add_collection(LineCollection(_segmentos(blon[u[g]], blat[u[g]], slon[s], slat[s]),
                                             colors='#2ca02c', linewidths=0.25, alpha=0.35, rasterized=True))
        e = np.flatnonzero(isl)
        s1, s2 = u[e] - nb, v[e] - nb
        cor_isl = [CORES_CAMADA[c % len(CORES_CAMADA)] for c in camada[s1]]
        ax.add_collection(LineCollection(_segmentos(slon[s1], slat[s1], slon[s2], slat[s2]),
                                         colors=cor_isl, linewidths=0.3, alpha=0.3, rasterized=True))
        ax.scatter(slon, slat, s=1.8 if not zoom else 7, alpha=0.9, c=[CORES_CAMADA[c % len(CORES_CAMADA)] for c in camada],
                   lw=0, zorder=3, rasterized=True)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set_title('(b) Região de maior densidade de estações' if zoom else f"(a) Rede completa em t = {a['t']:g} s")
    fig.tight_layout()
    fig.savefig(caminho, dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    adicionar_argumentos(ap)
    ap.add_argument('--instante-s', type=float, default=None, help='instante analisado (padrão: o primeiro)')
    ap.add_argument('--amostras', type=int, default=1000, help='fontes de BFS para a distância média')
    ap.add_argument('--semente', type=int, default=42)
    ap.add_argument('--saida', default='relatorio')
    args = ap.parse_args()
    nk.setNumberOfThreads(args.threads)
    os.makedirs(args.saida, exist_ok=True)
    t0 = time.perf_counter()

    def etapa(msg):
        print(f"[{time.perf_counter() - t0:7.1f} s] {msg}", flush=True)

    rede = Rede(config_de_args(args))
    k = rede.indice_instante(args.instante_s)
    rede.checar_memoria(k, args.max_ram_gb)
    a = rede.instante(k)
    rede.resumo(a)
    n, m = rede.n_nos, len(a['u'])
    etapa("montando o grafo no NetworKit")
    G = rede.grafo_networkit(k, a)
    assert G.numberOfEdges() == m, "arestas duplicadas na montagem"

    graus = (np.bincount(a['u'], minlength=n) + np.bincount(a['v'], minlength=n)).astype(np.int64)
    eh_sat = np.zeros(n, bool)
    eh_sat[rede.n_bs:] = True
    cont = np.bincount(a['tipo'], minlength=3)

    etapa("componentes")
    cc = nk.components.ConnectedComponents(G).run()
    tamanhos = np.sort(np.fromiter(cc.getComponentSizes().values(), dtype=np.int64))[::-1]
    etapa("clusterização local e transitividade")
    local = np.asarray(nk.centrality.LocalClusteringCoefficient(G, turbo=True).run().scores())
    # transitividade = 3 x triângulos / triplas = soma C_i d_i(d_i-1) / soma d_i(d_i-1): mesmo valor do
    # ClusteringCoefficient.exactGlobal, sem o custo quadrático no grau dos satélites
    pares = graus.astype(np.float64) * (graus - 1)
    transitividade = float((local * pares).sum() / pares.sum()) if pares.sum() > 0 else 0.0
    etapa("maior componente, diâmetro (exato)")
    L = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, compactGraph=True)
    diametro = int(nk.distance.Diameter(L, algo=nk.distance.DiameterAlgo.EXACT).run().getDiameter()[0])
    etapa(f"distância média ({min(args.amostras, L.numberOfNodes())} fontes de BFS)")
    dist, ic, exato, n_fontes = distancia_media(L, args.amostras, args.threads, args.semente)

    etapa("figuras")
    fig_graus(graus, eh_sat, os.path.join(args.saida, 'fig_graus.pdf'))
    fig_clustering(local, eh_sat, os.path.join(args.saida, 'fig_clustering.pdf'))
    if len(tamanhos) > 1:
        fig_componentes(tamanhos, os.path.join(args.saida, 'fig_componentes.pdf'))
    fig_mapa(rede, a, os.path.join(args.saida, 'fig_mapa.png'), args.semente)

    # ------------------------------------------------------------ resultados
    e = a['tipo'] == ISL
    A = a['pos_sat'][a['u'][e] - rede.n_bs]
    D = a['pos_sat'][a['v'][e] - rede.n_bs] - A
    s_ = np.clip(-np.einsum('ij,ij->i', A, D) / np.einsum('ij,ij->i', D, D), 0, 1)
    alt_min_isl = float((np.linalg.norm(A + s_[:, None] * D, axis=1).min() - R_TERRA) / 1000) if e.any() else float('nan')
    bs_sem_sat = int(np.sum(np.bincount(a['u'][a['tipo'] == GSL], minlength=rede.n_bs)[:rede.n_bs] == 0))
    r = {
        'instante_s': a['t'], 'vertices': n, 'arestas': m, 'n_bs': rede.n_bs, 'n_sat': rede.n_sat,
        'arestas_por_tipo': {NOMES_TIPO[i]: int(cont[i]) for i in range(3)},
        'grau_medio': 2 * m / n, 'grau_medio_bs': float(graus[~eh_sat].mean()),
        'grau_medio_sat': float(graus[eh_sat].mean()), 'grau_max': int(graus.max()),
        'grau_max_bs': int(graus[~eh_sat].max()), 'grau_max_sat': int(graus[eh_sat].max()),
        'isolados': int((graus == 0).sum()), 'transitividade': float(transitividade),
        'clust_medio': float(local.mean()), 'clust_medio_grau2': float(local[graus >= 2].mean()),
        'clust_medio_bs': float(local[~eh_sat].mean()), 'clust_medio_sat': float(local[eh_sat].mean()),
        'densidade': 2 * m / (n * (n - 1)), 'componentes': len(tamanhos), 'maior_componente': int(tamanhos[0]),
        'tamanhos_componentes_top10': tamanhos[:10].tolist(), 'diametro': diametro,
        'distancia_media': dist, 'distancia_media_ic95': ic, 'distancia_exata': exato, 'fontes_bfs': n_fontes,
        'bs_sem_satelite_visivel': bs_sem_sat, 'altitude_min_isl_km': alt_min_isl, 'diagnostico': rede.diag, 'camadas': rede.camadas,
        'radios_no_arquivo': rede.radios_arquivo, 'linhas_bs': rede.linhas_bs,
        'operadoras': int(rede.bs.groupby(['mcc', 'net']).ngroups), 'paises_mcc': int(rede.bs['mcc'].nunique()),
        'config': {k2: (list(v2) if isinstance(v2, tuple) else v2) for k2, v2 in vars(rede.cfg).items()},
    }
    with open(os.path.join(args.saida, 'metricas.json'), 'w') as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    escrever_tex(r, rede, os.path.join(args.saida, 'metricas.tex'))
    etapa(f"pronto: {args.saida}/metricas.tex, metricas.json e figuras")


def escrever_tex(r, rede, caminho):
    cfg, d = rede.cfg, r['diagnostico']
    cam = sorted(r['camadas'], key=lambda c: c['altitude_km'])

    def spp(c):
        a, b = c['sats_por_plano']
        return f"{a}" if a == b else f"{a}--{b}"
    desc_camadas = ', '.join(f"{fi(round(c['altitude_km']))}\\,km/{fd(c['inclinacao_graus'], 1)}$^\\circ$ "
                             f"({c['planos']}$\\times${spp(c)})" for c in cam)
    if len(cam) > 1:
        desc_camadas = desc_camadas[::-1].replace(' ,', ' e ', 1)[::-1]
    fatores = [c['fator_velocidade'] for c in cam] or [1.0]
    passo = (d['t_fim_s'] - d['t_ini_s']) / max(d['instantes'] - 1, 1)
    situ = d['referencial']
    if situ.startswith('rotação da Terra com sinal'):
        t_ref = (f"os planos orbitais giram para leste a {fd(d['rotacao_planos_graus_h'], 2)}$^\\circ$/h em relação "
                 f"ao referencial das estações em solo, quando deveriam regredir para oeste a "
                 f"{fd(np.degrees(OMEGA_TERRA) * 3600, 2)}$^\\circ$/h por causa da rotação da Terra. Isso é "
                 f"consistente com a implementação do modelo, que soma $+2\\pi t/24\\,\\mathrm{{h}}$ à longitude de "
                 f"cada plano. Corrigimos o traço girando as posições a {fd(abs(d['correcao_graus_h']), 2)}$^\\circ$/h "
                 f"para {'oeste' if d['correcao_graus_h'] < 0 else 'leste'} em torno do eixo polar, o que o leva a um "
                 f"referencial fixo à Terra (ECEF) coincidente com o do ns-3 em $t=0$")
    elif situ.startswith('inercial'):
        t_ref = ("as posições estão em referencial inercial; foram convertidas para um referencial fixo à Terra (ECEF) "
                 "girando-as de $-\\omega_\\oplus t$ em torno do eixo polar, com alinhamento em $t=0$")
    elif situ.startswith('fixo'):
        t_ref = "o traço já está em referencial fixo à Terra (ECEF); nenhuma correção foi necessária"
    else:
        t_ref = (f"os planos giram {fd(abs(d['rotacao_planos_graus_h']), 2)}$^\\circ$/h em longitude, uma taxa "
                 f"inesperada; giramos as posições a {fd(abs(d['correcao_graus_h']), 2)}$^\\circ$/h em torno do eixo "
                 f"polar para levá-lo a um referencial fixo à Terra (ECEF)")
    if max(abs(f - 1) for f in fatores) > 0.02:
        t_vel = (f"Além disso, o avanço angular no traço é {fd(min(fatores), 2)}--{fd(max(fatores), 2)} vezes o "
                 f"kepleriano (o modelo usa o raio da Terra, e não o da órbita, para converter velocidade em ângulo): "
                 f"cada instante é uma configuração válida da constelação, mas a escala de tempo da evolução fica "
                 f"comprimida, o que será considerado na análise temporal.")
    else:
        t_vel = ""
    radios = 'todas as tecnologias' if not cfg.radio else '/'.join(cfg.radio)
    ordem_r = ['GSM', 'CDMA', 'UMTS', 'LTE', 'NR']
    cont_r = sorted(r['radios_no_arquivo'].items(), key=lambda kv: (ordem_r.index(kv[0]) if kv[0] in ordem_r else 99, kv[0]))
    lista_r = [f"{nome} {fi(q)}" for nome, q in cont_r]
    lista_r = ', '.join(lista_r[:-1]) + ' e ' + lista_r[-1] if len(lista_r) > 1 else ''.join(lista_r)
    descartadas = r['linhas_bs'] - r['n_bs']
    if cfg.radio:
        t_filtro = (f"Do arquivo com {fi(r['linhas_bs'])} células ({lista_r}), mantivemos as de tecnologia {radios}: "
                    f"{fi(r['n_bs'])} células de {fi(r['operadoras'])} operadoras em {fi(r['paises_mcc'])} países "
                    f"(códigos MCC). Cada vértice terrestre é uma célula; células de um mesmo sítio não foram "
                    f"agregadas.")
    else:
        t_filtro = (f"Usamos todas as células do arquivo ({lista_r}"
                    + (f"; {fi(descartadas)} linhas com coordenadas ou códigos inválidos foram descartadas" if descartadas else "")
                    + f"), {fi(r['n_bs'])} ao todo, de {fi(r['operadoras'])} operadoras em {fi(r['paises_mcc'])} "
                    f"países (códigos MCC), e tratamos todas como estações-base 5G NR: a infraestrutura celular "
                    f"existente serve de aproximação para as posições de uma rede NR. Cada vértice terrestre é uma "
                    f"célula; células de um mesmo sítio, de setores ou tecnologias diferentes, não foram agregadas.")
    if cfg.terrestre == 'knn':
        t_terr = (f"cada estação liga-se às {cfg.k_vizinhos} estações mais próximas da mesma operadora (mesmos MCC e "
                  f"MNC), desde que estejam a até {fd(cfg.raio_terrestre_km, 0)}\\,km --- um substituto esparso para a rede de "
                  f"transporte (\\emph{{backhaul}}); ligar todos os pares a {fd(cfg.raio_terrestre_km, 0)}\\,km, como "
                  f"na proposta, formava cliques de milhares de vértices nas metrópoles")
    else:
        t_terr = (f"estações da mesma operadora (mesmos MCC e MNC) a até {fd(cfg.raio_terrestre_km, 0)}\\,km são "
                  f"todas ligadas entre si")
    if cfg.isl == 'grade':
        t_isl = ("topologia \\emph{+Grid} \\cite{hypatia}: cada satélite liga-se aos dois vizinhos do próprio plano e ao "
                 "satélite correspondente de cada plano adjacente da mesma camada (planos identificados a partir das "
                 "trajetórias)")
    else:
        t_isl = "cada satélite liga-se a todos os satélites dentro do alcance"
    t_maxsat = (f"cada estação mantém apenas os {cfg.max_sat_por_bs} satélites de maior elevação"
                if cfg.max_sat_por_bs else "cada estação liga-se a todos os satélites visíveis")
    if r['transitividade'] < r['clust_medio'] / 3:
        t_trans = (f"A transitividade ({fd(r['transitividade'], 4)}) é bem menor que a clusterização local média "
                   f"({fd(r['clust_medio'], 4)}): os satélites, de grau muito alto, concentram tríades abertas "
                   f"(pares de estações distantes que veem o mesmo satélite) e pesam mais no cálculo global, enquanto "
                   f"as estações, ligadas a vizinhas próximas que veem os mesmos satélites, têm $C_i$ médio "
                   f"{fd(r['clust_medio_bs'], 2)} contra {fd(r['clust_medio_sat'], 2)} dos satélites. ")
    else:
        t_trans = ""
    if not np.isfinite(r['altitude_min_isl_km']):
        t_vis = "No instante analisado não há ISLs."
    elif r['altitude_min_isl_km'] > cfg.altitude_rasante_km + 100:
        t_vis = (f"Nas altitudes desta constelação, o alcance é a restrição ativa: o ISL mais baixo do instante "
                 f"analisado passa a {fi(round(r['altitude_min_isl_km']))}\\,km de altitude.")
    else:
        t_vis = (f"No instante analisado, o ISL mais baixo passa a {fi(round(r['altitude_min_isl_km']))}\\,km de "
                 f"altitude, perto do limite de visada.")
    pct_gsl = 100 * r['arestas_por_tipo']['GSL'] / r['arestas']
    if r['componentes'] == 1:
        t_comp = ("A rede é conexa: os enlaces GSL ligam todas as estações-base à camada satelital, de modo que "
                  "a distribuição de tamanhos das componentes (item 9) não se aplica.")
    else:
        t_comp = (f"Há {fi(r['componentes'])} componentes; a maior reúne "
                  f"{fd(100 * r['maior_componente'] / r['vertices'], 2)}\\% dos vértices, e "
                  f"{fi(r['bs_sem_satelite_visivel'])} estações não têm satélite visível acima de "
                  f"{fd(cfg.elevacao_min_graus, 0)}$^\\circ$ neste instante (Figura~\\ref{{fig:dist}}c).")
    t_res = (f"No instante analisado, as arestas GSL são {fd(pct_gsl, 1)}\\% do total. Os graus das duas camadas "
             f"ocupam faixas distintas (Figura~\\ref{{fig:dist}}a): estações-base têm grau médio "
             f"{fd(r['grau_medio_bs'], 1)} (máximo {fi(r['grau_max_bs'])}), enquanto satélites têm grau médio "
             f"{fd(r['grau_medio_sat'], 1)} (máximo {fi(r['grau_max_sat'])}), pois cada satélite agrega as estações "
             f"que o veem acima de {fd(cfg.elevacao_min_graus, 0)}$^\\circ$. {t_trans}{t_comp}")
    dist_txt = (f"{fd(r['distancia_media'], 3)}" if r['distancia_exata'] else
                f"{fd(r['distancia_media'], 3)} $\\pm$ {fd(r['distancia_media_ic95'], 3)}")
    dist_nota = ("exata" if r['distancia_exata'] else
                 f"estimada com BFS a partir de {fi(r['fontes_bfs'])} vértices sorteados; IC 95\\%")

    macros = {
        'InstanteT': fd(r['instante_s'], 0), 'NumInstantes': fi(d['instantes']), 'PassoTraco': fd(passo, 0),
        'TIniTraco': fd(d['t_ini_s'], 0), 'TFimTraco': fd(d['t_fim_s'], 0), 'NumSat': fi(r['n_sat']),
        'DescricaoCamadas': desc_camadas, 'TextoReferencial': t_ref, 'TextoVelocidade': t_vel,
        'LinhasBS': fi(r['linhas_bs']), 'FiltroRadio': radios, 'NumBS': fi(r['n_bs']),
        'NumOperadoras': fi(r['operadoras']), 'TextoFiltroBS': t_filtro, 'NumPaises': fi(r['paises_mcc']),
        'TextoTerrestre': t_terr, 'TextoISL': t_isl, 'TextoMaxSat': t_maxsat,
        'ISLMaxKm': fi(cfg.isl_max_km), 'AltitudeRasante': fd(cfg.altitude_rasante_km, 0),
        'ElevacaoMin': fd(cfg.elevacao_min_graus, 0), 'TextoVisada': t_vis, 'NumVertices': fi(r['vertices']),
        'NumArestas': fi(r['arestas']), 'ArestasTerr': fi(r['arestas_por_tipo']['terrestre']),
        'ArestasISL': fi(r['arestas_por_tipo']['ISL']), 'ArestasGSL': fi(r['arestas_por_tipo']['GSL']),
        'GrauMedio': fd(r['grau_medio'], 2), 'GrauMedioBS': fd(r['grau_medio_bs'], 2),
        'GrauMedioSat': fd(r['grau_medio_sat'], 2), 'GrauMax': fi(r['grau_max']), 'GrauMaxBS': fi(r['grau_max_bs']),
        'GrauMaxSat': fi(r['grau_max_sat']), 'Transitividade': fd(r['transitividade'], 4),
        'ClustMedio': fd(r['clust_medio'], 4), 'ClustMedioGrauDois': fd(r['clust_medio_grau2'], 4),
        'ClustMedioBS': fd(r['clust_medio_bs'], 4), 'ClustMedioSat': fd(r['clust_medio_sat'], 4),
        'DistMedia': dist_txt, 'NotaDist': dist_nota, 'Diametro': fi(r['diametro']), 'Densidade': fc(r['densidade']),
        'NumComponentes': fi(r['componentes']), 'TamMaiorComp': fi(r['maior_componente']),
        'FracMaiorComp': fd(100 * r['maior_componente'] / r['vertices'], 2) + '\\%', 'TextoResultados': t_res,
    }
    with open(caminho, 'w') as f:
        f.write("% Gerado por analise_parcial.py - não edite à mão (rode a análise de novo)\n")
        f.write(f"\\newif\\ifVariasComponentes\\VariasComponentes{'true' if r['componentes'] > 1 else 'false'}\n")
        for nome, valor in macros.items():
            f.write(f"\\newcommand{{\\{nome}}}{{{valor}}}\n")


if __name__ == '__main__':
    main()

