#!/usr/bin/env python3
"""
falhas.py - fase final do projeto MO438: falhas, resiliência e redistribuição de carga.

Usa a mesma rede do rede.py (o modelo da entrega parcial) num instante do traço e simula:
  base       rede intacta: carga de cada satélite e medidas de referência
  regional   estações removidas em círculos de 50, 250 e 1.000 km nas regiões mais densas,
             comparadas com a remoção aleatória do mesmo número de estações
  hubs       falha do hub de maior grau e do de maior carga, e curvas de ataque
             (remoção por grau e por carga x aleatória, como em Albert, Jeong e Barabási, 2000)
  satelites  satélites aleatórios, os de maior grau, os de maior carga e cada plano orbital inteiro
  cascata    cascatas de sobrecarga: um satélite falha quando a carga passa da capacidade, que é
             (1+alfa) x a carga inicial dele (Motter-Lai) ou (1+alfa) x a maior carga inicial (uniforme)
  tempo      as medidas principais repetidas em vários instantes do traço
  todos      tudo acima, nessa ordem, e as figuras
  figuras    refaz as figuras e o resumo a partir dos CSVs, sem recalcular

Carga: o tráfego entre pares de estações-base segue o caminho de menor latência de propagação
(fibra a 2c/3 nos enlaces terrestres, c nos enlaces de rádio). A carga de um vértice é a fração dos
pares de estações cujo caminho passa por ele (intermediação restrita a pares de estações), estimada
com --fontes estações de origem sorteadas e todas as estações como destino. As mesmas origens e os
mesmos destinos são usados antes e depois de cada falha, de modo que a diferença mede só o
redirecionamento do tráfego.

Uso no servidor (na pasta do rede.py, com os.log e base_stations.csv):
  nohup nice -n 10 python -u falhas.py todos --threads 28 > falhas.log 2>&1 &
  python falhas.py figuras                  # depois, para refazer só as figuras e o resumo
Saída: resultados_falhas/ (CSVs, resumo_base.json, resumo.txt e figuras/)
"""
import os

for _var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_var, '1')      # o paralelismo vem dos processos, não das bibliotecas

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, shortest_path
from scipy.spatial import cKDTree

from rede import GSL, R_TERRA, TERRESTRE, Rede, adicionar_argumentos, config_de_args

V_FIBRA_KM_MS = 299_792.458 / 1.5 / 1000     # luz na fibra (2c/3), em km por ms
V_RADIO_KM_MS = 299_792.458 / 1000           # luz no vácuo, em km por ms
R_TERRA_KM = R_TERRA / 1000


@dataclass
class ConfigSim:
    instante_s: float = None
    fontes: int = 256
    fontes_bfs: int = 64
    reps: int = 5
    semente: int = 42
    atraso_salto_ms: float = 0.0
    raios_km: tuple = (50.0, 250.0, 1000.0)
    n_centros: int = 3
    sep_centros_km: float = 1500.0
    centros: tuple = None
    fracoes_ataque: tuple = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
    fracoes_sat: tuple = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
    alfas: tuple = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    max_passos_cascata: int = 50
    instantes_tempo: int = 10
    saida: str = 'resultados_falhas'


class Registro:
    def __init__(self):
        self.t0 = time.perf_counter()

    def __call__(self, msg):
        print(f"[{time.perf_counter() - self.t0:8.1f} s] {msg}", flush=True)


def gini(x):
    """Índice de Gini: 0 = carga igual em todos; perto de 1 = carga concentrada em poucos."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n, total = x.size, x.sum()
    if n == 0 or total <= 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, n + 1), x) / (n * total) - (n + 1) / n)


# ---------------------------------------------------------------- trabalho dos processos
_E = {}   # estado do instante, herdado pelos processos de trabalho via fork


def _blocos(x, k):
    for i in range(0, len(x), k):
        yield x[i:i + k]


def _removido(pacote):
    return np.unpackbits(pacote, count=_E['n']).astype(bool)


def _csr_sem(removido):
    """Grafo (CSR simétrico, peso = latência em ms) sem os vértices marcados em `removido`."""
    lin, col, w, n = _E['lin'], _E['col'], _E['w'], _E['n']
    if removido is not None:
        ok = ~(removido[lin] | removido[col])
        lin, col, w = lin[ok], col[ok], w[ok]
    ip = np.zeros(n + 1, np.int32)
    ip[1:] = np.cumsum(np.bincount(lin, minlength=n))
    return csr_matrix((w, col, ip), shape=(n, n))


def _grafo(chave, pacote):
    c = _E.get('cache')
    if c is None or c[0] != chave:
        c = (chave, _csr_sem(None if pacote is None else _removido(pacote)))
        _E['cache'] = c
    return c[1]


def _acumular(s, pred, dist, alvo, carga):
    """Soma em `carga` os vértices intermediários dos caminhos s -> t (t em `alvo`) da árvore `pred`.
    Devolve o número de destinos alcançados e as somas de latência e de saltos desses caminhos."""
    tg = np.flatnonzero(alvo & (pred >= 0))
    if tg.size == 0:
        return 0, 0.0, 0
    lat = float(dist[tg].sum(dtype=np.float64))
    saltos = tg.size
    cur = pred[tg]
    cur = cur[cur != s]
    pedacos = []
    while cur.size:                      # sobe a árvore, um nível por vez, até chegar à origem
        pedacos.append(cur)
        saltos += cur.size
        cur = pred[cur]
        cur = cur[cur != s]
    if pedacos:
        carga += np.bincount(np.concatenate(pedacos), minlength=carga.size)
    return tg.size, lat, saltos


def _t_arvores(linhas):
    """Árvores de caminhos mínimos da rede intacta para as origens `linhas` (usadas como referência)."""
    G = _grafo(-1, None)
    fontes = _E['fontes'][linhas]
    P = np.empty((len(linhas), _E['n']), np.int32)
    D = np.empty((len(linhas), _E['n']), np.float32)
    for i0 in range(0, len(linhas), 8):
        d, p = dijkstra(G, directed=True, indices=fontes[i0:i0 + 8], return_predecessors=True)
        P[i0:i0 + len(d)] = p
        D[i0:i0 + len(d)] = d
    return linhas, P, D


def _t_carga(args):
    chave, pacote_grafo, pacote_alvo, linhas = args
    n, nb = _E['n'], _E['nb']
    alvo = np.zeros(n, bool)
    alvo[:nb] = True
    if pacote_alvo is not None:
        alvo &= ~_removido(pacote_alvo)
    fontes = _E['fontes'][linhas]
    carga = np.zeros(n)
    atend, lat, saltos = 0, 0.0, 0
    if pacote_grafo is None and 'pred0' in _E:          # rede intacta: árvores já calculadas
        for j, s in zip(linhas, fontes):
            a, l_, h = _acumular(s, _E['pred0'][j], _E['dist0'][j], alvo, carga)
            atend += a
            lat += l_
            saltos += h
    else:
        G = _grafo(chave, pacote_grafo)
        for lote in _blocos(fontes, 8):
            d, p = dijkstra(G, directed=True, indices=lote, return_predecessors=True)
            for i, s in enumerate(lote):
                a, l_, h = _acumular(s, p[i], d[i], alvo, carga)
                atend += a
                lat += l_
                saltos += h
    demanda = int(alvo.sum()) * len(fontes) - int(alvo[fontes].sum())
    return carga, atend, lat, saltos, demanda


def _t_bfs(args):
    chave, pacote, origens = args
    G = _grafo(chave, pacote)
    soma, pares = 0.0, 0
    for lote in _blocos(origens, 8):
        h = shortest_path(G, method='D', directed=True, unweighted=True, indices=lote)
        ok = np.isfinite(h) & (h > 0)
        soma += float(h[ok].sum())
        pares += int(ok.sum())
    return soma, pares


# ---------------------------------------------------------------- simulador de um instante
class Simulador:
    """Rede de um instante, com as medidas de cada cenário de falha calculadas em paralelo."""

    def __init__(self, rede, k, cs, log=print):
        t0 = time.perf_counter()
        self.rede, self.k, self.cs, self.log = rede, k, cs, log
        a = rede.instante(k)
        self.t, self.pos_sat = a['t'], a['pos_sat']
        self.n, self.nb, self.ns = rede.n_nos, rede.n_bs, rede.n_sat
        self.u, self.v, self.tipo = a['u'], a['v'], a['tipo']
        self.n_arestas = len(self.u)
        km = a['comprimento_km'].astype(np.float64)
        w = np.where(self.tipo == TERRESTRE, km / V_FIBRA_KM_MS, km / V_RADIO_KM_MS) + cs.atraso_salto_ms
        w = np.maximum(w, 1e-9)                       # células no mesmo sítio têm comprimento zero
        lin = np.concatenate((self.u, self.v))
        col = np.concatenate((self.v, self.u))
        o = np.lexsort((col, lin))
        self.grau = np.bincount(lin, minlength=self.n)
        rng = np.random.default_rng(cs.semente)
        self.fontes = np.sort(rng.choice(self.nb, min(cs.fontes, self.nb), replace=False))
        self.ordem_bfs = rng.permutation(self.n)
        self.threads = max(1, rede.cfg.threads)
        _E.clear()
        _E.update(n=self.n, nb=self.nb, lin=lin[o].astype(np.int32), col=col[o].astype(np.int32),
                  w=np.concatenate((w, w))[o], fontes=self.fontes)
        del lin, col, o
        lat = np.degrees(np.arcsin(self.pos_sat[:, 2] / np.linalg.norm(self.pos_sat, axis=1)))
        lon = np.degrees(np.arctan2(self.pos_sat[:, 1], self.pos_sat[:, 0]))
        self.sat_lat, self.sat_lon = lat, lon

        # árvores de caminhos mínimos da rede intacta, uma por origem sorteada
        pred0 = np.empty((len(self.fontes), self.n), np.int32)
        dist0 = np.empty((len(self.fontes), self.n), np.float32)
        partes = [p for p in np.array_split(np.arange(len(self.fontes)), self.threads * 2) if len(p)]
        with mp.get_context('fork').Pool(self.threads) as pool:
            for linhas, P, D in pool.imap_unordered(_t_arvores, partes):
                pred0[linhas] = P
                dist0[linhas] = D
        _E['pred0'], _E['dist0'] = pred0, dist0
        self.pool = mp.get_context('fork').Pool(self.threads)
        self._chave = 0
        self.base = self.avaliar([], antes=False)
        self.carga0 = self.base['_carga']
        self._tot0 = self.base['_tot']
        log(f"Instante t = {self.t:g} s pronto em {time.perf_counter() - t0:.1f} s: {self.n:,} vértices, "
            f"{self.n_arestas:,} arestas, {len(self.fontes)} estações de origem, {self.threads} processos")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.terminate()
        self.pool.join()

    # ---------------------------------------------------------- partes paralelas
    def _partes(self, x):
        return [p for p in np.array_split(x, self.threads * 2) if len(p)]

    def _ell(self, chave, pacote, origens):
        if len(origens) == 0:
            return float('nan')
        res = self.pool.map(_t_bfs, [(chave, pacote, o) for o in self._partes(origens)])
        soma, pares = sum(r[0] for r in res), sum(r[1] for r in res)
        return soma / pares if pares else float('nan')

    def _carga(self, chave, pacote_grafo, pacote_alvo, linhas):
        res = self.pool.map(_t_carga, [(chave, pacote_grafo, pacote_alvo, p) for p in self._partes(linhas)])
        carga = np.zeros(self.n)
        for r in res:
            carga += r[0]
        return carga, sum(r[1] for r in res), sum(r[2] for r in res), sum(r[3] for r in res), \
            sum(r[4] for r in res)

    # ---------------------------------------------------------- um cenário
    def avaliar(self, removidos, com_ell=True, com_carga=True, antes=True, guardar=False):
        """Medidas da rede sem os vértices `removidos` (índices: estações 0..nb-1, satélites nb..n-1)."""
        removidos = np.unique(np.asarray(removidos, dtype=np.int64))
        self._chave += 1
        chave, n, nb = self._chave, self.n, self.nb
        rem = np.zeros(n, bool)
        rem[removidos] = True
        vivo = ~rem
        pacote = np.packbits(rem) if removidos.size else None
        B0, B = nb, int(vivo[:nb].sum())

        nc, lab = connected_components(_csr_sem(rem if removidos.size else None), directed=True,
                                       connection='strong')
        tam = np.bincount(lab[vivo], minlength=nc)
        lcc = int(np.argmax(tam))
        r = {'n_removidos': int(removidos.size), 'estacoes_removidas': int(rem[:nb].sum()),
             'satelites_removidos': int(rem[nb:].sum()), 'S': tam[lcc] / n, 'S_vivos': tam[lcc] / max(int(vivo.sum()), 1),
             'componentes': int((tam > 0).sum())}
        g = (self.tipo == GSL) & vivo[self.u] & vivo[self.v]
        tem = np.zeros(nb, bool)
        tem[self.u[g]] = True
        r['estacoes_sem_satelite_visivel'] = int((vivo[:nb] & ~tem).sum())
        comp_sat = np.zeros(nc, bool)
        comp_sat[lab[nb:][vivo[nb:]]] = True
        r['estacoes_sem_acesso_satelital'] = int((vivo[:nb] & ~comp_sat[lab[:nb]]).sum())
        bc = np.bincount(lab[:nb][vivo[:nb]], minlength=nc).astype(np.float64)
        conect = float((bc * (bc - 1)).sum())
        r['pares_conectados'] = conect / (B * (B - 1)) if B > 1 else 0.0
        r['pares_conectados_orig'] = conect / (B0 * (B0 - 1))

        if com_ell:
            o = self.ordem_bfs
            origens = o[(lab[o] == lcc) & vivo[o]][:self.cs.fontes_bfs]
            r['ell'] = self._ell(chave, pacote, origens)

        if com_carga:
            linhas = np.flatnonzero(vivo[self.fontes])
            if linhas.size == 0:
                return r
            esc = B / linhas.size / (B0 * (B0 - 1))     # estende a amostra a todas as estações vivas
            Ld, at, lat, h, dem = self._carga(chave, pacote, pacote, linhas)
            Ld *= esc
            r['_tot'] = (at, lat, h, dem)
            r['demanda_atendida'] = at / dem if dem else float('nan')
            r['demanda_atendida_orig'] = at * esc
            r['latencia_ms'] = lat / at if at else float('nan')
            r['saltos'] = h / at if at else float('nan')
            sv = vivo[nb:]
            cs_ = Ld[nb:][sv]
            r['carga_max_sat'] = float(cs_.max()) if cs_.size else 0.0
            r['gini_sat'] = gini(cs_)
            r['carga_total_sat'] = float(cs_.sum())
            r['_carga'] = Ld
            if antes:
                if removidos.size and rem[:nb].any():   # mesmos pares (sem as estações removidas), rede intacta
                    La, at_a, lat_a, h_a, _ = self._carga(chave, None, pacote, linhas)
                    La *= esc
                else:                                   # nenhuma estação removida: é a própria referência
                    La, (at_a, lat_a, h_a, _) = self.carga0, self._tot0
                r['demanda_atendida_antes'] = at_a / dem if dem else float('nan')
                r['latencia_ms_antes'] = lat_a / at_a if at_a else float('nan')
                r['saltos_antes'] = h_a / at_a if at_a else float('nan')
                r['carga_max_sat_antes'] = float(La[nb:].max())
                r['gini_sat_antes'] = gini(La[nb:])
                delta = (Ld - La)[nb:][sv]
                r['aumento_max_sat'] = float(delta.max()) if delta.size else 0.0
                r['carga_redistribuida'] = float(np.clip(delta, 0, None).sum())
                r['carga_satelites_falhos'] = float(La[nb:][~sv].sum())
                ref = La[nb:][sv]
                pos = ref > 0
                razao = Ld[nb:][sv][pos] / ref[pos]
                r['sat_aumento_50pct'] = int((razao > 1.5).sum())
                r['sat_aumento_100pct'] = int((razao > 2.0).sum())
                if guardar:
                    r['_La_sat'] = La[nb:].astype(np.float32)
            if guardar:
                r['_Ld_sat'] = Ld[nb:].astype(np.float32)
        return r

    # ---------------------------------------------------------- utilidades
    def estacoes_no_circulo(self, lat, lon, raio_km):
        if not hasattr(self, '_arv_bs'):
            self._arv_bs = cKDTree(self.rede.bs_unit)
        la, lo = np.radians(lat), np.radians(lon)
        c = [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        corda = 2 * np.sin(raio_km / (2 * R_TERRA_KM))
        return np.array(sorted(self._arv_bs.query_ball_point(c, corda)), dtype=np.int64)

    def info_satelite(self, j):
        """j = índice do satélite (0..ns-1)."""
        rd = self.rede
        cam = int(rd.camada[j]) if rd.camada is not None else -1
        return {'satelite': int(rd.ids_sat[j]), 'camada_km': round(rd.camadas[cam]['altitude_km']) if cam >= 0 else None,
                'plano': int(rd.plano[j]) if rd.plano is not None else -1,
                'lat': round(float(self.sat_lat[j]), 2), 'lon': round(float(self.sat_lon[j]), 2)}


def centros_densos(rede, cs):
    """Centros das falhas regionais: as estações com mais vizinhas a até 250 km, afastadas entre si."""
    if cs.centros:
        return [(float(la), float(lo)) for la, lo in cs.centros]
    unit = rede.bs_unit
    arv = cKDTree(unit)
    rng = np.random.default_rng(cs.semente)
    amostra = rng.choice(len(unit), min(len(unit), 20_000), replace=False)

    def corda(km):
        return 2 * np.sin(km / (2 * R_TERRA_KM))
    cont = arv.query_ball_point(unit[amostra], corda(250), return_length=True)
    escolhidos = []
    for i in amostra[np.argsort(-cont, kind='stable')]:
        if all(np.linalg.norm(unit[i] - unit[j]) >= corda(cs.sep_centros_km) for j in escolhidos):
            escolhidos.append(i)
            if len(escolhidos) == cs.n_centros:
                break
    lat, lon = rede.bs['lat'].to_numpy(), rede.bs['lon'].to_numpy()
    return [(round(float(lat[i]), 3), round(float(lon[i]), 3)) for i in escolhidos]


# ---------------------------------------------------------------- saída
def _limpo(r):
    return {k: v for k, v in r.items() if not k.startswith('_')}


def _salvar(linhas, caminho):
    pd.DataFrame([_limpo(r) for r in linhas]).to_csv(caminho, index=False)


def _resumo_linha(r):
    p = [f"S={r['S']:.4f}", f"comp.={r['componentes']:,}"]
    if 'ell' in r:
        p.append(f"<l>={r['ell']:.3f}")
    if 'latencia_ms' in r:
        p.append(f"lat.={r['latencia_ms']:.2f} ms")
    if 'demanda_atendida' in r:
        p.append(f"atendida={r['demanda_atendida']:.4f}")
    if 'aumento_max_sat' in r and r.get('carga_max_sat_antes'):
        p.append(f"aum.máx.={100 * r['aumento_max_sat'] / r['carga_max_sat_antes']:.1f}% da maior carga")
    p.append(f"sem acesso={r['estacoes_sem_acesso_satelital']:,}")
    return ' | '.join(p)


def salvar_base(sim, saida, centros):
    b, nb = sim.base, sim.nb
    L = sim.carga0[nb:]
    top = np.argsort(-L)[:10]
    resumo = {'instante_s': sim.t, 'vertices': sim.n, 'estacoes': nb, 'satelites': sim.ns, 'arestas': sim.n_arestas,
              'arestas_por_tipo': {t: int((sim.tipo == i).sum()) for i, t in enumerate(('terrestre', 'ISL', 'GSL'))},
              'fontes_carga': len(sim.fontes), 'fontes_bfs': sim.cs.fontes_bfs, 'centros_regionais': centros,
              'medidas': {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in _limpo(b).items()},
              'satelites_maior_carga': [dict(sim.info_satelite(j), grau=int(sim.grau[nb + j]), carga=float(L[j]))
                                        for j in top],
              'satelites_sem_carga': int((L == 0).sum()),
              'estacao_maior_carga': {'indice': int(np.argmax(sim.carga0[:nb])),
                                      'carga': float(sim.carga0[:nb].max())}}
    with open(os.path.join(saida, 'resumo_base.json'), 'w') as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    rd = sim.rede
    pd.DataFrame({'satelite': rd.ids_sat, 'camada': rd.camada, 'plano': rd.plano, 'lat': sim.sat_lat.round(3),
                  'lon': sim.sat_lon.round(3), 'grau': sim.grau[nb:], 'carga': L}).to_csv(
        os.path.join(saida, 'satelites_base.csv'), index=False)
    amostra = np.random.default_rng(0).choice(nb, min(nb, 60_000), replace=False)
    np.savez_compressed(os.path.join(saida, 'estacoes_amostra.npz'),
                        lat=rd.bs['lat'].to_numpy()[amostra].astype(np.float32),
                        lon=rd.bs['lon'].to_numpy()[amostra].astype(np.float32))
    sim.log(f"Rede intacta: latência média {b['latencia_ms']:.2f} ms, {b['saltos']:.1f} saltos, "
            f"<l> = {b['ell']:.3f} (estimada), Gini da carga dos satélites {b['gini_sat']:.3f}, "
            f"maior carga {100 * b['carga_max_sat']:.2f}% dos pares, {(L == 0).sum():,} satélites sem carga")


# ---------------------------------------------------------------- experimentos
def exp_regional(sim, cs, saida, centros):
    nb, log = sim.nb, sim.log
    linhas, total, feito = [], len(centros) * len(cs.raios_km) * (1 + cs.reps), 0
    log(f"== Falhas regionais: {len(centros)} centros x raios {list(cs.raios_km)} km, {cs.reps} sorteios aleatórios")
    for ic, (lat, lon) in enumerate(centros):
        for R in cs.raios_km:
            rem = sim.estacoes_no_circulo(lat, lon, R)
            rng = np.random.default_rng([cs.semente, 1, ic, int(R)])
            cenarios = [('regional', 0, rem)] + [('aleatoria', i, rng.choice(nb, len(rem), replace=False))
                                                 for i in range(cs.reps)]
            for estr, rep, rr in cenarios:
                r = sim.avaliar(rr)
                r.pop('_carga', None)
                r.update(centro=ic, lat_centro=lat, lon_centro=lon, raio_km=R, estrategia=estr, rep=rep)
                linhas.append(r)
                feito += 1
                _salvar(linhas, os.path.join(saida, 'regional.csv'))
                log(f"  [{feito}/{total}] centro {ic} ({lat:.2f}, {lon:.2f}) R={R:g} km {estr}"
                    f"{'' if estr == 'regional' else f' #{rep}'}: {len(rr):,} estações | {_resumo_linha(r)}")


def exp_hubs(sim, cs, saida):
    nb, n, log = sim.nb, sim.n, sim.log
    L0, grau = sim.carga0, sim.grau
    alvos = [('satelite de maior grau', nb + int(np.argmax(grau[nb:]))),
             ('satelite de maior carga', nb + int(np.argmax(L0[nb:]))),
             ('estacao de maior grau', int(np.argmax(grau[:nb]))),
             ('estacao de maior carga', int(np.argmax(L0[:nb])))]
    log("== Falha de um hub")
    linhas, red = [], {}
    for nome, no in alvos:
        r = sim.avaliar([no], guardar=True)
        r.pop('_carga', None)
        red[nome.replace(' ', '_')] = np.stack((r.pop('_La_sat'), r.pop('_Ld_sat')))
        r.update(cenario=nome, vertice=no, grau=int(grau[no]), carga_base=float(L0[no]))
        if no >= nb:
            r.update(sim.info_satelite(no - nb))
        linhas.append(r)
        log(f"  {nome} (grau {grau[no]:,}, carga {100 * L0[no]:.2f}% dos pares): {_resumo_linha(r)}")
    _salvar(linhas, os.path.join(saida, 'hubs.csv'))
    np.savez_compressed(os.path.join(saida, 'redistribuicao_hubs.npz'), **red)

    log(f"== Curvas de ataque: {len(cs.fracoes_ataque)} frações, por grau, por carga e {cs.reps} aleatórias")
    rank = {'grau': np.argsort(-grau, kind='stable'), 'carga': np.argsort(-L0, kind='stable')}
    perms = [np.random.default_rng([cs.semente, 2, i]).permutation(n) for i in range(cs.reps)]
    b = _limpo(sim.base)
    linhas = [dict(b, estrategia=e, fracao=0.0, rep=0) for e in ('grau', 'carga', 'aleatoria')]
    total, feito = len(cs.fracoes_ataque) * (2 + cs.reps), 0
    for f in cs.fracoes_ataque:
        k = max(1, int(round(f * n)))
        for estr, rr, rep in [('grau', rank['grau'][:k], 0), ('carga', rank['carga'][:k], 0)] + \
                [('aleatoria', p[:k], i) for i, p in enumerate(perms)]:
            r = sim.avaliar(rr, com_carga=False)
            r.update(estrategia=estr, fracao=f, rep=rep)
            linhas.append(r)
            feito += 1
            _salvar(linhas, os.path.join(saida, 'ataque.csv'))
            log(f"  [{feito}/{total}] f={f:g} {estr}: {r['estacoes_removidas']:,} estações e "
                f"{r['satelites_removidos']:,} satélites | {_resumo_linha(r)}")


def exp_satelites(sim, cs, saida):
    nb, ns, log, rd = sim.nb, sim.ns, sim.log, sim.rede
    rank_g = nb + np.argsort(-sim.grau[nb:], kind='stable')
    rank_c = nb + np.argsort(-sim.carga0[nb:], kind='stable')
    perms = [nb + np.random.default_rng([cs.semente, 3, i]).permutation(ns) for i in range(cs.reps)]
    log(f"== Camada satelital: frações {list(cs.fracoes_sat)} dos satélites, por grau, por carga e "
        f"{cs.reps} aleatórias")
    b = _limpo(sim.base)
    linhas = [dict(b, estrategia=e, fracao=0.0, rep=0) for e in ('maior grau', 'maior carga', 'aleatoria')]
    total, feito = len(cs.fracoes_sat) * (2 + cs.reps), 0
    for q in cs.fracoes_sat:
        k = max(1, int(round(q * ns)))
        for estr, rr, rep in [('maior grau', rank_g[:k], 0), ('maior carga', rank_c[:k], 0)] + \
                [('aleatoria', p[:k], i) for i, p in enumerate(perms)]:
            r = sim.avaliar(rr)
            r.pop('_carga', None)
            r.update(estrategia=estr, fracao=q, rep=rep)
            linhas.append(r)
            feito += 1
            _salvar(linhas, os.path.join(saida, 'satelites.csv'))
            log(f"  [{feito}/{total}] {k:,} satélites ({100 * q:g}%) {estr}: "
                f"{r['estacoes_sem_satelite_visivel']:,} estações sem satélite visível | {_resumo_linha(r)}")

    if rd.plano is None:
        log("  (sem planos identificados no traço: falhas por plano puladas)")
        return
    planos = np.unique(rd.plano)
    log(f"== Falha de cada plano orbital inteiro ({len(planos)} planos)")
    linhas, red = [], {}
    for i, p in enumerate(planos, 1):
        sats = np.flatnonzero(rd.plano == p)
        cam = int(rd.camada[sats[0]])
        r = sim.avaliar(nb + sats, guardar=True)
        r.pop('_carga', None)
        red[f'plano_{p}'] = np.stack((r.pop('_La_sat'), r.pop('_Ld_sat')))
        r.update(plano=int(p), camada=cam, altitude_km=round(rd.camadas[cam]['altitude_km'], 1),
                 inclinacao_graus=round(rd.camadas[cam]['inclinacao_graus'], 1), n_satelites_plano=len(sats),
                 carga_base_plano=float(sim.carga0[nb + sats].sum()))
        linhas.append(r)
        _salvar(linhas, os.path.join(saida, 'planos.csv'))
        log(f"  [{i}/{len(planos)}] plano {p} ({r['altitude_km']:g} km, {r['inclinacao_graus']:g} graus, "
            f"{len(sats)} satélites): {_resumo_linha(r)}")
    np.savez_compressed(os.path.join(saida, 'redistribuicao_planos.npz'), **red)


def exp_cascata(sim, cs, saida, centros):
    """Dois modelos de capacidade dos satélites:
    proporcional  Motter-Lai: (1+alfa) x a carga inicial do satélite, com piso na carga média dos satélites
                  (sem o piso, um satélite ocioso teria capacidade nula e cairia com qualquer tráfego)
    uniforme      satélites idênticos: (1+alfa) x a maior carga inicial da constelação"""
    nb, ns, log, rd = sim.nb, sim.ns, sim.log, sim.rede
    L0 = sim.carga0[nb:]
    modelos = (('proporcional', lambda a_: (1 + a_) * np.maximum(L0, L0.mean())),
               ('uniforme', lambda a_: np.full(ns, (1 + a_) * L0.max())))
    j = int(np.argmax(L0))
    gatilhos = [('satelite de maior carga', np.array([nb + j]))]
    if rd.plano is not None:
        gatilhos.append((f'plano do satelite de maior carga (plano {int(rd.plano[j])})',
                         nb + np.flatnonzero(rd.plano == rd.plano[j])))
    if centros:
        R = max(cs.raios_km)
        gatilhos.append((f'regional {R:g} km (centro 0)', sim.estacoes_no_circulo(*centros[0], R)))
    log(f"== Cascatas: capacidade proporcional (Motter-Lai, piso = carga média {L0.mean():.2e}) e uniforme "
        f"((1+alfa) x {L0.max():.2e}); alfa em {list(cs.alfas)}")
    linhas = []
    for nome, R0 in gatilhos:
        cache = {}                                   # o mesmo conjunto de falhas aparece em vários alfas
        for modelo, capacidade in modelos:
            for alfa in cs.alfas:
                cap = capacidade(alfa)
                removidos = np.unique(R0)
                historico = []
                convergiu = False
                for passo in range(1, cs.max_passos_cascata + 1):
                    chave = removidos.tobytes()
                    if chave not in cache:
                        r = sim.avaliar(removidos, com_ell=False, antes=False)
                        cache[chave] = (r, r.pop('_carga', np.zeros(sim.n))[nb:])
                    r, Lc = cache[chave]
                    vivos = np.ones(ns, bool)
                    vivos[removidos[removidos >= nb] - nb] = False
                    sobre = np.flatnonzero(vivos & (Lc > cap * (1 + 1e-9)))
                    historico.append(int(sobre.size))
                    if sobre.size == 0:
                        convergiu = True
                        break
                    removidos = np.union1d(removidos, nb + sobre)
                r = dict(r)
                falhas = int((removidos >= nb).sum()) - int((R0 >= nb).sum())
                r.update(gatilho=nome, modelo=modelo, alfa=alfa, passos=len(historico), convergiu=convergiu,
                         falhas_cascata=falhas, falhas_por_passo=' '.join(map(str, historico)))
                linhas.append(r)
                _salvar(linhas, os.path.join(saida, 'cascata.csv'))
                log(f"  {nome}, capacidade {modelo}, alfa={alfa:g}: {falhas:,} satélites falharam em cascata em "
                    f"{len(historico)} passos | atendida={r['demanda_atendida']:.4f} | lat.={r['latencia_ms']:.2f} ms")


def exp_tempo(rede, cs, saida, centros, log):
    T = len(rede.tempos)
    ks = np.unique(np.linspace(0, T - 1, min(cs.instantes_tempo, T)).round().astype(int))
    log(f"== Análise temporal: {len(ks)} instantes entre t = {rede.tempos[ks[0]]:g} s e {rede.tempos[ks[-1]]:g} s")
    linhas = []
    for i, k in enumerate(ks, 1):
        with Simulador(rede, int(k), cs, log) as sim:
            nb, ns = sim.nb, sim.ns
            b = _limpo(sim.base)
            linhas.append(dict(b, t_s=sim.t, cenario='rede intacta', rep=0))
            j = int(np.argmax(sim.carga0[nb:]))
            cen = [('satelite de maior carga', [nb + j], 0)]
            if rede.plano is not None:
                cen.append(('plano do satelite de maior carga', nb + np.flatnonzero(rede.plano == rede.plano[j]), 0))
            k10 = max(1, int(round(0.1 * ns)))
            cen += [('10% dos satelites ao acaso', nb + np.random.default_rng([cs.semente, 4, r_]).choice(ns, k10, replace=False), r_)
                    for r_ in range(min(cs.reps, 3))]
            if centros:
                R = max(cs.raios_km)
                cen.append((f'regional {R:g} km (centro 0)', sim.estacoes_no_circulo(*centros[0], R), 0))
            for nome, rr, rep in cen:
                r = sim.avaliar(rr)
                r.pop('_carga', None)
                linhas.append(dict(r, t_s=sim.t, cenario=nome, rep=rep))
                _salvar(linhas, os.path.join(saida, 'tempo.csv'))
            log(f"  [{i}/{len(ks)}] t = {sim.t:g} s: Gini {b['gini_sat']:.3f}, maior carga "
                f"{100 * b['carga_max_sat']:.2f}%, latência {b['latencia_ms']:.2f} ms")


# ---------------------------------------------------------------- figuras e resumo
CORES_CAMADA = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
COR = {'aleatoria': '0.3', 'grau': '#d62728', 'maior grau': '#d62728', 'carga': '#6a3d9a', 'maior carga': '#6a3d9a'}
ROTULO = {'aleatoria': 'aleatória', 'grau': 'por grau', 'carga': 'por carga', 'maior grau': 'maior grau',
          'maior carga': 'maior carga'}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'figure.dpi': 150,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})
    return plt


def _ler(pasta, nome):
    p = os.path.join(pasta, nome)
    return pd.read_csv(p) if os.path.exists(p) else None


def _bonito(nome):
    """Nomes de cenário dos CSVs (sem acento) para as legendas."""
    return str(nome).replace('satelite', 'satélite').replace('estacao', 'estação').replace('1000 km', '1.000 km')


def _curva(ax, df, x, y, estr, faixa=True, **kw):
    d = df[df['estrategia'] == estr]
    if d.empty:
        return
    g = d.groupby(x)[y]
    m = g.mean()
    ax.plot(m.index, m.values, marker='o', ms=2.8, lw=1.1, color=COR.get(estr, 'k'),
            label=kw.pop('label', ROTULO.get(estr, estr)), **kw)
    if faixa and g.size().max() > 1:
        ax.fill_between(m.index, g.min().values, g.max().values, color=COR.get(estr, 'k'), alpha=0.15, lw=0)


def fig_ataque(pasta, figs, plt):
    df = _ler(pasta, 'ataque.csv')
    if df is None:
        return
    base = df[df['fracao'] == 0].iloc[0]
    d = df[df['fracao'] > 0]
    with open(os.path.join(pasta, 'resumo_base.json')) as f:
        rb = json.load(f)
    f_sat = rb['satelites'] / rb['vertices']
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.15))
    for ax, y, rot, le in zip(axs, ('S', 'ell', 'pares_conectados'),
                              ('maior componente $S$ (fração de $N$)', 'distância média $\\langle\\ell\\rangle$ (saltos)',
                               'pares de estações conectados'), 'abc'):
        ax.set_title(f'({le})', loc='left')
        for e in ('grau', 'carga', 'aleatoria'):
            _curva(ax, d, 'fracao', y, e)
        ax.axhline(base[y], color='0.6', lw=0.7, ls=':')
        ax.axvline(f_sat, color='0.6', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlabel('fração de vértices removidos $f$')
        ax.set_ylabel(rot)
        ax.text(f_sat, 0.55, 'fração de \nsatélites ', fontsize=6, color='0.4', va='center', ha='right',
                transform=ax.get_xaxis_transform())
    axs[1].set_yscale('log')
    axs[0].legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(figs, 'fig_ataque.pdf'))
    plt.close(fig)


def fig_satelites(pasta, figs, plt):
    df = _ler(pasta, 'satelites.csv')
    if df is None:
        return
    with open(os.path.join(pasta, 'resumo_base.json')) as f:
        rb = json.load(f)
    nb = rb['estacoes']
    df = df.assign(sem_vis=100 * df['estacoes_sem_satelite_visivel'] / nb)
    base = df[df['fracao'] == 0].iloc[0]
    d = df[df['fracao'] > 0]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
    paineis = [('sem_vis', 'estações sem satélite\nvisível (%)'), ('demanda_atendida', 'pares de estações\natendidos'),
               ('latencia_ms', 'latência média\ndos pares (ms)'), ('gini_sat', 'Gini da carga\ndos satélites')]
    for ax, (y, rot), le in zip(axs.flat, paineis, 'abcd'):
        # latência e Gini só onde ao menos 10% dos pares ainda são atendidos (abaixo disso sobram só
        # pares terrestres vizinhos, e a média deixa de descrever a rede)
        dd = d[d['demanda_atendida'] >= 0.1] if y in ('latencia_ms', 'gini_sat') else d
        for e in ('maior grau', 'maior carga', 'aleatoria'):
            _curva(ax, dd, 'fracao', y, e)
        ax.axhline(base[y], color='0.6', lw=0.7, ls=':')
        ax.set_xscale('log')
        ax.set_xlim(d['fracao'].min() * 0.8, d['fracao'].max() * 1.25)
        ax.set_xlabel('fração $q$ dos satélites\nremovidos')
        ax.set_ylabel(rot)
        ax.set_title(f'({le})', loc='left')
    fig.tight_layout()
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.97))
    fig.savefig(os.path.join(figs, 'fig_satelites.pdf'))
    plt.close(fig)


def fig_planos(pasta, figs, plt):
    df = _ler(pasta, 'planos.csv')
    if df is None:
        return
    df = df.sort_values(['camada', 'plano']).reset_index(drop=True)
    aum = 100 * df['aumento_max_sat'] / df['carga_max_sat_antes']
    dlat = 100 * (df['latencia_ms'] / df['latencia_ms_antes'] - 1)
    cores = [CORES_CAMADA[c % len(CORES_CAMADA)] for c in df['camada']]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 2.6), sharex=True)
    # rótulos curtos: com dois painéis baixos, rótulos longos se sobrepõem (o texto completo vai na legenda da figura)
    for ax, y, rot, le in ((axs[0], aum, 'maior aumento\nde carga (%)', 'a'),
                           (axs[1], dlat, 'variação da\nlatência (%)', 'b')):
        ax.bar(np.arange(len(df)), y, color=cores, width=0.8)
        ax.set_ylabel(rot)
        ax.set_title(f'({le})', loc='left')
        ax.set_ylim(0, float(y.max()) * 1.2)            # espaço para o nome do plano dentro do painel
        i = int(np.argmax(y.to_numpy()))
        ax.annotate(f"plano {df['plano'][i]}", (i, y[i]), xytext=(0, 2), textcoords='offset points',
                    ha='center', fontsize=6)
    fig.align_ylabels(axs)
    from matplotlib.patches import Patch
    axs[1].set_xlabel('plano orbital (agrupado por camada)')
    axs[1].set_xticks([])
    fig.tight_layout()
    fig.legend(handles=[Patch(color=CORES_CAMADA[c % len(CORES_CAMADA)],
                              label=f"{g['altitude_km'].iloc[0]:g} km / {g['inclinacao_graus'].iloc[0]:g}°")
                        for c, g in df.groupby('camada')],
               frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 0.97))
    fig.savefig(os.path.join(figs, 'fig_planos.pdf'))
    plt.close(fig)


def fig_regional(pasta, figs, plt):
    df = _ler(pasta, 'regional.csv')
    if df is None:
        return
    with open(os.path.join(pasta, 'resumo_base.json')) as f:
        rb = json.load(f)
    nb = rb['estacoes']
    tot = rb['medidas']['carga_total_sat']           # soma das cargas dos satélites na rede intacta
    df = df.assign(rem=100 * df['estacoes_removidas'] / nb,
                   dlat=100 * (df['latencia_ms'] / df['latencia_ms_antes'] - 1),
                   red=100 * df['carga_redistribuida'] / tot)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1))
    paineis = [('rem', 'estações removidas (%)'),
               ('dlat', 'variação da latência média\ndos pares sobreviventes (%)'),
               ('red', 'carga redistribuída\n(% da carga dos satélites)')]
    for ax, (y, rot), le in zip(axs.flat, paineis, 'abc'):
        for c, g in df.groupby('centro'):
            cor = CORES_CAMADA[int(c) % len(CORES_CAMADA)]
            rg = g[g['estrategia'] == 'regional'].sort_values('raio_km')
            ax.plot(rg['raio_km'], rg[y], marker='o', ms=3, color=cor,
                    label=f"centro {c} ({g['lat_centro'].iloc[0]:.1f}°, {g['lon_centro'].iloc[0]:.1f}°)"
                    if y == 'rem' else None)
            if y != 'rem':
                al = g[g['estrategia'] == 'aleatoria'].groupby('raio_km')[y]
                ax.plot(al.mean().index, al.mean().values, ls='--', marker='x', ms=3, lw=0.9, color=cor)
        ax.set_xscale('log')
        ax.set_xticks(sorted(df['raio_km'].unique()))
        ax.set_xticklabels([f'{r:g}' for r in sorted(df['raio_km'].unique())])
        ax.set_xlabel('raio da região (km)')
        ax.set_ylabel(rot)
        ax.set_title(f'({le})', loc='left')
    axs[0].set_yscale('log')
    axs[0].legend(frameon=False, fontsize=6)
    axs[1].plot([], [], 'k-', label='falha regional')
    axs[1].plot([], [], 'k--', marker='x', ms=3, label='mesmo nº ao acaso')
    axs[1].legend(frameon=False, fontsize=6, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(figs, 'fig_regional.pdf'))
    plt.close(fig)


def fig_redistribuicao(pasta, figs, plt):
    sb = _ler(pasta, 'satelites_base.csv')
    pl = _ler(pasta, 'planos.csv')
    caminho = os.path.join(pasta, 'redistribuicao_planos.npz')
    if sb is None or pl is None or not os.path.exists(caminho):
        return
    from matplotlib.colors import LogNorm, SymLogNorm
    est = np.load(os.path.join(pasta, 'estacoes_amostra.npz'))
    rel = pl['aumento_max_sat'] / pl['carga_max_sat_antes']
    p = int(pl['plano'][int(np.argmax(rel.to_numpy()))])
    La, Ld = np.load(caminho)[f'plano_{p}']
    falhou = (sb['plano'] == p).to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for ax in axs:
        ax.scatter(est['lon'], est['lat'], s=0.3, c='0.75', lw=0, rasterized=True)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
    c = sb['carga'].to_numpy()
    pos = c > 0
    sc = axs[0].scatter(sb['lon'][pos], sb['lat'][pos], c=100 * c[pos], s=4, lw=0, cmap='viridis',
                        norm=LogNorm(vmin=max(100 * c[pos].min(), 1e-4), vmax=100 * c.max()), rasterized=True)
    axs[0].scatter(sb['lon'][~pos], sb['lat'][~pos], s=2, c='k', marker='.', lw=0, alpha=0.4)
    fig.colorbar(sc, ax=axs[0], shrink=0.75, label='carga (% dos pares)')
    axs[0].set_title('(a) Carga dos satélites na rede intacta')
    d = 100 * (Ld - La)
    lon_s, lat_s = sb['lon'].to_numpy(), sb['lat'].to_numpy()
    mudou = ~falhou & (np.abs(d) > 1e-5)
    axs[1].scatter(lon_s[~falhou & ~mudou], lat_s[~falhou & ~mudou], s=1.5, c='0.55', lw=0, rasterized=True)
    if mudou.any():
        lim = float(np.percentile(np.abs(d[mudou]), 99)) or 1.0
        ordem = np.argsort(np.abs(d[mudou]))                 # as maiores variações por cima
        sc = axs[1].scatter(lon_s[mudou][ordem], lat_s[mudou][ordem], c=d[mudou][ordem], s=7, lw=0,
                            cmap='RdBu_r', norm=SymLogNorm(linthresh=lim / 100, vmin=-lim, vmax=lim, base=10),
                            rasterized=True)
        fig.colorbar(sc, ax=axs[1], shrink=0.75, extend='both', label='variação (pontos percentuais)')
    axs[1].scatter(lon_s[falhou], lat_s[falhou], s=6, c='k', marker='x', lw=0.6)
    axs[1].set_title(f'(b) Variação após a falha do plano {p} (x)')
    fig.tight_layout()
    fig.savefig(os.path.join(figs, 'fig_redistribuicao.png'), dpi=250)
    plt.close(fig)


def fig_cascata(pasta, figs, plt):
    df = _ler(pasta, 'cascata.csv')
    if df is None:
        return
    if 'modelo' not in df:
        df = df.assign(modelo='proporcional')
    estilo = {'proporcional': '-', 'uniforme': '--'}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for i, (g, dg) in enumerate(df.groupby('gatilho', sort=False)):
        cor = CORES_CAMADA[i % len(CORES_CAMADA)]
        for m, d in dg.groupby('modelo'):
            d = d.sort_values('alfa')
            mk = 'o' if m == 'proporcional' else 'x'
            axs[0].plot(d['alfa'], d['falhas_cascata'], marker=mk, ms=3, color=cor, ls=estilo.get(m, ':'))
            axs[1].plot(d['alfa'], d['demanda_atendida'], marker=mk, ms=3, color=cor, ls=estilo.get(m, ':'))
        axs[0].plot([], [], color=cor, lw=4, label=_bonito(g))
    for m, ls in estilo.items():
        axs[1].plot([], [], color='k', ls=ls, marker='o' if m == 'proporcional' else 'x', ms=3,
                    label=f'capacidade {m}')
    axs[0].set_yscale('symlog', linthresh=1)        # mostra juntas as cascatas de ~7 e de ~2.000 satélites
    axs[0].set_ylim(-0.2, None)
    axs[0].set_ylabel('satélites que falharam em cascata')
    axs[1].set_ylabel('pares de estações atendidos')
    for ax, le in zip(axs, 'ab'):
        ax.set_xscale('log')
        ax.set_xlabel('tolerância $\\alpha$ (capacidade = $(1+\\alpha)\\times$ referência)')
        ax.set_title(f'({le})', loc='left')
    axs[1].legend(frameon=False, fontsize=6, loc='center right')
    fig.tight_layout()
    fig.legend(*axs[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False, fontsize=6,
               bbox_to_anchor=(0.5, 0.97))
    fig.savefig(os.path.join(figs, 'fig_cascata.pdf'))
    plt.close(fig)


def fig_tempo(pasta, figs, plt):
    df = _ler(pasta, 'tempo.csv')
    if df is None:
        return
    base = df[df['cenario'] == 'rede intacta'].sort_values('t_s')
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
    axs[0].plot(base['t_s'], base['gini_sat'], marker='o', ms=3, color='k')
    axs[0].set_ylabel('Gini da carga\ndos satélites')
    axs[1].plot(base['t_s'], 100 * base['carga_max_sat'], marker='o', ms=3, color='k')
    axs[1].set_ylabel('maior carga de\nsatélite (% dos pares)')
    outros = [c for c in df['cenario'].unique() if c != 'rede intacta']
    for i, c in enumerate(outros):
        d = df[df['cenario'] == c].groupby('t_s').mean(numeric_only=True).reset_index()
        cor = CORES_CAMADA[i % len(CORES_CAMADA)]
        axs[2].plot(d['t_s'], 100 * d['aumento_max_sat'] / d['carga_max_sat_antes'], marker='o', ms=3,
                    color=cor, label=_bonito(c))
        axs[3].plot(d['t_s'], 100 * (d['latencia_ms'] / d['latencia_ms_antes'] - 1), marker='o', ms=3, color=cor)
    axs[2].set_ylabel('maior aumento de carga\n(% da maior carga inicial)')
    axs[3].set_ylabel('variação da latência\nmédia (%)')
    for ax, le in zip(axs.flat, 'abcd'):
        ax.set_xlabel('instante do traço (s)')
        ax.set_title(f'({le})', loc='left')
    fig.tight_layout()
    if outros:
        fig.legend(*axs[2].get_legend_handles_labels(), loc='lower center', ncol=len(outros), frameon=False,
                   fontsize=6, bbox_to_anchor=(0.5, 0.97))
    fig.savefig(os.path.join(figs, 'fig_tempo.pdf'))
    plt.close(fig)


def escrever_resumo(pasta):
    """Os números principais de cada experimento, em texto, para escrever o relatório."""
    L = []
    caminho = os.path.join(pasta, 'resumo_base.json')
    if not os.path.exists(caminho):
        return ''
    with open(caminho) as f:
        rb = json.load(f)
    b = rb['medidas']
    tot = b['carga_total_sat']      # soma das cargas dos satélites = nº médio de satélites por caminho
    L.append(f"REDE INTACTA (t = {rb['instante_s']:g} s): {rb['vertices']:,} vértices, {rb['arestas']:,} arestas; "
             f"carga estimada com {rb['fontes_carga']} estações de origem")
    L.append(f"  latência média entre estações {b['latencia_ms']:.2f} ms em {b['saltos']:.1f} saltos; "
             f"<l> ~ {b['ell']:.3f}; Gini da carga dos satélites {b['gini_sat']:.3f}; "
             f"maior carga {100 * b['carga_max_sat']:.2f}% dos pares; {rb['satelites_sem_carga']:,} satélites sem carga")
    s0 = rb['satelites_maior_carga'][0]
    L.append(f"  satélite de maior carga: {s0['satelite']} (camada {s0['camada_km']} km, plano {s0['plano']}, "
             f"lat {s0['lat']}, lon {s0['lon']}, grau {s0['grau']:,})")
    hubs = _ler(pasta, 'hubs.csv')
    if hubs is not None:
        L.append("HUBS (falha de um vértice):")
        for _, r in hubs.iterrows():
            L.append(f"  {r['cenario']}: maior aumento de carga {100 * r['aumento_max_sat'] / r['carga_max_sat_antes']:.1f}% "
                     f"da maior carga inicial; carga redistribuída {100 * r['carga_redistribuida'] / tot:.2f}% da carga "
                     f"dos satélites; latência {r['latencia_ms_antes']:.2f} -> {r['latencia_ms']:.2f} ms; "
                     f"{int(r['sat_aumento_50pct'])} satélites com +50% de carga; "
                     f"{int(r['estacoes_sem_satelite_visivel'])} estações sem satélite visível")
    at = _ler(pasta, 'ataque.csv')
    if at is not None:
        L.append("ATAQUE (fração f de vértices removidos; S = maior componente):")
        for e in ('grau', 'carga', 'aleatoria'):
            d = at[(at['estrategia'] == e) & (at['fracao'] > 0)].groupby('fracao')['S'].mean()
            abaixo = d[d < 0.5]
            L.append(f"  {ROTULO[e]}: S < 0,5 a partir de f = {abaixo.index.min():g}" if len(abaixo) else
                     f"  {ROTULO[e]}: S >= 0,5 até f = {d.index.max():g} (S = {d.iloc[-1]:.3f})")
    sat = _ler(pasta, 'satelites.csv')
    if sat is not None:
        L.append("CAMADA SATELITAL (fração q dos satélites removidos):")
        for e in ('maior carga', 'maior grau', 'aleatoria'):
            d = sat[(sat['estrategia'] == e) & (sat['fracao'] > 0)].groupby('fracao').mean(numeric_only=True)
            partes = [f"q={q:g}: atendida {r['demanda_atendida']:.3f}, lat. {r['latencia_ms']:.1f} ms, "
                      f"{int(r['estacoes_sem_satelite_visivel']):,} sem satélite" for q, r in d.iterrows()]
            L.append(f"  {ROTULO[e]}: " + '; '.join(partes))
    pl = _ler(pasta, 'planos.csv')
    if pl is not None:
        rel = 100 * pl['aumento_max_sat'] / pl['carga_max_sat_antes']
        i = int(np.argmax(rel.to_numpy()))
        r = pl.iloc[i]
        L.append(f"PLANOS: pior plano {int(r['plano'])} ({r['altitude_km']:g} km, {r['inclinacao_graus']:g} graus): "
                 f"aumento máximo {rel[i]:.1f}% da maior carga inicial, carga redistribuída "
                 f"{100 * r['carga_redistribuida'] / tot:.1f}% da carga dos satélites, latência {r['latencia_ms_antes']:.2f} -> "
                 f"{r['latencia_ms']:.2f} ms; mediana dos planos {rel.median():.1f}%; "
                 f"máximo de estações sem satélite visível {int(pl['estacoes_sem_satelite_visivel'].max())}")
    rg = _ler(pasta, 'regional.csv')
    if rg is not None:
        L.append("REGIONAL (falha regional x mesmo número de estações ao acaso):")
        for (c, R), g in rg.groupby(['centro', 'raio_km']):
            a = g[g['estrategia'] == 'regional'].iloc[0]
            al = g[g['estrategia'] == 'aleatoria'].mean(numeric_only=True)
            L.append(f"  centro {c} ({a['lat_centro']:.2f}, {a['lon_centro']:.2f}) R={R:g} km: "
                     f"{int(a['estacoes_removidas']):,} estações; Δlatência {100 * (a['latencia_ms'] / a['latencia_ms_antes'] - 1):+.2f}% "
                     f"(acaso {100 * (al['latencia_ms'] / al['latencia_ms_antes'] - 1):+.2f}%); carga redistribuída "
                     f"{100 * a['carga_redistribuida'] / tot:.3f}% da carga dos satélites "
                     f"(acaso {100 * al['carga_redistribuida'] / tot:.3f}%)")
    ca = _ler(pasta, 'cascata.csv')
    if ca is not None:
        if 'modelo' not in ca:
            ca = ca.assign(modelo='proporcional')
        L.append("CASCATA (satélites que falharam além do gatilho):")
        for (g, m), d in ca.groupby(['gatilho', 'modelo'], sort=False):
            L.append(f"  {g}, capacidade {m}: " +
                     ', '.join(f"alfa={r['alfa']:g}: {int(r['falhas_cascata'])}" for _, r in d.iterrows()))
    te = _ler(pasta, 'tempo.csv')
    if te is not None:
        b = te[te['cenario'] == 'rede intacta']
        L.append(f"TEMPO ({len(b)} instantes): Gini {b['gini_sat'].mean():.3f} ± {b['gini_sat'].std():.3f}; "
                 f"maior carga {100 * b['carga_max_sat'].mean():.2f}% ± {100 * b['carga_max_sat'].std():.2f}%; "
                 f"latência {b['latencia_ms'].mean():.2f} ± {b['latencia_ms'].std():.2f} ms")
    texto = '\n'.join(L)
    with open(os.path.join(pasta, 'resumo.txt'), 'w') as f:
        f.write(texto + '\n')
    return texto


def figuras(pasta, log=print):
    plt = _plt()
    figs = os.path.join(pasta, 'figuras')
    os.makedirs(figs, exist_ok=True)
    for f in (fig_ataque, fig_satelites, fig_planos, fig_regional, fig_redistribuicao, fig_cascata, fig_tempo):
        try:
            f(pasta, figs, plt)
        except Exception as e:                          # uma figura com problema não impede as outras
            log(f"  figura {f.__name__} não foi gerada: {type(e).__name__}: {e}")
    log(f"Figuras em {figs}")
    texto = escrever_resumo(pasta)
    if texto:
        log("Resumo (também em resumo.txt):\n" + texto)


# ---------------------------------------------------------------- CLI
def main():
    comum = argparse.ArgumentParser(add_help=False)
    adicionar_argumentos(comum)
    c = ConfigSim()
    g = comum.add_argument_group('simulação')
    g.add_argument('--instante-s', type=float, default=None, help='instante analisado (padrão: o primeiro)')
    g.add_argument('--fontes', type=int, default=c.fontes, help='estações de origem sorteadas para a carga')
    g.add_argument('--fontes-bfs', type=int, default=c.fontes_bfs, help='origens para estimar <l>')
    g.add_argument('--reps', type=int, default=c.reps, help='sorteios de cada cenário aleatório')
    g.add_argument('--semente', type=int, default=c.semente)
    g.add_argument('--atraso-salto-ms', type=float, default=c.atraso_salto_ms,
                   help='atraso fixo por salto, somado à propagação (padrão: 0)')
    g.add_argument('--raios-km', default=','.join(f'{r:g}' for r in c.raios_km))
    g.add_argument('--centros', default=None, help="centros das falhas regionais, ex.: '-1.29,36.82;51,10'")
    g.add_argument('--n-centros', type=int, default=c.n_centros)
    g.add_argument('--alfas', default=','.join(f'{a:g}' for a in c.alfas))
    g.add_argument('--instantes-tempo', type=int, default=c.instantes_tempo)
    g.add_argument('--saida', default=c.saida)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for nome, ajuda in (('base', 'só a rede intacta'), ('regional', 'falhas regionais'),
                        ('hubs', 'falha de hubs e curvas de ataque'), ('satelites', 'falhas na camada satelital'),
                        ('cascata', 'cascatas de Motter-Lai'), ('tempo', 'análise temporal'),
                        ('todos', 'tudo, na ordem, e as figuras'), ('figuras', 'refaz figuras e resumo')):
        sub.add_parser(nome, parents=[comum], help=ajuda)
    a = ap.parse_args()
    cs = ConfigSim(instante_s=a.instante_s, fontes=a.fontes, fontes_bfs=a.fontes_bfs, reps=a.reps,
                   semente=a.semente, atraso_salto_ms=a.atraso_salto_ms,
                   raios_km=tuple(float(x) for x in a.raios_km.split(',')), n_centros=a.n_centros,
                   centros=tuple(tuple(float(y) for y in x.split(',')) for x in a.centros.split(';'))
                   if a.centros else None,
                   alfas=tuple(float(x) for x in a.alfas.split(',')), instantes_tempo=a.instantes_tempo,
                   saida=a.saida)
    os.makedirs(cs.saida, exist_ok=True)
    log = Registro()
    if a.cmd == 'figuras':
        figuras(cs.saida, log)
        return

    rede = Rede(config_de_args(a))
    k = rede.indice_instante(cs.instante_s)
    rede.checar_memoria(k, a.max_ram_gb)
    rede.resumo()
    centros = centros_densos(rede, cs)
    with open(os.path.join(cs.saida, f'config_{a.cmd}.json'), 'w') as f:
        json.dump({'comando': a.cmd, 'rede': asdict(rede.cfg), 'simulacao': asdict(cs), 'centros': centros},
                  f, indent=2, ensure_ascii=False, default=str)
    log(f"Centros das falhas regionais: {centros}")
    if a.cmd != 'tempo':
        with Simulador(rede, k, cs, log) as sim:
            salvar_base(sim, cs.saida, centros)
            if a.cmd in ('hubs', 'todos'):
                exp_hubs(sim, cs, cs.saida)
            if a.cmd in ('satelites', 'todos'):
                exp_satelites(sim, cs, cs.saida)
            if a.cmd in ('regional', 'todos'):
                exp_regional(sim, cs, cs.saida, centros)
            if a.cmd in ('cascata', 'todos'):
                exp_cascata(sim, cs, cs.saida, centros)
    if a.cmd in ('tempo', 'todos'):
        exp_tempo(rede, cs, cs.saida, centros, log)
    if a.cmd == 'todos':
        figuras(cs.saida, log)
    log("Concluído.")


if __name__ == '__main__':
    main()
