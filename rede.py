#!/usr/bin/env python3
"""
rede.py - rede integrada terrestre-satelital (projeto MO438).

Monta, para qualquer instante do traço de mobilidade do ns-3-leo, um grafo com
três tipos de enlace:
    TERRESTRE  estação-base <-> estação-base da mesma operadora (MCC, NET)
    ISL        satélite <-> satélite, com linha de visada geométrica
    GSL        estação-base <-> satélite acima da elevação mínima

Linha de comando:
    python rede.py diagnostico --threads 8            # unidades, referencial, camadas, tamanhos
    python rede.py exportar --passo 5 --threads 8     # snapshots .npz em paralelo

Biblioteca (ver analise_parcial.py):
    from rede import Config, Rede
    rede = Rede(Config(threads=8))
    a = rede.instante(0)            # dict: u, v, tipo, comprimento_km, pos_sat, t
    G = rede.grafo_networkit(0)     # ou rede.grafo_networkx(0) para redes pequenas
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

R_TERRA = 6_371_009.0        # m, mesmo raio terrestre do ns-3-leo
OMEGA_TERRA = 7.2921159e-5   # rad/s, rotação sideral
MU_TERRA = 3.986004418e14    # m^3/s^2
TERRESTRE, ISL, GSL = 0, 1, 2
NOMES_TIPO = ('terrestre', 'ISL', 'GSL')


@dataclass
class Config:
    log: str = 'os.log'
    bs: str = 'base_stations.csv'
    radio: tuple = None               # None = todas as células (tratadas igualmente); ('NR',) = só NR
    terrestre: str = 'knn'            # 'knn' ou 'raio' (critério do script original)
    k_vizinhos: int = 6
    raio_terrestre_km: float = 50.0
    isl: str = 'grade'                # 'grade' (+Grid) ou 'alcance' (todos no alcance)
    isl_max_km: float = 5000.0
    altitude_rasante_km: float = 80.0
    elevacao_min_graus: float = 40.0
    max_sat_por_bs: int = 0           # 0 = todos os satélites visíveis
    corrigir_referencial: bool = True
    threads: int = 8


# ---------------------------------------------------------------- utilidades
def _girar_z(P, ang):
    """Gira pontos P (..., 3) em torno do eixo z; `ang` deve ser compatível com P[..., 0]."""
    c, s = np.cos(ang), np.sin(ang)
    x, y = P[..., 0], P[..., 1]
    return np.stack((c * x - s * y, s * x + c * y, P[..., 2]), axis=-1)


def _norm(P):
    return np.sqrt(np.einsum('...i,...i->...', P, P))


def _linha_de_visada(A, D, r_min):
    """True se o segmento A -> A+D fica sempre a pelo menos r_min do centro da Terra."""
    dd = np.einsum('ij,ij->i', D, D)
    s = np.clip(-np.einsum('ij,ij->i', A, D) / np.where(dd > 0, dd, 1.0), 0.0, 1.0)
    Q = A + s[:, None] * D
    return np.einsum('ij,ij->i', Q, Q) >= r_min * r_min


_UNIDADES = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0, 'min': 60.0, 'h': 3600.0, 'd': 86400.0}


def _tempo_em_segundos(valores):
    """'+1e+09ns' (formato do ns-3) -> 1.0. Números sem unidade são lidos como segundos."""
    s = pd.Series(np.asarray(valores, dtype=str)).str.strip()
    p = s.str.extract(r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-z]*)$')
    if p[0].isna().any():
        raise ValueError(f"Formato de tempo não reconhecido no log: {s[p[0].isna()].iloc[0]!r}")
    fator = p[1].replace('', 's').map(_UNIDADES)
    if fator.isna().any():
        raise ValueError(f"Unidade de tempo desconhecida no log: {p[1][fator.isna()].iloc[0]!r}")
    return np.round(p[0].astype(float).to_numpy() * fator.to_numpy(), 9)   # resolução de 1 ns


# ---------------------------------------------------------------- satélites
def carregar_traco(caminho):
    """Lê o traço do ns-3-leo (Time:Satellite:x:y:z:Speed) -> tempos (T,), ids (S,), pos (T,S,3)."""
    df = pd.read_csv(caminho, sep=':', skiprows=1, header=None,
                     names=['t', 'sat', 'x', 'y', 'z', 'v'], usecols=['t', 'sat', 'x', 'y', 'z'],
                     dtype={'t': str, 'sat': np.int64})
    cod, rotulos = pd.factorize(df['t'], sort=False)
    t = _tempo_em_segundos(rotulos)[cod]
    sat = df['sat'].to_numpy()
    ordem = np.lexsort((sat, t))
    tempos, inicio = np.unique(t[ordem], return_index=True)
    ids = np.unique(sat)
    T, S = len(tempos), len(ids)
    if len(df) != T * S or not (sat[ordem].reshape(T, S) == ids).all():
        raise ValueError("O traço precisa ter exatamente uma linha por satélite em cada instante.")
    pos = df[['x', 'y', 'z']].to_numpy(dtype=np.float64)[ordem].reshape(T, S, 3)
    return tempos, ids, pos


def estimar_rotacao_planos(pos, tempos, n_sat=400, n_t=40):
    """Taxa rho (rad/s) tal que R_z(rho*t)·r(t) deixa cada órbita num plano fixo.
    rho = +Ω_Terra: traço fixo à Terra (ECEF); rho = 0: inercial; rho = -Ω_Terra: sinal invertido."""
    T, S, _ = pos.shape
    si = np.unique(np.linspace(0, S - 1, min(S, n_sat)).round().astype(int))
    ti = np.unique(np.linspace(0, T - 1, min(T, n_t)).round().astype(int))
    u = pos[np.ix_(ti, si)]
    u = u / _norm(u)[..., None]
    dt = (tempos[ti] - tempos[ti[0]])[:, None]

    def residuo(rho):
        V = _girar_z(u, rho * dt)
        M = np.einsum('tsi,tsj->sij', V, V) / len(ti)
        return float(np.sqrt(np.mean(np.clip(np.linalg.eigvalsh(M)[:, 0], 0, None))))

    grade = np.linspace(-2.0, 2.0, 81) * OMEGA_TERRA
    r = [residuo(x) for x in grade]
    i = int(np.argmin(r))
    res = minimize_scalar(residuo, bounds=(grade[max(i - 1, 0)], grade[min(i + 1, len(grade) - 1)]),
                          method='bounded', options={'xatol': 1e-11})
    return float(res.x), float(res.fun)


def inferir_orbitas(pos, tempos, rho):
    """Camada (por raio), plano (por normal orbital) e fase de cada satélite."""
    T, S, _ = pos.shape
    F0 = _girar_z(pos[0], rho * tempos[0])
    F1 = _girar_z(pos[1], rho * tempos[1])
    cosang = np.einsum('ij,ij->i', F0, F1) / (_norm(F0) * _norm(F1))
    w = np.median(np.arccos(np.clip(cosang, -1, 1))) / (tempos[1] - tempos[0])
    # segundo ponto ~30 graus de arco depois (bom condicionamento, sem passar de 180)
    b = int(np.clip(np.searchsorted(tempos - tempos[0], np.radians(30) / w, 'right') - 1, 1, T - 1))
    Fb = _girar_z(pos[b], rho * tempos[b])
    n = np.cross(F0, Fb)
    n /= _norm(n)[:, None]

    raio = _norm(F0)
    ordem = np.argsort(raio)
    corte = np.flatnonzero(np.diff(raio[ordem]) > 2000.0) + 1          # camadas separadas por > 2 km
    camada = np.empty(S, dtype=np.int64)
    for c, grupo in enumerate(np.split(ordem, corte)):
        camada[grupo] = c

    plano = np.empty(S, dtype=np.int64)
    fase = np.empty(S)
    camadas, prox = [], 0
    for c in range(camada.max() + 1):
        idx = np.flatnonzero(camada == c)
        pares = cKDTree(n[idx]).query_pairs(1e-3, output_type='ndarray').reshape(-1, 2)
        m = len(idx)
        nc, rot = connected_components(coo_matrix((np.ones(len(pares)), (pares[:, 0], pares[:, 1])),
                                                  shape=(m, m)), directed=False)
        normais = np.array([n[idx[rot == p]].mean(axis=0) for p in range(nc)])
        normais /= _norm(normais)[:, None]
        nodo = np.cross([0.0, 0.0, 1.0], normais)
        nodo[_norm(nodo) < 1e-9] = (1.0, 0.0, 0.0)
        nodo /= _norm(nodo)[:, None]
        ordem_planos = np.argsort(np.arctan2(nodo[:, 1], nodo[:, 0]))   # longitude do nodo
        novo = np.empty(nc, dtype=np.int64)
        novo[ordem_planos] = np.arange(nc) + prox
        plano[idx] = novo[rot]
        rh = F0[idx] / raio[idx, None]
        a, nn = nodo[rot], normais[rot]
        fase[idx] = np.arctan2(np.einsum('ij,ij->i', rh, np.cross(nn, a)), np.einsum('ij,ij->i', rh, a))
        # velocidade observada x kepleriana (no referencial em que o plano é fixo)
        v_obs = _norm(F1[idx] - F0[idx]) / (tempos[1] - tempos[0])
        v_kep = np.sqrt(MU_TERRA / raio[idx])
        por_plano = np.bincount(rot)
        camadas.append({'altitude_km': float(raio[idx].mean() - R_TERRA) / 1000,
                        'inclinacao_graus': float(np.degrees(np.median(np.arccos(np.clip(normais[:, 2], -1, 1))))),
                        'planos': int(nc), 'sats_por_plano': (int(por_plano.min()), int(por_plano.max())),
                        'satelites': int(m), 'fator_velocidade': float(np.median(v_obs / v_kep))})
        prox += nc
    return camada, plano, fase, camadas


def pares_grade(camada, plano, fase, P0):
    """Topologia +Grid: 2 vizinhos no mesmo plano e 1 em cada plano adjacente da mesma camada."""
    pares = []
    for c in np.unique(camada):
        ids_planos = np.unique(plano[camada == c])
        membros = []
        for p in ids_planos:
            m = np.flatnonzero(plano == p)
            m = m[np.argsort(fase[m])]
            membros.append(m)
            if len(m) >= 3:
                pares.append(np.column_stack((m, np.roll(m, -1))))
            elif len(m) == 2:
                pares.append(m[None, :])
        P = len(membros)
        adjacentes = [(p, (p + 1) % P) for p in range(P)] if P >= 3 else ([(0, 1)] if P == 2 else [])
        for p, q in adjacentes:
            A, B = membros[p], membros[q]
            if len(A) == len(B):        # mesmo nº de satélites: melhor deslocamento de índice
                d = [np.mean(_norm(P0[A] - P0[np.roll(B, -k)])) for k in range(len(B))]
                pares.append(np.column_stack((A, np.roll(B, -int(np.argmin(d))))))
            else:                       # planos desiguais: vizinho mais próximo
                _, j = cKDTree(P0[B]).query(P0[A])
                pares.append(np.column_stack((A, B[j])))
    pares = np.sort(np.concatenate(pares), axis=1)
    return np.unique(pares[pares[:, 0] != pares[:, 1]], axis=0)


# ---------------------------------------------------------------- estações-base
def carregar_estacoes(caminho, radio):
    """Lê o CSV do OpenCellID em blocos, filtrando a tecnologia de rádio."""
    usar = ['radio', 'mcc', 'net', 'lon', 'lat']
    partes, contagem, total = [], {}, 0
    for bloco in pd.read_csv(caminho, usecols=usar, dtype={'radio': str}, chunksize=1_000_000):
        total += len(bloco)
        for r, q in bloco['radio'].value_counts().items():
            contagem[r] = contagem.get(r, 0) + int(q)
        if radio:
            bloco = bloco[bloco['radio'].isin(radio)]
        bloco = bloco.dropna(subset=['mcc', 'net', 'lat', 'lon'])
        bloco = bloco[(bloco['lat'].abs() <= 90) & (bloco['lon'].abs() <= 180)]
        partes.append(bloco)
    bs = pd.concat(partes, ignore_index=True)
    bs['mcc'] = bs['mcc'].astype(np.int64)
    bs['net'] = bs['net'].astype(np.int64)
    bs['radio'] = bs['radio'].astype('category')
    if len(bs) == 0:
        raise ValueError(f"Nenhuma estação-base após o filtro radio={radio}. Tecnologias no arquivo: {contagem}")
    return bs, contagem, total


def enlaces_terrestres(bs, xyz, cfg):
    """Arestas entre estações da mesma operadora: k vizinhos mais próximos (até o raio) ou todas no raio."""
    raio = cfg.raio_terrestre_km * 1000.0
    blocos = []
    for idx in bs.groupby(['mcc', 'net'], sort=False).indices.values():
        if len(idx) < 2:
            continue
        X = xyz[idx]
        arvore = cKDTree(X)
        if cfg.terrestre == 'raio':
            n_pares = (arvore.count_neighbors(arvore, raio) - len(idx)) // 2
            if n_pares > 50_000_000:
                raise MemoryError(f"Critério 'raio' geraria {n_pares:,} arestas numa só operadora; use --terrestre knn.")
            p = arvore.query_pairs(raio, output_type='ndarray').reshape(-1, 2)
            blocos.append(idx[p])
        else:
            k = min(cfg.k_vizinhos + 1, len(idx))
            _, j = arvore.query(X, k=k, distance_upper_bound=raio, workers=cfg.threads)
            i = np.repeat(np.arange(len(idx)), k).reshape(-1, k)
            ok = (j < len(idx)) & (j != i)
            blocos.append(np.column_stack((idx[i[ok]], idx[j[ok]])))
    if not blocos:
        return np.empty((0, 2), dtype=np.int64)
    pares = np.sort(np.concatenate(blocos), axis=1)
    return np.unique(pares, axis=0)


# ---------------------------------------------------------------- rede
class Rede:
    def __init__(self, cfg=None, verbose=True):
        self.cfg = cfg = cfg or Config()
        self.log = print if verbose else (lambda *a, **k: None)
        t0 = time.perf_counter()

        self.log(f"Lendo o traço de satélites ({cfg.log})...", flush=True)
        self.tempos, self.ids_sat, self.pos = carregar_traco(cfg.log)
        T, S, _ = self.pos.shape
        self.n_sat = S
        raio_med = float(np.median(_norm(self.pos[0])))
        self.diag = {'instantes': T, 'satelites': S, 't_ini_s': float(self.tempos[0]),
                     't_fim_s': float(self.tempos[-1]), 'raio_orbital_mediano_km': raio_med / 1000}
        if not 6.5e6 < raio_med < 5e7:
            raise ValueError(f"Raio orbital mediano {raio_med:.4g}: as posições não parecem estar em metros.")

        if T >= 3:
            self.rho, residuo = estimar_rotacao_planos(self.pos, self.tempos)
        else:
            self.rho, residuo = OMEGA_TERRA, float('nan')
            self.log("  Aviso: poucos instantes para diagnosticar o referencial; assumindo ECEF.")
        rel = self.rho / OMEGA_TERRA
        if abs(rel - 1) < 0.05:
            situacao, self.taxa_correcao = 'fixo à Terra (ECEF)', 0.0
        elif abs(rel) < 0.05:
            situacao, self.taxa_correcao = 'inercial (ECI)', self.rho - OMEGA_TERRA
        elif abs(rel + 1) < 0.05:
            situacao, self.taxa_correcao = 'rotação da Terra com sinal invertido', self.rho - OMEGA_TERRA
        else:
            situacao, self.taxa_correcao = 'taxa de rotação inesperada', self.rho - OMEGA_TERRA
        if not cfg.corrigir_referencial:
            self.taxa_correcao = 0.0
        self.diag.update({'rotacao_planos_graus_h': float(-np.degrees(self.rho) * 3600),
                          'residuo_planaridade': residuo, 'referencial': situacao,
                          'correcao_graus_h': float(np.degrees(self.taxa_correcao) * 3600)})

        if T >= 2:
            self.camada, self.plano, self.fase, self.camadas = inferir_orbitas(self.pos, self.tempos, self.rho)
        else:
            self.camada = self.plano = self.fase = None
            self.camadas = []
        self.pares_grade = None
        if cfg.isl == 'grade':
            if self.plano is None:
                raise ValueError("A topologia +Grid precisa de pelo menos 2 instantes no traço; use --isl alcance.")
            self.pares_grade = pares_grade(self.camada, self.plano, self.fase, self.posicoes_satelites(0))

        self.log(f"Lendo as estações-base ({cfg.bs})...", flush=True)
        self.bs, self.radios_arquivo, self.linhas_bs = carregar_estacoes(cfg.bs, cfg.radio)
        lat, lon = np.radians(self.bs['lat'].to_numpy()), np.radians(self.bs['lon'].to_numpy())
        self.bs_unit = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
        self.bs_xyz = R_TERRA * self.bs_unit
        self.n_bs = len(self.bs)
        self.n_nos = self.n_bs + self.n_sat

        self.terr = enlaces_terrestres(self.bs, self.bs_xyz, cfg)
        self.comp_terr = _norm(self.bs_xyz[self.terr[:, 1]] - self.bs_xyz[self.terr[:, 0]])

        # busca de GSL: KD-trees estáticas por blocos de estações (limita a memória por consulta)
        e = np.radians(cfg.elevacao_min_graus)
        r_max = float(_norm(self.pos[0]).max())
        self.seno_min = np.sin(e)
        self.raio_busca_gsl = np.sqrt(r_max**2 - (R_TERRA * np.cos(e))**2) - R_TERRA * np.sin(e) + 1000.0
        passo = 100_000
        self._blocos_bs = [(i, cKDTree(self.bs_xyz[i:i + passo])) for i in range(0, self.n_bs, passo)]
        self.log(f"  Rede pronta em {time.perf_counter() - t0:.1f} s.", flush=True)

    # ---------------------------------------------------------- estimativa
    def estimar(self, k=0, amostra=20_000, semente=0):
        """Arestas e pico de RAM (GB) previstos para o instante k, a partir de uma amostra de estações.
        Constantes calibradas nos testes: ~70 B por aresta e ~(100 + 8*threads) B por vértice,
        além da memória já em uso e de ~0,4 GB transitórios no cálculo dos GSL (blocos de 100 mil estações)."""
        idx = np.random.default_rng(semente).choice(self.n_bs, min(amostra, self.n_bs), replace=False)
        P = self.posicoes_satelites(k)
        r = cKDTree(P).sparse_distance_matrix(cKDTree(self.bs_xyz[idx]), self.raio_busca_gsl, output_type='ndarray')
        seno = (np.einsum('ij,ij->i', P[r['i']], self.bs_unit[idx][r['j']]) - R_TERRA) / r['v']
        vis = np.bincount(r['j'][seno >= self.seno_min], minlength=len(idx))
        if self.cfg.max_sat_por_bs:
            vis = np.minimum(vis, self.cfg.max_sat_por_bs)
        arestas = int(len(self.terr) + 2 * self.n_sat + vis.mean() * self.n_bs)
        try:
            em_uso = int(open('/proc/self/statm').read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError):
            em_uso = 300 * self.n_bs + 4e8
        gb = (em_uso + 70 * arestas + (100 + 8 * self.cfg.threads) * self.n_nos) / 1e9 + 0.4
        return arestas, gb, float(vis.mean())

    def checar_memoria(self, k=0, limite_gb=60.0):
        arestas, gb, por_bs = self.estimar(k)
        self.log(f"Estimativa: ~{arestas:,} arestas ({por_bs:.1f} satélites por estação), "
                 f"pico de ~{gb:.1f} GB de RAM (limite --max-ram-gb {limite_gb:g})", flush=True)
        if gb > limite_gb:
            raise SystemExit(
                "Estimativa acima do limite de RAM. Opções: --max-sat-por-bs 2 (cada estação liga-se só aos 2 "
                "satélites de maior elevação), um filtro de tecnologia (--radio), ou aumentar --max-ram-gb se "
                "o servidor comportar.")
        return arestas, gb

    # ---------------------------------------------------------- por instante
    def indice_instante(self, t_s=None):
        return 0 if t_s is None else int(np.argmin(np.abs(self.tempos - t_s)))

    def posicoes_satelites(self, k):
        """Posições (S, 3) no referencial fixo à Terra, já corrigidas."""
        return _girar_z(self.pos[k], self.taxa_correcao * self.tempos[k])

    def _isl(self, P):
        cand = self.pares_grade if self.pares_grade is not None else \
            cKDTree(P).query_pairs(self.cfg.isl_max_km * 1000, output_type='ndarray').reshape(-1, 2)
        A = P[cand[:, 0]]
        D = P[cand[:, 1]] - A
        comp = _norm(D)
        ok = (comp <= self.cfg.isl_max_km * 1000) & \
            _linha_de_visada(A, D, R_TERRA + self.cfg.altitude_rasante_km * 1000)
        return cand[ok], comp[ok]

    def _gsl(self, P):
        arvore = cKDTree(P)
        out_bs, out_sat, out_d = [], [], []
        kmax = self.cfg.max_sat_por_bs
        for ini, arv_bs in self._blocos_bs:
            r = arvore.sparse_distance_matrix(arv_bs, self.raio_busca_gsl, output_type='ndarray')
            sat, bs, d = r['i'], r['j'] + ini, r['v']
            seno = (np.einsum('ij,ij->i', P[sat], self.bs_unit[bs]) - R_TERRA) / d
            ok = seno >= self.seno_min
            sat, bs, d, seno = sat[ok], bs[ok], d[ok], seno[ok]
            if kmax:
                o = np.lexsort((-seno, bs))
                bo = bs[o]
                novo = np.r_[True, bo[1:] != bo[:-1]]
                pos_grupo = np.arange(len(bo)) - np.maximum.accumulate(np.where(novo, np.arange(len(bo)), 0))
                o = o[pos_grupo < kmax]
                sat, bs, d = sat[o], bs[o], d[o]
            out_bs.append(bs)
            out_sat.append(sat)
            out_d.append(d)
        return np.concatenate(out_bs), np.concatenate(out_sat), np.concatenate(out_d)

    def instante(self, k):
        """Arestas do instante k. Vértices: 0..n_bs-1 = estações; n_bs.. = satélites (ordem de ids_sat)."""
        P = self.posicoes_satelites(k)
        isl, c_isl = self._isl(P)
        g_bs, g_sat, c_gsl = self._gsl(P)
        nb = self.n_bs
        return {
            'u': np.concatenate((self.terr[:, 0], isl[:, 0] + nb, g_bs)).astype(np.int64),
            'v': np.concatenate((self.terr[:, 1], isl[:, 1] + nb, g_sat + nb)).astype(np.int64),
            'tipo': np.concatenate((np.full(len(self.terr), TERRESTRE, np.int8),
                                    np.full(len(isl), ISL, np.int8), np.full(len(g_bs), GSL, np.int8))),
            'comprimento_km': (np.concatenate((self.comp_terr, c_isl, c_gsl)) / 1000).astype(np.float32),
            'pos_sat': P, 't': float(self.tempos[k]),
        }

    def grafo_networkit(self, k, arestas=None):
        import networkit as nk
        a = arestas if arestas is not None else self.instante(k)
        G = nk.Graph(self.n_nos, weighted=False, directed=False)
        G.addEdges((a['u'].astype(np.uint64), a['v'].astype(np.uint64)))
        return G

    def grafo_networkx(self, k, arestas=None):
        """nx.Graph com atributos (lento para milhões de arestas; útil em recortes)."""
        import networkx as nx
        a = arestas if arestas is not None else self.instante(k)
        nomes = np.array([f"BS_{i}" for i in range(self.n_bs)] + [f"SAT_{s}" for s in self.ids_sat], dtype=object)
        G = nx.Graph(t=a['t'])
        b = self.bs
        G.add_nodes_from((nomes[i], {'type': 'BaseStation', 'lat': la, 'lon': lo, 'mcc': mc, 'net': ne, 'radio': ra})
                         for i, (la, lo, mc, ne, ra) in enumerate(zip(b['lat'].tolist(), b['lon'].tolist(),
                                                                      b['mcc'].tolist(), b['net'].tolist(),
                                                                      b['radio'].tolist())))
        for j, p in enumerate(a['pos_sat'].tolist()):
            G.add_node(nomes[self.n_bs + j], type='Satellite', pos=p,
                       camada=None if self.camada is None else int(self.camada[j]),
                       plano=None if self.plano is None else int(self.plano[j]))
        G.add_edges_from(zip(nomes[a['u']], nomes[a['v']],
                             ({'tipo': NOMES_TIPO[t], 'comprimento_km': float(c)}
                              for t, c in zip(a['tipo'].tolist(), a['comprimento_km'].tolist()))))
        return G

    def resumo(self, a=None):
        d = self.diag
        self.log(f"\nTraço: {d['satelites']:,} satélites x {d['instantes']:,} instantes "
                 f"(t = {d['t_ini_s']:g} a {d['t_fim_s']:g} s)")
        self.log(f"Unidades: raio orbital mediano {d['raio_orbital_mediano_km']:,.0f} km -> posições em metros")
        for c in self.camadas:
            spp = c['sats_por_plano']
            spp = spp[0] if spp[0] == spp[1] else f"{spp[0]}-{spp[1]}"
            self.log(f"  camada {c['altitude_km']:7.1f} km  {c['inclinacao_graus']:5.1f} graus  "
                     f"{c['planos']:3d} planos x {spp} sat.  (velocidade no traço = "
                     f"{c['fator_velocidade']:.3f} x kepleriana)")
        self.log(f"Referencial: planos orbitais giram {d['rotacao_planos_graus_h']:+.2f} graus/h em longitude "
                 f"(físico em relação à Terra: {-np.degrees(OMEGA_TERRA) * 3600:+.2f}) -> {d['referencial']}")
        self.log(f"  correção aplicada: {d['correcao_graus_h']:+.2f} graus/h em torno do eixo polar")
        self.log(f"Estações-base: {self.linhas_bs:,} linhas no arquivo {self.radios_arquivo}; "
                 + (f"{self.n_bs:,} após filtro radio={self.cfg.radio}" if self.cfg.radio else
                    f"{self.n_bs:,} usadas, todas as tecnologias tratadas igualmente (sem filtro)"))
        if a is not None:
            cont = np.bincount(a['tipo'], minlength=3)
            self.log(f"Instante t = {a['t']:g} s: {self.n_nos:,} vértices; arestas: " +
                     ", ".join(f"{NOMES_TIPO[i]} {cont[i]:,}" for i in range(3)) + f" (total {cont.sum():,})")


# ---------------------------------------------------------------- CLI
def adicionar_argumentos(ap):
    c = Config()
    ap.add_argument('--log', default=c.log, help='traço do ns-3-leo')
    ap.add_argument('--bs', default=c.bs, help='CSV do OpenCellID')
    ap.add_argument('--radio', default='todos',
                    help="'todos' (padrão) usa todas as células como estações-base; ou filtre: NR, LTE,NR...")
    ap.add_argument('--terrestre', choices=['knn', 'raio'], default=c.terrestre)
    ap.add_argument('--k-vizinhos', type=int, default=c.k_vizinhos)
    ap.add_argument('--raio-terrestre-km', type=float, default=c.raio_terrestre_km)
    ap.add_argument('--isl', choices=['grade', 'alcance'], default=c.isl)
    ap.add_argument('--isl-max-km', type=float, default=c.isl_max_km)
    ap.add_argument('--altitude-rasante-km', type=float, default=c.altitude_rasante_km)
    ap.add_argument('--elevacao-min', type=float, default=c.elevacao_min_graus, help='graus')
    ap.add_argument('--max-sat-por-bs', type=int, default=c.max_sat_por_bs, help='0 = todos os visíveis')
    ap.add_argument('--sem-correcao-referencial', action='store_true')
    ap.add_argument('--threads', type=int, default=c.threads, help='núcleos usados (padrão: 8)')
    ap.add_argument('--max-ram-gb', type=float, default=48.0,
                    help='para antes de montar a rede se a estimativa de RAM passar disso (padrão: 48)')


def config_de_args(a):
    radio = None if a.radio.lower() in ('todos', 'all', '') else tuple(r.strip().upper() for r in a.radio.split(','))
    return Config(log=a.log, bs=a.bs, radio=radio, terrestre=a.terrestre, k_vizinhos=a.k_vizinhos,
                  raio_terrestre_km=a.raio_terrestre_km, isl=a.isl, isl_max_km=a.isl_max_km,
                  altitude_rasante_km=a.altitude_rasante_km, elevacao_min_graus=a.elevacao_min,
                  max_sat_por_bs=a.max_sat_por_bs, corrigir_referencial=not a.sem_correcao_referencial,
                  threads=a.threads)


_REDE = None  # herdada pelos workers via fork


def _exportar_instante(args):
    k, pasta = args
    a = _REDE.instante(k)
    nb = _REDE.n_bs
    m = a['tipo'] != TERRESTRE
    np.savez(os.path.join(pasta, f"instante_{k:05d}.npz"), t=a['t'], pos_sat=a['pos_sat'].astype(np.float32),
             u=a['u'][m].astype(np.uint32), v=a['v'][m].astype(np.uint32), tipo=a['tipo'][m],
             comprimento_km=a['comprimento_km'][m])
    return int(m.sum()), nb


def main():
    global _REDE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('diagnostico', help='confere unidades, referencial, camadas e tamanhos')
    e = sub.add_parser('exportar', help='grava snapshots .npz em paralelo')
    for p in (d, e):
        adicionar_argumentos(p)
    e.add_argument('--passo', type=int, default=5, help='1 a cada N instantes')
    e.add_argument('--saida', default='snapshots')
    e.add_argument('--max-gb', type=float, default=20.0, help='aborta se a estimativa de disco passar disso')
    a = ap.parse_args()

    rede = Rede(config_de_args(a))
    rede.checar_memoria(0, a.max_ram_gb)
    if a.cmd == 'diagnostico':
        rede.resumo(rede.instante(0))
        return

    os.makedirs(a.saida, exist_ok=True)
    ks = list(range(0, len(rede.tempos), a.passo))
    arestas0 = rede.instante(0)
    m0 = int((arestas0['tipo'] != TERRESTRE).sum())
    gb = len(ks) * (m0 * 13 + rede.n_sat * 12) / 1e9
    rede.resumo(arestas0)
    print(f"Estimativa de disco: {gb:.1f} GB para {len(ks)} instantes.")
    if gb > a.max_gb:
        raise SystemExit("Acima de --max-gb. Aumente --passo, use --max-sat-por-bs ou calcule os instantes "
                         "sob demanda com rede.instante(k).")
    b = rede.bs
    np.savez(os.path.join(a.saida, 'estatico.npz'), lat=b['lat'].to_numpy(), lon=b['lon'].to_numpy(),
             mcc=b['mcc'].to_numpy(), net=b['net'].to_numpy(), radio=b['radio'].to_numpy(dtype=str),
             ids_sat=rede.ids_sat, tempos=rede.tempos,
             camada=rede.camada if rede.camada is not None else np.empty(0),
             plano=rede.plano if rede.plano is not None else np.empty(0),
             terr=rede.terr.astype(np.uint32), comprimento_terr_km=(rede.comp_terr / 1000).astype(np.float32),
             config=json.dumps(asdict(rede.cfg)))
    _REDE = rede
    t0 = time.perf_counter()
    workers = max(1, min(a.threads, len(ks)))
    with mp.get_context('fork').Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_exportar_instante, [(k, a.saida) for k in ks]), 1):
            if i % max(1, len(ks) // 20) == 0 or i == len(ks):
                dt = time.perf_counter() - t0
                print(f"  {i}/{len(ks)} instantes ({dt:.0f} s, faltam ~{dt / i * (len(ks) - i):.0f} s)", flush=True)


if __name__ == '__main__':
    main()
