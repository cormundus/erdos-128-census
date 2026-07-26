"""Erdos #128 phase 2: the triangle-free SRG bestiary.

Known triangle-free strongly regular graphs (complete list as of 2026 — only
seven exist, per Biggs / Brouwer-van Lint-Wilson):

    C5           srg(5, 2, 0, 1)    [census: 0.0200]
    Petersen     srg(10, 3, 0, 1)   [census: 0.0200]
    Clebsch      srg(16, 5, 0, 2)   [census: 0.0156]
    Hoffman-Singleton srg(50, 7, 0, 1)
    Gewirtz      srg(56, 10, 0, 2)
    M22 (Mesner) srg(77, 16, 0, 4)
    Higman-Sims  srg(100, 22, 0, 6)

This module builds the last four and runs the calibrated weighted-blowup
search from search.py on each. Constructions:

  * Hoffman-Singleton: Robertson's pentagons/pentagrams construction.
  * S(3,6,22) Steiner system: points = PG(2,4) + {infinity}; blocks =
    21 (line + infinity) + one PSL(3,4)-orbit of 56 hyperovals (same-orbit
    hyperovals meet in an even number of points).
  * M22 graph: 77 blocks of S(3,6,22), adjacent iff disjoint.
  * Gewirtz: the 56 blocks avoiding a fixed point, adjacent iff disjoint.
  * Higman-Sims: {*} + 22 points + 77 blocks; * ~ every point;
    point ~ block iff point in block; block ~ block iff disjoint.

Every object is verified programmatically (design property, SRG parameters,
triangle-freeness) before any search touches it; a construction bug would
otherwise silently invalidate the census.
"""

import argparse
import itertools
import json
import time

import numpy as np
import networkx as nx

from search import optimize_weights, inner_min, triangle_free


# ---------- PG(2,4) ----------

def gf4():
    """GF(4) as ints 0..3 with add/mul tables. 2 = x, 3 = x+1, x^2 = x+1."""
    add = [[a ^ b for b in range(4)] for a in range(4)]
    mul = [[0] * 4 for _ in range(4)]
    for a in range(1, 4):
        for b in range(1, 4):
            # log table for GF(4)*: 1->0, 2->1, 3->2 (generator 2)
            log = {1: 0, 2: 1, 3: 2}
            exp = {0: 1, 1: 2, 2: 3}
            mul[a][b] = exp[(log[a] + log[b]) % 3]
    return add, mul


ADD, MUL = gf4()


def pg24():
    """Points and lines of PG(2,4). Returns (points, lines) — points as
    normalized homogeneous triples, lines as frozensets of point indices."""
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append((1, x, y))
    for y in range(4):
        pts.append((0, 1, y))
    pts.append((0, 0, 1))
    assert len(pts) == 21
    idx = {p: i for i, p in enumerate(pts)}

    def dot(u, v):
        s = 0
        for a, b in zip(u, v):
            s = ADD[s][MUL[a][b]]
        return s

    lines = set()
    for line in pts:  # duality: lines are also normalized triples
        on = frozenset(idx[p] for p in pts if dot(line, p) == 0)
        assert len(on) == 5
        lines.add(on)
    lines = sorted(lines, key=sorted)
    assert len(lines) == 21
    return pts, lines


def hyperovals(pts, lines):
    """All 168 hyperovals (6-point sets meeting every line in 0 or 2 points)."""
    line_sets = [set(l) for l in lines]
    # A hyperoval = conic + nucleus in PG(2,4); enumerate greedily: extend
    # 4-point arcs (no 3 collinear). 21 points is small enough to brute-force
    # over 6-subsets containing a fixed structure; do smart DFS instead.
    collinear = {}
    for l in line_sets:
        for trip in itertools.combinations(sorted(l), 3):
            collinear[trip] = True

    def is_arc(s):
        return not any(collinear.get(t, False)
                       for t in itertools.combinations(sorted(s), 3))

    ovals = set()

    def extend(arc, start):
        if len(arc) == 6:
            ovals.add(frozenset(arc))
            return
        for p in range(start, 21):
            if all(not collinear.get(tuple(sorted((a, b, p))), False)
                   for a, b in itertools.combinations(arc, 2)):
                extend(arc + [p], p + 1)

    extend([], 0)
    ovals = [o for o in ovals if is_arc(o)]
    assert len(ovals) == 168, f'expected 168 hyperovals, got {len(ovals)}'
    return ovals


def hyperoval_orbit(ovals):
    """Split hyperovals into the 3 orbits of 56: same orbit <=> even meet.
    Returns one orbit."""
    o0 = ovals[0]
    orbit = [o for o in ovals if len(o0 & o) % 2 == 0]
    assert len(orbit) == 56, f'orbit size {len(orbit)} != 56'
    # sanity: even-intersection must be an equivalence on this orbit
    for a, b in itertools.combinations(orbit[:20], 2):
        assert len(a & b) % 2 == 0
    return orbit


def steiner_3_6_22():
    """S(3,6,22): 22 points (21 PG(2,4) points + infinity=21), 77 blocks."""
    pts, lines = pg24()
    orbit = hyperoval_orbit(hyperovals(pts, lines))
    INF = 21
    blocks = [frozenset(l | {INF}) for l in lines] + list(orbit)
    assert len(blocks) == 77
    # verify the design property: every 3-subset of 22 points in EXACTLY one block
    cover = {}
    for b in blocks:
        assert len(b) == 6
        for trip in itertools.combinations(sorted(b), 3):
            assert trip not in cover, f'triple {trip} covered twice'
            cover[trip] = b
    from math import comb
    assert len(cover) == comb(22, 3), \
        f'covered {len(cover)} triples, expected {comb(22, 3)}'
    return blocks


# ---------- the four big triangle-free SRGs ----------

def hoffman_singleton():
    """Robertson construction: pentagons P0..P4, pentagrams Q0..Q4.
    P_h vertex j ~ Q_i vertex (h*i + j) mod 5."""
    G = nx.Graph()
    P = lambda h, j: ('P', h, j)
    Q = lambda i, j: ('Q', i, j)
    for h in range(5):
        for j in range(5):
            G.add_edge(P(h, j), P(h, (j + 1) % 5))          # pentagon
            G.add_edge(Q(h, j), Q(h, (j + 2) % 5))          # pentagram
    for h in range(5):
        for i in range(5):
            for j in range(5):
                G.add_edge(P(h, j), Q(i, (h * i + j) % 5))
    return nx.convert_node_labels_to_integers(G)


def m22_graph(blocks):
    G = nx.Graph()
    G.add_nodes_from(range(77))
    for i, j in itertools.combinations(range(77), 2):
        if not (blocks[i] & blocks[j]):
            G.add_edge(i, j)
    return G


def gewirtz_graph(blocks, point=21):
    sub = [b for b in blocks if point not in b]
    assert len(sub) == 56
    G = nx.Graph()
    G.add_nodes_from(range(56))
    for i, j in itertools.combinations(range(56), 2):
        if not (sub[i] & sub[j]):
            G.add_edge(i, j)
    return G


def higman_sims(blocks):
    """Vertices: 0 = *, 1..22 = points 0..21, 23..99 = blocks."""
    G = nx.Graph()
    G.add_nodes_from(range(100))
    for p in range(22):
        G.add_edge(0, 1 + p)
    for bi, b in enumerate(blocks):
        for p in b:
            G.add_edge(1 + p, 23 + bi)
    for i, j in itertools.combinations(range(77), 2):
        if not (blocks[i] & blocks[j]):
            G.add_edge(23 + i, 23 + j)
    return G


def check_srg(G, v, k, lam, mu, name):
    """Assert G is srg(v,k,lambda,mu)."""
    assert G.number_of_nodes() == v, f'{name}: n={G.number_of_nodes()} != {v}'
    degs = {d for _, d in G.degree()}
    assert degs == {k}, f'{name}: degrees {degs} != {k}'
    A = nx.to_numpy_array(G, dtype=np.int64)
    A2 = A @ A
    for i in range(v):
        for j in range(i + 1, v):
            common = A2[i, j]
            want = lam if A[i, j] else mu
            assert common == want, \
                f'{name}: common({i},{j})={common}, adj={A[i,j]}, want {want}'
    assert triangle_free(G) == (lam == 0), f'{name}: triangle-freeness mismatch'
    print(f'  {name}: srg({v},{k},{lam},{mu}) VERIFIED, triangle-free={lam == 0}')
    return A.astype(float)


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results.jsonl')
    ap.add_argument('--uniform-starts', type=int, default=512,
                    help='extra inner starts for the uniform-weight probe')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print('Building S(3,6,22)...')
    t0 = time.time()
    blocks = steiner_3_6_22()
    print(f'  S(3,6,22) VERIFIED: 77 blocks, every triple once  [{time.time()-t0:.1f}s]')

    print('Building + verifying the four large triangle-free SRGs...')
    zoo = [
        ('hoffman_singleton', hoffman_singleton(), (50, 7, 0, 1)),
        ('gewirtz', gewirtz_graph(blocks), (56, 10, 0, 2)),
        ('m22_mesner', m22_graph(blocks), (77, 16, 0, 4)),
        ('higman_sims', higman_sims(blocks), (100, 22, 0, 6)),
    ]
    mats = {}
    for name, G, (v, k, lam, mu) in zoo:
        mats[name] = check_srg(G, v, k, lam, mu, name)

    done = set()
    try:
        with open(args.out, encoding='utf-8') as fh:
            done = {json.loads(line)['graph'] for line in fh if line.strip()}
    except FileNotFoundError:
        pass

    target = 1.0 / 50.0
    with open(args.out, 'a', encoding='utf-8') as fh:
        for name, G, params in zoo:
            if name in done:
                print(f'{name}: already in {args.out}, skipping')
                continue
            A = mats[name]
            h = A.shape[0]
            t0 = time.time()
            # probe 1: uniform weights (vertex-transitive; census optima were uniform)
            w_uni = np.full(h, 1.0 / h)
            f_uni, _ = inner_min(A, w_uni, n_starts=args.uniform_starts, rng=rng)
            # probe 2: full weight hill-climb
            f_opt, w_opt = optimize_weights(A, rng, rounds=args.rounds)
            f_best = max(f_uni, f_opt)
            rec = {'graph': name, 'n': h, 'f': float(f_best),
                   'f_uniform': float(f_uni), 'f_hillclimb': float(f_opt),
                   'gap_vs_1_50': float(f_best - target), 'srg': params,
                   'w': [round(float(x), 6) for x in
                         (w_opt if f_opt >= f_uni else w_uni)]}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            flag = '  <<< ABOVE 1/50!' if f_best > target + 1e-9 else ''
            print(f'{name:20s} n={h:3d}  f_uni={f_uni:.6f}  f_climb={f_opt:.6f}  '
                  f'best={f_best:.6f} vs 0.02{flag}  [{time.time()-t0:.0f}s]',
                  flush=True)


if __name__ == '__main__':
    main()
