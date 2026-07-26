"""Erdos problem #128 ($250): does every triangle-free graph on n vertices
contain an induced subgraph on floor(n/2) vertices with at most n^2/50 edges?

Counterexample hunt over weighted blowups. A pattern graph H (triangle-free,
adjacency A) with weights w on the simplex defines a blowup whose half-subset
edge minimum, in the continuous limit, is

    f(H, w) = min { 1/2 s^T A s  :  0 <= s <= w,  sum(s) = 1/2 }.

A counterexample needs f > 1/50 = 0.02. C5 with uniform weights gives exactly
0.02 (as does Petersen), so the game is finding any (H, w) strictly above it.

Inner minimization: multi-start projected gradient (heuristic upper bound on
the true min -- good enough for search; finalists get exact face-enumeration
verification via verify.py). Outer: per-H hill climb on w.
"""

import argparse
import itertools
import json
import sys
import time

import numpy as np
import networkx as nx


# ---------- inner problem: min 1/2 s'As over the capped-simplex slice ----------

def project_capped_simplex(X, w, c):
    """Row-wise projection of X onto {s : 0 <= s <= w, sum(s) = c}."""
    lo = X.min(axis=1) - c  # tau bounds: sum clip(x - tau) is monotone in tau
    hi = X.max(axis=1)
    for _ in range(60):
        tau = 0.5 * (lo + hi)
        s = np.clip(X - tau[:, None], 0.0, w[None, :])
        too_big = s.sum(axis=1) > c
        lo = np.where(too_big, tau, lo)
        hi = np.where(too_big, hi, tau)
    return np.clip(X - (0.5 * (lo + hi))[:, None], 0.0, w[None, :])


def inner_min(A, w, c=0.5, n_starts=64, iters=300, rng=None):
    """Heuristic min of 1/2 s'As s.t. 0<=s<=w, sum s = c. Returns (val, s)."""
    rng = rng or np.random.default_rng()
    h = len(w)
    if w.sum() < c - 1e-12:
        return np.inf, None  # infeasible weight vector (shouldn't happen: sum w = 1 > c)
    # starts: random points + each "cheapest corner" greedy seed
    X = rng.random((n_starts, h)) * w[None, :]
    S = project_capped_simplex(X, w, c)
    eta = 1.0 / (np.abs(A).sum(axis=1).max() + 1e-9)  # 1/||A||_inf
    for _ in range(iters):
        S = project_capped_simplex(S - eta * (S @ A), w, c)
    vals = 0.5 * np.einsum('ij,jk,ik->i', S, A, S)
    i = np.argmin(vals)
    return vals[i], S[i]


# ---------- outer problem: per-H maximization of f over the weight simplex ----------

def score(A, w, rng, n_starts=64, iters=300):
    v, _ = inner_min(A, w, n_starts=n_starts, iters=iters, rng=rng)
    return v


def optimize_weights(A, rng, rounds=400, pop=8):
    """Hill-climb w to maximize the (heuristic) inner min. Returns (best_f, best_w)."""
    h = A.shape[0]
    # seeds: uniform + Dirichlet samples
    seeds = [np.full(h, 1.0 / h)]
    seeds += [rng.dirichlet(np.ones(h)) for _ in range(pop - 1)]
    best_w, best_f = None, -1.0
    for w in seeds:
        f = score(A, w, rng)
        if f > best_f:
            best_f, best_w = f, w
    sigma = 0.25
    for r in range(rounds):
        w = np.clip(best_w + sigma * rng.standard_normal(h) / h, 1e-6, None)
        w /= w.sum()
        f = score(A, w, rng)
        if f > best_f:
            best_f, best_w = f, w
        if r % 80 == 79:
            sigma *= 0.6  # anneal
    return best_f, best_w


# ---------- pattern graph supply ----------

def named_specials():
    g = nx.generators.small
    out = {
        'petersen': nx.petersen_graph(),
        'heawood': nx.heawood_graph(),
        'pappus': nx.pappus_graph(),
        'desargues': nx.desargues_graph(),
        'moebius_kantor': nx.moebius_kantor_graph(),
        'grotzsch': mycielski_of(nx.cycle_graph(5)),  # Grotzsch = Mycielskian of C5
        'clebsch': clebsch_graph(),
        'kneser_7_3': nx.kneser_graph(7, 3),
        'c7': nx.cycle_graph(7),
        'c9': nx.cycle_graph(9),
        'c11': nx.cycle_graph(11),
        'mycielski_c7': mycielski_of(nx.cycle_graph(7)),
    }
    try:
        out['mcgee'] = g.LCF_graph(24, [12, 7, -7], 8)
    except Exception:
        pass
    return {k: v for k, v in out.items() if v is not None and triangle_free(v)}


def mycielski_of(G):
    return nx.mycielskian(G)


def clebsch_graph():
    # vertices = F_2^4, edges between x,y iff x+y in {e_i} u {1111}
    diffs = {0b0001, 0b0010, 0b0100, 0b1000, 0b1111}
    G = nx.Graph()
    G.add_nodes_from(range(16))
    for x in range(16):
        for y in range(x + 1, 16):
            if (x ^ y) in diffs:
                G.add_edge(x, y)
    return G


def triangle_free(G):
    return all(t == 0 for t in nx.triangles(G).values())


def atlas_triangle_free(max_n=7):
    """All connected triangle-free graphs on 3..max_n vertices (atlas <= 7)."""
    seen = []
    for G in nx.graph_atlas_g()[2:]:
        n = G.number_of_nodes()
        if n < 3 or n > max_n:
            continue
        if G.number_of_edges() < n - 1 or not nx.is_connected(G):
            continue
        if triangle_free(G):
            seen.append(G)
    return seen


# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=400)
    ap.add_argument('--specials-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results.jsonl')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    done = set()
    try:
        with open(args.out, encoding='utf-8') as fh:
            done = {json.loads(line)['graph'] for line in fh if line.strip()}
    except FileNotFoundError:
        pass

    jobs = []
    for name, G in named_specials().items():
        jobs.append((name, G))
    if not args.specials_only:
        for i, G in enumerate(atlas_triangle_free()):
            jobs.append((f'atlas_{i}_n{G.number_of_nodes()}e{G.number_of_edges()}', G))

    target = 1.0 / 50.0
    best_overall = (-1.0, None, None)
    t0 = time.time()
    with open(args.out, 'a', encoding='utf-8') as fh:
        for name, G in jobs:
            if name in done:
                continue
            if nx.is_bipartite(G):
                # one side of any bipartition has weight >= 1/2, so a
                # half-measure subset fits inside an independent set: f = 0
                fh.write(json.dumps({'graph': name, 'n': G.number_of_nodes(),
                                     'f': 0.0, 'bipartite': True}) + '\n')
                print(f'{name:28s} n={G.number_of_nodes():3d}  f=0 (bipartite, skipped)',
                      flush=True)
                continue
            A = nx.to_numpy_array(G)
            f, w = optimize_weights(A, rng, rounds=args.rounds)
            rec = {'graph': name, 'n': G.number_of_nodes(), 'f': float(f),
                   'gap_vs_1_50': float(f - target), 'w': [round(float(x), 6) for x in w]}
            fh.write(json.dumps(rec) + '\n')
            fh.flush()
            flag = '  <<< ABOVE 1/50!' if f > target + 1e-9 else ''
            print(f'{name:28s} n={G.number_of_nodes():3d}  f={f:.6f}  (1/50={target:.6f}){flag}',
                  flush=True)
            if f > best_overall[0]:
                best_overall = (f, name, w)
    f, name, w = best_overall
    print(f'\nBEST: {name}  f={f:.6f}  vs target 0.020000   [{time.time()-t0:.0f}s]')


if __name__ == '__main__':
    main()
