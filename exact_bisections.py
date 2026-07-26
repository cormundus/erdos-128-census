"""Erdos #128: integer certificates for the SRG spectral floors.

For a d-regular triangle-free H on 2m vertices, an m-subset S with every
vertex of H having exactly r neighbors on its own side (r inside S for S's
vertices, r inside the complement for the rest) makes the +-1 indicator an
adjacency eigenvector with eigenvalue 2r - d = lambda_min, and the blowup
value f(H) = e(S)/n^2 = (m*r/2)/n^2 hits the eigenvalue lower bound
(d + lambda_min)/(8n) exactly.  Such an S is an *integer certificate*:
anyone can recheck it by counting edges.

Targets:
  hoffman_singleton srg(50,7,0,1),  lambda_min=-3 -> r=2, e(S)=25,  f=0.0100
  gewirtz           srg(56,10,0,2), lambda_min=-4 -> r=3, e(S)=42,  f=0.013393
  higman_sims       srg(100,22,0,6),lambda_min=-8 -> r=7, e(S)=175, f=0.0175
                    (known: the split into two Hoffman-Singleton subgraphs)

Search: annealing on the symmetric difference — swap one vertex in / one out,
cost = sum over vertices of (deg_own_side - r)^2, target cost 0.
"""

import json

import numpy as np
import networkx as nx

from srg import (steiner_3_6_22, hoffman_singleton, gewirtz_graph,
                 m22_graph, higman_sims, check_srg)


def find_regular_bisection(A, r, rng, tries=40, steps=200000):
    """Find S (half the vertices) with every vertex having exactly r
    neighbors on its own side. Returns sorted vertex list or None."""
    n = A.shape[0]
    m = n // 2
    Ai = A.astype(np.int32)
    for t in range(tries):
        side = np.zeros(n, dtype=bool)
        side[rng.choice(n, m, replace=False)] = True
        # deg_same[v] = neighbors of v on v's side
        deg_same = np.where(side, Ai @ side, Ai @ (~side))
        cost = int(((deg_same - r) ** 2).sum())
        T = 3.0
        for step in range(steps):
            if cost == 0:
                return sorted(np.flatnonzero(side).tolist())
            T = max(0.02, 3.0 * (1 - step / steps))
            u = int(rng.choice(np.flatnonzero(side)))
            v = int(rng.choice(np.flatnonzero(~side)))
            new_side = side.copy()
            new_side[u], new_side[v] = False, True
            nd = np.where(new_side, Ai @ new_side, Ai @ (~new_side))
            nc = int(((nd - r) ** 2).sum())
            if nc <= cost or rng.random() < np.exp((cost - nc) / T):
                side, deg_same, cost = new_side, nd, nc
        if cost == 0:
            return sorted(np.flatnonzero(side).tolist())
    return None


def verify_certificate(A, S, r):
    """Independently recheck: S is half, every vertex has r same-side nbrs,
    and e(S) = |S|*r/2. Returns (ok, e_inside)."""
    n = A.shape[0]
    S = np.array(sorted(S))
    side = np.zeros(n, dtype=bool)
    side[S] = True
    if side.sum() * 2 != n:
        return False, -1
    deg_same = np.where(side, A @ side, A @ (~side))
    if not np.all(deg_same == r):
        return False, -1
    e_inside = int(A[np.ix_(S, S)].sum()) // 2
    return e_inside * 2 == len(S) * r, e_inside


def main():
    rng = np.random.default_rng(1)
    blocks = steiner_3_6_22()
    from search import clebsch_graph
    jobs = [
        ('clebsch', clebsch_graph(), (16, 5, 0, 2), 1),
        ('hoffman_singleton', hoffman_singleton(), (50, 7, 0, 1), 2),
        ('gewirtz', gewirtz_graph(blocks), (56, 10, 0, 2), 3),
        ('higman_sims', higman_sims(blocks), (100, 22, 0, 6), 7),
    ]
    out = {}
    for name, G, (v, k, lam, mu), r in jobs:
        A = check_srg(G, v, k, lam, mu, name)
        S = find_regular_bisection(A, r, rng)
        if S is None:
            print(f'{name}: NO {r}-regular bisection found (may not exist)')
            continue
        ok, e = verify_certificate(A, S, r)
        f_exact = e / v ** 2
        floor = (k + (2 * r - k)) / (8 * v)  # = (k + lambda_min)/(8n)
        print(f'{name}: bisection FOUND & VERIFIED  e(S)={e}  '
              f'f = {e}/{v}^2 = {f_exact:.6f}  floor={floor:.6f}  match={ok}')
        out[name] = {'r': r, 'e_inside': e, 'f_exact': f_exact,
                     'n': v, 'S': S}
    with open('bisection_certificates.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('certificates written to bisection_certificates.json')


if __name__ == '__main__':
    main()
