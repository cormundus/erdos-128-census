"""Erdos #128, direct search: SA over raw triangle-free graphs on n vertices
(n even, n <= 22), EXACT objective = min over all C(n, n/2) half-subsets of
the edges they span. Counterexample needs min > n^2/50.

Incremental trick: E[S] (edges inside each half-subset S) changes by +-1
exactly on subsets containing both endpoints of a toggled edge, so one move
costs one boolean mask AND + add, not a full recompute.

Also scores fixed graphs: Petersen (calibration, min must be 2 at n=10),
all triangle-free circulants for each n, and C5-balanced-blowups as seeds.
"""

import argparse
import itertools
import json
import time

import numpy as np
import networkx as nx


def half_subset_matrix(n):
    combos = list(itertools.combinations(range(n), n // 2))
    M = np.zeros((len(combos), n), dtype=bool)
    for r, c in enumerate(combos):
        M[r, list(c)] = True
    return M


def subset_edge_counts(A, M):
    return np.einsum('ij,ij->i', M @ A, M).astype(np.int32) // 2


def is_triangle_free_after_add(adj_sets, u, v):
    return not (adj_sets[u] & adj_sets[v])


def c5_blowup(n):
    """Balanced C5 blowup on n vertices (n divisible by 5 preferred)."""
    parts = np.array_split(np.arange(n), 5)
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(5):
        for u in parts[i]:
            for v in parts[(i + 1) % 5]:
                A[u, v] = A[v, u] = 1
    return A


def random_triangle_free(n, rng, tries=400):
    A = np.zeros((n, n), dtype=np.int8)
    adj = [set() for _ in range(n)]
    for _ in range(tries):
        u, v = rng.integers(0, n, 2)
        if u != v and not A[u, v] and not (adj[u] & adj[v]):
            A[u, v] = A[v, u] = 1
            adj[u].add(v); adj[v].add(u)
    return A


def sa_run(n, M, rng, steps=20000, seed_graph=None, t0=1.5, t1=0.01):
    A = seed_graph.copy() if seed_graph is not None else random_triangle_free(n, rng)
    adj = [set(np.flatnonzero(A[i])) for i in range(n)]
    E = subset_edge_counts(A, M)
    cur = E.min()
    best, bestA = cur, A.copy()
    for step in range(steps):
        T = t0 * (t1 / t0) ** (step / steps)
        u, v = rng.integers(0, n, 2)
        if u == v:
            continue
        if A[u, v]:  # removal always stays triangle-free
            delta = -1
        else:
            if adj[u] & adj[v]:
                continue  # would create a triangle
            delta = 1
        mask = M[:, u] & M[:, v]
        E2min = (E + delta * mask).min()
        # objective: maximize min; accept by SA rule on integer diff
        d = E2min - cur
        if d >= 0 or rng.random() < np.exp(d / max(T, 1e-9)):
            E += delta * mask
            cur = E2min
            if delta == 1:
                A[u, v] = A[v, u] = 1
                adj[u].add(v); adj[v].add(u)
            else:
                A[u, v] = A[v, u] = 0
                adj[u].discard(v); adj[v].discard(u)
            if cur > best:
                best, bestA = cur, A.copy()
    return best, bestA


def circulant_sweep(n, M):
    """All triangle-free circulants on n vertices; returns best (min, diffs)."""
    best = (-1, None)
    half = n // 2
    diffs_all = list(range(1, half + 1))
    for r in range(1, half + 1):
        for combo in itertools.combinations(diffs_all, r):
            s = set()
            for d in combo:
                s.add(d); s.add(n - d)
            # triangle check: a+b=c (mod n) within connection set
            conn = sorted(s)
            if any((a + b) % n in s for a in conn for b in conn):
                continue
            A = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for d in conn:
                    A[i, (i + d) % n] = 1
            m = int(subset_edge_counts(A, M).min())
            if m > best[0]:
                best = (m, combo)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--restarts', type=int, default=12)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-circulants', action='store_true')
    args = ap.parse_args()
    n = args.n
    # n=26 fits desktop RAM: C(26,13) ~ 10.4M subsets, M ~ 270 MB bool.
    # int8 einsum accumulator is safe: 2*E[S] <= 2*floor((n/2)^2/4) = 84 < 127
    # at n=26 by Mantel on the half-subset.
    assert n % 2 == 0 and n <= 26
    thr = n * n / 50.0
    rng = np.random.default_rng(args.seed)
    print(f'n={n}: need min half-subset edges > {thr:.2f} (i.e. >= {int(thr) + 1})', flush=True)

    t0 = time.time()
    M = half_subset_matrix(n)
    print(f'{M.shape[0]} half-subsets enumerated [{time.time()-t0:.1f}s]', flush=True)

    # calibration: Petersen at n=10
    if n == 10:
        Ap = nx.to_numpy_array(nx.petersen_graph()).astype(np.int8)
        print('Petersen min =', int(subset_edge_counts(Ap, M).min()), '(expect 2)', flush=True)

    if not args.skip_circulants and n <= 20:
        m, diffs = circulant_sweep(n, M)
        print(f'best circulant: min={m} diffs={diffs}  ({m/n**2:.5f} vs 0.02)', flush=True)

    results = []
    seeds = [c5_blowup(n), None] + [None] * (args.restarts - 2)
    for i, sg in enumerate(seeds[:args.restarts]):
        b, bA = sa_run(n, M, rng, steps=args.steps, seed_graph=sg)
        tag = 'c5seed' if i == 0 else 'random'
        print(f'restart {i:2d} ({tag}): best min={b}  ({b/n**2:.5f} vs 0.02)'
              + ('   <<< ABOVE n^2/50!' if b > thr else ''), flush=True)
        results.append(int(b))
        if b > thr:
            np.save(f'counterexample_n{n}_min{b}.npy', bA)
            print('SAVED candidate adjacency matrix!', flush=True)
    print(f'n={n} done: best={max(results)} need>{thr:.1f}  [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
