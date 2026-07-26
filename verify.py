"""Exact verification for #128 blowup candidates (h <= ~12).

The inner problem  min 1/2 s'As  s.t. 0 <= s <= w, sum(s) = c  attains its
global minimum at a point that is stationary within the relative interior of
some face of the polytope. Faces = assignments of each coordinate to
{LO (s_i=0), UP (s_i=w_i), FREE}. For each of the 3^h patterns we solve the
bordered KKT system on the FREE block and keep every feasible stationary
point; the global min is the least objective among them. Float arithmetic
with tight tolerances -- a genuine counterexample claim would be re-done in
exact rationals, this is the gate before that effort.
"""

import itertools
import numpy as np


def exact_inner_min(A, w, c=0.5, tol=1e-10):
    h = len(w)
    best_val, best_s = np.inf, None
    idx = np.arange(h)
    for pattern in itertools.product((0, 1, 2), repeat=h):  # 0=LO 1=UP 2=FREE
        pat = np.array(pattern)
        up = idx[pat == 1]
        fr = idx[pat == 2]
        base = w[up].sum()
        rem = c - base
        if rem < -tol or (len(fr) == 0 and abs(rem) > tol):
            continue
        s = np.zeros(h)
        s[up] = w[up]
        if len(fr) > 0:
            if rem > w[fr].sum() + tol:
                continue
            # stationarity on the face: A_ff s_f + A_fu w_u = lambda * 1
            k = len(fr)
            M = np.zeros((k + 1, k + 1))
            M[:k, :k] = A[np.ix_(fr, fr)]
            M[:k, k] = -1.0
            M[k, :k] = 1.0
            rhs = np.zeros(k + 1)
            rhs[:k] = -A[np.ix_(fr, up)] @ w[up] if len(up) else 0.0
            rhs[k] = rem
            try:
                sol, res, rank, _ = np.linalg.lstsq(M, rhs, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if np.linalg.norm(M @ sol - rhs) > 1e-8:
                continue  # no stationary point interior to this face
            sf = sol[:k]
            if (sf < -tol).any() or (sf > w[fr] + tol).any():
                continue  # stationary point lies outside the face
            s[fr] = np.clip(sf, 0.0, w[fr])
        val = 0.5 * s @ A @ s
        if val < best_val:
            best_val, best_s = val, s.copy()
    return best_val, best_s


if __name__ == '__main__':
    import networkx as nx
    A5 = nx.to_numpy_array(nx.cycle_graph(5))
    v, s = exact_inner_min(A5, np.full(5, 0.2))
    print('C5 uniform exact min:', v, ' (expect 0.02)  s =', np.round(s, 4))
    Ap = nx.to_numpy_array(nx.petersen_graph())
    v, s = exact_inner_min(Ap, np.full(10, 0.1))
    print('Petersen uniform exact min:', round(v, 10), ' (expect 0.02)')
