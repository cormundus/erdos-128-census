# A computational census for Erdős problem #128

**Problem** ([erdosproblems.com/128](https://www.erdosproblems.com/128), $250):
if every induced subgraph on ≥ ⌊n/2⌋ vertices of an n-vertex graph G spans
more than n²/50 edges, must G contain a triangle? Equivalently: does every
triangle-free graph contain a half-sized vertex subset spanning at most n²/50
edges? The constant 1/50 is attained by balanced blowups of C₅ and of the
Petersen graph.

This repository documents a systematic computational search for a
counterexample (none found), including some intermediate values that appear
not to be recorded in the literature.

## Setup

For a triangle-free pattern graph H on h vertices with vertex weights w on
the simplex, the balanced-blowup local density at one half is

    f(H) = min { ½ sᵀA_H s : 0 ≤ s ≤ w, Σs = ½ }.

A counterexample to the conjecture requires f(H) > 1/50. For d-regular H with
uniform weights, writing s = (1/2h)·𝟙 + x gives the elementary spectral
bounds

    (d + λ_min) / 8h  ≤  f(H)  ≤  d / 8h,

and the lower bound is attained **iff** the λ_min-eigenspace contains a
balanced ±1 vector — i.e. iff H has a bisection in which every vertex has
exactly (d + λ_min)/2 neighbours on its own side.

## Census of the known triangle-free strongly regular graphs

Only seven triangle-free SRGs are known. Constructions are verified
programmatically in `srg.py` (the S(3,6,22) Steiner system underlying the
Gewirtz / M22 / Higman–Sims constructions is checked to cover all 1540
triples exactly once; all SRG parameters are verified from the adjacency
matrix). Values of f at one half:

| H | srg | f(H) | status |
|---|-----|------|--------|
| C₅ | (5,2,0,1) | 1/50 = 0.0200 | exact (known extremal) |
| Petersen | (10,3,0,1) | 1/50 = 0.0200 | exact (known extremal) |
| Clebsch | (16,5,0,2) | 1/64 = 0.015625 | exact, certificate |
| Hoffman–Singleton | (50,7,0,1) | 1/100 = 0.0100 | exact, certificate |
| Gewirtz | (56,10,0,2) | 3/224 ≈ 0.013393 | exact, certificate |
| M22 (Mesner) | (77,16,0,4) | (0.016234, 0.017710] | floor unattainable (odd order) |
| Higman–Sims | (100,22,0,6) | 7/400 = 0.0175 | exact, certificate |

Every "certificate" is an explicit half-subset inducing an r-regular subgraph
(r = 1, 2, 3, 7 for Clebsch, Hoffman–Singleton, Gewirtz, Higman–Sims), stored
in `bisection_certificates.json` — anyone can recheck by counting edges. In
particular:

* the Gewirtz graph has a bisection into two 3-regular 28-vertex halves;
* Hoffman–Singleton has a bisection into two 2-regular 25-vertex halves;
* for Higman–Sims the certificate recovers the classical split into two
  Hoffman–Singleton subgraphs (the ±1 indicator is a tight λ_min = −8
  eigenvector), giving f = 175/100² exactly.

For M22, 77 is odd, so no balanced ±1 eigenvector exists and f sits strictly
above its spectral floor 10/616; the stated upper value is the best feasible
point found by the optimizer (a certified upper bound on the true minimum,
since any feasible point is one).

None of the seven reaches 1/50 except C₅ and the Petersen graph.

## Weighted-blowup census

`search.py` runs a max-min search (inner: multistart projected gradient on
the capped simplex; outer: weight hill-climb), calibrated to reproduce
f(C₅) = f(Petersen) = 1/50 exactly, over all 88 connected triangle-free
graphs on ≤ 7 vertices plus named sporadics (Kneser(7,3), McGee, cages,
Mycielskians, …). Results in `results.jsonl`. Nothing exceeds 1/50.

## Direct search

`direct_sa.py` runs simulated annealing over raw triangle-free graphs with
the **exact** objective (minimum over all C(n, n/2) half-subsets, maintained
incrementally), for even n ≤ 26. Best minima found vs the counterexample
threshold: n=22: 8 (needs ≥ 10), n=24: 10 (needs ≥ 12), n=26: 10 (needs ≥ 14).
The value 1/50 is attained only when 10 | n, via C₅-blowups (e.g. the
circulant C₂₀(1,4,6,9) ties at n = 20).

## Reproducing

```
pip install numpy scipy networkx
python srg.py                 # build + verify SRGs, run blowup search
python exact_bisections.py    # find + verify the bisection certificates
python direct_sa.py --n 22    # direct SA at a given n (even, <= 26)
python verify.py              # exact face-enumeration verifier (small h)
```

## Attribution

Computations, code, and the spectral-bound derivations by **Claude**
(Anthropic Claude Fable 5), directed by **Cormundus** as part
of an ongoing human+AI collaboration on computational attacks on open
problems. The numbers come from the runs; the certificates are checkable by
hand.

## License

MIT
