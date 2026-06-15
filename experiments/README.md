# Experiments

Research scripts exploring **asymptotic (in N = number of TOAs) improvements**
to the lattice search. Run from the repo root with the g6k environment active:

```
source ~/g6k/activate
PYTHONPATH=. python3 experiments/density_sweep.py
PYTHONPATH=. python3 experiments/lever1.py
```

Both reconstruct lattices from `span_vecs`/`coeff_std` (the constructor is not in
the repo; it was reverse-engineered — see `density_sweep.build_il`).

## `modq_reduction.py` — does exploiting Z^n (reducing mod the integer lattice) help?

The lattice contains `q·Z^N` (q = `mul_factor`), so any vector can be shortened
by reducing its first N coordinates mod q and stays in the lattice. Two checks:
post-processing the full sieve database, and a modq-vs-raw reinsertion control.

**Finding:** it does **not** help. Reducing the final database mod q shortens
~10% of vectors but reveals no new detection (the sieve already returns the
solution in reduced form), and reinserting *reduced* vs *raw* short vectors
behaves identically — because the LLL size reduction already performs the mod-q
reduction. So "reduce mod Z^n" is redundant with LLL. (Note: uses exact integer
arithmetic — `q ≈ 1e16` exceeds float64's exact range.)

## `progressive_solver.py` — the lever-1 prototype (NEGATIVE RESULT)

A progressive-baseline solver: sieve a small `(k0+p)`-dim sub-lattice on the
first `k0` time-sorted TOAs (bootstrap), then phase-connect the rest — predict
each next wrap `k_i = round(-(b·A[:,i])/q)` and least-squares-refine `b`.

**It does not recover the pulsar.** Two obstructions:

1. The trivial **constant-phase** vector (b in the φ direction → all wraps equal,
   `std(wraps)=0`) folds *perfectly* (residual ~1e-13) and hits the Q coherence
   ceiling (~2110). So selecting the "tightest global fold" returns this
   non-pulsar — a physical mask (`std(wraps) > threshold`) is required to exclude
   it. (An earlier version of this script mistook this Q=2110 for a detection.)
2. With the physical mask applied, **no** bootstrap candidate (hundreds, k0=30–45)
   connects to the real pulsar — the best physical connection has Q~1. The pulsar
   is recovered *only* when the connection is seeded with the exact true `b*`
   (then Q~400). The connection geometry works, but a short-baseline bootstrap
   cannot **acquire** the solution: a short baseline is consistent with many
   physical timing solutions (the lever-1 test's huge `g_far`), and connecting a
   wrong one diverges.

This is consistent with `density_sweep.py`: identifying the solution genuinely
needs ~N-dimensional information, so a bounded bootstrap cannot escape the
`2^Theta(N)` cost. The lever-1 leverage test was necessary (forward connection
from the truth works) but not sufficient (it said nothing about acquisition).

## `density_sweep.py` — how does the required sieving dimension scale with N?

Subsamples the 70 lattice TOAs, rebuilds the lattice, and finds the minimum
sieving dimension `d_sieve` that recovers the tight pulsar solution.

**Finding:** `d_sieve ≈ N + 1` (linear, slope ≈ 0.99); the "dimensions for free"
is a constant ≈ p−1 ≈ 6, independent of N. So sieving cost is `2^Θ(N)` —
exponential in the number of TOAs. The q-ary / slaved structure does **not**
collapse the effective dimension, so reducing to a p-dimensional problem
("lever 2") is not viable as-is.

## `lever1.py` — can a short baseline phase-connect forward?

Computes the forward phase-prediction leverage `g_j(k) = sqrt(s_j^T (S_k^T S_k)^{-1} s_j)`
from the timing design matrix (scale-invariant).

**Finding:** predicting an *arbitrary* remaining TOA (`g_far`) is hopeless from a
short baseline (~8e3 at k=8), but predicting the *next* time-sorted TOA
(`g_next`) drops below the ½-cycle limit by a bootstrap of **k0 ≈ 10** TOAs and
stays bounded thereafter (measured `σ_phase ≈ 0.08` cycle). Because `k0` is set
by parameter conditioning, not by N, this *suggested* a **progressive-baseline**
algorithm: O(1) bootstrap solve + O(N) cheap phase-connections = `poly(N)`
("lever 1"). **However, building that solver disproved it** — see
`progressive_solver.py`. This leverage test measures forward connection *from the
true solution*; it says nothing about whether a short-baseline bootstrap can
*acquire* the true solution in the first place. It cannot.
