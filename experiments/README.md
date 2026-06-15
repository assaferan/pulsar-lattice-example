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
by parameter conditioning, not by N, this supports a **progressive-baseline**
algorithm: O(1) bootstrap solve + O(N) cheap phase-connections = `poly(N)`
("lever 1"). Building that solver is the next step.
