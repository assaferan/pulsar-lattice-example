# Asymptotic-scaling investigation — findings

Goal: find an **asymptotic** (in N = number of TOAs) improvement to the lattice
search — keep cost `poly(N)` rather than `2^Theta(N)`. Summary of what we tried
and learned. Scripts referenced live in this directory; see `README.md`.

## Setup facts

- The lattice is q-ary block lower-triangular: `[[q·I_N, 0], [A, L]]`, with N
  phase-wrap dims (one per TOA), p=7 timing-parameter dims, `q = mul_factor`.
  Cost is dominated by sieving, `~2^(c·d_sieve)` in the sieving dimension.
- The constructor isn't in the repo; it was reverse-engineered (see
  `density_sweep.build_il`): `lattice = [[I_N,0],[s·Qortho, diag(c/coeff_std)]]`,
  `s=8.105e-4`, `c=6.570e-5`, `Qortho` = orthonormalized `span_vecs`.
- Phase model (validated exactly): parameters `b` fold TOA i to
  `k_i = round(-(b·A[:,i])/q)`. Use exact ints — `q·k` overflows int64.

## What worked (constant factor, not asymptotic)

- **`fold(fast=True)`** — LLL + a shallow pump instead of BKZ + deep pump.
  ~100x faster sieving, same detection. Shipped in `fermi_fold.py`. This is a
  constant-factor win; `d_sieve` (hence the exponent) is unchanged.

## What didn't help

- **Reduce mod Z^n** (`modq_reduction.py`): redundant with LLL — no gain.
- **Lever 2 — reduce to the p-dim problem** (`density_sweep.py`): measured
  `d_sieve(N) ≈ N + 1` (linear, slope 0.99); dimensions-for-free constant ≈ p−1,
  independent of N. So sieving cost stays `2^Theta(N)`; the slaved structure does
  not collapse the effective dimension. **Dead.**
- **Lever 1 — progressive baseline** (`progressive_solver.py`): the leverage test
  (`lever1.py`) showed forward phase-connection *from the true solution* is
  feasible after a small bootstrap (`g_next` bounded). But building the solver
  disproved the algorithm: (1) the trivial constant-phase vector folds perfectly
  and wins any L2 selection — need a physical mask; (2) with that mask, no
  short-baseline bootstrap candidate connects to the pulsar (best Q~1); the
  pulsar is recovered only when seeded with the exact true `b*`. **Dead.**

## The core obstruction

Identifying the solution genuinely needs ~N-dimensional information. A short
baseline is consistent with many physical timing solutions (the lever-1 test's
huge `g_far`), so a bounded bootstrap cannot **acquire** the solution, and the
linear-`d_sieve` cost reasserts itself. The hard part is *acquisition from
partial data*, not *connection once you have the solution*.

## Open directions

- **Longer-baseline simulator.** Everything here is fixed N=70 / fixed span.
  The regime that actually matters (longer observation → bigger parameter ranges,
  more wraps) is unreachable from `data.npy`. A forward simulator that emits
  `span_vecs`/`coeff_std` at a chosen baseline would let us measure the true
  N-scaling and test ideas in the right regime.
- **Change the search objective (L1 vs L2)** (`l1_search.py`): **DEAD.** Two
  tests. (i) Re-ranking the full L2-sieve database by L1 separates the pulsar
  from nulls *worse* than L2 (6.1σ vs 7.8σ) — the residuals aren't sparse
  (~0.08-cycle pulse-width scatter), so L1's premise fails. (ii) Decomposing
  acquisition into generation vs selection: on the bootstrap, the true `b*` is
  the **worst** of 200 candidates under *both* L1 and L2 (overfit solutions reach
  ~0 residual), so no norm-min search emits it; globally `b*` is rank 1 under
  *both* norms, so selection is fine. The bottleneck is **generation from a short
  baseline**, which is norm-independent (it is about generalising to unseen TOAs).
  Also: the no-signal solution has zero residual under any norm, so residual-min
  never prefers a pulsar — detection needs the coherence/physical criterion.
