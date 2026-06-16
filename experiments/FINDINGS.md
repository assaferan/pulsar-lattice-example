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
- **Reduce mod q inside the sieve** (`pure_sieve.py` `mod_q=`; `modq_firing.py`):
  reduce pair- and triple-candidates mod q (centered, over the wrap coords)
  *before* the length check, so a candidate that is long raw but short after
  wrapping is kept. **Inert for detection** — Q is identical with/without mod-q
  in every (n, p) cell tested. Firing is governed by the lattice *dimension*,
  not the data quality: triple fire-rate ≈0% (n≤16), 0.25% (n=18), 0.5% (n=20),
  1.8% (n=22) as bigger triples reach the ±q/2 boundary; the pair step fires more
  (it also wraps raw samples) but equally harmlessly. Across association
  probability p (1.0→0.4, i.e. more background, wider σ_eff) the fire-rate is
  ~flat at fixed n and `maxcoord/q` stays ~constant — the sieve still works with
  small-residual (overfit) vectors regardless of p. Confirms at the sieve *core*
  what `modq_reduction` found post-hoc: the wrap structure is fully absorbed by
  ordinary reduction.
- **Modulus switching — exploit q·Z^n via the value of q** (`q_invariance.py`):
  above the precision floor (`q ≫ f_prior/(d_f σ²)`, which matches the paper's
  footnote), `d_sieve` is q-invariant. q is only the integer SCALE of a
  scale-invariant continuous problem (phase residuals mod 1), not an exploitable
  LWE-style modulus — which is exactly why `modq_reduction` was redundant with
  LLL. **Dead.**
- **Longer observation baseline** (`baseline_scaling.py`): a *controlled*
  comparison of two ways to add TOAs — grow the baseline (`span ∝ N`) vs.
  subsample a fixed span (the `density_sweep` setup). In the easy/large-gap
  regime (`σ=0.03`) the two scale identically (`d_sieve`-vs-`N` slope `0.275`
  vs. `0.300`), so a longer baseline confers no advantage over more density. An
  uncontrolled grow-baseline run looked sub-linear, but the control disproved it.
  **No win shown** (the hard/`gap≈1` regime at `N≤56` is too noisy to resolve —
  see Open directions).

## The core obstruction

Identifying the solution genuinely needs ~N-dimensional information. A short
baseline is consistent with many physical timing solutions (the lever-1 test's
huge `g_far`), so a bounded bootstrap cannot **acquire** the solution, and the
linear-`d_sieve` cost reasserts itself. The hard part is *acquisition from
partial data*, not *connection once you have the solution*.

In lattice-crypto terms: the lattice is q-ary with a rank-`p` code mod q
(LWE/SIS shape), and the difficulty is the **near-unit gap** `σ/σ_exp ≈ 2^0.2`
at the Table-1 detection threshold — the true solution is barely shorter than a
random short vector, which forces full-dimension sieving and a `2^(0.2N)`
candidate database. The gap is set by the *physics* (pulse width σ vs. photon
background), so no lattice-construction trick — q-ary structure, modulus
switching, or baseline — can move it. The GH/`σ_exp` model underlying this is
validated empirically in `../complexity_empirical.py`; Table 1 is reproduced in
`../complexity_table.py`. **The only lever that touches the gap is data quality
(narrower σ / better photon weighting), i.e. statistics, not lattices.**

## Tooling

- **Forward simulator (built).** `../model_b_full_timing.py` emits
  `span_vecs`/`coeff_std`/`integer_lattice` for the full 7-parameter model at any
  chosen baseline, photon count, and pulse width — the simulator the open
  directions below used to call for. `../model_a_constant_frequency.py` is the
  2-parameter version.
- **Tweakable LLL + pump (`lattice_tools.py`).** Transparent, hackable versions
  of the two pipeline stages: `pure_lll` (exact integer basis, float GSO, both
  reduction rules editable; returns the transform `U`), `fpylll_lll` (fast),
  and a `pump` over g6k primitives with a per-round `on_round` hook (inspect the
  DB, inject vectors, switch sieve alg, stop early). Verified to recover the
  pulsar end-to-end on `data.npy` (Q≈404). Substrate for algorithm experiments.
- **Pure-Python sieve (`pure_sieve.py`).** g6k's `hk3` triple sieve cracked open:
  a database reduced by editable **pair** (Gauss) and **triple** (hk3 3-tuple)
  moves, exact-integer, coefficients carried for an exact map-back. With
  `pure_lll` it gives a fully g6k-free transparent stack; recovers a small
  synthetic pulsar (Q≈384). Small-dimension only (slow); g6k for scale.

## Open directions

- **Hard-regime baseline test (postponed).** `baseline_scaling.py`'s grow-vs-
  fixed-span comparison is clean in the easy/large-gap regime (A≈B). Settling it
  at `gap≈1` (the regime that matters) needs `N≈80–120` with averaging
  (minute-scale sieves) to beat the detection-edge noise seen at `N≤56`.
- **Widen the gap (the only live lever).** Since difficulty is gap-limited and
  physical, the payoff is in `σ_exp/σ`: better per-photon weighting, sharper
  pulse modeling, or dropping low-probability photons. This is statistics, not
  lattice structure — but it is the one thing that moves the exponent.
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
