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
- LLL/BKZ before sieving is *essential*, not just a speedup (`sieve_only.py`):
  sieving the raw lattice needs ~30x more iterations at N=14 and fails to converge
  by N≈16–18, because the fold's wrap counts (~1e10) can't be built up by the
  sieve's `v±w±u` moves — LLL pre-packages them into short basis vectors.

## What worked (constant factor, not asymptotic)

- **`fold(fast=True)`** — LLL + a shallow pump instead of BKZ + deep pump.
  ~100x faster sieving, same detection. Shipped in `fermi_fold.py`. This is a
  constant-factor win; `d_sieve` (hence the exponent) is unchanged.
- **Right sieve algorithm at high dimension** (`sieve_algo.py`): the default
  `hk3` triple sieve is memory-optimized (smallest database, ~2^{0.19 d}) and
  **fails** (`SaturationError` / misses) once `d_sieve` passes ~80 on this skewed
  q-ary lattice — e.g. model B, p=0.7, N=111 (needs `d_sieve~88`). `bgj1` detects
  there in ~80 s; `bdgl2` also detects but is **no faster at this dimension** (its
  best-known 0.292 time exponent is dominated by LSF overhead until much larger
  N). `fermi_fold._sieve` now auto-falls-back `hk3 -> bgj1 -> bdgl2`. The real
  feasibility lever is the **pump depth** (`d_sieve ~ 0.8 N`, not full-dim) — so
  the harder p=0.7 / larger-N cases run well short of the worst-case `2^(0.36 N)`.

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
  *before* the length check — the idea being to reach short vectors faster
  (fewer iterations / shallower pumps), not to change the endpoint.
  **It gives no speedup, and no change to the endpoint.**
  * *Speed (the point):* samples-to-first-detection are **identical** with and
    without mod-q, in every (n, seed) at n=18–30 — including the high-firing end
    (n=30 fires ~37% of candidates). Detection is reached in the first ~5–35
    samples via small-coordinate combinations that never engage a wrap; the
    firing happens later on longer/irrelevant candidates, off the detection path.
    So the shortcut does not reach short vectors faster, even where it fires often.
  * *Why a g6k-pump test at n≈70 is unnecessary:* mod-q subtracts wrap rows
    `q*e_j`, changing the wrap counts k (norm) but **not** the timing parameters
    p — and detection (Q) depends only on p. So applying mod-q to any found
    vector (post-hoc, or by insertion) cannot change Q or `d_sieve` — it is
    algebraically vacuous. The only non-vacuous mechanism is in-sieve
    keep/discard, which is exactly what the pure sieve tests above (no speedup).
  * *Endpoint:* Q is identical with/without mod-q in every (n, p) cell.
  * *Firing rate is set by lattice dimension, not data quality:* triple fire-rate
    ≈0% (n≤16) → 0.2% (18) → 2.3% (22) → 37% (30); a sigmoid saturating to ~100%
    by n≈36–40 (`maxcoord/q ≈ 0.0208 n − 0.176` crosses ½ at n≈33), so at the
    target n=70–80 essentially **every** candidate fires. Across p (1.0→0.4,
    more background, wider σ_eff) the rate is ~flat at fixed n. The pair step
    fires more (it also wraps raw samples) but equally harmlessly.
  * *q-invariant above the precision floor:* firing% is also independent of the
    modulus q (n=26: ~10–14% for q=1e13…1e16) — `coord/q` is a phase residual,
    q-independent. Below the floor (q≲1e12, where `round(q·s/coeff_std)`→0) the
    lattice degenerates: detection fails and mod-q stops being inert but only
    makes the already-failed result worse. So no q makes the shortcut useful.

  Net: the explicit mod-q is a *shortcut* for what pair-reduction against the
  wrap vectors `q*e_j` already does; firing more often (large n) just means the
  shortcut triggers more, not that it accomplishes anything reduction wouldn't —
  confirming at the sieve *core* what `modq_reduction` found post-hoc.
- **Modulus switching — exploit q·Z^n via the value of q** (`q_invariance.py`):
  above the precision floor (`q ≫ f_prior/(d_f σ²)`, which matches the paper's
  footnote), `d_sieve` is q-invariant. q is only the integer SCALE of a
  scale-invariant continuous problem (phase residuals mod 1), not an exploitable
  LWE-style modulus — which is exactly why `modq_reduction` was redundant with
  LLL. **Dead.**
- **mod-1 in a bare pair-sieve** (`pair_sieve_mod1.py`, `pair_sieve_scaling.py`,
  `pair_sieve_triple.py`; a collaborator's experiment): on a random bank of
  uniform vectors reduced by `+/-1` pair moves to a target length, wrapping
  `v +/- w` mod 1 *does* lower the minimal viable database size `L_min`, by ~1.4x.
  This is the one place the wrap helps — the bare 2-sieve has no LLL to absorb it.
  But it is a **pair-sieve-only** effect with **no pipeline relevance**: (i) the
  `L_min` advantage is small and roughly **constant** (the log-slope gap bounced
  0.006–0.029 across runs; the no-mod-1 exponent ~0.21 matches the textbook
  `(4/3)^{n/2}`); an asymptotic (growing) advantage is *not* established. (ii)
  Adding the triple (hk3) move collapses `L_min` to a constant (~4) at every
  dimension — the 3-tuple trivialises the metric, so there is no exponent left for
  mod-1 to touch. The real sieve uses triple + LLL, where the wrap was already
  found inert (`modq_reduction`, `modq_firing`). **Real in the toy, dead in the
  pipeline.**
  * *Re-checked for the bgj1 (pair) pipeline* (`modq_pair.py`): once we moved off
    hk3 to bgj1 (pure pairs), the natural worry was that the pair-sieve win above
    would now transfer. It does **not**. On the synthetic pulsar lattice the pair
    sieve is **identical with and without mod-q** by both metrics — work-to-
    detection (10/10, 17/17 samples) and L_min (15/15, 18/18, 12/12) — with the
    *physical* detection control (Q>50 AND std(k)>1e3) so mod-q can't cheat via the
    trivial k=0 vector. The collaborator's win is for *geometric* shortness; the
    pulsar needs a *large-k* solution, which is sieve-type-independent. Caveat:
    decisive only at small n (pure_sieve's reach), where detection comes from
    small-coordinate combinations that never wrap; the large-n regime is
    untestable (g6k's bgj1 is C++ and won't take a custom mod-q).
- **mod-q in a sieve WITHOUT LLL** (`sieve_only.py`): tested because the bare
  pair-sieve above *benefits* from mod-1, so maybe mod-q helps once LLL isn't
  there to absorb it. On the actual pulsar lattice it is the **opposite — mod-q
  destroys detection.** The plain raw sieve finds the fold at N=14 (~180 iters);
  *with* mod-q it fails at every N (and at every p). Diagnostic (N=14): mod-q
  wraps each candidate by decrementing the wrap counts, driving median `k-std`
  from `1.6e9` to `0` and collapsing the database (~25→3, via collisions). The
  surviving high-Q vectors are the trivial constant-phase (`k=0`) ones; the
  *physical* solution needs large, varying `k` (`~1e10`, std>1e3), which wrapping
  annihilates. The pair-sieve benefits because its target is pure geometric
  shortness; detecting the pulsar needs a physical large-wrap-count solution, so
  "mod-q shortens vectors" and "mod-q finds the pulsar" are different claims —
  only the former holds. **mod-q harmful here.**
  * *Refinement — excluding the redundant wrap rows doesn't rescue it.* Under
    mod-q the wrap rows `q*e_j` reduce to 0, so they are dead generators; one
    might hope removing them (sieve the quotient `L/qZ^N`, via the new
    `gauss_sieve(active_rows=...)`) avoids the collapse. It does not — db goes to
    ~2 with `|b|~1`, `k=0` (even more degenerate). The physical solution is short
    only via *large near-cancelling* coeffs (`|b|~8e11`, `k-std~1.6e9`); mod-q
    reaches shortness the easy way (`k=0`) regardless of how you seed. So the
    failure is **not** the wrap-row seeding — mod-q simply targets the wrong
    short-vector class. (`sieve_only.py` parts [3]–[4].)
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
- **g6k saturates only up to dim ~100 on this lattice; subsample past that
  (`fermi_fold.fold_subsampled`).** On the highly skewed q-ary pulsar lattice
  (`q=1e15` wrap rows vs a tiny prior block), g6k's `hk3` fails to saturate once
  the dimension exceeds ~100 -- `SaturationError`, "saturation 0.000" (zero short
  vectors found), early in the pump. It is **not** a tunable knob: identical
  failure across `saturation_ratio ∈ {0.5,0.3,0.15}`, pump depth, and with BKZ(30)
  -- a GH/conditioning limit on the skew, not compute. Fix: `fold_subsampled`
  sieves a random TOA subset (default 85 -> dim 87, in the working window ~58-95)
  and scores detection on the full verify set (the parameters are shared, so a
  subset pins them). On Model A `n_toas=111, p=0.7` (σ≈0.16): direct `fold`
  crashes at dim 113, while `fold_subsampled` detects **6/6, Q≈387, ~1.7 s/seed**
  -- stronger and faster than the subset-LLL helper (`simple_lattice`, coh ~50).

## p=0.5 at scale — the practical wall (`p05_lava.py`, lovelace)

Direct attempt at Table 1's near-infeasible row, `p=0.5` (σ=√(0.5/12)=0.204), on a
256-core / 2 TiB workstation with a fresh g6k build (`p05_lava.py` runs one
`(model,n)` per process; `run_p05_scan.sh` fans them out). Three compounding walls
stop it — and they line up with `cost_vs_p.py` (the `n_threshold` for `p=0.52` was
already off the `n≤64` grid):

- **Cost.** Detection needs sieve dim `d_sieve ≈ 0.85 N ≈ 112–124` for model A
  `n≥130`. That is hundreds× the `dim≈88` sieves we timed for `p=0.7` (bgj1 exponent
  0.349): the model-A `n=130…180` jobs ran **>1.8 h each with zero completing**.
- **Stability.** bgj1 **C-aborts** (libstdc++ `terminate`) on the skewed q-ary
  lattice once the sieving dim exceeds ~110 — `n=130/140/150` all crashed mid-sieve.
  Same failure class as the dim>100 `hk3` saturation collapse (see Tooling). One
  `(model,n)` per process is essential so an abort kills only that point.
- **bdgl2 unavailable there.** The asymptotically-faster sieve (exponent 0.292)
  needs precomputed spherical-coding tables `g6k/spherical_coding/sc_<d>_256.def`;
  the build only generated them into the ~150s, so `n=160/170` died on
  `Cannot open … sc_162_256.def`.

**Detection was never observed.** The deepest *clean* run (`n=110`, `d_sieve=95`)
gave best **physical** `Q≈24 < 50` and not climbing with `n`. Note the decoy: the
*unmasked* maxQ can be huge (≈2562) for a near-zero-wrap "fold everything into one
phase" vector — correctly rejected by the `std(k) > 1e5` physical mask (the mask
threshold is irrelevant here: the wrap-std distribution is bimodal, ~0 or >1e11).

**Verdict.** Model A `p=0.5` sits at/beyond the practical feasibility edge for this
build. Reaching it would need (a) a g6k rebuilt for *stable* bdgl2 at dim 112–128
(spherical codes generated higher, possibly a larger `MAX_SIEVING_DIM`), and (b)
likely far more TOAs than the early ~110 projection — and even then detection is
unproven. This is the gap-limited core obstruction made concrete: the physics sets
`σ/σ_exp`, and at `p=0.5` it forces a sieve this build cannot reach.

## Open directions

- **Hard-regime baseline test (postponed).** `baseline_scaling.py`'s grow-vs-
  fixed-span comparison is clean in the easy/large-gap regime (A≈B). Settling it
  at `gap≈1` (the regime that matters) needs `N≈80–120` with averaging
  (minute-scale sieves) to beat the detection-edge noise seen at `N≤56`.
- **Widen the gap (the only live lever).** Since difficulty is gap-limited and
  physical, the payoff is in `σ_exp/σ`: better per-photon weighting, sharper
  pulse modeling, or dropping low-probability photons. This is statistics, not
  lattice structure — but it is the one thing that moves the exponent.
  *Quantified* (`cost_vs_p.py`): lowering the association probability `p` widens
  `σ` (`σ²=(1-p)/12`) and the measured minimum detection dimension `n_threshold`
  climbs with it — 26→30→40→58→(>64) for `p`=0.98→0.92→0.83→0.69→0.52 — so the
  pump cost `2^(0.36 n)` explodes (660 → 1.9e6, ~2900×, from `p`=0.98 to 0.69;
  off the `n≤64` grid by `p`=0.52). The growth tracks the closed-form `n ∝ 1/D`,
  `D=-1.280-ln σ` (ratio `n/(1/D)` converges to ~32 as `n` grows; inflated at
  small `n` by the finite-size GH correction). You pay in the *exponent*, with a
  hard wall at `σ→0.278` (`p≈0.07`). Real `data.npy` (`p≈0.9`) sits at the cheap
  end, so association quality is precious.
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
