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

## `lattice_tools.py` — tweakable LLL + g6k pump (TOOLING)

Transparent, hackable versions of the two pipeline stages for experiments:
`pure_lll` (pure-Python, exact integer basis + float GSO, both reduction rules
editable, returns the transform `U` so coefficients map back to the original
basis), `fpylll_lll` (fast, same signature), and `pump` over g6k primitives with
a per-round `on_round` hook (inspect the DB, inject vectors, switch sieve `alg`,
stop early). Self-test recovers the pulsar end-to-end on `data.npy` (Q≈404).

## `pure_sieve.py` — the hk3 sieve cracked open (TOOLING)

A pure-Python sieve mirroring g6k's `hk3` triple sieve: a database of lattice
vectors reduced by **pair** (Gauss, 2-reduction) and optional **triple** (the
hk3 3-tuple move) combinations until saturation. Exact-integer (for the q≈1e15
entries), so small-dimension only; g6k stays for scale. Each entry carries its
coefficients in the input basis, so mapping results back is exact. Self-test:
`pure_lll` + `gauss_sieve` recover a small synthetic pulsar end-to-end (Q≈384),
with **no g6k dependency** — the fully transparent stack for algorithm tweaks.
Includes a `mod_q=(n_per, q)` option that reduces **pair- and triple-**candidates
mod q before the length check (a structure-aware experiment), with a built-in
firing counter (`MODQ_STATS`, `reset_modq_stats`); the self-test compares plain
vs mod-q for both. **Result: inert** (see FINDINGS) — the wrap structure is
already absorbed by ordinary reduction; it never changes detection. Also an
`active_rows=` knob restricting the seed/sampler to a subset of basis rows: under
mod-q the wrap rows `q*e_j` reduce to 0, so `active_rows=` the timing rows sieves
the quotient `L/qZ^N` without the dead generators.

## `sieve_only.py` — iterations to the short vector, LLL vs none, and mod-q (NEGATIVE)

Feeds the pure triple-sieve either an LLL-reduced basis or the **raw**
`integer_lattice`, counting samples until detection (held-out Q>50). **(1) LLL is
essential:** LLL+sieve ~5–17 iters (flat in n); sieve-only ~180 at n=14 (~36×) and
fails to converge by n≈16–18 — the raw fold has wrap counts `k~1e10` a sieve can't
build up; LLL pre-packages them. **(2) mod-q *hurts* the raw sieve** (opposite of
the bare pair-sieve): the physical solution is short only via *large near-cancelling*
coeffs (`|b|~8e11`, k-std 1.6e9), but mod-q reaches shortness the easy way — wrapping
to `k=0` — so it parks on the trivial constant-phase family and the physical solution
never forms; fails at every n. **(3) Excluding the redundant wrap rows** (`active_rows`)
does *not* rescue it (db→2, `|b|~1`): the failure is mod-q targeting the wrong
short-vector class, not the wrap-row seeding. **(4) Lower p** breaks the raw sieve
regardless. So "mod-q shortens vectors" ≠ "mod-q finds the pulsar".

## `p07_threshold.py` — how many TOAs to detect at association probability p?

Sweeps `n` and reports the max reasonable Q (>50 = detected) at the effective
pulse width `σ=√((1−p)/12)`, for model A or B. **At p=0.7 (σ=0.158): model A
needs n≈60, model B (7 params) n≈80** — both well below Table 1's blind-7-param
n=111 (our synthetic has tight astrometry priors). Acquisition is a sharp
threshold (found → Q~250–350; not → Q~10–25), ~4× the p=1 requirement.
`... p07_threshold.py [A|B] [p]` sweeps n; `... p07_threshold.py check` runs the
shipped `fermi_fold.fold` on p=0.7 data and shows full mode detects (Q~900) while
`fast` mode needs a **deeper pump** (it's pump depth, not LLL-vs-BKZ — `fast` with
`pump_stop≲10` detects too).

## `p05_lava.py` — heavy p=0.5 detection sweep (run on a workstation, not a laptop)

Tackles Table 1's hardest row, p=0.5 (σ=0.204). Projected thresholds: model A
~n110/d_sieve88 (laptop-feasible), model B ~n145/d_sieve115 (needs RAM+cores; dim
~115 is the bdgl regime, near g6k's MAX_SIEVING_DIM=128). BKZ-30 + a **tuned**
pump depth (`d_sieve≈0.85·N`, capped <128, *not* full-mode), `bgj1→bdgl2` per
case with retries; appends rows to `p05_results.txt` (gitignored) as they finish.
`PYTHONPATH=. python experiments/p05_lava.py [A|B] [p] [n1,n2,...]`.

## `sieve_algo.py` — which g6k sieve when complexity bites? (hk3 fails, bgj1/bdgl2 win)

Compares `hk3` / `bgj1` / `bdgl2` at the cost-wall case (model B, p=0.7, N=111,
sieve dim 88). **`hk3` fails** (SaturationError/misses — its memory-optimized
database is too small at this dimension); **`bgj1` detects in ~80 s**; **`bdgl2`
detects but isn't faster here** (its 0.292 exponent only wins at larger N).
`fermi_fold._sieve` now auto-falls-back `hk3→bgj1→bdgl2`; the key lever is pump
depth (`d_sieve≈0.8·N`), not full-dim sieving.

## `modq_firing.py` — does the mod-q shortcut help (firing, scaling, speed)? (NEGATIVE)

Four measurements: (1) firing vs `n` at p=1 — rises 0.2%→37% over n=18→30, a
sigmoid saturating to ~100% by n≈36–40 (so ~100% at the n=70–80 target);
(2) firing vs association probability `p` (modelling `1-p` background photons) —
**driven by `n`, flat in `p`**; (3) **work-to-detection** — samples until the
first Q>50 solution, with vs without mod-q. **Identical** in every (n, seed):
the shortcut gives no speedup. Detection is reached early via small-coordinate
combinations that never wrap; firing happens later on irrelevant candidates.
(4) firing vs modulus `q` — **q-invariant above the precision floor** (coord/q is
a phase residual); below it the lattice degenerates and detection fails.

## `q_invariance.py` — is the modulus q a lever? (NEGATIVE)

Measures `d_sieve` for the same problem at several `q`. **Above the precision
floor** (`q ≫ f_prior/(d_f σ²)`, matching the paper's footnote) `d_sieve` is
**q-invariant**: q is the integer scale of a scale-invariant continuous problem,
not an exploitable modulus. Below the floor, detection fails (rounding error
swamps the signal). Confirms why `modq_reduction` was redundant with LLL.

## `baseline_scaling.py` — does a longer baseline lower the sieving cost? (NO, in the clean regime)

Controlled comparison of two ways to add TOAs under one detection methodology
(held-out verify-set Q): **A** grow the baseline (`span ∝ N`) vs. **B** subsample
a fixed span. In the easy/large-gap regime they scale identically (`d_sieve`-vs-
`N` slope `0.275` vs. `0.300`) — baseline confers no advantage over density. The
hard/`gap≈1` regime is left as a postponed open question (needs larger `N`).

## `l1_search.py` — would an L1 objective acquire the pulsar? (NEGATIVE)

Tests whether minimising the L1 (rather than L2) phase residual helps acquisition.
**It does not.** On the bootstrap the true `b*` is the *worst* of 200 candidates
under both L1 and L2 (overfit solutions reach ~0 residual), so no norm-min search
emits it; globally `b*` is rank 1 under both norms, so selection is fine. The
bottleneck is **generation from a short baseline**, which is norm-independent.
(The no-signal solution has zero residual under any norm, so residual-min never
prefers a pulsar — detection needs the coherence/physical criterion.)

## `cost_vs_p.py` — how much does lowering the association probability p cost the pump?

Empirically confirms the Table-1 cost model under varying data quality. For each
`p` (via `σ²=(1-p)/12`) it grows the number of TOAs until the pump detects, giving
the minimum detection dimension `n_threshold` and pump cost `2^(0.36 n)`.

**Finding:** cost climbs steeply and accelerates — `n_threshold` 26→30→40→58→(>64)
for `p`=0.98→0.92→0.83→0.69→0.52, i.e. cost 660 → 1.9e6 (~2900×) by `p`=0.69 and
off the `n≤64` grid by `p`=0.52. Tracks the closed form `n ∝ 1/D`, `D=-1.280-ln σ`
(the `n/(1/D)` ratio converges to ~32 as `n` grows; inflated at small `n` by the
finite-size GH correction). You pay in the *exponent*, with a hard wall at `p≈0.07`.

## `pair_sieve_mod1.py` / `pair_sieve_scaling.py` / `pair_sieve_triple.py` — does mod-1 help the pair-sieve? (YES ~1.4x, but pair-only)

Reproduces a collaborator's experiment: a bank of `L` random vectors uniform in
`[-1/2,1/2]^n`, reduced by single `+/-1` (Gauss/2-) pair moves to saturation, is
*viable* if the saturated mean `|v|^2` reaches `n*((1-w)/12 + w sigma^2)` (a
mix of uniform background and a width-`sigma` signal; `w=0.5` is the collaborator's
setting). `L_min(n)` is the smallest viable database, measured **with and without
reducing `v +/- w` mod 1** (the q=1 wrap). `pair_sieve_scaling.py` fits
`log2 L_min` vs `n`; `pair_sieve_triple.py` adds the 3-tuple (hk3) move.

**Finding:** mod-1 lowers `L_min` by **~1.4x** in the bare 2-sieve (no-mod-1
exponent ~0.21 matches the textbook `(4/3)^(n/2)=2^0.2075`, validating the setup).
Whether the advantage *grows* with `n` is **not settled** — the log-slope gap
bounced between 0.006 and 0.029 across runs/targets, so it leans **constant**, not
asymptotic. Crucially it is **pair-only**: adding the triple (hk3) move collapses
`L_min` to a constant (~4) at every `n` (the 3-tuple trivialises the random-bank
metric — no exponent left for mod-1 to touch). The real sieve uses triple + LLL,
where the wrap was already found inert (`modq_reduction`, `modq_firing`). So:
real in the toy, dead in the pipeline.

## `modq_pair.py` — does mod-q help the PAIR (bgj1) sieve? (NEGATIVE)

Re-opens the mod-q question now that the pipeline uses bgj1 (pairs) not hk3
(triples): `pair_sieve_triple` showed mod-1 helps a pair sieve but triples erase
it, so pairs might revive it. Measures work-to-detection and minimal database
`L_min` in the pure-Python pair sieve, mod-q vs not, with the **physical**
detection control (Q>50 AND std(k)>1e3). **Finding: inert** — identical by both
metrics on the pulsar lattice (work 10/10, 17/17; L_min 15/15, 18/18, 12/12). The
collaborator's pair-sieve win is for *geometric* shortness; the pulsar's *large-k*
solution is sieve-type-independent. (Small-n only; g6k's bgj1 can't take a custom
mod-q, so the large-n regime is untestable.)
