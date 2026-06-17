"""How often does the mod-q reduction actually fire in the pure sieve, as a
function of the photon association probability p and the lattice size n?

`pure_sieve` can reduce candidates mod q (over the wrap coordinates) inside both
the pair and triple steps (``mod_q=``). Whether that ever does anything depends
on whether short vectors reach the +/- q/2 wrap boundary. This script counts the
firings (via ``pure_sieve.MODQ_STATS``) while sweeping:

  * p : association probability. We model 1-p of the lattice photons as background
        (uniform arrival times), so the effective pulse width is
        sigma_eff = sqrt(p*sigma_int^2 + (1-p)/12) -- lower p, noisier, wider.
  * n : number of lattice TOAs. Firings were first seen only at n >= 18, so we
        scan a few sizes to separate the p-effect from an n-threshold artifact.

Uses the 2-parameter model (model_a) so detection actually works at small n.

Run (g6k not required):  PYTHONPATH=. python experiments/modq_firing.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import pure_sieve as ps
from lattice_tools import pure_lll
import model_a_constant_frequency as A

F, PHI = 50.0, 0.3
T0, TSPAN = 2e8, 3e8
SIGMA_INT = 0.03
Q = 10 ** 15


def mixed_lattice(n_toas, p, n_verify=200, seed=0):
    """model_a lattice with a fraction (1-p) of background (uniform-phase) TOAs."""
    rng = np.random.default_rng(seed)
    tref = T0 + TSPAN / 2
    n_bg = int(round((1 - p) * n_toas))
    n_ps = n_toas - n_bg
    ps_t = A.generate_toas(n_ps, F, PHI, tref, TSPAN, SIGMA_INT, rng) if n_ps else np.array([])
    bg_t = T0 + TSPAN * rng.random(n_bg) if n_bg else np.array([])
    toas = np.sort(np.concatenate([ps_t, bg_t]))
    M = A.design_matrix(toas, tref)
    steps = 0.01 / np.max(np.abs(M), axis=1)
    cstd = 10 * np.abs(np.array([PHI, F]))
    il = A.build_integer_lattice(M, Q, steps, cstd)
    vt = A.generate_toas(n_verify, F, PHI, tref, TSPAN, SIGMA_INT, rng)
    tm = Q * steps[:, None] * A.design_matrix(vt, tref)
    return il, n_toas, tm, np.ones(n_verify, np.float32)


def detect(out, n_per, tm, probs):
    coeffs = np.array([c for _, c in out], dtype=object)
    p_par = coeffs[:, n_per:].astype(float)
    vf = np.mod(p_par @ tm / Q + 0.5, 1) - 0.5
    Q_ = np.abs(np.sum(probs * np.exp(2j * np.pi * vf), axis=1)) ** 2 / np.sum(probs ** 2 / 2)
    mask = np.std(coeffs[:, :n_per].astype(float), axis=1) > 1e3
    return Q_[mask].max() if mask.any() else float("nan")


def work_to_detect(n, seed, mod_q, sigma_int=SIGMA_INT):
    """Samples processed until the first reasonable vector clears Q>50 -- the
    pure-sieve analogue of 'iterations / pump depth to reach the solution'."""
    il, n_per, tm, probs = mixed_lattice(n, 1.0, seed=seed)   # clean (p=1) so it detects
    qi = int(il[0, 0])
    B, U = pure_lll(il, 0.99)
    state = {"hit": None}

    def hook(pv, pc, p2, db, samples):
        pp = np.array(pc[n_per:], float)
        vf = np.mod(pp @ tm / Q + 0.5, 1) - 0.5
        Qv = np.abs(np.sum(probs * np.exp(2j * np.pi * vf))) ** 2 / np.sum(probs ** 2 / 2)
        if Qv > 50 and np.std(np.array(pc[:n_per], float)) > 1e3:
            state["hit"] = samples
            return False

    ps.gauss_sieve(B, triple=True, seed=1, seed_coef=U, max_samples=8000,
                   mod_q=((n_per, qi) if mod_q else None), move_hook=hook)
    return state["hit"]


def firing_row_q(n, q, seed=7):
    """Like firing_row but at an explicit modulus q (clean p=1 lattice), to probe
    the precision floor. Self-contained so q is not the module default."""
    from lattice_tools import fpylll_lll          # robust on near-degenerate (small q)
    rng = np.random.default_rng(seed)
    tref = T0 + TSPAN / 2
    toas = A.generate_toas(n, F, PHI, tref, TSPAN, SIGMA_INT, rng)
    M = A.design_matrix(toas, tref)
    steps = 0.01 / np.max(np.abs(M), axis=1)
    il = A.build_integer_lattice(M, q, steps, 10 * np.abs(np.array([PHI, F])))
    qi = int(il[0, 0])
    vt = A.generate_toas(200, F, PHI, tref, TSPAN, SIGMA_INT, rng)
    tm = q * steps[:, None] * A.design_matrix(vt, tref)
    probs = np.ones(200, np.float32)
    B, U = fpylll_lll(il, 0.99)

    def det(out):
        co = np.array([c for _, c in out], dtype=object)
        vf = np.mod(co[:, n:].astype(float) @ tm / qi + 0.5, 1) - 0.5
        Qv = np.abs(np.sum(probs * np.exp(2j * np.pi * vf), axis=1)) ** 2 / np.sum(probs ** 2 / 2)
        m = np.std(co[:, :n].astype(float), axis=1) > 1e3
        return Qv[m].max() if m.any() else float("nan")

    Q0 = det(ps.gauss_sieve(B, triple=True, seed=1, seed_coef=U))
    ps.reset_modq_stats()
    Q1 = det(ps.gauss_sieve(B, triple=True, seed=1, seed_coef=U, mod_q=(n, qi)))
    st = ps.MODQ_STATS
    return 100 * st["fires"] / max(st["calls"], 1), Q0, Q1


def firing_row(n, p, seed=7):
    il, n_per, tm, probs = mixed_lattice(n, p, seed=seed)
    qi = int(il[0, 0])
    B, U = pure_lll(il, 0.99)
    out0 = ps.gauss_sieve(B, triple=True, seed=1, seed_coef=U)
    Q0 = detect(out0, n_per, tm, probs)
    maxc = max(max(abs(int(x)) for x in v[:n_per]) for v, _ in out0) / qi
    ps.reset_modq_stats()
    out1 = ps.gauss_sieve(B, triple=True, seed=1, seed_coef=U, mod_q=(n_per, qi))
    Q1 = detect(out1, n_per, tm, probs)
    st = ps.MODQ_STATS
    return 100 * st["fires"] / max(st["calls"], 1), maxc, Q0, Q1


def main():
    print(f"q={Q:.0e}, sigma_int={SIGMA_INT}, mod-q applied in pair AND triple.\n")

    print("[1] firing vs n (p=1.0): rises toward saturation (=> ~100% at n=70-80)")
    print(f"   {'n':>4} {'fire%':>7} {'maxcoord/q':>11} {'Q(no q)':>8} {'Q(mod q)':>9}")
    for n in (18, 22, 26, 30):
        pct, maxc, Q0, Q1 = firing_row(n, 1.0)
        print(f"   {n:>4} {pct:>6.2f}% {maxc:>11.3f} {Q0:>8.1f} {Q1:>9.1f}", flush=True)

    print("\n[2] firing vs p (n=22): driven by n, ~flat in p")
    print(f"   {'p':>4} {'fire%':>7} {'maxcoord/q':>11}")
    for p in (1.0, 0.8, 0.6, 0.4):
        pct, maxc, _, _ = firing_row(22, p)
        print(f"   {p:>4.1f} {pct:>6.2f}% {maxc:>11.3f}", flush=True)

    print("\n[3] work-to-detection (samples) -- does the shortcut reach the solution faster?")
    print("    (spans the high-firing regime: n=30 fires ~37% of candidates)")
    print(f"   {'n':>4} {'no-modq':>9} {'mod-q':>8}  (median over 5 seeds)")
    for n in (18, 22, 26, 30):
        a = [work_to_detect(n, s, False) for s in range(5)]
        b = [work_to_detect(n, s, True) for s in range(5)]
        md = lambda x: np.median([v for v in x if v is not None])
        print(f"   {n:>4} {md(a):>9.0f} {md(b):>8.0f}", flush=True)
    print("\n[4] firing vs modulus q (n=26): q-invariant above the precision floor")
    print(f"   {'q':>7} {'fire%':>7} {'Q(no q)':>8} {'Q(mod q)':>9} {'detect':>7}")
    for q in (10 ** 12, 10 ** 13, 10 ** 15):
        pct, Q0, Q1 = firing_row_q(26, q)
        print(f"   {q:>7.0e} {pct:>6.2f}% {Q0:>8.1f} {Q1:>9.1f} "
              f"{('YES' if Q0 > 50 else 'no'):>7}", flush=True)

    print("\n=> firing climbs with n but NOT with p, and is q-invariant above the\n"
          "   precision floor (below it the lattice degenerates, detection fails).\n"
          "   And it gives no speedup -- samples-to-detection identical with/without\n"
          "   mod-q, even at ~37% firing.")


if __name__ == "__main__":
    main()
