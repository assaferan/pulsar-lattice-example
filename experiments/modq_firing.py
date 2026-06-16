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


def main(ns=(18, 20, 22), ps_vals=(1.0, 0.8, 0.6, 0.4), seed=7):
    print(f"mod-q firing vs (n, p).  q={Q:.0e}, sigma_int={SIGMA_INT}, "
          f"mod-q applied in BOTH pair and triple.")
    print(f"{'n':>4} {'p':>5} {'sig_eff':>8} {'fires':>7} {'calls':>8} {'fire%':>7} "
          f"{'maxcoord/q':>11} {'Q(no q)':>8} {'Q(mod q)':>9}")
    for n in ns:
        for p in ps_vals:
            il, n_per, tm, probs = mixed_lattice(n, p, seed=seed)
            qi = int(il[0, 0])
            B, U = pure_lll(il, 0.99)
            # baseline (no mod-q) for detection + boundary diagnostic
            out0 = ps.gauss_sieve(B, triple=True, seed=1, seed_coef=U)
            Q0 = detect(out0, n_per, tm, probs)
            maxc = max(max(abs(int(x)) for x in v[:n_per]) for v, _ in out0) / qi
            # with mod-q (pair + triple), counting firings
            ps.reset_modq_stats()
            out1 = ps.gauss_sieve(B, triple=True, seed=1, seed_coef=U, mod_q=(n_per, qi))
            Q1 = detect(out1, n_per, tm, probs)
            st = ps.MODQ_STATS
            pct = 100 * st["fires"] / max(st["calls"], 1)
            sig = np.sqrt(p * SIGMA_INT ** 2 + (1 - p) / 12)
            print(f"{n:>4} {p:>5.1f} {sig:>8.3f} {st['fires']:>7} {st['calls']:>8} "
                  f"{pct:>6.2f}% {maxc:>11.3f} {Q0:>8.1f} {Q1:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
