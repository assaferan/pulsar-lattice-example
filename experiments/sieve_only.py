"""How many sieve iterations to find the short vector -- WITH vs WITHOUT an LLL
warm-start?

The real pipeline always LLL/BKZ-reduces before sieving. This asks what the LLL
step actually buys, by feeding the pure triple-sieve (`pure_sieve.gauss_sieve`)
either an LLL-reduced basis or the *raw* `integer_lattice`, and counting the
samples (iterations) until the first reasonable vector clears Q>50 (held-out
verify set). Iterations are the pure-sieve analogue of pump depth / work.

Finding (model_a, clean, 3 seeds):

      n     LLL+sieve         sieve-only (no LLL)
     14     5,  5,  5         179, 198, 171     (~36x more)
     16    17, 17, 17         X,   191,  X
     18     4,  4,  4         X,   X,    X       (X = not found in 40000)

So sieving alone is far slower and stops converging by n~16-18, while LLL+sieve
stays at a handful of iterations. (n<=12 fails for BOTH -- below the detection
threshold, too few photons -- so it is not informative about LLL.)

Why: the true fold vector in the *raw* basis is (k, p) with wrap counts k ~ 1e10
-- enormous integer coefficients. A sieve only forms v +/- w (+/- u), so reaching
those coefficients means building them up combination by combination, which blows
up. LLL's job is exactly to pre-package those huge combinations into short basis
vectors, after which the fold is a *small* combination found in a few steps. That
is why no real sieve runs on a raw skewed q-ary basis.

Run (g6k not required):  PYTHONPATH=. python experiments/sieve_only.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # experiments/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import pure_sieve as ps
from lattice_tools import pure_lll
from modq_firing import mixed_lattice, Q


def samples_to_detect(il, n_per, tm, probs, use_lll, max_samples, seed):
    """Iterations until the first reasonable vector clears Q>50, or None.

    use_lll : LLL-reduce the basis first (and track the transform) vs. sieve the
              raw lattice directly (coeffs tracked in the original basis).
    """
    state = {"hit": None}

    def hook(pv, pc, p2, db, samples):
        pp = np.array(pc[n_per:], float)
        vf = np.mod(pp @ tm / Q + 0.5, 1) - 0.5
        Qv = np.abs(np.sum(probs * np.exp(2j * np.pi * vf))) ** 2 / np.sum(probs ** 2 / 2)
        if Qv > 50 and np.std(np.array(pc[:n_per], float)) > 1e3:
            state["hit"] = samples
            return False

    if use_lll:
        B, U = pure_lll(il, 0.99)
        ps.gauss_sieve(B, triple=True, seed=seed, seed_coef=U,
                       max_samples=max_samples, move_hook=hook)
    else:
        # raw lattice: no LLL, coeffs tracked in the original basis (seed_coef=None)
        ps.gauss_sieve(il, triple=True, seed=seed,
                       max_samples=max_samples, move_hook=hook)
    return state["hit"]


def main(ns=(14, 16, 18), seeds=(1, 2, 3), lll_budget=20000, raw_budget=40000):
    print("samples-to-detection (model_a, clean); 'X' = not found in budget")
    print(f"  LLL budget={lll_budget}, sieve-only budget={raw_budget}")
    print(f"{'n':>4} {'LLL+sieve':>18} {'sieve-only (no LLL)':>26}")
    fmt = lambda x: str(x) if x is not None else "X"
    for n in ns:
        il, n_per, tm, probs = mixed_lattice(n, 1.0, seed=0)
        a = [samples_to_detect(il, n_per, tm, probs, True, lll_budget, s) for s in seeds]
        b = [samples_to_detect(il, n_per, tm, probs, False, raw_budget, s) for s in seeds]
        print(f"{n:>4} {str([fmt(v) for v in a]):>18} {str([fmt(v) for v in b]):>26}",
              flush=True)
    print("\n=> LLL warm-start is essential: without it the sieve must build up the\n"
          "   huge wrap-count coefficients (~1e10) combination by combination, so it\n"
          "   needs ~30x more iterations at n=14 and stops converging by n~16-18.")


if __name__ == "__main__":
    main()
