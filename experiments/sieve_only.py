"""How many sieve iterations to find the short vector -- and does the LLL warm-start
or the mod-q wrap shortcut matter -- when we sieve WITHOUT LLL?

The real pipeline always LLL/BKZ-reduces before sieving. This feeds the pure
triple-sieve (`pure_sieve.gauss_sieve`) either an LLL-reduced basis or the *raw*
`integer_lattice`, and counts the samples (iterations) until the first reasonable
vector clears Q>50 on the held-out verify set. Iterations are the pure-sieve
analogue of pump depth / work. It also tests whether mod-q reduction helps the
raw (no-LLL) sieve.

Findings (model_a, clean unless noted):

1. LLL is *essential*, not just a speedup.
       n     LLL+sieve         sieve-only (no LLL)
      14     5,  5,  5         179, 198, 171     (~36x more)
      16    17, 17, 17         X,   191,  X
      18     4,  4,  4         X,   X,    X        (X = not found in budget)
   The raw fold vector is (k, p) with wrap counts k ~ 1e10 -- enormous integer
   coefficients a sieve (only v +/- w +/- u) can't build up. LLL pre-packages them
   into short basis vectors, after which the fold is a small combination.

2. mod-q does NOT help the raw sieve -- it DESTROYS it (the opposite of the bare
   pair-sieve result in `pair_sieve_*`). With mod-q the raw sieve fails even at
   n=14. Diagnostic (n=14): the physical solution is short only via LARGE near-
   cancelling coeffs (|b| ~ 8e11, k-std ~ 1.6e9); mod-q reaches shortness the easy
   way -- wrapping each phase coord to a small residual with k=0 -- so it parks on
   the trivial constant-phase family (|b|~60, k=0) and the physical solution never
   forms. The pair-sieve benefits because its target is pure geometric shortness;
   here it is a physical large-wrap-count solution, which wrapping annihilates.

3. Excluding the wrap rows `q*e_j` (which are *redundant* under mod-q -- each
   reduces to 0) does NOT rescue it (`gauss_sieve(active_rows=...)`): db collapses
   to ~2 with |b|~1, k=0. So the failure is not the wrap-row seeding; mod-q simply
   targets the wrong short-vector class (k=0), independent of how you seed.

4. Lower p (more background) breaks the raw sieve regardless of mod-q.

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


def samples_to_detect(il, n_per, tm, probs, mode, max_samples, seed):
    """Iterations until the first reasonable vector clears Q>50, or None.

    mode : "lll"      -- LLL-reduce first (warm start), then sieve;
           "raw"      -- sieve the raw lattice directly (coeffs in original basis);
           "raw_modq" -- raw sieve with mod-q reduction in pair + triple.
    """
    n = il.shape[0]
    qi = int(il[0, 0])
    state = {"hit": None}

    def hook(pv, pc, p2, db, samples):
        pp = np.array(pc[n_per:], float)
        vf = np.mod(pp @ tm / Q + 0.5, 1) - 0.5
        Qv = np.abs(np.sum(probs * np.exp(2j * np.pi * vf))) ** 2 / np.sum(probs ** 2 / 2)
        if Qv > 50 and np.std(np.array(pc[:n_per], float)) > 1e3:
            state["hit"] = samples
            return False

    kw = dict(triple=True, seed=seed, max_samples=max_samples, move_hook=hook)
    if mode == "lll":
        B, U = pure_lll(il, 0.99)
        ps.gauss_sieve(B, seed_coef=U, **kw)
    else:
        kw["seed_coef"] = np.eye(n, dtype=object)        # coeffs in original basis
        if mode in ("raw_modq", "raw_modq_noqv"):
            kw["mod_q"] = (n_per, qi)
        if mode == "raw_modq_noqv":                      # drop the dead q*e_j wrap rows
            kw["active_rows"] = range(n_per, n)
        ps.gauss_sieve(il, **kw)
    return state["hit"]


def kstd_diagnostic(il, n_per, tm, probs, use_modq, noqv=False, max_samples=4000, seed=1):
    """Why mod-q breaks the raw sieve: report db size, max Q, the wrap-count
    spread (k-std) and the largest timing coeff |b| it leaves behind. ``noqv``
    additionally drops the wrap rows from the seed/sampler."""
    n = il.shape[0]
    qi = int(il[0, 0])
    kw = dict(triple=True, seed=seed, max_samples=max_samples,
              seed_coef=np.eye(n, dtype=object))
    if use_modq:
        kw["mod_q"] = (n_per, qi)
    if noqv:
        kw["active_rows"] = range(n_per, n)
    out = ps.gauss_sieve(il, **kw)
    co = np.array([c for _, c in out], dtype=object)
    pp = co[:, n_per:].astype(float)
    vf = np.mod(pp @ tm / Q + 0.5, 1) - 0.5
    Qv = np.abs(np.sum(probs * np.exp(2j * np.pi * vf), axis=1)) ** 2 / np.sum(probs ** 2 / 2)
    kstd = np.std(co[:, :n_per].astype(float), axis=1)
    bmax = float(np.max(np.abs(pp))) if pp.size else 0.0
    n_phys = int(((Qv > 50) & (kstd > 1e3)).sum())
    return len(out), float(Qv.max()), float(np.median(kstd)), bmax, n_phys


def main(seeds=(1, 2), lll_budget=20000, raw_budget=20000):
    fmt = lambda x: str(x) if x is not None else "X"

    print("[1] iterations to detection: LLL+sieve vs sieve-only (no LLL)")
    print(f"   {'n':>4} {'LLL+sieve':>14} {'sieve-only':>18}")
    for n in (14, 16, 18):
        il, n_per, tm, probs = mixed_lattice(n, 1.0, seed=0)
        a = [samples_to_detect(il, n_per, tm, probs, "lll", lll_budget, s) for s in seeds]
        b = [samples_to_detect(il, n_per, tm, probs, "raw", raw_budget, s) for s in seeds]
        print(f"   {n:>4} {str([fmt(v) for v in a]):>14} {str([fmt(v) for v in b]):>18}",
              flush=True)

    print("\n[2] sieve-only: does mod-q help? (no -- it destroys detection)")
    print(f"   {'n':>4} {'no mod-q':>14} {'with mod-q':>14}")
    for n in (14, 16):
        il, n_per, tm, probs = mixed_lattice(n, 1.0, seed=0)
        a = [samples_to_detect(il, n_per, tm, probs, "raw", raw_budget, s) for s in seeds]
        b = [samples_to_detect(il, n_per, tm, probs, "raw_modq", raw_budget, s) for s in seeds]
        print(f"   {n:>4} {str([fmt(v) for v in a]):>14} {str([fmt(v) for v in b]):>14}",
              flush=True)

    print("\n[3] why mod-q breaks it (n=14): the physical solution needs LARGE near-"
          "cancelling\n    coeffs (big |b|, big k-std); mod-q wraps to k=0 instead. "
          "Dropping the dead\n    wrap rows (noqv) doesn't help -- it starves the db further.")
    il, n_per, tm, probs = mixed_lattice(14, 1.0, seed=0)
    for label, modq, noqv in [("raw", False, False), ("modq", True, False),
                              ("modq_noqv", True, True)]:
        db, mq, mks, bmax, nphys = kstd_diagnostic(il, n_per, tm, probs, modq, noqv=noqv)
        print(f"   {label:>10}: db={db:>3} maxQ={mq:>5.0f} median_k-std={mks:.1e} "
              f"max|b|={bmax:.1e} #physical={nphys}", flush=True)

    print("\n[4] excluding the q*e_j wrap rows under mod-q does NOT rescue detection")
    print(f"   {'n':>4} {'mod-q':>14} {'mod-q (no wrap rows)':>22}")
    for n in (14, 16):
        il, n_per, tm, probs = mixed_lattice(n, 1.0, seed=0)
        a = [samples_to_detect(il, n_per, tm, probs, "raw_modq", raw_budget, s) for s in seeds]
        b = [samples_to_detect(il, n_per, tm, probs, "raw_modq_noqv", raw_budget, s) for s in seeds]
        print(f"   {n:>4} {str([fmt(v) for v in a]):>14} {str([fmt(v) for v in b]):>22}",
              flush=True)

    print("\n[5] lower association probability p (n=14): breaks regardless of mod-q")
    print(f"   {'p':>4} {'no mod-q':>14} {'with mod-q':>14}")
    for p in (1.0, 0.8, 0.6):
        il, n_per, tm, probs = mixed_lattice(14, p, seed=0)
        a = [samples_to_detect(il, n_per, tm, probs, "raw", raw_budget, s) for s in seeds]
        b = [samples_to_detect(il, n_per, tm, probs, "raw_modq", raw_budget, s) for s in seeds]
        print(f"   {p:>4.1f} {str([fmt(v) for v in a]):>14} {str([fmt(v) for v in b]):>14}",
              flush=True)

    print("\n=> LLL warm-start is essential (sieve builds k~1e10 only via LLL's prepackaging);\n"
          "   mod-q HURTS the raw sieve by wrapping away the large wrap counts the physical\n"
          "   solution needs -- and dropping the (redundant) wrap rows doesn't rescue it:\n"
          "   mod-q targets the wrong short-vector class (k=0), not the wrap rows per se.")


if __name__ == "__main__":
    main()
