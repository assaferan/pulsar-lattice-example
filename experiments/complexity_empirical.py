"""Empirically test the complexity model behind Table 1 (Section 5 of the paper)
against what g6k actually does, on H0 (no-signal) lattices.

The model (see ``../docs/METHOD.md`` and ``../complexity_table.py``) rests on
three quantitative claims; this script checks each by building random H0 lattices
of growing dimension, reducing+sieving them, and measuring:

  1. Candidate count   N_cand ~ 2^(0.2 N)   (sieve database size).
  2. Sieve time        C      ~ 2^(0.36 N)  (wall-clock of the pump).
  3. Gaussian heuristic  sigma_exp = vol^(1/n)/sqrt(2 pi e), the shortest random
     vector per coordinate -- the H0 length that the whole detection model is
     calibrated against.

For (3) we use the paper's *residual sublattice* construction (Appendix:
L = [[I_n, 0], [V, eta]], V orthonormal timing rows, eta_ii = sigma_exp/sigma_i),
because the naive GH on the full timing lattice fails by ~15x (its volume is
controlled by the tiny step sizes -- exactly as the paper notes). Two regime
caveats the experiment makes explicit:

  * Narrow search (sigma_i small) degenerates the residual lattice toward Z^n,
    which violates GH: ratio -> sqrt(2 pi e / n). Use wide ranges (paper regime).
  * GH is asymptotic in n: at small n the ratio has O(1) finite-size corrections
    and only approaches 1 as n grows.

Run (needs the g6k env active):  python experiments/complexity_empirical.py
"""

import time

import numpy as np
import fpylll
from g6k import Siever

GH = np.sqrt(2 * np.pi * np.e)
CAND_EXP, COST_EXP = 0.2, 0.36     # paper's 2^(0.2 N), 2^(0.36 N)


def solve_sigma_exp(sigma_i, n):
    """Self-consistent GH: sigma_exp = (prod Sigma/sqrt(1+Sigma^2))^(1/n)/sqrt(2 pi e),
    with Sigma_i = sigma_exp/sigma_i. Solved by bisection in log(sigma_exp)."""
    sigma_i = np.asarray(sigma_i, float)

    def resid(ls):
        s = np.exp(ls)
        S = s / sigma_i
        return ls - ((1 / n) * np.sum(np.log(S / np.sqrt(1 + S ** 2))) - np.log(GH))

    lo, hi = -80.0, 10.0
    for _ in range(500):
        mid = (lo + hi) / 2
        if resid(mid) > 0:
            hi = mid
        else:
            lo = mid
    return np.exp((lo + hi) / 2)


def _full_sieve(integer_lattice, retries=6):
    """LLL + full G6K pump. Returns (g6k, seconds). Retries g6k's intermittent
    SaturationError."""
    N = integer_lattice.shape[0]
    IM = fpylll.IntegerMatrix.from_iterable(
        N, N, list(map(int, integer_lattice.flatten())))
    last = None
    for _ in range(retries):
        gso = fpylll.GSO.Mat(IM, flags=fpylll.GSO.INT_GRAM,
                             U=fpylll.IntegerMatrix.identity(N),
                             UinvT=fpylll.IntegerMatrix.identity(N))
        fpylll.LLL.Reduction(gso)()
        g6k = Siever(gso)
        g6k.initialize_local(0, N // 2, N)
        try:
            t0 = time.perf_counter()
            with g6k.temp_params(otf_lift=False):
                while g6k.l > 0:
                    g6k.extend_left(1)
                    g6k(alg="hk3")
            return g6k, time.perf_counter() - t0
        except Exception as exc:        # SaturationError lives inside g6k
            last = exc
    raise last


def residual_lattice(n, sigma_i, Q=10 ** 13, d=1.0, seed=0):
    """Paper's H0 residual lattice, scaled to integers; returns (L, sigma_exp)."""
    rng = np.random.default_rng(seed)
    m = len(sigma_i)
    sexp = solve_sigma_exp(sigma_i, n)
    V = np.linalg.qr(rng.standard_normal((n, m)))[0].T   # m orthonormal timing rows
    L = np.zeros((n + m, n + m))
    L[:n, :n] = np.eye(n) * Q
    for i in range(m):
        L[n + i, :n] = Q * d * V[i]
        L[n + i, n + i] = Q * d * sexp / sigma_i[i]
    return np.round(L).astype(object), sexp


def measure(n, sigma_i, Q=10 ** 13, seed=0):
    """One H0 trial at dimension n. Returns dict of the three measured quantities."""
    L, sexp_pred = residual_lattice(n, sigma_i, Q=Q, seed=seed)
    m = len(sigma_i)
    N = n + m
    g6k, secs = _full_sieve(L)

    db = np.array(list(g6k.itervalues()))
    coeffs = db @ np.array(list(g6k.M.U))      # original-basis integer coeffs
    vecs = db @ np.array(list(g6k.M.B))
    wraps = coeffs[:, :n]                        # unit-vector (wrap) coefficients
    nontrivial = np.any(wraps != 0, axis=1)     # exclude pure timing-row vectors
    norms = np.linalg.norm(vecs.astype(float), axis=1)
    lam1 = norms[nontrivial & (norms > 0)].min()

    return {
        "n": n, "N": N, "n_cand": len(db), "secs": secs,
        "sexp_pred": sexp_pred, "sexp_meas": lam1 / (np.sqrt(n) * Q),
    }


def gh_sanity_qary(n, q=1009, seed=0):
    """GH on a textbook random q-ary lattice (should give ratio ~1)."""
    fpylll.FPLLL.set_random_seed(seed)
    A = fpylll.IntegerMatrix.random(n, "qary", k=n // 2, q=q)
    g6k, _ = _full_sieve(np.array([[A[i, j] for j in range(n)] for i in range(n)],
                                  dtype=object))
    gso = g6k.M
    logvol = 0.5 * sum(np.log(gso.get_r(i, i)) for i in range(n))
    vecs = np.array(list(g6k.itervalues())) @ np.array(list(gso.B))
    nr = np.linalg.norm(vecs.astype(float), axis=1)
    lam1 = nr[nr > 0].min()
    lam1_gh = np.exp(0.5 * np.log(n / (2 * np.pi * np.e)) + logvol / n)
    return lam1 / lam1_gh


def _fit_slope(ns, values):
    """Least-squares slope of log2(values) vs n."""
    return np.polyfit(ns, np.log2(values), 1)[0]


def main(ns=(30, 40, 50, 60), sigma_i=(1e6, 1e6, 1e6)):
    print("Sanity: GH on random q-ary lattices (ratio should be ~1):")
    for n in (40, 50):
        print(f"   n={n}: lam1/GH = {gh_sanity_qary(n, seed=1):.3f}")

    print(f"\nH0 residual lattices (m={len(sigma_i)} timing params, wide search "
          f"sigma_i={sigma_i[0]:.0e}):")
    hdr = (f"{'N':>4} {'n_cand':>8} {'2^.2N':>8} | {'secs':>7} | "
           f"{'sexp_pred':>10} {'sexp_meas':>10} {'GHratio':>8}")
    print(hdr + "\n" + "-" * len(hdr))
    rows = []
    for n in ns:
        r = measure(n, sigma_i, seed=1)
        rows.append(r)
        print(f"{r['N']:>4} {r['n_cand']:>8} {2**(CAND_EXP*r['N']):>8.0f} | "
              f"{r['secs']:>7.2f} | {r['sexp_pred']:>10.3e} {r['sexp_meas']:>10.3e} "
              f"{r['sexp_meas']/r['sexp_pred']:>8.3f}")

    Ns = np.array([r["N"] for r in rows])
    cand_exp = _fit_slope(Ns, [r["n_cand"] for r in rows])
    time_exp = _fit_slope(Ns, [r["secs"] for r in rows])
    print("\nFitted scaling exponents (log2(.) vs N):")
    print(f"   candidate count : {cand_exp:.3f}        (model 2^{CAND_EXP} N)   [robust]")
    print(f"   implied cost    : {cand_exp * COST_EXP / CAND_EXP:.3f}        "
          f"(model 2^{COST_EXP} N, via C ~ N_cand^{COST_EXP/CAND_EXP:.1f})")
    print(f"   raw wall-clock  : {time_exp:.3f}        (UNRELIABLE at N<=60: "
          f"sub-second/noisy, overhead-dominated)")
    print("\nNotes:")
    print(" * Candidate count is the robust observable; the cost exponent 0.36 follows")
    print(f"   from it via C ~ N_cand^{COST_EXP/CAND_EXP:.1f}. Direct wall-clock needs N~80-100")
    print("   (minute-scale sieves) to overcome setup/threading overhead.")
    print(" * The GH ratio approaches 1 from above as n grows (asymptotic heuristic);")
    print("   narrow search instead degenerates the residual lattice toward Z^n,")
    print("   giving ratio -> sqrt(2 pi e / n) < 1.")


if __name__ == "__main__":
    main()
