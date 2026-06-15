"""Reproduce Table 1 (the lattice-sieve complexity) from Section 5 of

    Gazith, Pearlman & Zackay (2025), ApJ 979, 48; arXiv:2402.07228,
    "Recovering Pulsar Periodicity ... by Finding the Shortest Vector in a Lattice".

See ``docs/METHOD.md`` for the surrounding method.  This script rebuilds the
complexity estimate from the underlying model and compares it to the published
table.

The model
---------
1. **Pulse width <-> association probability** (Table 1 footnote).  A photon is
   from the pulsar with probability ``p`` (phase scatter ``sigma_int``) or is
   background with probability ``1 - p`` (uniform phase, variance ``1/12``):

       sigma**2 = p * sigma_int**2 + (1 - p) / 12.

   For an intrinsically infinitely narrow pulse (``sigma_int = 0``):

       p = 1 - 12 * sigma**2.

2. **Gaussian heuristic (GH).**  The shortest non-trivial vector of an
   n-dimensional lattice has per-coordinate length

       sigma_exp = vol(L)**(1/n) / sqrt(2*pi*e).

   With the "entropy" / number of independent options ``Lambda = 1 / vol(L)``,

       sigma_exp = Lambda**(-1/n) / sqrt(2*pi*e).

3. **Detecting a non-shortest solution.**  The true solution has per-coordinate
   length ``sigma`` (the pulse width).  The number of spurious lattice points
   shorter than it is ``(sigma / sigma_exp)**n`` (the false-alarm count).  The
   sieve yields ``N_cand ~ 2**(0.2 n)`` candidates at time cost
   ``C ~ 2**(0.36 n)`` (Ducas et al. 2021, measured for n ~ 100).  The solution
   is recovered once it is among the generated candidates:

       (sigma / sigma_exp)**n <= N_cand = 2**(0.2 n).

4. **Threshold (minimum data n).**  At equality, taking the n-th root removes n:

       sigma / sigma_exp = 2**0.2
       => sigma * sqrt(2*pi*e) * Lambda**(1/n) = 2**0.2
       => n = ln(Lambda) / ln( 2**0.2 / (sigma * sqrt(2*pi*e)) ).

   The cost then follows, and writing ``C = Lambda**a``:

       C = 2**(0.36 n),   a = 0.36 * n * ln(2) / ln(Lambda).
"""

import numpy as np

# Sieve scaling measured by Ducas et al. (2021) for n ~ 100.
CAND_EXP = 0.2      # N_candidates ~ 2**(CAND_EXP * n)
COST_EXP = 0.36     # time cost    ~ 2**(COST_EXP * n)
GH = np.sqrt(2 * np.pi * np.e)   # Gaussian-heuristic per-coordinate factor

# Published Table 1: (sigma, p, a in C=Lambda**a, log10 C at Lambda=1e28, n at 1e28).
PAPER_TABLE = [
    (0.20, 0.50, 0.81, 22.5, 207),
    (0.16, 0.70, 0.44, 12.0, 111),
    (0.11, 0.85, 0.27,  7.5,  69),
    (0.06, 0.95, 0.17,  4.7,  42),
]


def prob_from_sigma(sigma, sigma_int=0.0):
    """Association probability p for a pulse width sigma (Table 1 footnote).

    Inverting  sigma**2 = p*sigma_int**2 + (1 - p)/12  gives
    p = (sigma**2 - 1/12) / (sigma_int**2 - 1/12); at sigma_int = 0 this is
    p = 1 - 12*sigma**2.
    """
    return (sigma ** 2 - 1.0 / 12.0) / (sigma_int ** 2 - 1.0 / 12.0)


def sigma_from_prob(p):
    """Inverse of prob_from_sigma at sigma_int = 0:  sigma = sqrt((1 - p)/12)."""
    return np.sqrt((1.0 - p) / 12.0)


def required_dimension(sigma, Lambda):
    """Minimum lattice dimension n to recover the solution (model step 4)."""
    b = 2 ** CAND_EXP / (sigma * GH)
    return np.log(Lambda) / np.log(b)


def cost_exponent(sigma):
    """Exponent a such that the time cost is C = Lambda**a (Lambda-independent)."""
    b = 2 ** CAND_EXP / (sigma * GH)
    return COST_EXP * np.log(2) / np.log(b)


def complexity(sigma, Lambda):
    """Return (n, log10 C, a) for a pulse width sigma and entropy Lambda."""
    n = required_dimension(sigma, Lambda)
    log10C = COST_EXP * n * np.log10(2.0)
    return n, log10C, cost_exponent(sigma)


def main(Lambda=1e28):
    print(f"Reproducing Table 1 at Lambda = {Lambda:.0e}\n")

    print("Column 2  (p = 1 - 12 sigma^2, sigma_int = 0):")
    for sigma, p_pap, *_ in PAPER_TABLE:
        print(f"  sigma={sigma:.2f}  ->  p={1 - 12 * sigma**2:.3f}   (paper {p_pap})")

    print("\nColumns 3-5 (sigma = sqrt((1 - p)/12) from the round p values):")
    hdr = f"{'p':>5} {'sigma':>7} | {'n':>6} {'n_pap':>6} | {'a':>6} {'a_pap':>6} | {'log10C':>7} {'pap':>5}"
    print(hdr)
    print("-" * len(hdr))
    for sigma_pap, p, a_pap, logC_pap, n_pap in PAPER_TABLE:
        sigma = sigma_from_prob(p)
        n, logC, a = complexity(sigma, Lambda)
        print(f"{p:>5.2f} {sigma:>7.4f} | {n:>6.1f} {n_pap:>6d} | "
              f"{a:>6.3f} {a_pap:>6.2f} | {logC:>7.1f} {logC_pap:>5.1f}")

    print("\nThe cost exponent a matches the paper to ~0.01; n agrees to within a\n"
          "few. See diagnostics() for where that residual comes from.")


def diagnostics(Lambda=1e28):
    """Trace the small residual between this reproduction and the paper's n.

    Conclusion: it is NOT a missing analytic term. The sqrt(1+Sigma^2) and
    (n - m) corrections are absorbed into Lambda and are numerically ~1 here, so
    the exact sigma_exp equals Lambda^(-1/n)/sqrt(2 pi e) (already used). The
    residual is the empirical sieve constants (calibrated for n ~ 100) plus the
    table's own ~1% rounding.
    """
    lnL = np.log(Lambda)

    print("\n[1] The paper's own table is internally consistent only to ~1%")
    print("    (a implied by the listed n,  a = 0.36 n log2 / log Lambda):")
    for _, p, a_pap, _, n_pap in PAPER_TABLE:
        a_from_n = COST_EXP * n_pap * np.log10(2) / np.log10(Lambda)
        print(f"      p={p:.2f}: n={n_pap} -> a={a_from_n:.3f}  (listed {a_pap})")

    print("\n[2] The sqrt(1+Sigma^2)/(n-m) corrections are negligible: with search")
    print("    ranges sigma_i >> sigma_exp, Sigma_i = sigma_exp/sigma_i -> 0, so")
    print("    prod sqrt(1+Sigma_i^2) -> 1 and vol (hence sigma_exp) is unchanged:")
    for m, smax in [(2, 1e3), (7, 1e3)]:
        Sig = (0.05 / smax) * np.ones(m)     # sigma_exp~0.05, sigma_i~1e3 (very tight ranges)
        print(f"      m={m}, sigma_i~{smax:.0e}: prod sqrt(1+Sigma^2)="
              f"{np.prod(np.sqrt(1+Sig**2)):.8f}  (timing ranges are far larger still)")

    print("\n[3] sigma/sigma_exp at the paper's (n, sigma) is NOT the constant 2^0.2")
    print(f"    (= {2**CAND_EXP:.4f}); the effective candidate exponent c drifts as n")
    print("    leaves the n~100 calibration point:")
    for _, p, _, _, n_pap in PAPER_TABLE:
        sigma = sigma_from_prob(p)
        sexp = np.exp(-lnL / n_pap) / GH
        c = np.log2(np.exp(lnL / n_pap) * sigma * GH)
        print(f"      p={p:.2f} (n={n_pap}): sigma/sigma_exp={sigma/sexp:.4f},  c={c:.3f}")

    print("\n=> The residual is the empirical sieve scaling (0.2, 0.36 for n~100),")
    print("   not a dropped analytic correction.")


if __name__ == "__main__":
    main()
    diagnostics()
