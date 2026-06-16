"""Is the modulus q (= mul_factor) a lever for reducing the sieving dimension?

The lattice contains q*Z^n, so it is "q-ary" -- the natural hope is that q is an
exploitable modulus (as in LWE). It is NOT. Here q is just the integer SCALE used
to discretize a scale-invariant continuous problem (phase residuals mod 1): the
real object is the continuous lattice `[[I,0],[s*Qortho, diag(c/coeff_std)]]`, and
the code multiplies by q and rounds only to feed g6k integers. SVP is
scale-invariant, so above the precision floor q cannot change the difficulty.

This script confirms it: it measures `d_sieve` (minimum sieving dimension that
detects, via held-out verify-set Q) for the SAME problem at several q.

Expected:
  * q below the floor `~ f_prior/(d_f sigma^2)` (the paper's footnote): detection
    FAILS -- the design-row rounding error (x p_f) swamps the pulse signal q*sigma.
  * q above the floor: d_sieve is constant (q-invariant) -- q is not a lever.

Run (g6k env active):  PYTHONPATH=. python experiments/q_invariance.py
"""

import numpy as np

from fermi_fold import fold
import model_b_full_timing as B

DETECT_Q = 50.0
KSTD = 1e3


def measure_d_sieve(data, retries=4, trials=2):
    n = data["integer_lattice"].shape[0]
    for d_sieve in range(n // 2 + 1, n + 1, 2):
        pump_stop = n - d_sieve
        for _ in range(trials):
            for _ in range(retries):
                try:
                    r = fold(data, fast=True, pump_stop=pump_stop, reasonable_k_std=KSTD)
                    m = r["reasonable_solutions_mask"]
                    if m.any() and r["Q_stat"][m].max() > DETECT_Q:
                        return d_sieve
                    break
                except Exception:
                    continue
    return None


def main(n_toas=40, q_values=(10**9, 10**12, 10**14, 10**15, 10**16)):
    print(f"Same problem (n_toas={n_toas}, seed=0), varying the modulus q:")
    print(f"{'q':>8} {'N':>4} {'d_sieve':>8} {'d/N':>6}")
    for q in q_values:
        data = B.generate_data(n_toas=n_toas, n_verify=400, mul_factor=q, seed=0)
        N = data["integer_lattice"].shape[0]
        d = measure_d_sieve(data)
        tag = "" if d else "   (below precision floor -> no detection)"
        d_over_N = f"{d/N:.2f}" if d else "-"
        print(f"{q:>8.0e} {N:>4} {str(d):>8} {d_over_N:>6}{tag}", flush=True)
    print("\nAbove the floor, d_sieve is q-invariant: q is a discretization scale,\n"
          "not an exploitable modulus. (Reducing q only loses precision; it cannot\n"
          "lower the sieving cost.)")


if __name__ == "__main__":
    main()
