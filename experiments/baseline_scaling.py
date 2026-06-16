"""Does growing the observation BASELINE (not just the photon count) change how the
sieving dimension scales -- i.e. can a longer baseline reduce the complexity?

`density_sweep.py` measured `d_sieve ~ N` (slope ~1) by SUBSAMPLING the 70 TOAs at
a FIXED span. FINDINGS.md flagged the untested regime: a longer baseline (bigger
parameter ranges, more wraps), which was unreachable from `data.npy`. The
`model_b_full_timing` generator now reaches it, so we contrast two ways of adding
TOAs under one identical detection methodology:

  A. grow-baseline : span proportional to n (more TOAs over a longer observation)
  B. fixed-span    : span fixed at the largest value, n TOAs subsampled from it
                     (this reproduces density_sweep's setup)

Both regimes coincide at the largest n. `d_sieve` is the minimum sieving dimension
at which the *held-out* verify-set Q exceeds the detection threshold (verify-set Q
is robust to lattice-set overfitting). If regime A has a smaller slope than B, a
longer baseline genuinely lowers the per-TOA sieving cost.

Run (g6k env active):  PYTHONPATH=. python experiments/baseline_scaling.py
"""

import numpy as np

from fermi_fold import fold
import model_b_full_timing as B

DETECT_Q = 50.0
KSTD = 1e3
Q_MUL = 10 ** 15          # well above the precision floor for these baselines
PHASE_STD = 0.03
SPAN_PER_TOA = 2e8 / 30   # baseline added per TOA in regime A
N_VALUES = (24, 32, 40, 48, 56)


def measure_d_sieve(data, retries=4, trials=2):
    """Minimum sieving dimension whose best *reasonable* solution clears the
    verify-set Q threshold. None if undetected up to full dimension."""
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


def run(regime, n, span_fixed):
    span = span_fixed if regime == "fixed-span" else SPAN_PER_TOA * n
    data = B.generate_data(n_toas=n, n_verify=400, t_start=2e8, t_span=span,
                           phase_std=PHASE_STD, mul_factor=Q_MUL, seed=0)
    N = data["integer_lattice"].shape[0]
    return N, measure_d_sieve(data)


def main():
    span_fixed = SPAN_PER_TOA * max(N_VALUES)     # both regimes coincide at max n
    print(f"detection: verify-set Q>{DETECT_Q}; q={Q_MUL:.0e}; sig(pulse)={PHASE_STD}")
    print(f"{'n':>4} | {'A grow-baseline':>20} | {'B fixed-span':>20}")
    print(f"{'':>4} | {'N  d_sieve  d/N':>20} | {'N  d_sieve  d/N':>20}")
    print("-" * 52)
    A, Bd = [], []
    for n in N_VALUES:
        Na, da = run("grow", n, span_fixed)
        Nb, db = run("fixed-span", n, span_fixed)
        if da:
            A.append((n, da))
        if db:
            Bd.append((n, db))
        fa = f"{Na} {str(da):>5} {da/Na:.2f}" if da else f"{Na}  none   -"
        fb = f"{Nb} {str(db):>5} {db/Nb:.2f}" if db else f"{Nb}  none   -"
        print(f"{n:>4} | {fa:>20} | {fb:>20}", flush=True)

    def slope(pts):
        a = np.array(pts)
        return np.polyfit(a[:, 0], a[:, 1], 1)[0] if len(a) >= 2 else float("nan")

    print(f"\nd_sieve-vs-n slope:  A grow-baseline = {slope(A):.3f}   "
          f"B fixed-span = {slope(Bd):.3f}")
    print("If slope(A) < slope(B), a longer baseline lowers the per-TOA sieving cost\n"
          "(cost ~ 2^(0.36 d_sieve), so a smaller d_sieve slope is a smaller exponent).")


if __name__ == "__main__":
    main()
