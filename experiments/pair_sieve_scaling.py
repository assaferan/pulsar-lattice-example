"""Is the mod-1 advantage in the pair-sieve a constant factor or does it GROW?

Extends pair_sieve_mod1.py to larger n with finer L sampling: estimate L_min(n)
by the interpolated 50%-viability crossing (smooth, not grid-quantised), for both
modes, then fit log2(L_min) = slope*n + intercept.

  * equal slopes  -> mod-1 is a CONSTANT factor (2^(intercept gap));
  * smaller mod-1 slope -> mod-1 is an ASYMPTOTIC (exponential) advantage.

Run:  PYTHONPATH=. python3 experiments/pair_sieve_scaling.py
"""
import numpy as np
from experiments.pair_sieve_mod1 import viable

SIGMA = 0.03
SEEDS = 9
N_GRID = [12, 15, 18, 21, 24, 27, 30]
L_GRID = np.unique(np.round(8 * 1.25 ** np.arange(0, 27)).astype(int))


def L_min(n, mod1):
    """Interpolated smallest L with viability rate >= 0.5 (in log L)."""
    Ls, rates, ones = [], [], 0
    for L in L_GRID:
        rate = np.mean([viable(n, int(L), mod1, SIGMA, s) for s in range(SEEDS)])
        Ls.append(float(L)); rates.append(rate)
        ones = ones + 1 if rate >= 1.0 else 0
        if ones >= 2:                       # well past the threshold; stop
            break
    Ls, rates = np.array(Ls), np.array(rates)
    if rates.max() < 0.5:
        return np.nan
    i = int(np.argmax(rates >= 0.5))
    if i == 0:
        return Ls[0]
    r0, r1 = rates[i - 1], rates[i]
    l0, l1 = np.log(Ls[i - 1]), np.log(Ls[i])
    lc = l0 + (0.5 - r0) * (l1 - l0) / (r1 - r0)
    return float(np.exp(lc))


if __name__ == "__main__":
    print(f"sigma={SIGMA}, {SEEDS} seeds, L_min = interpolated 50% viability\n", flush=True)
    print(f"{'n':>4} {'L_min(no mod1)':>15} {'L_min(mod1)':>13} {'ratio':>7}", flush=True)
    a, b = [], []
    for n in N_GRID:
        la = L_min(n, False); lb = L_min(n, True)
        a.append(la); b.append(lb)
        print(f"{n:>4} {la:>15.1f} {lb:>13.1f} {la/lb:>7.2f}", flush=True)

    N = np.array(N_GRID, float)
    sa, ia = np.polyfit(N, np.log2(a), 1)
    sb, ib = np.polyfit(N, np.log2(b), 1)
    print(f"\nlog2 L_min fits:")
    print(f"  no mod1: slope = {sa:.4f} / dim   (L_min ~ 2^({sa:.3f} n))", flush=True)
    print(f"  mod1   : slope = {sb:.4f} / dim   (L_min ~ 2^({sb:.3f} n))", flush=True)
    print(f"\n  slope ratio mod1/noMod1 = {sb/sa:.3f}", flush=True)
    if sb < sa - 0.01:
        print("  => mod-1 has a SMALLER exponent: a GROWING (asymptotic) advantage.", flush=True)
    else:
        print(f"  => same exponent within noise: mod-1 is a CONSTANT factor "
              f"~2^({ia-ib:.2f}) = {2**(ia-ib):.1f}x.", flush=True)
