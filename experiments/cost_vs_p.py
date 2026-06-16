"""Empirical check: how much does decreasing the association probability p cost
the pump?

Theory (docs/METHOD.md): lower p widens the effective pulse, sigma^2=(1-p)/12,
which raises the search dimension n = lnLambda / D with D = -1.280 - ln(sigma),
and the pump costs 2^(0.36 n). So the *minimum dimension to detect* should grow
like 1/D as sigma grows (p falls).

We measure n_threshold(sigma) directly: for each sigma, grow the number of TOAs
until the pump detects (max Q over physical solutions > 50) over >=2/3 seeds.
Uses Model A (2 params -> small fast lattices); the sigma-dependence is model
independent.

Run from the repo root with the g6k env active:
    PYTHONPATH=. python3 experiments/cost_vs_p.py
"""
import numpy as np
from fermi_fold import fold
import model_a_constant_frequency as A

GH = np.sqrt(2 * np.pi * np.e)
def D_of(sigma):
    return 0.2 * np.log(2) - np.log(sigma * GH)

SIGMAS = [0.04, 0.08, 0.12, 0.16, 0.20]
N_GRID = [12, 16, 20, 24, 28, 32, 38, 44, 50, 56, 62]
SEEDS = 3
QTHRESH = 50.0


def detects(sigma, n_toas, seed):
    data = A.generate_data(n_toas=n_toas, phase_std=sigma, seed=seed, n_verify=400)
    try:
        r = fold(data, reasonable_k_std=1e3)
    except Exception:
        return False
    m = r["reasonable_solutions_mask"]; q = r["Q_stat"][m]
    return bool(q.size and q.max() > QTHRESH)


print(f"{'p':>5} {'sigma':>6} {'n_thresh':>9} {'pump cost 2^(0.36n)':>20} {'1/D (predicted)':>16}", flush=True)
ref = None
for sigma in SIGMAS:
    p = 1 - 12 * sigma ** 2
    nthr = None
    for n in N_GRID:
        hits = sum(detects(sigma, n, s) for s in range(SEEDS))
        if hits >= 2:
            nthr = n + 2          # lattice dimension = n_toas + p_params(2)
            break
    invD = 1.0 / D_of(sigma)
    if ref is None and nthr is not None:
        ref = (nthr, invD)
    if nthr is None:
        print(f"{p:>5.2f} {sigma:>6.3f} {'>64':>9} {'--':>20} {invD:>16.2f}", flush=True)
    else:
        cost = 2 ** (0.36 * nthr)
        print(f"{p:>5.2f} {sigma:>6.3f} {nthr:>9} {cost:>20.3e} {invD:>16.2f}", flush=True)

if ref:
    print(f"\nIf n_threshold ~ 1/D, the n_thresh column and the 1/D column should be"
          f"\nproportional (ratio ~{ref[0]/ref[1]:.1f}). Pump cost = 2^(0.36 n_thresh).", flush=True)
