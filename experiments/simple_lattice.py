"""Recover a single frequency f from v_i = (K_i + eps_i)/f (eps_i ~ N(0, sigma)).

`f` is the shared multiplier that makes `f*v` simultaneously near-integer -- a
simultaneous-Diophantine / shared-denominator problem. `recover_frequency`
solves it with subset-LLL + coherent verification (no sieving). See the module
self-test for a vec_toas(111, 0.16) demo.
"""
import numpy as np
import fpylll
from numpy.random import Generator, PCG64


def vec_toas(n, sigma, f_true=50.0, t_span=3.0e8, seed=None):
    k_max = int(f_true * t_span / 2)
    rng = Generator(PCG64(seed))
    Ks = rng.integers(-k_max, k_max, size=n)
    eps = rng.normal(scale=sigma, size=n)
    v = (Ks + eps) / f_true
    return v


def _simul_diophantine_lll(v, weight):
    """Shortest vector of the simultaneous-Diophantine lattice of ``v``; its first
    coordinate is the candidate multiplier f (smallest a with a*v ~ integer).

    A lattice point ``a*row0 - sum_i K_i*row_i`` equals
    ``(a, weight*(a*v_1 - K_1), ...)`` -- short when ``a*v`` is near-integer. This
    is the shortest vector only while ``len(v)*sigma^2 < 1`` (else a single unit
    wrap is shorter), which is why the caller subsamples to ~1/sigma^2 coords.
    """
    n = len(v)
    D = n + 1
    M = fpylll.IntegerMatrix(D, D)
    M[0, 0] = 1
    for i in range(n):
        M[0, i + 1] = int(round(weight * v[i]))
    for j in range(n):
        M[j + 1, j + 1] = int(weight)
    fpylll.LLL.reduction(M)
    rows = [[M[i, j] for j in range(D)] for i in range(D)]
    rows = [r for r in rows if any(r)]
    rows.sort(key=lambda r: sum(x * x for x in r))
    return abs(rows[0][0])


def coherent_power(v, f):
    """|sum exp(2*pi*i f v_j)|^2 / n -- ~n for the true f, ~1 for a wrong one."""
    v = np.asarray(v, float)
    return float(np.abs(np.sum(np.exp(2j * np.pi * f * v))) ** 2 / len(v))


def recover_frequency(v, sigma, n_sub=None, tries=16, weight=10 ** 8, seed=0):
    """Recover the multiplier f such that ``f*v`` is simultaneously near-integer.

    Why subsetting: with all coordinates the f-solution is the shortest lattice
    vector only while ``n*sigma^2 < 1``; past that a unit wrap is shorter and LLL
    misses it. So we run LLL on random subsets of ``n_sub ~ 1/sigma^2`` coords
    (where it *is* shortest), then verify each candidate's coherent power on the
    *full* vector and keep the best -- recovering the SNR the subset gave up.

    Parameters
    ----------
    v : array of the noisy samples ``(K_i + eps_i)/f``.
    sigma : the per-sample noise on ``f*v`` (the eps scale).
    n_sub : coords per LLL subset (default ``min(len(v), int(0.9/sigma**2))``).
    tries : number of random subsets to try (ignored if ``n_sub == len(v)``).
    weight : LLL lattice weight (large; default 1e8).

    Returns
    -------
    (f_hat, coherence) -- the recovered multiplier and its coherent power on the
    full vector (high == confident; ~1 means failure / no signal).
    """
    v = np.asarray(v, float)
    n = len(v)
    if n_sub is None:
        n_sub = max(8, int(0.9 / sigma ** 2))
    n_sub = min(n_sub, n)
    rng = np.random.default_rng(seed)

    best_f, best_power = 0, -1.0
    for _ in range(tries):
        idx = np.arange(n) if n_sub == n else rng.choice(n, n_sub, replace=False)
        f = _simul_diophantine_lll(v[idx], weight)
        if f != 0:
            p = coherent_power(v, f)
            if p > best_power:
                best_f, best_power = f, p
        if n_sub == n:                 # deterministic -- one pass suffices
            break
    return best_f, best_power


if __name__ == "__main__":
    f_true = 50.0
    print(f"recover_frequency on vec_toas(111, sigma), true f = {f_true:g}")
    for sigma in (0.05, 0.10, 0.16, 0.20):
        ok = 0
        for s in range(10):
            v = vec_toas(111, sigma, f_true=f_true, seed=s)
            f_hat, power = recover_frequency(v, sigma, seed=s)
            ok += (f_hat == round(f_true))
        nsub = min(111, max(8, int(0.9 / sigma ** 2)))
        print(f"  sigma={sigma:.2f}  n_sub={nsub:>3}  recovered {ok:>2}/10")
