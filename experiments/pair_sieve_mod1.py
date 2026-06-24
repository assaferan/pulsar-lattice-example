"""Minimal database size L for a pair-sieve to reach a target length -- with and
without mod-1 reduction. (Reproducing a collaborator's experiment.)

Setup:
  * a bank of L random vectors in R^n, each coordinate ~ Uniform(-1/2, 1/2);
  * pair reduction: for each v, pick the w minimising |v +/- w| and replace v by
    it if that shortens v (single +/-1 combination -- the Gauss/2-reduction move);
  * iterate to saturation (no reducing pair left), and call the run *viable* if
    the saturated mean squared length reaches the target
        target = n * (0.5 * (1/12) + 0.5 * sigma^2)
    i.e. the p=0.5 mix of the uniform background (per-coord var 1/12) and a
    signal of per-coord width sigma (sigma small -> target ~ n/24).
  * For each n there is a minimal L for viability; we measure L_min(n).

The question: does reducing v +/- w **mod 1** (wrapping each coordinate back into
[-1/2, 1/2]) lower L_min?  mod-1 models the q-ary (Z^n) wrap with q=1.

NOTE -- interpretation choices (flag for the collaborator): target uses
per-coordinate uniform variance 1/12; viability = saturated *mean* |v|^2 <=
target; reduction is a single +/-1 combination applied to all v per round.

Run:  PYTHONPATH=. python3 experiments/pair_sieve_mod1.py
"""
import numpy as np

SIGMA = 0.03
SEEDS = 7
N_GRID = [8, 10, 12, 14, 16, 18, 20, 22, 24]
L_GRID = [10, 14, 20, 28, 40, 56, 80, 113, 160, 226, 320, 452, 640, 905, 1280]
MAX_ROUNDS = 400


def sieve_round(DB, mod1):
    norms = (DB * DB).sum(1)
    G = DB @ DB.T
    s = norms[:, None] + norms[None, :]
    plus, minus = s + 2 * G, s - 2 * G
    np.fill_diagonal(plus, np.inf)
    np.fill_diagonal(minus, np.inf)
    jp, jm = plus.argmin(1), minus.argmin(1)
    bp = plus[np.arange(len(DB)), jp]
    bm = minus[np.arange(len(DB)), jm]
    use_m = bm < bp
    j = np.where(use_m, jm, jp)
    best = np.where(use_m, bm, bp)
    sign = np.where(use_m, -1.0, 1.0)
    imp = best < norms - 1e-12
    new = DB.copy()
    new[imp] = DB[imp] + sign[imp, None] * DB[j[imp]]
    if mod1:
        new -= np.round(new)
    return new, int(imp.sum())


def viable(n, L, mod1, sigma, seed):
    rng = np.random.default_rng(seed)
    DB = rng.uniform(-0.5, 0.5, (L, n))
    target = n * (0.5 * (1.0 / 12) + 0.5 * sigma ** 2)
    for _ in range(MAX_ROUNDS):
        DB, nimp = sieve_round(DB, mod1)
        if (DB * DB).sum(1).mean() <= target:
            return True
        if nimp == 0:
            return False
    return False


def L_min(n, mod1, sigma):
    for L in L_GRID:
        hits = sum(viable(n, L, mod1, sigma, s) for s in range(SEEDS))
        if hits > SEEDS // 2:
            return L
    return None


if __name__ == "__main__":
    print(f"target = n*(0.5/12 + 0.5*{SIGMA}^2);  L_min = smallest L viable in "
          f">{SEEDS//2}/{SEEDS} seeds\n", flush=True)
    print(f"{'n':>4} {'L_min(no mod1)':>15} {'L_min(mod1)':>13} {'ratio':>7}", flush=True)
    for n in N_GRID:
        a = L_min(n, False, SIGMA)
        b = L_min(n, True, SIGMA)
        ratio = (a / b) if (a and b) else float('nan')
        print(f"{n:>4} {str(a):>15} {str(b):>13} {ratio:>7.2f}", flush=True)
