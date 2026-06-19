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


def _modelA_lattice(dt, q, f_prior, phi_prior, res_frac=0.01):
    """The 2-parameter (phi, f) q-ary lattice for a TOA subset, with design
    Phi = phi*1 + f*dt. Returns (integer_lattice, steps). A short vector folds the
    photons (q*K_i + param phase ~ 0) while a diagonal prior keeps the params
    within ``[phi_prior, f_prior]`` (the prior just needs the right *scale*)."""
    dt = np.asarray(dt, float)
    n = len(dt)
    M = np.vstack([np.ones(n), dt])                 # (2, n): dPhi/dphi, dPhi/df
    steps = res_frac / np.max(np.abs(M), axis=1)
    cstd = 10.0 * np.array([phi_prior, f_prior])    # loose prior std per param
    D = n + 2
    il = np.zeros((D, D), dtype=object)
    for j in range(n):
        il[j, j] = int(q)                           # wrap rows q*e_j
    for i in range(2):                              # parameter rows
        for j in range(n):
            il[n + i, j] = int(round(q * steps[i] * M[i, j]))
        il[n + i, n + i] = int(round(q * steps[i] / cstd[i]))
    return il, steps


def recover_frequency_modelA(toas, sigma, f_prior=50.0, phi_prior=1.0, t_ref=None,
                             mul_factor=10 ** 15, n_sub=None, tries=28, retries=4,
                             conf=40.0, seed=0):
    """Recover the spin frequency f of a constant-frequency (Model A) pulsar from
    arrival times, where ``Phi(t) = phi + f*(t - t_ref)`` is near-integer for the
    true (phi, f).

    Extends :func:`recover_frequency` to two parameters so the phase offset
    ``phi`` is modelled (a constant offset otherwise stops the f-solution being
    the shortest vector). Per round: LLL on ``tries`` random ``n_sub``-TOA subsets
    of the 2-parameter lattice proposes candidate f's; each is checked
    harmonically (``f/k``) by coherent power on the *full* set (phi-invariant,
    background-robust); rounds repeat until coherence exceeds ``conf``.

    Parameters
    ----------
    toas, sigma : arrival times and per-sample phase noise (the eps scale).
    f_prior, phi_prior : rough parameter *scales* for the loose lattice prior
        (only the order of magnitude matters).
    conf : coherent-power threshold for a confident hit (~40 for ~100 TOAs).
    retries : number of fresh-subset rounds before giving up.

    Returns
    -------
    (f_hat, coherence). ``coherence > conf`` is a confident recovery; a low
    coherence means f could not be recovered (escalate to the full sieve).
    """
    toas = np.asarray(toas, float)
    if t_ref is None:
        t_ref = 0.5 * (toas.min() + toas.max())
    dt = toas - t_ref
    n = len(toas)
    if n_sub is None:
        n_sub = max(8, int(0.9 / sigma ** 2))
    n_sub = min(n_sub, n)
    q = int(mul_factor)
    prior_f = int(round(q * (0.01 / np.max(np.abs(dt))) / (10.0 * f_prior)))

    best_f, best_c = 0.0, -1.0
    for r in range(retries):
        rng = np.random.default_rng(seed + r)
        raw = set()
        for _ in range(tries):
            idx = np.sort(rng.choice(n, n_sub, replace=False))
            il, steps = _modelA_lattice(dt[idx], q, f_prior, phi_prior)
            D = il.shape[0]
            IM = fpylll.IntegerMatrix.from_iterable(D, D, [int(x) for row in il for x in row])
            fpylll.LLL.reduction(IM)
            pf = int(round(q * steps[1] / (10.0 * f_prior)))
            rows = sorted(([IM[i, j] for j in range(D)] for i in range(D)),
                          key=lambda rr: sum(x * x for x in rr))
            rows = [rr for rr in rows if any(rr)][:4]
            for rr in rows:
                if pf:
                    f = abs(round(rr[n_sub + 1] / pf) * steps[1])
                    if 1 < f < 1e6:
                        raw.add(round(f, 6))
        for f0 in raw:                              # harmonic-aware verification
            for k in range(1, 7):
                f = f0 / k
                if f >= 1:
                    c = coherent_power(dt, f)
                    if c > best_c:
                        best_f, best_c = f, c
        if best_c > conf:                           # confident -- stop early
            break
    return best_f, best_c


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

    # Model A demo: constant-frequency pulsar with phase offset + background.
    phi_true, t_ref, t_span, sig_int = 0.3, 3.5e8, 3.0e8, 0.03

    def _modelA_toas(n, p, seed):
        rng = np.random.default_rng(seed)
        n_bg = int(round((1 - p) * n))
        k_max = int(f_true * t_span / 2)
        K = rng.integers(-k_max, k_max, size=n - n_bg)
        eps = rng.normal(0, sig_int, size=n - n_bg)
        sig_t = t_ref + (K + eps - phi_true) / f_true
        bg_t = (t_ref - t_span / 2) + t_span * rng.random(n_bg)
        return np.sort(np.concatenate([sig_t, bg_t]))

    print(f"\nrecover_frequency_modelA (phi={phi_true}, retry-until-confident), true f = {f_true:g}")
    for p, sigma in ((1.0, 0.03), (0.7, 0.16)):
        ok = 0
        for s in range(10):
            f_hat, c = recover_frequency_modelA(_modelA_toas(111, p, s), sigma, seed=s)
            ok += (abs(f_hat - f_true) < 0.5)
        print(f"  p={p:.1f} sigma={sigma:.2f}: recovered {ok:>2}/10")
