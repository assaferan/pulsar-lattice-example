"""Baby Model A: the standard basis q*Z^N plus k parameter vectors.

The simplest controlled setting for the mod-q / shortest-vector question. The
lattice is the wrap lattice ``q*Z^N`` (the "standard basis") plus ``k`` parameter
rows (a truncated, FREQUENCY-FIRST Taylor phase model: k=1 frequency only, k=2
phi+f = Model A, k>=3 toward Model B). A planted pulsar makes the phase
near-integer for the true parameters; a fraction ``1-p`` of background photons
(uniform phase) can be mixed in.

Two things this model makes exact:

* **Shortest-vector sense.** With **k=1** the wraps are slaved to one multiplier,
  so the shortest lattice vector IS the mod-q reduction of the best integer
  multiple of that vector -- the sense in which LLL == mod-q. mod-q reduction
  handles the N-dimensional wrap part deterministically; what remains is a
  k-dimensional search over parameter multiples. Run ``__main__`` part 1.
* **Where it breaks.** At **k>=2** the planted signal stops being the shortest
  vector (the phi row introduces a trivial constant-phase short vector), so
  detection needs verification + the physical mask, not shortness. Part 2 folds
  the verification Q in and sweeps the background fraction p, with vs without
  mod-q sieving.

Run:  PYTHONPATH=.:experiments python3 experiments/baby_model.py
"""
import math

import numpy as np
import fpylll

Q = 10 ** 15


def _design(dt, k):
    """Phase design, FREQUENCY-FIRST so k=1 is the single-frequency model (the
    one with large wraps): rows = [f:dt, phi:1, fdot:dt^2/2, ...][:k]."""
    terms = [dt, np.ones_like(dt)] + [dt ** i / math.factorial(i) for i in range(2, 8)]
    return np.vstack(terms[:k])                          # (k, N)


def _dphase(dt, theta):
    """d(phase)/d(dt) for the frequency-first Taylor model (for Newton)."""
    k = len(theta)
    dterms = [np.ones_like(dt), np.zeros_like(dt)] + \
        [dt ** (i - 1) / math.factorial(i - 1) for i in range(2, 8)]
    return theta @ np.vstack(dterms[:k])


def _signal_times(n, theta, sigma, t_span, rng):
    """Photon times at near-integer phase: pick integer wraps K, add scatter,
    Newton-solve dt so the full k-parameter phase(dt) == K + eps."""
    f = theta[0]
    K = rng.integers(-int(f * t_span / 2), int(f * t_span / 2), n).astype(float)
    target = K + rng.normal(0, sigma, n)
    phi0 = theta[1] if len(theta) >= 2 else 0.0
    dt = (target - phi0) / f
    for _ in range(60):
        dt = dt - (theta @ _design(dt, len(theta)) - target) / _dphase(dt, theta)
    return dt, K


def baby_lattice(N, k, sigma, p=1.0, t_span=3.0e8, theta=None,
                 prior_factor=100.0, res_frac=0.1, seed=0):
    """Build the (N+k)-dim q-ary lattice: q*Z^N wraps + k Taylor parameter rows,
    with a planted pulsar and a fraction ``1-p`` of uniform-phase background.

    Returns (il, steps, c_true, K) where ``c_true`` is the planted parameter
    multiple (the thing a search must find; the wraps are slaved to it).
    """
    rng = np.random.default_rng(seed)
    if theta is None:                                   # f, phi, fdot, ... (freq-first)
        theta = np.array([50.0, 0.3, -1e-14, 5e-7, 5e-7, 3e-16, 3e-16])[:k]
    n_bg = int(round((1 - p) * N))
    n_ps = N - n_bg
    dt_s, K_s = _signal_times(n_ps, theta, sigma, t_span, rng) if n_ps else \
        (np.array([]), np.array([]))
    dt_b = rng.uniform(-t_span / 2, t_span / 2, n_bg) if n_bg else np.array([])
    dt = np.concatenate([dt_s, dt_b])
    M = _design(dt, k)
    K = np.round(theta @ M)                             # wraps (signal ~exact, bg ~random)
    order = np.argsort(dt)
    dt, M, K = dt[order], M[:, order], K[order]
    steps = res_frac / np.max(np.abs(M), axis=1)
    cstd = prior_factor * np.abs(theta)
    c_true = np.round(theta / steps).astype(object)

    il = np.zeros((N + k, N + k), dtype=object)
    for j in range(N):
        il[j, j] = int(Q)
    for i in range(k):
        for j in range(N):
            il[N + i, j] = int(round(Q * steps[i] * M[i, j]))
        il[N + i, N + i] = int(round(Q * steps[i] / cstd[i]))
    return il, steps, c_true, K.astype(object)


def verify_tm(steps, k, sigma, t_span=3.0e8, theta=None, n_verify=200, seed=12345):
    """Held-out pure-signal TOAs, returned as the scaled design ``tm`` used to
    fold a candidate's parameter multiple into a coherence Q."""
    if theta is None:
        theta = np.array([50.0, 0.3, -1e-14, 5e-7, 5e-7, 3e-16, 3e-16])[:k]
    rng = np.random.default_rng(seed)
    tv, _ = _signal_times(n_verify, theta, sigma, t_span, rng)
    return Q * steps[:, None] * _design(tv, k), np.ones(n_verify, np.float32)


def planted_vector(il, N, k, c_true, K):
    """The planted solution: parameter combination ``c_true`` with the true wraps
    ``K`` subtracted. Also reports whether mod-q reduction recovers the SAME wraps
    (i.e. whether the shortest vector IS the mod-q reduction)."""
    v = np.zeros(il.shape[0], dtype=object)
    for i in range(k):
        v += int(c_true[i]) * il[N + i]
    modq_same = True
    for j in range(N):
        if round(int(v[j]) / Q) != int(K[j]):
            modq_same = False
        v[j] = int(v[j]) - int(K[j]) * Q
    return v, modq_same


def _coherence(pc, n_per, tm, probs, k_std_min=1e3):
    """Coherence Q of a candidate's parameter multiple on the held-out TOAs, but
    only for PHYSICAL candidates (wrap std above ``k_std_min``) -- this excludes
    the trivial constant-phase vector the phi row introduces at k>=2."""
    if np.std(np.array(pc[:n_per], float)) <= k_std_min:
        return 0.0
    pp = np.array(pc[n_per:], float)
    vf = np.mod(pp @ tm / Q + 0.5, 1) - 0.5
    return float(np.abs(np.sum(probs * np.exp(2j * np.pi * vf))) ** 2 / np.sum(probs ** 2 / 2))


def solve(il, n_per, tm, probs, mode, seed, max_samples=2500):
    """Best physical coherence Q reachable by ``mode``:
      'lll'        -- scan the LLL-reduced basis only (no sieve);
      'sieve'      -- LLL + pure pair sieve;
      'sieve_modq' -- LLL + pair sieve with mod-q candidate reduction;
      'raw_sieve'  -- pair sieve on the RAW basis (no LLL);
      'raw_modq'   -- raw pair sieve with mod-q reduction.
    All apply the physical wrap-std mask via :func:`_coherence`."""
    from lattice_tools import pure_lll
    import pure_sieve as ps
    qi = int(il[0, 0])
    if mode.startswith("raw"):
        B, U = il, np.eye(il.shape[0], dtype=object)
    else:
        B, U = pure_lll(il, 0.99)
    if mode == "lll":
        return max(_coherence(np.array(u, float), n_per, tm, probs)
                   for u in np.array(U, dtype=object))
    use_modq = mode.endswith("modq")
    best = {"q": 0.0}

    def hook(pv, pc, p2, db, samples):
        q = _coherence(pc, n_per, tm, probs)
        if q > best["q"]:
            best["q"] = q
        if q > 100:
            return False
    ps.gauss_sieve(B, triple=False, seed=seed, seed_coef=U, max_samples=max_samples,
                   mod_q=((n_per, qi) if use_modq else None), move_hook=hook)
    return best["q"]


def _lll_rows(il):
    D = il.shape[0]
    IM = fpylll.IntegerMatrix.from_iterable(D, D, [int(x) for row in il for x in row])
    fpylll.LLL.reduction(IM)
    return [[IM[i, j] for j in range(D)] for i in range(D)]


def _norm(v):
    return float(np.sqrt(float(sum(int(x) * int(x) for x in v))))


if __name__ == "__main__":
    print("PART 1 -- is the planted signal the shortest vector, and is it the")
    print("mod-q reduction? (clean, p=1)\n")
    print(f"{'k':>2} {'N':>4} {'||planted||':>13} {'||shortest||':>13} {'short/planted':>13} {'modq=wraps':>11}")
    for k in (1, 2, 3):
        for N in (16, 24, 32):
            il, steps, c_true, K = baby_lattice(N, k, 0.03, seed=1)
            planted, modq_same = planted_vector(il, N, k, c_true, K)
            rows = sorted((r for r in _lll_rows(il) if any(r)),
                          key=lambda r: sum(int(x) * int(x) for x in r))
            ratio = _norm(rows[0]) / _norm(planted)
            print(f"{k:>2} {N:>4} {_norm(planted):>13.3e} {_norm(rows[0]):>13.3e} "
                  f"{ratio:>13.3f} {str(modq_same):>11}", flush=True)

    print("\nPART 2 -- recovery with background: physical max Q (>50 = recovered),")
    print("median over 4 seeds, N=24.  Does the sieve / mod-q help beyond LLL?\n")
    print(f"{'k':>2} {'p':>5} {'LLL only':>10} {'LLL+sieve':>11} {'+mod-q':>9}")
    for k in (1, 2):
        for p in (1.0, 0.7, 0.5):
            cols = []
            for mode in ("lll", "sieve", "sieve_modq"):
                qs = []
                for s in range(4):
                    il, steps, c_true, K = baby_lattice(24, k, 0.03, p=p, seed=s)
                    tm, probs = verify_tm(steps, k, 0.03)
                    qs.append(solve(il, 24, tm, probs, mode, seed=s))
                cols.append(np.median(qs))
            print(f"{k:>2} {p:>5} {cols[0]:>10.1f} {cols[1]:>11.1f} {cols[2]:>9.1f}", flush=True)
    print("\nDONE")
