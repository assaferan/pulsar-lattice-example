"""Baby Model A: the standard basis q*Z^N plus k parameter vectors.

The simplest controlled setting for the mod-q / shortest-vector question. The
lattice is the wrap lattice ``q*Z^N`` (the "standard basis") plus ``k`` parameter
rows (a truncated Taylor phase model: k=1 frequency only, k=2 phi+f = Model A,
k>=3 toward Model B). A planted pulsar makes the phase near-integer for the true
parameters.

The point of starting here:

* With **k=1** (a single extra vector) the wraps are slaved to one multiplier, so
  the shortest lattice vector IS the mod-q reduction of the best integer multiple
  of that vector -- the "shortest-vector sense" in which LLL == mod-q. No sieve.
* mod-q reduction handles the **N-dimensional wrap part** deterministically; what
  remains is a **k-dimensional search** over parameter multiples. So this model
  separates the two cleanly and lets us watch where mod-q stops sufficing as k
  grows -- and whether N then enters only polynomially (the asymptotic goal).

Run:  PYTHONPATH=.:experiments python3 experiments/baby_model.py
"""
import math

import numpy as np
import fpylll


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


def baby_lattice(N, k, sigma, q=10 ** 12, t_span=3.0e8, theta=None,
                 prior_factor=10.0, res_frac=0.01, seed=0):
    """Build the (N+k)-dim q-ary lattice: q*Z^N wraps + k Taylor parameter rows,
    with a planted pulsar (phase near-integer for ``theta``).

    Returns (il, dt, M, steps, c_true) where ``c_true`` is the integer parameter
    multiple of the planted solution (in step units) -- the thing a search must
    find; the wraps are slaved to it.
    """
    rng = np.random.default_rng(seed)
    if theta is None:                                   # f, phi, fdot, ... (freq-first)
        theta = np.array([50.0, 0.3, -1e-14, 5e-7, 5e-7, 3e-16, 3e-16])[:k]
    f = theta[0]
    # PLANT a signal: photons arrive at near-integer phase. Pick integer wraps K,
    # add intrinsic scatter eps, then Newton-solve the time dt where the full
    # k-parameter phase(dt) == K + eps (so the signal exists for any k).
    K = rng.integers(-int(f * t_span / 2), int(f * t_span / 2), N).astype(float)
    target = K + rng.normal(0, sigma, N)
    phi0 = theta[1] if k >= 2 else 0.0
    dt = (target - phi0) / f                             # linear (f*dt+phi) init
    for _ in range(60):                                 # Newton on the full model
        dt = dt - (theta @ _design(dt, k) - target) / _dphase(dt, theta)
    order = np.argsort(dt)
    dt, K = dt[order], K[order]
    M = _design(dt, k)                                  # (k, N)
    steps = res_frac / np.max(np.abs(M), axis=1)
    cstd = prior_factor * np.abs(theta)
    # planted parameter multiple (offset from the prior centre 0, in step units):
    c_true = np.round(theta / steps).astype(object)

    D = N + k
    il = np.zeros((D, D), dtype=object)
    for j in range(N):
        il[j, j] = int(q)                               # wrap rows q*e_j
    for i in range(k):                                  # parameter rows
        for j in range(N):
            il[N + i, j] = int(round(q * steps[i] * M[i, j]))
        il[N + i, N + i] = int(round(q * steps[i] / cstd[i]))
    return il, dt, M, steps, c_true, K.astype(object)


def _lll_rows(il):
    D = il.shape[0]
    IM = fpylll.IntegerMatrix.from_iterable(D, D, [int(x) for row in il for x in row])
    fpylll.LLL.reduction(IM)
    return [[IM[i, j] for j in range(D)] for i in range(D)]


def planted_vector(il, N, k, c_true, K, q):
    """The planted solution: the parameter combination ``c_true`` with the true
    integer wraps ``K`` subtracted from the first N coords (so they cancel to the
    intrinsic residual ~q*eps). Also returns whether mod-q reduction recovers the
    SAME wraps -- i.e. whether the shortest vector IS the mod-q reduction."""
    v = np.zeros(il.shape[0], dtype=object)
    for i in range(k):
        v += int(c_true[i]) * il[N + i]
    modq_same = True
    for j in range(N):
        k_modq = round(int(v[j]) / q)                   # what mod-q reduction picks
        if k_modq != int(K[j]):
            modq_same = False
        v[j] = int(v[j]) - int(K[j]) * q                # subtract the TRUE wraps
    return v, modq_same


def _norm(v):
    return float(np.sqrt(float(sum(int(x) * int(x) for x in v))))


if __name__ == "__main__":
    q = 10 ** 15
    print("Baby Model A: q*Z^N + k parameter rows.  Does the shortest vector equal")
    print("the mod-q reduction of the planted parameter multiple? (k=1 => yes, exactly)\n")
    print(f"{'k':>2} {'N':>4} {'||planted||':>13} {'||shortest LLL||':>17} {'LLL/planted':>12} {'modq=wraps':>11}")
    for k in (1, 2, 3):
        for N in (16, 24, 32):
            il, dt, M, steps, c_true, K = baby_lattice(
                N, k, 0.03, q=q, seed=1, res_frac=0.1, prior_factor=100.0)
            planted, modq_same = planted_vector(il, N, k, c_true, K, q)
            rows = [r for r in _lll_rows(il) if any(r)]
            rows.sort(key=lambda r: sum(int(x) * int(x) for x in r))
            shortest = rows[0]
            ratio = _norm(shortest) / _norm(planted)
            print(f"{k:>2} {N:>4} {_norm(planted):>13.3e} {_norm(shortest):>17.3e} "
                  f"{ratio:>12.3f} {str(modq_same):>11}", flush=True)
    print("\nDONE")
