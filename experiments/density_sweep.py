"""Density scaling study: d_sieve(N') for subsampled TOA sets.

For each subset size N', rebuild the lattice from a random subset of the 70
lattice TOAs (preserving the span), BKZ-reduce once, then increase the sieving
dimension until the tight pulsar solution is recovered (per-coordinate phase
residual << q).  d_sieve(N') is the minimum sieving dimension that detects.
"""
import sys, numpy as np, fpylll
from g6k import Siever
from fermi_fold import load_data

d = load_data('data/data.npy')
sv_full = np.asarray(d['span_vecs']); cs = np.asarray(d['coeff_std']); mul = d['mul_factor']
NTOA = sv_full.shape[1]
S, C = 8.105450e-04, 6.569831e-05
TIGHT = 0.01; TRIALS = 2; BLOCK = 20

def orth(M):
    Q, _ = np.linalg.qr(M.T); return Q.T

def build_il(span):
    p, N = span.shape
    L = np.zeros((N + p, N + p))
    L[:N, :N] = np.eye(N); L[N:, :N] = S * orth(span); L[N:, N:] = np.diag(C / cs)
    return np.round(L * mul).astype(np.int64)

def bkz_basis(il, block):
    n = il.shape[0]
    im = fpylll.IntegerMatrix.from_iterable(n, n, [int(x) for r in il for x in r])
    gso = fpylll.GSO.Mat(im, flags=fpylll.GSO.INT_GRAM,
        U=fpylll.IntegerMatrix.identity(n), UinvT=fpylll.IntegerMatrix.identity(n))
    fpylll.BKZ.Reduction(gso, fpylll.LLL.Reduction(gso, delta=0.95),
        fpylll.BKZ.Param(block_size=block, strategies=fpylll.BKZ.DEFAULT_STRATEGY, delta=0.95))()
    return np.array([[im[i, j] for j in range(n)] for i in range(n)], dtype=np.int64)

def percoord_resid(B, ntoa, pump_stop, q):
    """Per-coordinate phase residual of the best vector. inf on sieve failure."""
    n = B.shape[0]
    gso = fpylll.GSO.Mat(
        fpylll.IntegerMatrix.from_iterable(n, n, [int(x) for r in B for x in r]),
        flags=fpylll.GSO.INT_GRAM, U=fpylll.IntegerMatrix.identity(n),
        UinvT=fpylll.IntegerMatrix.identity(n))
    fpylll.LLL.Reduction(gso, delta=0.95)()
    g = Siever(gso); g.initialize_local(0, n // 2, n)
    try:
        with g.temp_params(otf_lift=False):
            while g.l > pump_stop:      # only called with pump_stop < n//2
                g.extend_left(1); g(alg='hk3')
            g.extend_left(g.l)
    except Exception:
        return np.inf                   # g6k saturation failures are spurious
    db = np.array(list(g.itervalues()))
    if db.size == 0:
        return np.inf
    V = (db @ np.array(list(g.M.B))).astype(np.int64)
    per = ((V[:, :ntoa] + q // 2) % q) - q // 2
    return np.linalg.norm(per.astype(float), axis=1).min() / np.sqrt(ntoa) / q

rng = np.random.RandomState(0)
print(f"{'N':>4} {'n':>4} {'d_sieve':>8} {'d/n':>5} {'resid/q':>9}", flush=True)
for Np in (70, 55, 40, 30, 20):
    idx = np.arange(NTOA) if Np >= NTOA else np.sort(rng.choice(NTOA, Np, replace=False))
    il = build_il(sv_full[:, idx]); n = il.shape[0]; q = int(il[0, 0])
    B = bkz_basis(il, BLOCK)
    found = None
    # sweep increasing sieving dimension; gradual pump needs d_sieve > n//2
    for d_sieve in range(n // 2 + 1, n + 1, 2):
        pump_stop = n - d_sieve
        best = min(percoord_resid(B, Np, pump_stop, q) for _ in range(TRIALS))
        if best < TIGHT:
            found = (d_sieve, best); break
    if found:
        ds, best = found
        print(f"{Np:>4} {n:>4} {ds:>8} {ds/n:>5.2f} {best:>9.2e}", flush=True)
    else:
        print(f"{Np:>4} {n:>4} {'NONE':>8} {'-':>5} {'-':>9}  (no tight detection)", flush=True)
