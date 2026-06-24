"""Heavy p=0.5 detection scan -- one (model, n) per process, for a workstation.

p=0.5 is Table 1's hardest row: effective pulse width sigma = sqrt(0.5/12) =
0.204. An earlier projection put model A's threshold at n~110, but direct runs
showed n=100/110/120 do NOT detect (best *physical* Q ~ 24 < 50), so the real
threshold is higher -- this scans upward to find it (and whether it is reachable
under g6k's MAX_SIEVING_DIM=128 cap).

Design notes (learned the hard way):
  * ONE (model, n) per process. The deep q-ary sieve can abort at C level
    (siever.so) past dim ~110; isolating each n means a crash kills only that
    point, not the whole scan. Launch many in parallel (see the driver).
  * bgj1 only. hk3 SaturationErrors and bdgl2 C-aborts at these dims; bgj1 is the
    robust workhorse. (bdgl2 is tried only if bgj1 itself raises.)
  * Detection = max Q over PHYSICAL vectors (wrap-count std > reasonable_k_std),
    exactly as fermi_fold.fold does. We also report the unmasked maxQ and its
    wrap-std to expose the trivial low-wrap decoy (Q can be huge for a vector that
    folds everything into one phase -- not a real pulsar).
  * Retries: detection at the wall is stochastic, so each n gets ATTEMPTS draws.

Usage (g6k env active, repo + g6k on PYTHONPATH):
    python experiments/p05_lava.py A 0.5 140        # one point -> p05_A_140.txt
    python experiments/p05_lava.py B 0.5 160
A driver loops these over an n-grid with a concurrency cap (see run_p05_scan.sh).
"""

import os
import sys
import time

import numpy as np
import fpylll
from g6k import Siever
from g6k.siever_params import SieverParams

import model_a_constant_frequency as model_a
import model_b_full_timing as model_b

# ---- config ----
THREADS = 24             # g6k sieve threads per process (parallelism is across n)
DEPTH_FRAC = 0.85         # sieve dimension as a fraction of the lattice dimension
DIM_CAP = 124            # keep under g6k MAX_SIEVING_DIM (default 128)
ATTEMPTS = 2             # stochastic retries per n
BLOCK = 30              # BKZ block size
MUL = 10 ** 16          # q
N_VERIFY = 1281
K_STD = 1e5            # physical-solution wrap-count-std threshold (== fold default)
DETECT_Q = 50.0
OUTDIR = os.path.dirname(__file__)


def sigma_of_p(p):
    return float(np.sqrt((1 - p) / 12))


def _bkz(il):
    n = il.shape[0]
    IM = fpylll.IntegerMatrix.from_iterable(n, n, [int(x) for r in il for x in r])
    gso = fpylll.GSO.Mat(IM, flags=fpylll.GSO.INT_GRAM,
                         U=fpylll.IntegerMatrix.identity(n), UinvT=fpylll.IntegerMatrix.identity(n))
    fpylll.BKZ.Reduction(gso, fpylll.LLL.Reduction(gso, delta=0.95),
                         fpylll.BKZ.Param(block_size=BLOCK, strategies=fpylll.BKZ.DEFAULT_STRATEGY,
                                          delta=0.95))()
    return [[IM[i, j] for j in range(n)] for i in range(n)]


def _sieve_detect(reduced, pump_stop, alg, n_per, tm, q, probs):
    """Pump to pump_stop and return (detQ, unmaskedQ, wstd_at_unmasked_argmax, db_size)."""
    n = len(reduced)
    IM = fpylll.IntegerMatrix.from_iterable(n, n, [int(x) for r in reduced for x in r])
    gso = fpylll.GSO.Mat(IM, flags=fpylll.GSO.INT_GRAM,
                         U=fpylll.IntegerMatrix.identity(n), UinvT=fpylll.IntegerMatrix.identity(n))
    gso.update_gso()
    g = Siever(gso, SieverParams(threads=THREADS))
    g.initialize_local(0, n // 2, n)
    with g.temp_params(otf_lift=False):
        while g.l > pump_stop:
            g.extend_left(1)
            g(alg=alg)
        g.extend_left(g.l)
    db = np.array(list(g.itervalues()))
    tr = db @ np.array(list(g.M.U))
    vf = np.mod(tr[:, n_per:] @ tm / q + 0.5, 1) - 0.5
    Q = np.abs(np.sum(probs * np.exp(2j * np.pi * vf), axis=1)) ** 2 / np.sum(probs ** 2 / 2)
    wstd = np.std(tr[:, :n_per].astype(float), axis=1)
    mask = wstd > K_STD
    detQ = float(Q[mask].max()) if mask.any() else 0.0
    uidx = int(Q.argmax())
    return detQ, float(Q[uidx]), float(wstd[uidx]), len(db)


def run_case(model, n_toas, p):
    sig = sigma_of_p(p)
    gen = model_a.generate_data if model == "A" else model_b.generate_data
    data = gen(n_toas=n_toas, n_verify=N_VERIFY, phase_std=sig, mul_factor=MUL, seed=0)
    il = data["integer_lattice"]; dim = il.shape[0]
    n_per = len(data["toas_met_lattice"]); q = data["mul_factor"]
    tm, probs = data["transformation_matrix"], data["probs_verify"]
    d_sieve = min(round(DEPTH_FRAC * dim), DIM_CAP)
    pump_stop = dim - d_sieve
    reduced = _bkz(il)
    best, uq, uw, alg_used, total_t = 0.0, 0.0, 0.0, "bgj1", 0.0
    for attempt in range(ATTEMPTS):
        for alg in ("bgj1", "bdgl2"):       # bdgl2 only if bgj1 raises
            try:
                t0 = time.perf_counter()
                detQ, unmQ, wstd, _ = _sieve_detect(reduced, pump_stop, alg, n_per, tm, q, probs)
                total_t += time.perf_counter() - t0
                alg_used = alg
                if detQ > best:
                    best, uq, uw = detQ, unmQ, wstd
                break
            except Exception as exc:
                alg_used = f"{alg}:{type(exc).__name__}"
                if alg == "bdgl2":
                    break
        if best > DETECT_Q:
            break
    return dict(model=model, n=n_toas, dim=dim, d_sieve=d_sieve, detQ=best,
                unmaskedQ=uq, decoy_wstd=uw, alg=alg_used, secs=round(total_t, 1),
                detected=best > DETECT_Q)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "A"
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    n = int(sys.argv[3])
    r = run_case(model, n, p)
    line = (f"{r['model']} n={r['n']} dim={r['dim']} d_sieve={r['d_sieve']} "
            f"detQ={r['detQ']:.1f} unmaskedQ={r['unmaskedQ']:.1f} "
            f"decoy_wstd={r['decoy_wstd']:.3g} alg={r['alg']} secs={r['secs']} "
            f"-> {'DETECTED' if r['detected'] else 'no'}")
    print(line, flush=True)
    with open(os.path.join(OUTDIR, f"p05_{model}_{n}.txt"), "w") as fh:
        fh.write(line + "\n")


if __name__ == "__main__":
    main()
