"""Heavy p=0.5 detection sweep -- meant for a workstation (e.g. lava), not a laptop.

p=0.5 is Table 1's hardest, near-"infeasible" row: effective pulse width
sigma = sqrt((1-0.5)/12) = 0.204. Projected detection thresholds (from the
p=0.7 / p=1 scaling, n ∝ 1/ln(2^0.2/(sigma*sqrt(2 pi e)))):

    model A (phi, f) :  n ~ 110,  d_sieve ~ 0.8 n ~ 88   (feasible on a laptop)
    model B (7 param):  n ~ 145,  d_sieve ~ 0.8 n ~ 115  (needs RAM + cores;
                        dim ~115 is the bdgl regime and near g6k's default
                        MAX_SIEVING_DIM=128 -- rebuild g6k higher if you cap out)

For each (model, n) it BKZ-30 reduces, then pumps a TUNED depth
`d_sieve = min(round(depth_frac * N), dim_cap)` -- NOT full-mode (which would
exceed the sieving cap and cost the worst-case 2^{0.36 N}). It tries `bgj1`
first (the robust workhorse here) and falls back to `bdgl2` (asymptotically
fastest, wins only at large dim). Detection at the wall is stochastic, so each is
retried; results are appended to RESULTS_PATH as they complete (crash-safe).

Usage on lava (g6k env active, repo on PYTHONPATH):
    PYTHONPATH=. python experiments/p05_lava.py            # both models, p=0.5
    PYTHONPATH=. python experiments/p05_lava.py A           # model A only
    PYTHONPATH=. python experiments/p05_lava.py B 0.5 130,145,160
Results also stream to experiments/p05_results.txt.
"""

import os
import sys
import time

import numpy as np
import fpylll
from g6k import Siever

import model_a_constant_frequency as model_a
import model_b_full_timing as model_b

# ---- config (edit for lava) ----
DEPTH_FRAC = 0.85          # sieve dimension as a fraction of the lattice dimension
DIM_CAP = 124             # keep under g6k MAX_SIEVING_DIM (default 128)
ATTEMPTS = 3              # retries (detection at the wall is stochastic)
BLOCK = 30               # BKZ block size for pre-reduction
MUL = 10 ** 16           # q (above the precision floor for these N)
N_VERIFY = 1281
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "p05_results.txt")
DEFAULT_NS = {"A": (100, 110, 120, 130), "B": (120, 135, 150, 165)}


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


def _pump_Q(reduced, pump_stop, alg, n_per, tm, q, probs):
    n = len(reduced)
    IM = fpylll.IntegerMatrix.from_iterable(n, n, [int(x) for r in reduced for x in r])
    gso = fpylll.GSO.Mat(IM, flags=fpylll.GSO.INT_GRAM,
                         U=fpylll.IntegerMatrix.identity(n), UinvT=fpylll.IntegerMatrix.identity(n))
    gso.update_gso()
    g = Siever(gso)
    g.initialize_local(0, n // 2, n)
    t0 = time.perf_counter()
    with g.temp_params(otf_lift=False):
        while g.l > pump_stop:
            g.extend_left(1)
            g(alg=alg)
        g.extend_left(g.l)
    dt = time.perf_counter() - t0
    db = np.array(list(g.itervalues()))
    tr = db @ np.array(list(g.M.U))
    vf = np.mod(tr[:, n_per:] @ tm / q + 0.5, 1) - 0.5
    Q = np.abs(np.sum(probs * np.exp(2j * np.pi * vf), axis=1)) ** 2 / np.sum(probs ** 2 / 2)
    m = np.std(tr[:, :n_per].astype(float), axis=1) > 1e5
    return (float(Q[m].max()) if m.any() else 0.0), dt


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
    best, used_alg, total_t = 0.0, "-", 0.0
    for alg in ("bgj1", "bdgl2"):
        for _ in range(ATTEMPTS):
            try:
                Q, dt = _pump_Q(reduced, pump_stop, alg, n_per, tm, q, probs)
            except Exception as exc:
                total_t += 0.0
                Q, dt = None, 0.0
                used_alg = f"{alg}:{type(exc).__name__}"
                break
            total_t += dt
            if Q > best:
                best, used_alg = Q, alg
            if best > 50:
                break
        if best > 50:
            break
    return dict(model=model, n=n_toas, dim=dim, d_sieve=d_sieve,
                maxQ=best, alg=used_alg, secs=round(total_t, 1),
                detected=best > 50)


def main(models=("A", "B"), p=0.5, ns=None):
    with open(RESULTS_PATH, "a") as fh:
        hdr = (f"# p={p} sigma={sigma_of_p(p):.3f} depth_frac={DEPTH_FRAC} "
               f"dim_cap={DIM_CAP} attempts={ATTEMPTS}")
        print(hdr, flush=True); fh.write(hdr + "\n")
        cols = f"{'model':>5} {'n':>4} {'dim':>4} {'d_sieve':>7} {'maxQ':>8} {'alg':>14} {'secs':>8} {'det':>4}"
        print(cols, flush=True); fh.write(cols + "\n"); fh.flush()
        for model in models:
            for n in (ns or DEFAULT_NS[model]):
                r = run_case(model, n, p)
                line = (f"{r['model']:>5} {r['n']:>4} {r['dim']:>4} {r['d_sieve']:>7} "
                        f"{r['maxQ']:>8.1f} {r['alg']:>14} {r['secs']:>8.1f} "
                        f"{('YES' if r['detected'] else 'no'):>4}")
                print(line, flush=True); fh.write(line + "\n"); fh.flush()


if __name__ == "__main__":
    models = (sys.argv[1],) if len(sys.argv) > 1 else ("A", "B")
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    ns = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    main(models, p, ns)
