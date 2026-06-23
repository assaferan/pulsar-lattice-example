"""Which g6k sieve algorithm to use when complexity bites (high sieving dimension)?

At small dimension the default `hk3` triple sieve is fastest. But `hk3` is
*memory-optimized* -- it saturates the smallest database (~2^{0.19 d} vectors) --
so on the skewed q-ary timing lattice, once the sieving dimension passes ~80 its
database is too small to contain the (buried) physical solution: it misses, or
raises `SaturationError`. This script compares the algorithms at the cost-wall
case model B / p=0.7 / N=111 (detection needs `d_sieve ~ 0.8 N ~ 88`).

Findings (N=111, p=0.7, sieve dim 88, BKZ-30 pre-reduction):

      alg      max Q     pump time     result
      hk3        --        --          SaturationError (or misses Q~22)
      bgj1     ~925-989    ~79 s        DETECTED
      bdgl2    ~925-989    ~88-97 s     DETECTED

So:
  * `hk3` is the wrong sieve here -- avoid it past dim ~80.
  * `bgj1` (the pipeline's auto-fallback) detects N=111 in ~80s. Sufficient.
  * `bdgl2` also detects but is NOT faster at d=88 -- its asymptotically-best
    time exponent (~0.292 vs hk3's 0.36) is dominated by LSF overhead at this
    dimension; it pays off only at substantially larger N. `fermi_fold._sieve`
    falls back hk3 -> bgj1 -> bdgl2, so this is the last resort for very large N.
  * The real feasibility lever is the PUMP DEPTH: `d_sieve ~ 0.8 N` (dim 88)
    suffices -- no need for full-mode's `pump_stop=7` (dim ~N, the expensive path).

Note: detection right at the `d_sieve` wall (dim ~88) is stochastic per g6k run
(~50/50), so each algorithm is retried a few times and the best result reported;
`hk3` SaturationErrors on every attempt, `bgj1`/`bdgl2` detect within a couple.

Run (g6k env active):  PYTHONPATH=. python experiments/sieve_algo.py
"""

import time

import numpy as np
import fpylll
from g6k import Siever

import model_b_full_timing as B


def sigma_of_p(p):
    return float(np.sqrt((1 - p) / 12))


def _bkz_reduced(il, block=30, delta=0.95):
    n = il.shape[0]
    IM = fpylll.IntegerMatrix.from_iterable(n, n, [int(x) for r in il for x in r])
    gso = fpylll.GSO.Mat(IM, flags=fpylll.GSO.INT_GRAM,
                         U=fpylll.IntegerMatrix.identity(n),
                         UinvT=fpylll.IntegerMatrix.identity(n))
    fpylll.BKZ.Reduction(gso, fpylll.LLL.Reduction(gso, delta=delta),
                         fpylll.BKZ.Param(block_size=block,
                                          strategies=fpylll.BKZ.DEFAULT_STRATEGY,
                                          delta=delta))()
    return [[IM[i, j] for j in range(n)] for i in range(n)]


def _one_pump(reduced_basis, pump_stop, alg, n_per, tm, q, probs):
    n = len(reduced_basis)
    IM = fpylll.IntegerMatrix.from_iterable(
        n, n, [int(x) for r in reduced_basis for x in r])
    gso = fpylll.GSO.Mat(IM, flags=fpylll.GSO.INT_GRAM,
                         U=fpylll.IntegerMatrix.identity(n),
                         UinvT=fpylll.IntegerMatrix.identity(n))
    gso.update_gso()
    g = Siever(gso)
    g.initialize_local(0, n // 2, n)
    t0 = time.perf_counter()
    try:
        with g.temp_params(otf_lift=False):
            while g.l > pump_stop:
                g.extend_left(1)
                g(alg=alg)
            g.extend_left(g.l)
    except Exception as exc:
        return None, time.perf_counter() - t0, type(exc).__name__
    dt = time.perf_counter() - t0
    db = np.array(list(g.itervalues()))
    tr = db @ np.array(list(g.M.U))
    vf = np.mod(tr[:, n_per:] @ tm / q + 0.5, 1) - 0.5
    Q = np.abs(np.sum(probs * np.exp(2j * np.pi * vf), axis=1)) ** 2 / np.sum(probs ** 2 / 2)
    m = np.std(tr[:, :n_per].astype(float), axis=1) > 1e5
    return (float(Q[m].max()) if m.any() else 0.0), dt, "ok"


def sieve_detect(reduced_basis, pump_stop, alg, n_per, tm, q, probs, attempts=3):
    """Pump a (pre-reduced) basis with `alg` down to `pump_stop`, up to
    `attempts` times. Detection right at the d_sieve wall is stochastic per g6k
    run, so we report the BEST result over attempts (stop early on detection).
    Returns (max reasonable Q, cumulative pump seconds, status)."""
    best_Q, total_t, status = None, 0.0, "ERR"
    for _ in range(attempts):
        Q, dt, st = _one_pump(reduced_basis, pump_stop, alg, n_per, tm, q, probs)
        total_t += dt
        if st == "ok":
            status = "ok"
            best_Q = Q if best_Q is None else max(best_Q, Q)
            if Q > 50:
                break
        elif status != "ok":
            status = st                      # remember the failure type (e.g. SaturationError)
    return best_Q, total_t, status


def main(n_toas=111, p=0.7, pump_stop=30, seed=0, algs=("hk3", "bgj1", "bdgl2")):
    sig = sigma_of_p(p)
    data = B.generate_data(n_toas=n_toas, n_verify=1281, phase_std=sig,
                           mul_factor=10 ** 16, seed=seed)
    il = data["integer_lattice"]
    n_per = len(data["toas_met_lattice"])
    q = data["mul_factor"]
    tm, probs = data["transformation_matrix"], data["probs_verify"]
    dim = il.shape[0]

    print(f"model B, p={p} (sigma={sig:.3f}), N={n_toas}, lattice dim={dim}; "
          f"BKZ-30, pump to sieve_dim={dim - pump_stop}")
    reduced = _bkz_reduced(il)        # reduce once, reuse for each algorithm
    print(f"{'alg':>7} {'max Q':>8} {'pump_s':>8} {'result':>16}")
    for alg in algs:
        Q, dt, st = sieve_detect(reduced, pump_stop, alg, n_per, tm, q, probs)
        if st != "ok":
            verdict = st
        else:
            verdict = "DETECTED" if Q > 50 else "no"
        print(f"{alg:>7} {('%.1f' % Q) if Q is not None else '-':>8} {dt:>8.1f} "
              f"{verdict:>16}", flush=True)
    print("=> hk3 fails at this dimension; bgj1 detects (~80s); bdgl2 detects but is\n"
          "   no faster here (its 0.292 exponent only wins at larger N).")


if __name__ == "__main__":
    main()
