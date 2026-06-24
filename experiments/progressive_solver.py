"""Progressive-baseline solver (lever 1) -- NEGATIVE RESULT.

Idea: the density sweep showed the monolithic sieve needs dimension ~N. The
lever-1 leverage test suggested that, after a small bootstrap, each next
time-sorted TOA's wrap is *determined*. The hope:

    bootstrap: sieve a small (k0+p)-dim sub-lattice  -> coarse parameters b
    connect:   for each later TOA, predict k_i = round(-(b.A[:,i])/q),
               then least-squares refine b on the assigned wraps

which would be O(1) bootstrap + O(N) cheap steps = poly(N).

**It does not work.** Two obstructions, both demonstrated by __main__:

1. The trivial constant-phase vector (b in the phi direction -> all wraps equal,
   std(wraps)=0) folds *perfectly* (residual ~1e-13) and hits the Q coherence
   ceiling (~2110). So "tightest global fold" always selects this non-pulsar.
   The phyical mask std(wraps)>threshold is required to exclude it.

2. With the physical mask applied, NO bootstrap candidate (out of hundreds,
   k0=30..45) connects to the real pulsar -- the best physical connection has
   Q~1. The pulsar is recovered *only* when the connection is seeded with the
   exact true b* (then Q~400, physical). I.e. the connection geometry works, but
   a short-baseline bootstrap cannot *acquire* the solution: a short baseline is
   consistent with many physical timing solutions (the lever-1 test's huge
   g_far), and connecting a wrong one diverges.

This is consistent with the density sweep: identifying the solution genuinely
needs ~N-dimensional information, so a bounded bootstrap cannot escape it.

Phase model (validated exactly against the full sieve): with A = integer_lattice
timing block and q = mul_factor, parameters b fold TOA i to
k_i = round(-(b . A[:,i]) / q)  (use exact ints; q*k overflows int64).

Run from the repo root with the g6k environment active:
    PYTHONPATH=. python3 experiments/progressive_solver.py
"""
import numpy as np
from fermi_fold import load_data, fold, _sieve

STD_PHYSICAL = 1e5            # std(wraps) above this == physical (wraps vary)


def _bootstrap(il, ntoa, q, sub, n_candidates, n_svp=7):
    """Sieve the sub-lattice on TOAs `sub`; return the best candidate parameter
    vectors b (by fold residual on the bootstrap TOAs)."""
    k0 = len(sub)
    A_sub = il[ntoa:, sub].astype(np.int64)
    L = il[ntoa:, ntoa:].astype(np.int64)
    p = L.shape[0]
    boot = np.zeros((k0 + p, k0 + p), dtype=np.int64)
    boot[:k0, :k0] = q * np.eye(k0, dtype=np.int64)
    boot[k0:, :k0] = A_sub
    boot[k0:, k0:] = L
    _, db_transformation, db_vectors = _sieve(boot, n_svp, 30, 0.95, do_bkz=True)
    b = db_transformation[:, k0:]
    resid = np.abs(db_vectors[:, :k0]).max(axis=1) / q
    nz = np.linalg.norm(b.astype(float), axis=1) > 0
    idx = np.where(nz)[0][np.argsort(resid[nz])[:n_candidates]]
    return b[idx]


def connect(b0, Acol, k0, ntoa):
    """Phase-connect forward from b0; return (b, wraps, global_residual)."""
    b = b0.astype(float).copy()
    wraps = [int(round(-(b @ Acol[i]))) for i in range(k0)]
    for i in range(k0, ntoa):
        wraps.append(int(round(-(b @ Acol[i]))))
        b, *_ = np.linalg.lstsq(Acol[:i + 1], -np.array(wraps, float), rcond=None)
    wraps = np.array(wraps)
    resid = float(np.linalg.norm(-(Acol @ b) - wraps))
    return b, wraps, resid


if __name__ == "__main__":
    d = load_data()
    il = d["integer_lattice"]; ntoa = len(d["toas_met_lattice"]); q = int(il[0, 0])
    tm = d["transformation_matrix"]; mul = d["mul_factor"]
    pv = d["probs_verify"]; Qn = np.sum(pv ** 2 / 2)
    order = np.argsort(np.asarray(d["toas_met_lattice"]))
    Acol = il[ntoa:, order].T / q

    def Qof(b):
        vf = np.mod(np.round(b).astype(np.int64) @ tm / mul + 0.5, 1) - 0.5
        return float(np.abs(np.sum(pv * np.exp(1j * 2 * np.pi * vf))) ** 2 / Qn)

    # ground-truth physical pulsar from the full sieve
    full = fold(d); m = full["reasonable_solutions_mask"]
    bstar = full["db_transformation_p"][m][np.argmax(full["Q_stat"][m])].astype(float)
    print(f"reference: full sieve finds pulsar Q={full['Q_stat'][m].max():.1f}")

    # connection works when seeded with the TRUE b*
    b, wraps, r = connect(bstar, Acol, 30, ntoa)
    print(f"connect from TRUE b*:        Q={Qof(b):7.1f}  std(wraps)={np.std(wraps):.1e}  resid={r:.1e}")

    # but the bootstrap cannot acquire it
    cands = _bootstrap(il, ntoa, q, order[:30], n_candidates=500)
    rows = [(*[None], *connect(b0, Acol, 30, ntoa)) for b0 in cands]
    by_resid = sorted(((r, np.std(w), Qof(b)) for _, b, w, r in rows))
    phys = [t for t in by_resid if t[1] > STD_PHYSICAL]
    print(f"bootstrap k0=30 ({len(cands)} candidates):")
    print(f"  best by residual (any):  Q={by_resid[0][2]:7.1f}  std(wraps)={by_resid[0][1]:.1e}"
          f"  resid={by_resid[0][0]:.1e}   <- trivial constant-phase vector")
    print(f"  best by residual (phys): Q={phys[0][2]:7.1f}  std(wraps)={phys[0][1]:.1e}"
          f"  resid={phys[0][0]:.1e}   <- pulsar NOT acquired")
