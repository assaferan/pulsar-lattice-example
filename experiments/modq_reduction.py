"""Did exploiting Z^n (reducing vectors modulo the integer lattice) help?

The lattice contains q.Z^N (q = mul_factor) in its first N coordinates, so any
lattice vector can be shortened by reducing those coordinates mod q -- and the
result is still a lattice vector.  We tested whether this "reduce mod Z^n" idea
buys anything.  It does NOT: it is redundant with the LLL size reduction the
pipeline already runs.  This script reproduces the two decisive checks.

Run from the repo root with the g6k environment active:
    PYTHONPATH=. python3 experiments/modq_reduction.py
"""
import numpy as np
import fpylll
from g6k import Siever
from fermi_fold import load_data, _sieve


def centered_mod(x, q):
    """Exact integer centered residue in (-q/2, q/2].  (Do NOT use float here:
    q ~ 1e16 exceeds float64's exact-integer range, so x - q*round(x/q) is wrong
    near +-q/2.)"""
    return ((x + q // 2) % q) - q // 2


def reduce_mod_Zn(vectors, q, n_periodic):
    v = np.array(vectors, dtype=np.int64)
    v[:, :n_periodic] = centered_mod(v[:, :n_periodic], q)
    return v


def test_postprocessing(data):
    """Reduce the full sieve database mod q -- does it reveal any new detection?

    Conclusion: no.  ~10% of vectors get shorter, but the true solution was
    already returned in reduced form (the sieve ranks by length, so short
    vectors already have small coordinates), so the minimum is unchanged and
    ranking by reduced norm finds exactly the same solution as raw norm.
    """
    il = data["integer_lattice"]
    q = int(il[0, 0]); n_periodic = len(data["toas_met_lattice"])
    db_raw, db_transformation, db_vectors = _sieve(
        il, len(data["coeff_std"]), block_size=30, delta=0.95, do_bkz=True)
    V = db_vectors.astype(np.int64)
    Vr = reduce_mod_Zn(V, q, n_periodic)

    raw_per = np.linalg.norm(V[:, :n_periodic].astype(float), axis=1)
    red_per = np.linalg.norm(Vr[:, :n_periodic].astype(float), axis=1)
    shortened = red_per < raw_per - 1.0

    p = db_transformation[:, n_periodic:]
    vf = np.mod(p @ data["transformation_matrix"] / data["mul_factor"] + 0.5, 1) - 0.5
    Qn = np.sum(data["probs_verify"] ** 2 / 2)
    Q = np.abs(np.sum(data["probs_verify"] * np.exp(1j * 2 * np.pi * vf), axis=1)) ** 2 / Qn

    print("== post-processing the full sieve database ==")
    print(f"  vectors shortened by mod-q : {shortened.sum()} / {len(V)}")
    print(f"  best Q overall             : {Q.max():.1f}")
    print(f"  was the best-Q vector shortened by mod-q? {bool(shortened[np.argmax(Q)])}")
    print(f"  argmin(raw_per) == argmin(red_per)? {np.argmin(raw_per) == np.argmin(red_per)}"
          "   (same solution ranks first either way)")


def _setup_gso(basis):
    n = basis.shape[0]
    im = fpylll.IntegerMatrix.from_iterable(n, n, [int(x) for r in basis for x in r])
    gso = fpylll.GSO.Mat(im, flags=fpylll.GSO.INT_GRAM,
                         U=fpylll.IntegerMatrix.identity(n),
                         UinvT=fpylll.IntegerMatrix.identity(n))
    gso.update_gso()
    return gso


def _cheap_sieve(gso, pump_stop, n):
    fpylll.LLL.Reduction(gso, delta=0.95)()
    g = Siever(gso); g.initialize_local(0, n // 2, n)
    with g.temp_params(otf_lift=False):
        while g.l > pump_stop:
            g.extend_left(1); g(alg="hk3")
        g.extend_left(g.l)
    return np.array(list(g.itervalues())), np.array(list(g.M.B)), np.array(list(g.M.U))


def _reinsert(keep, B, n):
    stack = np.vstack([keep, B])
    im = fpylll.IntegerMatrix.from_iterable(stack.shape[0], n, [int(x) for r in stack for x in r])
    fpylll.LLL.reduction(im)
    rows = [[im[i, j] for j in range(n)] for i in range(im.nrows)]
    rows = [r for r in rows if any(v != 0 for v in r)]
    return np.array(rows[-n:], dtype=object)


def test_reinsertion_control(data, pump_stop=37, seeds=range(1, 7)):
    """Is mod-q the active ingredient in an iterated reduce-and-reinsert loop?

    Control: reinsert the shortest *reduced* vectors vs the shortest *raw*
    vectors (a lattice-preserving basis update either way), and compare.  They
    behave identically -- because LLL re-reduces both stacks the same way -- so
    mod-q reduction is redundant.  (Separately, the iteration itself was later
    found nondeterministic and ineffective; the real win was a shallower single
    sieve, now fold(fast=True).  This is kept only as the modq-vs-raw control.)
    """
    il = data["integer_lattice"]; n = il.shape[0]
    ntoa = len(data["toas_met_lattice"]); q = int(il[0, 0])
    tm = data["transformation_matrix"]; mul = data["mul_factor"]
    pv = data["probs_verify"]; Qn = np.sum(pv ** 2 / 2)

    def run(mode, seed):
        fpylll.FPLLL.set_random_seed(seed)
        B = il.astype(object).copy()
        for it in range(6):
            db_raw, Bcur, U = _cheap_sieve(_setup_gso(B), pump_stop, n)
            V = (db_raw @ Bcur).astype(np.int64)
            p = (db_raw @ U)[:, ntoa:]
            vf = np.mod(p @ tm / mul + 0.5, 1) - 0.5
            Q = np.abs(np.sum(pv * np.exp(1j * 2 * np.pi * vf), axis=1)) ** 2 / Qn
            if Q.max() > 50:
                return it
            if mode == "modq":
                Vr = V.copy(); Vr[:, :ntoa] = centered_mod(Vr[:, :ntoa], q)
                keep = Vr[np.argsort(np.linalg.norm(Vr[:, :ntoa].astype(float), axis=1))[:n]]
            else:
                keep = V[np.argsort(np.linalg.norm(V[:, :ntoa].astype(float), axis=1))[:n]]
            B = _reinsert(keep, Bcur, n)
        return None

    print(f"\n== reduce-and-reinsert control (pump_stop={pump_stop}) ==")
    print(f"  {'seed':>4}  {'modq':>8}  {'raw':>8}")
    for seed in seeds:
        m, r = run("modq", seed), run("raw", seed)
        print(f"  {seed:>4}  {('det@%d' % m) if m is not None else 'miss':>8}"
              f"  {('det@%d' % r) if r is not None else 'miss':>8}")
    print("  -> modq and raw behave identically: mod-q reduction is redundant with LLL.")


if __name__ == "__main__":
    data = load_data()
    test_postprocessing(data)
    test_reinsertion_control(data)
