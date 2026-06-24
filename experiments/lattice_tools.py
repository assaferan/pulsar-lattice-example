"""Tweakable, transparent versions of the two lattice stages the pipeline uses:
LLL reduction and the G6K pump. These mirror what ``fermi_fold._sieve`` does, but
broken open with explicit steps and hooks so experiments can modify them
(q-aware reduction, custom pump shapes, vector injection, alternative selection,
early stopping, ...).

Contents
--------
* ``pure_lll``    -- pure-Python LLL. Exact integer basis, float64 GSO. Every
                     decision (size reduction, Lovasz swap) is explicit and
                     editable. Use for algorithm experiments on small lattices.
* ``fpylll_lll``  -- fast LLL via fpylll, same signature. Use for big lattices.
* ``make_siever`` -- build a g6k Siever from an integer basis.
* ``pump``        -- transparent pump over g6k primitives, with an ``on_round``
                     hook called after every sieve (inspect / inject / stop).
* ``database``    -- read the sieve DB as (raw coeffs, original-basis coeffs,
                     natural vectors).

Why float64 GSO is safe here: LLL keeps the basis as exact integers and only uses
the GSO (mu, ||b*||^2) to decide roundings and swaps. Those are ratios of inner
products; float64 carries ~1e-16 *relative* precision, which is plenty even
though the entries are ~1e15 (the absolute magnitudes cancel in the ratios).

Run the self-test (g6k env active):  PYTHONPATH=. python experiments/lattice_tools.py
"""

import numpy as np
import fpylll
# g6k is imported lazily inside make_siever/pump so the pure-Python tools
# (pure_lll) can be used without a g6k build.


# --------------------------------------------------------------------------- #
# LLL                                                                          #
# --------------------------------------------------------------------------- #
def _gso(B):
    """Gram-Schmidt of integer rows B (list of int-lists). Returns float
    (mu, bstar, norm2). Recomputed from scratch -- simple and tweak-friendly."""
    n = len(B)
    Bf = np.array([[float(x) for x in row] for row in B])
    bstar = np.zeros_like(Bf)
    mu = np.zeros((n, n))
    norm2 = np.zeros(n)
    for i in range(n):
        bstar[i] = Bf[i]
        for j in range(i):
            # Guard against a (near-)zero Gram-Schmidt norm, which arises for
            # (near-)rank-deficient bases -- e.g. a q-ary lattice below the
            # precision floor, where small entries round to 0. Treat mu = 0 there.
            mu[i, j] = (Bf[i] @ bstar[j] / norm2[j]) if norm2[j] > 1e-9 else 0.0
            bstar[i] -= mu[i, j] * bstar[j]
        norm2[i] = bstar[i] @ bstar[i]
    return mu, norm2


def pure_lll(basis, delta=0.99, verbose=False):
    """Pure-Python LLL. ``basis`` is an iterable of integer row vectors.

    Returns ``(B, U)``: the reduced integer basis and the unimodular transform
    with ``B = U @ basis`` (so a vector's coefficients map back to the original
    basis via ``c_orig = c_reduced @ U`` -- needed to read the timing parameters
    for the fold). The two LLL rules are right here to edit:

      * size reduction:  B[k] -= round(mu[k,j]) * B[j]
      * Lovasz swap:      ||b*_k||^2 < (delta - mu[k,k-1]^2) ||b*_{k-1}||^2

    GSO is recomputed after each basis change (clarity over speed); fine for
    experiment-scale lattices (n up to ~60). For larger, use ``fpylll_lll``.
    """
    B = [[int(x) for x in row] for row in basis]
    n = len(B)
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # tracks B = U@basis
    mu, norm2 = _gso(B)
    k = 1
    while k < n:
        # --- size-reduce B[k] against B[k-1..0] (apply same op to U) ---
        for j in range(k - 1, -1, -1):
            r = int(round(mu[k, j]))
            if r != 0:
                B[k] = [B[k][t] - r * B[j][t] for t in range(n)]
                U[k] = [U[k][t] - r * U[j][t] for t in range(n)]
                mu, norm2 = _gso(B)
        # --- Lovasz condition ---
        if norm2[k] >= (delta - mu[k, k - 1] ** 2) * norm2[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            U[k], U[k - 1] = U[k - 1], U[k]
            mu, norm2 = _gso(B)
            k = max(k - 1, 1)
        if verbose:
            print(f"  LLL k={k}", end="\r")
    return np.array(B, dtype=object), np.array(U, dtype=object)


def fpylll_lll(basis, delta=0.99):
    """Fast LLL via fpylll. Returns ``(B, U)`` like ``pure_lll`` (B = U @ basis)."""
    n = len(basis)
    IM = fpylll.IntegerMatrix.from_iterable(
        n, n, [int(x) for row in basis for x in row])
    U = fpylll.IntegerMatrix.identity(n)
    M = fpylll.GSO.Mat(IM, U=U, UinvT=fpylll.IntegerMatrix.identity(n),
                       flags=fpylll.GSO.INT_GRAM)
    fpylll.LLL.Reduction(M, delta=delta)()
    B = np.array([[IM[i, j] for j in range(n)] for i in range(n)], dtype=object)
    Umat = np.array([[U[i, j] for j in range(n)] for i in range(n)], dtype=object)
    return B, Umat


# --------------------------------------------------------------------------- #
# G6K pump                                                                     #
# --------------------------------------------------------------------------- #
def make_siever(basis):
    """A g6k Siever over the integer lattice with rows ``basis``."""
    from g6k import Siever
    n = len(basis)
    IM = fpylll.IntegerMatrix.from_iterable(
        n, n, [int(x) for row in basis for x in row])
    gso = fpylll.GSO.Mat(IM, flags=fpylll.GSO.INT_GRAM,
                         U=fpylll.IntegerMatrix.identity(n),
                         UinvT=fpylll.IntegerMatrix.identity(n))
    gso.update_gso()
    return Siever(gso)


def pump(siever, left_stop=0, start=None, alg="hk3", lift=True, on_round=None):
    """Transparent G6K pump.

    Sieve on a context ``[l, n]`` that grows leftward one coordinate per round,
    from ``start`` down to ``left_stop`` (smaller left_stop = deeper, more
    expensive; ``len(coeff_std)`` mimics the full pump, ``n//2 - k`` a shallow
    one). Mirrors ``fermi_fold._sieve`` but with a hook.

    Parameters
    ----------
    siever     : g6k Siever (from ``make_siever``).
    left_stop  : left bound to descend to (default 0 = full lattice).
    start      : initial left bound (default n//2).
    alg        : sieve algorithm -- "hk3", "bgj1", "gauss", ...
    lift       : final ``extend_left`` to read full-context vectors.
    on_round   : optional callback ``on_round(siever, l)`` after each sieve;
                 return False to stop early. Use it to inspect the DB, inject
                 vectors, change ``alg``, exploit q*Z^n, etc.
    """
    n = siever.M.B.nrows
    if start is None:
        start = n // 2
    siever.initialize_local(0, start, n)
    with siever.temp_params(otf_lift=False):
        while siever.l > left_stop:
            siever.extend_left(1)
            siever(alg=alg)
            if on_round is not None and on_round(siever, siever.l) is False:
                break
        if lift:
            siever.extend_left(siever.l)
    return siever


def database(siever, U_pre=None):
    """Read the sieve database. Returns (raw, coeffs, vectors):
      raw     : coefficients in the sieve's reduced basis (what it stores),
      coeffs  : coefficients in the basis passed to ``make_siever`` (raw @ M.U);
                if ``U_pre`` (an earlier LLL transform, B = U_pre @ original) is
                given, these are further mapped to the ORIGINAL basis
                (``@ U_pre``) -- which is what you need to read the timing
                parameters for the fold,
      vectors : the lattice vectors themselves (raw @ M.B).
    """
    raw = np.array(list(siever.itervalues()))
    coeffs = raw @ np.array(list(siever.M.U), dtype=object)
    if U_pre is not None:
        coeffs = coeffs @ U_pre
    vectors = raw @ np.array(list(siever.M.B), dtype=object)
    return raw, coeffs, vectors


# --------------------------------------------------------------------------- #
# self-test                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # 1) pure_lll vs fpylll_lll agree on a small lattice (shortest-vector length).
    rng = np.random.default_rng(0)
    A = rng.integers(-50, 50, size=(12, 12))
    while round(np.linalg.det(A)) == 0:
        A = rng.integers(-50, 50, size=(12, 12))
    def shortest(B):
        return min(np.linalg.norm(np.array(r, float)) for r in B if any(r))
    p, Up = pure_lll(A, delta=0.99)
    f, Uf = fpylll_lll(A, delta=0.99)
    print(f"pure_lll shortest={shortest(p):.3f}  fpylll_lll shortest={shortest(f):.3f}")
    assert np.array_equal(p, Up @ np.array(A, dtype=object)), "pure_lll: B != U@basis"
    assert np.array_equal(f, Uf @ np.array(A, dtype=object)), "fpylll_lll: B != U@basis"
    print("transform check: B == U @ basis for both  (OK)")

    # 2) full custom stack (fpylll_lll + pump) recovers the pulsar on real data.
    from fermi_fold import load_data
    data = load_data("data/data.npy")
    il = data["integer_lattice"]
    n_per = len(data["toas_met_lattice"])
    q = data["mul_factor"]
    B, U = fpylll_lll(il, delta=0.95)
    s = make_siever(B)
    pump(s, left_stop=27)                            # shallow pump (fast-mode depth)
    _, coeffs, _ = database(s, U_pre=U)              # map coeffs back to original il
    p_par = coeffs[:, n_per:].astype(float)
    vf = np.mod(p_par @ data["transformation_matrix"] / q + 0.5, 1) - 0.5
    Qn = np.sum(data["probs_verify"] ** 2 / 2)
    Q = np.abs(np.sum(data["probs_verify"] * np.exp(2j * np.pi * vf), axis=1)) ** 2 / Qn
    kstd = np.std(coeffs[:, :n_per].astype(float), axis=1)
    reasonable = kstd > 1e5
    best = Q[reasonable].max() if reasonable.any() else float("nan")
    print(f"custom stack on data.npy: {int(reasonable.sum())} reasonable, max Q = {best:.1f}")
