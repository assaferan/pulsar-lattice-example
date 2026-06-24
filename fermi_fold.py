"""Fold Fermi-LAT TOAs using lattice algorithms.

This module turns the pipeline from ``fermi_fold_demo_paper.ipynb`` into an
importable function.  Given the data dictionary shipped in ``data/data.npy``
it reduces and sieves the timing lattice, folds the verification TOAs and
computes the Q statistic used to identify true pulsar solutions.

Two search strategies are available:

* the default **full sieve** (one BKZ reduction followed by a deep G6K pump);
* a **fast sieve** (``fast=True``): LLL plus a much shallower pump.  The solution
  vector is found at a modest sieving dimension, so the shallow pump recovers it
  reliably about two orders of magnitude faster than the full pump (~0.2 s vs
  ~20 s), at a still-decisive Q statistic.

Example
-------
>>> from fermi_fold import load_data, fold
>>> data = load_data("data/data.npy")
>>> result = fold(data)                 # full sieve
>>> result["Q_stat"].max()
>>> fast = fold(data, fast=True)        # shallow sieve, same detection, faster
"""

import numpy as np

import fpylll
from g6k import Siever


def load_data(path="data/data.npy"):
    """Load the data dictionary used by :func:`fold`.

    Parameters
    ----------
    path : str
        Path to the ``.npy`` file containing the pickled data dictionary.

    Returns
    -------
    dict
        The data dictionary (see the notebook / README for the keys).
    """
    return np.load(path, allow_pickle=True).tolist()


def _reduced_gso(integer_lattice, block_size, delta, do_bkz):
    """Build a fresh GSO (tracking U) and BKZ- (``do_bkz``) or LLL-reduce it."""
    n = integer_lattice.shape[0]
    integer_matrix = fpylll.IntegerMatrix.from_iterable(
        n, n, list(map(int, list(integer_lattice.flatten()))),
    )
    gso = fpylll.GSO.Mat(
        integer_matrix,
        flags=fpylll.GSO.INT_GRAM,
        U=fpylll.IntegerMatrix.identity(n),
        UinvT=fpylll.IntegerMatrix.identity(n),
    )
    if do_bkz:
        fpylll.BKZ.Reduction(
            gso,
            fpylll.LLL.Reduction(gso, delta=delta),
            fpylll.BKZ.Param(
                block_size=block_size,
                strategies=fpylll.BKZ.DEFAULT_STRATEGY,
                delta=delta,
            ),
        )()
    else:
        fpylll.LLL.Reduction(gso, delta=delta)()
    return gso


def _sieve(integer_lattice, pump_stop, block_size, delta, do_bkz, alg="hk3"):
    """Reduce and sieve the lattice; return (db_raw, db_transformation, db_vectors).

    With ``do_bkz`` the basis is reduced with BKZ (the full strategy); otherwise
    only LLL is used (the fast strategy).  ``pump_stop`` is the left bound the
    G6K pump descends to: smaller means a deeper, more expensive sieve.

    ``alg`` is the G6K sieve algorithm (``"hk3"``, ``"bgj1"``, ``"bdgl2"``, ...).
    ``hk3`` (the default triple sieve) is memory-optimized and fastest at low
    dimension, but its small database raises ``SaturationError`` / misses on the
    highly skewed q-ary lattice once the sieving dimension exceeds ~80-100. On
    that failure we automatically rebuild and retry, first with ``bgj1`` (robust,
    detects p=0.7 at N~111 in ~80s with a tuned pump), then with ``bdgl2`` whose
    asymptotically-best time exponent (~0.292 vs hk3's 0.36) pays off only at
    substantially larger dimension. So large-N searches work without changing
    ``alg``.
    """
    n = integer_lattice.shape[0]
    # try the requested alg, then fall back hk3->bgj1->bdgl2 (dedup, keep order)
    algorithms = list(dict.fromkeys([alg, "bgj1", "bdgl2"]))
    g6k = None
    for a in algorithms:
        gso = _reduced_gso(integer_lattice, block_size, delta, do_bkz)
        g6k = Siever(gso)
        g6k.initialize_local(0, n // 2, n)
        try:
            with g6k.temp_params(otf_lift=False):   # sieve over larger sublattices
                while g6k.l > pump_stop:
                    g6k.extend_left(1)
                    g6k(alg=a)
                g6k.extend_left(g6k.l)
            break
        except Exception:
            if a == algorithms[-1]:                 # fallback also failed
                raise

    # Read the sieve results in the various formats.
    db_raw = np.array(list(g6k.itervalues()))
    db_transformation = db_raw @ np.array(list(g6k.M.U))
    db_vectors = db_raw @ np.array(list(g6k.M.B))
    return db_raw, db_transformation, db_vectors


def fold(data_needed, fast=False, pump_stop=27, block_size=30, delta=0.95,
         reasonable_k_std=1e5, alg="hk3"):
    """Run the lattice folding pipeline on the given data.

    Parameters
    ----------
    data_needed : dict
        The data dictionary, as returned by :func:`load_data`.  Must contain
        the keys ``integer_lattice``, ``coeff_std``, ``toas_met_lattice``,
        ``transformation_matrix``, ``mul_factor`` and ``probs_verify``.
    fast : bool, optional
        If ``True`` use the fast strategy (LLL + shallow pump) instead of the
        full sieve (BKZ + deep pump). Default ``False``.
    pump_stop : int, optional
        Fast mode only: left bound the shallow pump descends to (larger ==
        cheaper/weaker; default 27, i.e. a sieving dimension of ``n - 27``).
    block_size : int, optional
        Block size for the BKZ lattice reduction in the full sieve (default 30).
    delta : float, optional
        LLL/BKZ ``delta`` reduction parameter (default 0.95).
    reasonable_k_std : float, optional
        Minimum standard deviation of the integer coefficients ``k`` for a
        solution to be considered physical (default 1e5).
    alg : str, optional
        G6K sieve algorithm (default ``"hk3"``). ``hk3`` is fastest at low
        dimension but raises ``SaturationError`` / misses on this skewed q-ary
        lattice past sieving dimension ~80-100; the sieve then falls back
        automatically to ``"bgj1"`` and then ``"bdgl2"`` (which handle the larger
        dimensions), so large-N searches work without changing this.

    Returns
    -------
    dict
        Dictionary with the following entries:

        ``db_raw``
            Coefficients of the reduced basis for every sieved vector.
        ``db_transformation_k`` / ``db_transformation_p``
            Coefficients of the original basis, split into the integer part
            ``k`` and the timing-parameter part ``p``.
        ``db_vectors``
            The folds (sieved vectors in the natural basis).
        ``verify_fold``
            The verification TOAs folded according to each solution.
        ``Q_stat``
            The Q statistic for every solution.
        ``reasonable_solutions_mask``
            Boolean mask selecting physically reasonable solutions.
    """
    integer_lattice = data_needed["integer_lattice"]
    coeff_std = data_needed["coeff_std"]
    toas_met_lattice = data_needed["toas_met_lattice"]
    transformation_matrix = data_needed["transformation_matrix"]
    mul_factor = data_needed["mul_factor"]
    probs_verify = data_needed["probs_verify"]
    n_periodic = len(toas_met_lattice)

    if fast:
        db_raw, db_transformation, db_vectors = _sieve(
            integer_lattice, pump_stop, block_size, delta, do_bkz=False, alg=alg)
    else:
        db_raw, db_transformation, db_vectors = _sieve(
            integer_lattice, len(coeff_std), block_size, delta, do_bkz=True, alg=alg)

    db_transformation_k = db_transformation[:, :n_periodic]
    db_transformation_p = db_transformation[:, n_periodic:]

    # Fold the verification TOAs and compute the Q statistic.
    verify_fold = (
        np.mod(db_transformation_p @ transformation_matrix / mul_factor + 0.5, 1) - 0.5
    )
    Q_stat_normalization = np.sum(probs_verify ** 2 / 2)
    Q_stat = (
        np.abs(np.sum(probs_verify * np.exp(1j * 2 * np.pi * verify_fold), axis=1)) ** 2
        / Q_stat_normalization
    )

    # Filter out unphysical solutions (too few phase revolutions).
    reasonable_solutions_mask = np.std(db_transformation_k, axis=1) > reasonable_k_std

    return {
        "db_raw": db_raw,
        "db_transformation_k": db_transformation_k,
        "db_transformation_p": db_transformation_p,
        "db_vectors": db_vectors,
        "verify_fold": verify_fold,
        "Q_stat": Q_stat,
        "reasonable_solutions_mask": reasonable_solutions_mask,
    }


def fold_subsampled(data_needed, n_sub=85, tries=4, detect_q=50.0, seed=0,
                    **fold_kwargs):
    """Run :func:`fold` on a random subset of the lattice TOAs.

    g6k's sieve fails to saturate on this (highly skewed, q-ary) lattice once the
    dimension exceeds ~100, so the full pipeline cannot be sieved directly for
    large numbers of TOAs. Because the timing parameters are shared across all
    TOAs, a subset of them still pins the solution -- and detection is scored on
    the full (unchanged) verification set. We sieve ``n_sub`` randomly chosen TOAs
    (keeping the dimension tractable), retrying fresh subsets (this also rides out
    g6k's intermittent ``SaturationError``), and return the result with the
    highest max Q.

    The sub-lattice is the original ``[[q*I, 0], [A, L]]`` restricted to the
    chosen TOA rows/columns plus the parameter rows/columns.

    Parameters
    ----------
    n_sub : number of TOAs to sieve (default 85; keep below ~95).
    tries : number of random subsets to attempt; stops early once a subset
        detects (max Q over physical solutions > ``detect_q``).

    Returns the best :func:`fold` result dict, or ``None`` if every attempt
    raised.
    """
    integer_lattice = data_needed["integer_lattice"]
    toas = np.asarray(data_needed["toas_met_lattice"])
    n_toas = len(toas)
    n_params = integer_lattice.shape[0] - n_toas
    n_sub = min(n_sub, n_toas)
    rng = np.random.default_rng(seed)

    best, best_q = None, -1.0
    for _ in range(tries):
        sub = np.sort(rng.choice(n_toas, n_sub, replace=False))
        ix = np.concatenate([sub, np.arange(n_toas, n_toas + n_params)])
        sub_data = dict(data_needed)
        sub_data["integer_lattice"] = integer_lattice[np.ix_(ix, ix)]
        sub_data["toas_met_lattice"] = toas[sub]
        try:
            result = fold(sub_data, **fold_kwargs)
        except Exception:
            continue
        mask = result["reasonable_solutions_mask"]
        max_q = float(result["Q_stat"][mask].max()) if mask.any() else 0.0
        if max_q > best_q:
            best, best_q = result, max_q
        if max_q > detect_q:
            break
    return best


if __name__ == "__main__":
    import sys
    fast = "--fast" in sys.argv[1:]
    data = load_data()
    result = fold(data, fast=fast)
    mask = result["reasonable_solutions_mask"]
    Q_stat = result["Q_stat"][mask]
    label = "fast" if fast else "full"
    print(f"[{label}] Sieved {len(result['Q_stat'])} vectors, "
          f"{mask.sum()} physically reasonable.")
    if Q_stat.size:
        print(f"Max Q statistic: {Q_stat.max():.2f}")
