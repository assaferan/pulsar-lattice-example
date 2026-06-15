"""Fold Fermi-LAT TOAs using lattice algorithms.

This module turns the pipeline from ``fermi_fold_demo_paper.ipynb`` into an
importable function.  Given the data dictionary shipped in ``data/data.npy``
it reduces and sieves the timing lattice, folds the verification TOAs and
computes the Q statistic used to identify true pulsar solutions.

Example
-------
>>> from fermi_fold import load_data, fold
>>> data = load_data("data/data.npy")
>>> result = fold(data)
>>> result["Q_stat"].max()
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


def fold(data_needed, block_size=30, delta=0.95, reasonable_k_std=1e5):
    """Run the lattice folding pipeline on the given data.

    Parameters
    ----------
    data_needed : dict
        The data dictionary, as returned by :func:`load_data`.  Must contain
        the keys ``integer_lattice``, ``coeff_std``, ``toas_met_lattice``,
        ``transformation_matrix``, ``mul_factor`` and ``probs_verify``.
    block_size : int, optional
        Block size for the BKZ lattice reduction (default 30).
    delta : float, optional
        LLL/BKZ ``delta`` reduction parameter (default 0.95).
    reasonable_k_std : float, optional
        Minimum standard deviation of the integer coefficients ``k`` for a
        solution to be considered physical (default 1e5).

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

    # 1. Set up the lattice in fpylll's format.
    integer_matrix = fpylll.IntegerMatrix.from_iterable(
        *integer_lattice.shape,
        list(map(int, list(integer_lattice.flatten()))),
    )
    gso = fpylll.GSO.Mat(
        integer_matrix,
        flags=fpylll.GSO.INT_GRAM,
        U=fpylll.IntegerMatrix.identity(integer_matrix.nrows),
        UinvT=fpylll.IntegerMatrix.identity(integer_matrix.nrows),
    )

    # 2. Reduce the lattice with BKZ for a more balanced basis.
    bkz = fpylll.BKZ.Reduction(
        gso,
        fpylll.LLL.Reduction(gso, delta=delta),
        fpylll.BKZ.Param(
            block_size=block_size,
            strategies=fpylll.BKZ.DEFAULT_STRATEGY,
            delta=delta,
        ),
    )
    bkz()

    # 3. Sieve on successively larger sublattices (G6K "pump").
    g6k = Siever(gso)
    g6k.initialize_local(0, integer_lattice.shape[0] // 2, integer_lattice.shape[0])
    with g6k.temp_params(otf_lift=False):
        while g6k.l > len(coeff_std):
            g6k.extend_left(1)
            g6k(alg="hk3")
        g6k.extend_left(len(coeff_std))

    # Read the sieve results in the various formats.
    db_raw = np.array(list(g6k.itervalues()))
    db_transformation = db_raw @ np.array(list(g6k.M.U))
    db_transformation_k = db_transformation[:, : len(toas_met_lattice)]
    db_transformation_p = db_transformation[:, len(toas_met_lattice):]
    db_vectors = db_raw @ np.array(list(g6k.M.B))

    # 4. Fold the verification TOAs and compute the Q statistic.
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


if __name__ == "__main__":
    data = load_data()
    result = fold(data)
    mask = result["reasonable_solutions_mask"]
    Q_stat = result["Q_stat"][mask]
    print(f"Sieved {len(result['Q_stat'])} vectors, "
          f"{mask.sum()} physically reasonable.")
    if Q_stat.size:
        print(f"Max Q statistic: {Q_stat.max():.2f}")
