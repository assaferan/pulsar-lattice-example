"""Model A -- synthetic data for a *constant-frequency* pulsar.

The pulse phase of a photon arriving at (mission-elapsed) time ``t`` is

    Phi(t) = phi + f * (t - t_ref)

with two parameters: the reference phase ``phi`` and spin frequency ``f``.  A
genuine pulsar photon arrives near an integer phase, so for the true ``(phi, f)``
every ``Phi(t_j)`` is close to a whole number.  Recovering ``(phi, f)`` from the
arrival times -- without knowing the integer pulse counts -- is the short-vector
problem :func:`fermi_fold.fold` solves.

Lattice layout (n = n_toas + 2; q == ``mul_factor``):

                      n_toas cols                 2 param cols
        [ q * I_ntoa                       0                     ]  wrap rows
        [ round(q * s_i * M_i(t_j))        diag(round(q s_i/sd)) ]  param rows

* the ``q * I`` rows subtract whole pulse rotations;
* each parameter row carries that parameter's (scaled, rounded) phase
  contribution to every TOA -- the design matrix ``M_i = dPhi/dtheta_i`` -- plus
  a diagonal prior penalty ``round(q s_i / coeff_std_i)`` that keeps the
  parameter within its prior ``coeff_std_i``.

A short vector folds every photon while keeping the parameters sane: the timing
solution.  See ``model_b_full_timing.py`` for the full 7-parameter model.

Run this file to write ``data/data_generated_A.npy`` and verify it round-trips.
"""

import os

import numpy as np

PARAM_LABELS = ("phi", "f")


def design_matrix(toas, t_ref):
    """Partials ``dPhi/dtheta`` for ``theta = (phi, f)``, shape (2, len(toas))."""
    toas = np.asarray(toas, dtype=float)
    return np.vstack([
        np.ones_like(toas),     # dPhi/dphi
        toas - t_ref,           # dPhi/df
    ])


def generate_toas(n_toas, f_true, phi_true, t_ref, t_span, phase_std, rng):
    """Sample photon arrival times from a constant-frequency pulsar.

    Each photon gets a random integer pulse number ``K`` and small intra-pulse
    scatter ``eps``; inverting ``Phi(t) = K + eps`` gives
    ``t = t_ref + (K + eps - phi)/f``.
    """
    k_max = int(f_true * t_span / 2)
    Ks = rng.integers(-k_max, k_max, size=n_toas)
    eps = rng.normal(0.0, phase_std, size=n_toas)
    return np.sort(t_ref + (Ks + eps - phi_true) / f_true)


def build_integer_lattice(M, mul_factor, steps, coeff_std):
    """Assemble the ``(n, n)`` integer basis from design matrix ``M`` (object
    dtype keeps the big integers exact)."""
    n_params, n_toas = M.shape
    lattice = np.zeros((n_toas + n_params, n_toas + n_params), dtype=object)
    for j in range(n_toas):                              # wrap rows: q * I
        lattice[j, j] = int(mul_factor)
    for i in range(n_params):                            # parameter rows
        row = n_toas + i
        for j in range(n_toas):
            lattice[row, j] = int(round(mul_factor * steps[i] * M[i, j]))
        lattice[row, n_toas + i] = int(round(mul_factor * steps[i] / coeff_std[i]))
    return lattice


def generate_data(
    n_toas=70,
    n_verify=1281,
    f_true=50.0,            # spin frequency [Hz]
    phi_true=0.3,           # reference phase [cycles]
    t_start=2.0e8,          # MET window start [s]
    t_span=3.0e8,           # MET window length [s]
    phase_std=0.03,         # intra-pulse phase scatter [cycles]
    mul_factor=10 ** 15,    # q: integerisation scale
    res_frac=0.01,          # parameter-step resolution [cycles]
    prior_factor=10.0,      # prior std as a multiple of |true value|
    seed=0,
):
    """Build a data dict (same keys as ``data/data.npy``) for Model A."""
    rng = np.random.default_rng(seed)
    t_ref = t_start + t_span / 2.0
    theta = np.array([phi_true, f_true])

    toas_lattice = generate_toas(n_toas, f_true, phi_true, t_ref, t_span, phase_std, rng)
    toas_verify = generate_toas(n_verify, f_true, phi_true, t_ref, t_span, phase_std, rng)

    M = design_matrix(toas_lattice, t_ref)               # (2, n_toas)
    steps = res_frac / np.max(np.abs(M), axis=1)         # phase-resolution per param
    coeff_std = prior_factor * np.abs(theta)             # loose prior centred on truth

    integer_lattice = build_integer_lattice(M, mul_factor, steps, coeff_std)

    # transformation_matrix[i,m] == q * s_i * M_i(verify TOA m) so that fold()'s
    # `p @ transformation_matrix / q` reconstructs the photon phases.
    M_verify = design_matrix(toas_verify, t_ref)
    transformation_matrix = mul_factor * steps[:, None] * M_verify
    span_vecs = mul_factor * steps[:, None] * M

    return {
        "toas_met_lattice": toas_lattice,
        "span_vecs": span_vecs,
        "coeff_std": coeff_std,
        "integer_lattice": integer_lattice,
        "mul_factor": float(mul_factor),
        "transformation_matrix": transformation_matrix,
        "toas_met_verify": toas_verify,
        "probs_verify": np.ones(n_verify, dtype=np.float32),
        "theta_true": theta,
        "steps": steps,
    }


if __name__ == "__main__":
    from fermi_fold import fold

    out = os.path.join(os.path.dirname(__file__), "data", "data_generated_A.npy")
    data = generate_data()
    np.save(out, data, allow_pickle=True)
    print(f"wrote {out}")
    for fast in (False, True):
        r = fold(data, fast=fast, reasonable_k_std=1e3)
        m = r["reasonable_solutions_mask"]; q = r["Q_stat"][m]
        print(f"[{'fast' if fast else 'full'}] {int(m.sum())} reasonable, "
              f"max Q = {q.max() if q.size else float('nan'):.1f}")
