"""Model B -- synthetic data for the full 7-parameter pulsar timing model.

The pulse phase of a photon arriving at (mission-elapsed) time ``t`` is modelled
as a *linear* combination of seven design functions,

    Phi(t) = sum_i  theta_i * M_i(t),

with parameters ``theta = (phi, f, fdot, dalpha, ddelta, dalpha_dot, ddelta_dot)``
and design functions ``M_i(t) = dPhi/dtheta_i``:

    M_phi        = 1
    M_f          = dt                      (dt = t - t_ref)
    M_fdot       = dt**2 / 2
    M_alpha      = f0/c * r_earth(t) . dnhat/dalpha     (Roemer delay, RA)
    M_delta      = f0/c * r_earth(t) . dnhat/ddelta     (Roemer delay, Dec)
    M_alpha_dot  = dt * M_alpha            (proper motion in RA)
    M_delta_dot  = dt * M_delta            (proper motion in Dec)

The spin terms (phi, f, fdot) make the phase a quadratic in time.  The
astrometric terms are the *Roemer delay* -- the light-travel time across the
Earth's orbit, ``r_earth(t) . nhat / c`` -- which modulates the arrival phase
annually; ``alpha, delta`` are searched as small offsets of the line of sight
``nhat`` about a reference position, and ``alpha_dot, delta_dot`` let that offset
drift (proper motion).  The model is exactly linear in all seven parameters, so
the same lattice construction as Model A applies (see
``model_a_constant_frequency.py``), with seven parameter rows.

Lattice layout (n = n_toas + 7; q == ``mul_factor``):

                      n_toas cols                 7 param cols
        [ q * I_ntoa                       0                     ]  wrap rows
        [ round(q * s_i * M_i(t_j))        diag(round(q s_i/sd)) ]  param rows

A short vector folds every photon while keeping the parameters within their
priors -- the timing solution.

IMPORTANT -- prior penalties.  With seven parameter rows the lattice is rich
enough to contain many spurious short vectors (large-coefficient combinations
that wrap repeatedly but fold poorly).  The diagonal penalty
``round(q s_i / coeff_std_i)`` suppresses them: a prior std of ``coeff_std_i``
makes any excursion beyond it expensive.  The default ``coeff_std = prior_factor
* |theta|`` (prior ~10x the true value) gives a true-solution penalty ~0.1 q,
short enough to keep the true vector while pricing junk out.  Too tight a prior
(prior_factor <~ 3) makes the sieve fail to saturate; too loose (-> blind) lets
the junk back in.

Run this file to write ``data/data_generated_B.npy`` and verify it round-trips.
"""

import os

import numpy as np

PARAM_LABELS = ("phi", "f", "fdot", "alpha", "delta", "alpha_dot", "delta_dot")

# Physical constants.
AU_OVER_C = 499.0047837              # light travel time across 1 AU [s]
YEAR = 365.25 * 86400.0             # [s]
OBLIQUITY = np.deg2rad(23.4392811)  # Earth axial tilt


def _earth_unit_position(toas, consts):
    """Earth's heliocentric direction (equatorial frame), shape (3, len(toas)).

    Circular orbit: ecliptic longitude ``lam = 2 pi (t - t_equinox)/year``; the
    ecliptic position ``(cos lam, sin lam, 0)`` is tilted into equatorial
    coordinates by the obliquity.  (The 1 AU magnitude is folded into AU/c.)
    """
    lam = 2 * np.pi * (toas - consts["t_equinox"]) / YEAR
    eps = OBLIQUITY
    return np.vstack([
        np.cos(lam),
        np.cos(eps) * np.sin(lam),
        np.sin(eps) * np.sin(lam),
    ])


def design_matrix(toas, consts):
    """The 7 design functions ``M_i(t) = dPhi/dtheta_i``, shape (7, len(toas))."""
    toas = np.asarray(toas, dtype=float)
    dt = toas - consts["t_ref"]

    a0, d0, f0 = consts["alpha0"], consts["delta0"], consts["f0"]
    nhat_dalpha = np.array([-np.cos(d0) * np.sin(a0),  np.cos(d0) * np.cos(a0), 0.0])
    nhat_ddelta = np.array([-np.sin(d0) * np.cos(a0), -np.sin(d0) * np.sin(a0), np.cos(d0)])

    r = _earth_unit_position(toas, consts)               # (3, n)
    roemer_scale = f0 * AU_OVER_C
    M_alpha = roemer_scale * (nhat_dalpha @ r)           # (n,)
    M_delta = roemer_scale * (nhat_ddelta @ r)           # (n,)

    return np.vstack([
        np.ones_like(dt),       # phi
        dt,                     # f
        0.5 * dt ** 2,          # fdot
        M_alpha,                # alpha
        M_delta,                # delta
        dt * M_alpha,           # alpha_dot
        dt * M_delta,           # delta_dot
    ])


def generate_toas(n_toas, theta, consts, phase_std, rng):
    """Sample photon arrival times whose phases fold to a common value.

    ``Phi`` is quadratic-plus-orbital in ``t`` and has no closed-form inverse, so
    each photon is placed by Newton iteration: pick a random base time, assign
    the nearest integer pulse number ``K`` plus small intra-pulse scatter
    ``eps``, then slide ``t`` until ``Phi(t) = K + eps`` (the local derivative
    ``dPhi/dt`` is dominated by the spin frequency ``f``).
    """
    theta = np.asarray(theta, dtype=float)
    f = theta[1]
    toas = consts["t_start"] + consts["t_span"] * rng.random(n_toas)
    Ks = np.round(theta @ design_matrix(toas, consts))
    target = Ks + rng.normal(0.0, phase_std, size=n_toas)
    for _ in range(5):                                   # Newton; converges fast
        toas = toas + (target - theta @ design_matrix(toas, consts)) / f
    return np.sort(toas)


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
    # true parameters (phi, f, fdot, dalpha, ddelta, dalpha_dot, ddelta_dot)
    theta=(0.3, 50.0, -1e-14, 5e-7, 5e-7, 3e-16, 3e-16),
    alpha0=0.85, delta0=0.05,        # reference sky position [rad]
    t_start=2.0e8, t_span=3.0e8,     # MET window [s]  (~9.5 yr baseline)
    t_equinox=0.0,                   # phase of Earth's orbit
    phase_std=0.03,                  # intra-pulse phase scatter [cycles]
    mul_factor=10 ** 15,             # q
    res_frac=0.01,                   # parameter-step resolution [cycles]
    prior_factor=10.0,               # prior std as a multiple of |true value|
    coeff_std=None,                  # explicit prior std per parameter (overrides)
    seed=0,
):
    """Build a data dict (same keys as ``data/data.npy``) for Model B."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta, dtype=float)
    consts = dict(
        t_ref=t_start + t_span / 2.0, t_start=t_start, t_span=t_span,
        t_equinox=t_equinox, alpha0=alpha0, delta0=delta0, f0=theta[1],
    )

    toas_lattice = generate_toas(n_toas, theta, consts, phase_std, rng)
    toas_verify = generate_toas(n_verify, theta, consts, phase_std, rng)

    M = design_matrix(toas_lattice, consts)              # (7, n_toas)
    steps = res_frac / np.max(np.abs(M), axis=1)

    if coeff_std is None:
        # Loose prior centred on the truth; falls back to "blind" for any
        # parameter whose true value is zero (avoids a divide-by-zero penalty).
        coeff_std = prior_factor * np.abs(theta)
        coeff_std[coeff_std == 0] = (mul_factor * steps)[coeff_std == 0]
    coeff_std = np.asarray(coeff_std, dtype=float)

    integer_lattice = build_integer_lattice(M, mul_factor, steps, coeff_std)

    M_verify = design_matrix(toas_verify, consts)
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


def _fold_robust(data, retries=5, **kwargs):
    """Run ``fold`` with retries: g6k's shallow pump occasionally raises a
    (non-deterministic) SaturationError on this lattice; just try again."""
    from fermi_fold import fold
    last = None
    for _ in range(retries):
        try:
            return fold(data, **kwargs)
        except Exception as exc:        # SaturationError lives inside g6k
            last = exc
    raise last


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "data", "data_generated_B.npy")
    data = generate_data()
    np.save(out, data, allow_pickle=True)
    print(f"wrote {out}")
    # fast mode uses pump_stop=30 here: the deeper default (27) is fast enough
    # but trips g6k's saturation check on this lattice more often.
    for fast, kw in [(False, {}), (True, {"pump_stop": 30})]:
        r = _fold_robust(data, fast=fast, reasonable_k_std=1e3, **kw)
        m = r["reasonable_solutions_mask"]; q = r["Q_stat"][m]
        print(f"[{'fast' if fast else 'full'}] {int(m.sum())} reasonable, "
              f"max Q = {q.max() if q.size else float('nan'):.1f}")
