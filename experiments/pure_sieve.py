"""A pure-Python sieve -- "hk3 cracked open".

g6k's `hk3` is a fast C++ *triple* sieve (Herold-Kirshanova 3-tuple): it keeps a
database of lattice vectors and repeatedly replaces a vector by a shorter
combination of itself with one (pair) or two (triple) database vectors, until the
database saturates with short vectors. `lattice_tools.pump` only orchestrates the
context around that C++ core; this module reimplements the core itself in plain,
editable Python so experiments can change the reduction rule, the tuple search,
the sampler, or add structure-aware moves (e.g. q*Z^n).

It is exact-integer (norms reach ~q^2 ~ 1e30, far past float64's exact range) and
therefore slow -- use it at small dimension (lattice dim <~ 30). For scale, use
the g6k `hk3` via `lattice_tools`.

Each database entry carries both the lattice vector and its integer coefficients
in the input basis (vec = coef @ basis), so mapping a result back to the original
basis is exact -- no float matrix inversion.

The two moves to edit:
  * pair  (Gauss / 2-reduction):  p <- p - round(<p,v>/<v,v>) v
  * triple (the hk3 essence):     p <- p (+/-) v (+/-) w   if that is shorter

Run the self-test (g6k env not required):  PYTHONPATH=. python experiments/pure_sieve.py
"""

import numpy as np


def _n2(v):
    """Exact squared Euclidean norm of an integer vector."""
    return sum(int(x) * int(x) for x in v)


def _sub(a, s, b):
    """Exact a - s*b, elementwise."""
    return [int(a[t]) - s * int(b[t]) for t in range(len(a))]


def _pair_reduce(pv, pc, p2, db, mod_q=None):
    """Gauss (2-)reduction of (vector pv, coeffs pc) against the database. A move
    is accepted only if it strictly reduces the norm, so float rounding of the
    Gauss coefficient can never produce a longer vector.

    If ``mod_q=(n_per, q)``, each candidate is reduced mod q before its length is
    checked, and a standalone wrap of p itself is tried each pass (subtracting
    q*e_j directly). Requires coefficients in the original/wrap basis."""
    changed = True
    while changed and p2 > 0:
        changed = False
        for v, c, v2 in db:
            if v2 == 0:
                continue
            ip = sum(int(a) * int(b) for a, b in zip(pv, v))
            if 2 * abs(ip) <= v2:                 # already reduced wrt this v
                continue
            s = int(round(ip / v2))
            if s == 0:
                continue
            cand, cc = _sub(pv, s, v), _sub(pc, s, c)
            if mod_q is not None:
                cand, cc, c2 = _mod_q(cand, cc, mod_q[0], mod_q[1])
            else:
                c2 = _n2(cand)
            if c2 < p2:
                pv, pc, p2 = cand, cc, c2
                changed = True
        if mod_q is not None and p2 > 0:          # try wrapping p itself
            wv, wc, w2 = _mod_q(pv, pc, mod_q[0], mod_q[1])
            if w2 < p2:
                pv, pc, p2 = wv, wc, w2
                changed = True
    return pv, pc, p2


# Firing counter: how often _mod_q is invoked and how often it actually wraps a
# coordinate. Reset with ``reset_modq_stats()``; read ``MODQ_STATS`` afterwards.
MODQ_STATS = {"calls": 0, "fires": 0}


def reset_modq_stats():
    MODQ_STATS["calls"] = 0
    MODQ_STATS["fires"] = 0


def _mod_q(vec, coef, n_per, q):
    """Reduce the first ``n_per`` coordinates mod q to their centered
    representative -- a lattice move, since the wrap rows q*e_j are in the
    lattice. Requires ``coef`` to be in the ORIGINAL basis (whose j-th row is
    q*e_j for j < n_per), so the bookkeeping is just ``coef[j] -= s``. Returns
    new (vec, coef, norm2)."""
    MODQ_STATS["calls"] += 1
    vec = list(vec)
    coef = list(coef)
    fired = False
    for j in range(n_per):
        vj = int(vec[j])
        s = int(round(vj / q))
        if s:
            vec[j] = vj - s * q
            coef[j] = int(coef[j]) - s
            fired = True
    if fired:
        MODQ_STATS["fires"] += 1
    return vec, coef, _n2(vec)


def _triple_reduce(pv, pc, p2, db, topk, mod_q=None):
    """3-tuple reduction (the hk3 move): try p (+/-) v (+/-) w over the topk
    shortest database vectors; keep the shortest result below ||p||.

    If ``mod_q=(n_per, q)``, each candidate is reduced mod q (over its first
    n_per coordinates) BEFORE its length is checked -- so a triple that is long
    in raw coordinates but short after wrapping is still accepted. (Requires
    coefficients in the original/wrap basis; see ``_mod_q``.)"""
    short = sorted(db, key=lambda e: e[2])[:topk]
    bv, bc, b2 = pv, pc, p2
    for i in range(len(short)):
        vi, ci, _ = short[i]
        for j in range(i + 1, len(short)):
            vj, cj, _ = short[j]
            for sv in (1, -1):
                for sw in (1, -1):
                    cand = [int(pv[t]) - sv * int(vi[t]) - sw * int(vj[t])
                            for t in range(len(pv))]
                    cc = [int(pc[t]) - sv * int(ci[t]) - sw * int(cj[t])
                          for t in range(len(pc))]
                    if mod_q is not None:
                        cand, cc, c2 = _mod_q(cand, cc, mod_q[0], mod_q[1])
                    else:
                        c2 = _n2(cand)
                    if c2 < b2:
                        bv, bc, b2 = cand, cc, c2
    return bv, bc, b2


def gauss_sieve(basis, triple=False, target=None, max_samples=20000,
                sat_ratio=0.5, topk=40, seed=0, sampler_coeff=3, move_hook=None,
                seed_coef=None, mod_q=None, active_rows=None):
    """Sieve the lattice with rows ``basis``; return a list of (vector, coeffs),
    shortest first.

    triple        : also do 3-tuple reductions (the hk3 move).
    target        : stop once the database reaches this size (default 4*dim).
    sat_ratio     : stop when collisions exceed sat_ratio*target (saturation).
    topk          : how many shortest db vectors the triple search scans.
    sampler_coeff : range of random basis-combination coeffs for new samples.
    move_hook     : optional callback move_hook(pv, pc, p2, db, samples) after each
                    insert; return False to stop the sieve early (e.g. on first
                    detection, to measure work-to-detection).
    seed_coef     : (n x n) matrix; row i = coordinates of ``basis[i]`` in the
                    basis you want ``coeffs`` reported in (default identity ->
                    coeffs in ``basis`` itself). Pass the LLL transform ``U`` to
                    report coeffs in the original lattice basis.
    mod_q         : ``(n_per, q)`` to enable q-reduction of triple candidates
                    before the length check (the structure-aware experiment).
                    Requires ``seed_coef`` mapping to the original/wrap basis.
    active_rows   : restrict the seed stack and sampler to this subset of basis
                    rows (default all). Under ``mod_q`` the wrap rows ``q*e_j``
                    reduce to 0, so excluding them (active_rows = the timing rows)
                    sieves the quotient lattice L/qZ^N without the dead generators.
    """
    rng = np.random.default_rng(seed)
    B = [[int(x) for x in row] for row in basis]
    dim = len(B)
    if target is None:
        target = 4 * dim
    if mod_q is not None and seed_coef is None:
        raise ValueError("mod_q requires seed_coef (the original/wrap basis transform)")
    C = (np.eye(dim, dtype=object) if seed_coef is None
         else np.array(seed_coef, dtype=object))   # report-basis coords of each B row
    idx = list(range(dim)) if active_rows is None else list(active_rows)

    # work stack seeded with the (active) basis vectors and negatives, as (vec, coef)
    stack = [(list(B[i]), [int(x) for x in C[i]]) for i in idx]
    stack += [([-x for x in B[i]], [-int(x) for x in C[i]]) for i in idx]

    db = []                          # (vec, coef, norm2), pairwise-reduced
    collisions = 0

    def sample():
        coeffs = np.zeros(dim, dtype=int)
        coeffs[idx] = rng.integers(-sampler_coeff, sampler_coeff + 1, size=len(idx))
        pv = [0] * dim
        for c, b in zip(coeffs, B):
            if c:
                pv = [pv[t] + int(c) * b[t] for t in range(dim)]
        coef = [int(x) for x in (coeffs.astype(object) @ C)]
        return pv, coef

    samples = 0
    while len(db) < target and samples < max_samples:
        pv, pc = stack.pop() if stack else sample()
        samples += 1
        p2 = _n2(pv)
        pv, pc, p2 = _pair_reduce(pv, pc, p2, db, mod_q=mod_q)
        if triple and p2 > 0:
            pv, pc, p2 = _triple_reduce(pv, pc, p2, db, topk, mod_q=mod_q)
            pv, pc, p2 = _pair_reduce(pv, pc, p2, db, mod_q=mod_q)
        if p2 == 0:                                # collision
            collisions += 1
            if collisions > sat_ratio * target:
                break
            continue
        kept = []
        for v, c, v2 in db:                        # reduce db against newcomer
            rv, rc, rv2 = _pair_reduce(v, c, v2, [(pv, pc, p2)], mod_q=mod_q)
            if rv2 < v2:
                stack.append((rv, rc))
            else:
                kept.append((v, c, v2))
        db = kept
        db.append((pv, pc, p2))
        if move_hook is not None and move_hook(pv, pc, p2, db, samples) is False:
            break

    return [(v, c) for v, c, _ in sorted(db, key=lambda e: e[2])]


# --------------------------------------------------------------------------- #
# self-test: recover a small synthetic pulsar with the pure stack             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_a_constant_frequency import generate_data
    from lattice_tools import pure_lll

    data = generate_data(n_toas=14, n_verify=200, seed=0)      # small => fast
    il = data["integer_lattice"]
    n_per = len(data["toas_met_lattice"])
    q = data["mul_factor"]

    B, U = pure_lll(il, delta=0.99)                            # transparent LLL
    qi = int(il[0, 0])                                         # exact integer q
    Qn = np.sum(data["probs_verify"] ** 2 / 2)

    def evaluate(out):
        # seed_coef=U => coeffs already in the original il basis
        coeffs = np.array([c for _, c in out], dtype=object)
        norms = np.array([float(_n2(v)) ** 0.5 for v, _ in out])
        p_par = coeffs[:, n_per:].astype(float)
        vf = np.mod(p_par @ data["transformation_matrix"] / q + 0.5, 1) - 0.5
        Q = np.abs(np.sum(data["probs_verify"] * np.exp(2j * np.pi * vf), axis=1)) ** 2 / Qn
        kstd = np.std(coeffs[:, :n_per].astype(float), axis=1)
        reasonable = kstd > 1e3
        best = Q[reasonable].max() if reasonable.any() else float("nan")
        return len(out), int(reasonable.sum()), best, norms.min()

    runs = [
        ("pair-only (Gauss)", dict(triple=False)),
        ("pair + mod-q",      dict(triple=False, mod_q=(n_per, qi))),
        ("triple (hk3)",      dict(triple=True)),
        ("triple + mod-q",    dict(triple=True, mod_q=(n_per, qi))),
    ]
    print(f"{'':>18}  {'db':>4} {'reason':>6} {'max Q':>7} {'min ||v||':>12} {'modq fires':>11}")
    for label, kw in runs:
        reset_modq_stats()
        out = gauss_sieve(B, seed=1, seed_coef=U, **kw)
        n_db, n_reas, best, mn = evaluate(out)
        print(f"{label:>18}: {n_db:>4} {n_reas:>6} {best:>7.1f} {mn:>12.3e} "
              f"{MODQ_STATS['fires']:>11}")
