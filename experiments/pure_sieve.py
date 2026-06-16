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


def _pair_reduce(pv, pc, p2, db):
    """Gauss (2-)reduction of (vector pv, coeffs pc) against the database. A move
    is accepted only if it strictly reduces the norm, so float rounding of the
    Gauss coefficient can never produce a longer vector."""
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
            cand = _sub(pv, s, v)
            c2 = _n2(cand)
            if c2 < p2:
                pv, pc, p2 = cand, _sub(pc, s, c), c2
                changed = True
    return pv, pc, p2


def _triple_reduce(pv, pc, p2, db, topk):
    """3-tuple reduction (the hk3 move): try p (+/-) v (+/-) w over the topk
    shortest database vectors; keep the shortest result below ||p||."""
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
                    c2 = _n2(cand)
                    if c2 < b2:
                        bc = [int(pc[t]) - sv * int(ci[t]) - sw * int(cj[t])
                              for t in range(len(pc))]
                        bv, b2 = cand, c2
    return bv, bc, b2


def gauss_sieve(basis, triple=False, target=None, max_samples=20000,
                sat_ratio=0.5, topk=40, seed=0, sampler_coeff=3, move_hook=None):
    """Sieve the lattice with rows ``basis``; return a list of (vector, coeffs),
    shortest first. ``coeffs`` are integer coordinates in ``basis``.

    triple        : also do 3-tuple reductions (the hk3 move).
    target        : stop once the database reaches this size (default 4*dim).
    sat_ratio     : stop when collisions exceed sat_ratio*target (saturation).
    topk          : how many shortest db vectors the triple search scans.
    sampler_coeff : range of random basis-combination coeffs for new samples.
    move_hook     : optional callback move_hook(pv, pc, p2, db) after each insert.
    """
    rng = np.random.default_rng(seed)
    B = [[int(x) for x in row] for row in basis]
    dim = len(B)
    if target is None:
        target = 4 * dim

    def e(i):
        return [1 if t == i else 0 for t in range(dim)]

    # work stack seeded with the basis vectors (and negatives), as (vec, coef)
    stack = [(list(B[i]), e(i)) for i in range(dim)]
    stack += [([-x for x in B[i]], [-x for x in e(i)]) for i in range(dim)]

    db = []                          # (vec, coef, norm2), pairwise-reduced
    collisions = 0

    def sample():
        coeffs = rng.integers(-sampler_coeff, sampler_coeff + 1, size=dim)
        pv = [0] * dim
        for c, b in zip(coeffs, B):
            if c:
                pv = [pv[t] + int(c) * b[t] for t in range(dim)]
        return pv, [int(c) for c in coeffs]

    samples = 0
    while len(db) < target and samples < max_samples:
        pv, pc = stack.pop() if stack else sample()
        samples += 1
        p2 = _n2(pv)
        pv, pc, p2 = _pair_reduce(pv, pc, p2, db)
        if triple and p2 > 0:
            pv, pc, p2 = _triple_reduce(pv, pc, p2, db, topk)
            pv, pc, p2 = _pair_reduce(pv, pc, p2, db)
        if p2 == 0:                                # collision
            collisions += 1
            if collisions > sat_ratio * target:
                break
            continue
        kept = []
        for v, c, v2 in db:                        # reduce db against newcomer
            rv, rc, rv2 = _pair_reduce(v, c, v2, [(pv, pc, p2)])
            if rv2 < v2:
                stack.append((rv, rc))
            else:
                kept.append((v, c, v2))
        db = kept
        db.append((pv, pc, p2))
        if move_hook is not None:
            move_hook(pv, pc, p2, db)

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
    for label, triple in [("pair-only (Gauss)", False), ("triple (hk3-style)", True)]:
        out = gauss_sieve(B, triple=triple, seed=1)
        coeffs = np.array([c for _, c in out], dtype=object) @ U   # -> original basis
        p_par = coeffs[:, n_per:].astype(float)
        vf = np.mod(p_par @ data["transformation_matrix"] / q + 0.5, 1) - 0.5
        Qn = np.sum(data["probs_verify"] ** 2 / 2)
        Q = np.abs(np.sum(data["probs_verify"] * np.exp(2j * np.pi * vf), axis=1)) ** 2 / Qn
        kstd = np.std(coeffs[:, :n_per].astype(float), axis=1)
        reasonable = kstd > 1e3
        best = Q[reasonable].max() if reasonable.any() else float("nan")
        print(f"{label:>20}: db={len(out):4d}  reasonable={int(reasonable.sum()):3d}  "
              f"max Q={best:.1f}")
