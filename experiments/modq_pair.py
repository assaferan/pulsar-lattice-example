"""Does mod-q help a PAIR sieve (the bgj1 analogue) on the pulsar lattice?

The "mod-q is inert/harmful" results were entangled with hk3 (triples). But the
pipeline now uses bgj1 (pairs), and `pair_sieve_triple` showed mod-1 *does* lower
a pair-sieve's database exponent while the triple move *erases* it. So with a pair
sieve the benefit might be recoverable -- re-test on the real (synthetic) lattice.

Crux (`modq_reduction`): mod-q changes the wrap counts k (the norm), NOT the
timing parameters p, and detection Q depends only on p. So mod-q's only lever is
*in-sieve keep/discard*. The pulsar is short via LARGE near-cancelling k (~1e9),
which mod-q collapses to 0 (`sieve_only`) -- so the prediction is inert-or-harmful
even for pairs, despite the collaborator's pair-sieve win (which was for *geometric*
shortness, the opposite of what the pulsar needs). This settles which wins.

Tool: `pure_sieve` (pure-Python exact-integer pair/triple sieve) is the only place
we can inject our custom mod-q (g6k's bgj1 is C++; post-hoc mod-q is vacuous).
Small-dimension only.

CONTROL: detection requires Q>50 AND std(k)>1e3 (a *physical* solution), never
just "a short vector" -- else mod-q spuriously "wins" by parking on the trivial
k=0 constant-phase vector (the progressive_solver bug).

Run:  PYTHONPATH=.:experiments python3 experiments/modq_pair.py
"""
import numpy as np
import pure_sieve as ps
from lattice_tools import pure_lll
from modq_firing import mixed_lattice, Q

SEEDS = range(8)
MAX_SAMPLES = 15000


def samples_to_detect(n, p, triple, use_modq, seed, use_lll=True, target=None):
    """Samples until the first PHYSICAL solution clears Q>50, or None.

    With ``target`` set, the sieve database is capped at that size, so a None
    return means the physical solution was not reached within a db of ``target``
    -- which is how :func:`min_database` measures the minimal viable L_min.
    """
    il, n_per, tm, probs = mixed_lattice(n, p, seed=seed)
    qi = int(il[0, 0])
    if use_lll:
        B, U = pure_lll(il, 0.99)
    else:
        B, U = il, np.eye(il.shape[0], dtype=object)
    state = {"hit": None}

    def hook(pv, pc, p2, db, samples):
        pp = np.array(pc[n_per:], float)
        vf = np.mod(pp @ tm / Q + 0.5, 1) - 0.5
        Qv = np.abs(np.sum(probs * np.exp(2j * np.pi * vf))) ** 2 / np.sum(probs ** 2 / 2)
        if Qv > 50 and np.std(np.array(pc[:n_per], float)) > 1e3:   # physical only
            state["hit"] = samples
            return False

    kw = dict(triple=triple, seed=seed, seed_coef=U, max_samples=MAX_SAMPLES,
              mod_q=((n_per, qi) if use_modq else None), move_hook=hook)
    if target is not None:
        kw["target"] = target
    ps.gauss_sieve(B, **kw)
    return state["hit"]


def summarize(n, p, triple, use_modq):
    res = [samples_to_detect(n, p, triple, use_modq, s) for s in SEEDS]
    hits = [r for r in res if r is not None]
    med = int(np.median(hits)) if hits else None
    return med, len(hits)


L_GRID = [5, 6, 8, 10, 12, 15, 18, 22, 28, 36, 48, 64]


def min_database(n, use_modq, p=1.0, triple=False):
    """Smallest database cap ``L`` at which the (pair) sieve still reaches a
    physical detection in >= half the seeds -- the collaborator's L_min metric,
    on the pulsar lattice. Returns None if none in L_GRID suffices."""
    for L in L_GRID:
        rate = np.mean([samples_to_detect(n, p, triple, use_modq, s, target=L) is not None
                        for s in SEEDS])
        if rate >= 0.5:
            return L
    return None


if __name__ == "__main__":
    print("Work-to-detection: median samples to first PHYSICAL Q>50 (LLL + sieve).")
    print("Question: does mod-q reduce the work of the PAIR sieve (bgj1 analogue)?\n")
    print(f"{'sieve':>14} {'p':>5} {'no mod-q (det/8)':>20} {'mod-q (det/8)':>20}", flush=True)
    for n in (14, 16):
        print(f"-- n={n} --", flush=True)
        for triple in (False, True):
            name = "triple(hk3)" if triple else "PAIR(bgj1)"
            for p in (1.0, 0.7):
                a_med, a_hit = summarize(n, p, triple, False)
                b_med, b_hit = summarize(n, p, triple, True)
                fa = f"{a_med} ({a_hit}/8)" if a_med else f"miss (0/8)"
                fb = f"{b_med} ({b_hit}/8)" if b_med else f"miss (0/8)"
                print(f"{name:>14} {p:>5} {fa:>20} {fb:>20}", flush=True)

    # Metric 2: minimal database size L_min (the collaborator's pair-sieve metric,
    # where mod-1 lowered the exponent on *random* vectors). On the pulsar lattice
    # it is identical with/without mod-q -- the win does not transfer.
    print("\nMinimal database L_min for PHYSICAL detection (PAIR sieve, clean p=1):")
    print(f"{'n':>4} {'L_min no-modq':>14} {'L_min mod-q':>12}", flush=True)
    for n in (14, 16, 18):
        a = min_database(n, False)
        b = min_database(n, True)
        print(f"{n:>4} {str(a):>14} {str(b):>12}", flush=True)
    print("\nDONE", flush=True)
