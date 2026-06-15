"""Definitive L1 verdict: would an L1 search acquire the pulsar where L2 fails?

G6K minimises L2. The hypothesis was that L1 (rewarding sparse residuals) would
acquire the pulsar from a short baseline. A search emits its *shortest* vectors,
so the question is whether the true solution b* is short under L1. We split the
acquisition into generation vs selection, under both norms:

B. **Generation** (bootstrap): is b* shorter than the spurious overfit vectors
   under L1 / L2?  (Would any norm-min search emit b*?)
C. **Selection** (global): among physical candidates, is b* best under L1 / L2?

Note up front: the no-signal solution (b=0) has *zero* residual under any norm,
so residual-min alone never prefers a pulsar -- detection needs the physical /
coherence criterion regardless of the norm.

Run from the repo root with the g6k environment active:
    PYTHONPATH=. python3 experiments/l1_search.py
"""
import numpy as np
from fermi_fold import load_data, fold
from experiments.progressive_solver import _bootstrap, connect

d = load_data()
il = d["integer_lattice"]; ntoa = len(d["toas_met_lattice"]); q = int(il[0, 0])
order = np.argsort(np.asarray(d["toas_met_lattice"]))
Acol = il[ntoa:, order].T / q
full = fold(d); m = full["reasonable_solutions_mask"]
bstar = full["db_transformation_p"][m][np.argmax(full["Q_stat"][m])].astype(float)

k0 = 30


def norms(b, idx):
    phi = -(b @ Acol[idx].T)
    r = phi - np.round(phi)
    return np.abs(r).sum(), np.linalg.norm(r)


# ---- B. generation on the bootstrap ---------------------------------------
cands = _bootstrap(il, ntoa, q, order[:k0], n_candidates=200)
boot = np.arange(k0)
bstar_L1, bstar_L2 = norms(bstar, boot)
cand_L1 = np.array([norms(b, boot)[0] for b in cands])
cand_L2 = np.array([norms(b, boot)[1] for b in cands])
print(f"B. GENERATION -- bootstrap residual on first k0={k0} TOAs (shorter = emitted first):")
print(f"   true b*:            L1={bstar_L1:7.3f}   L2={bstar_L2:7.3f}")
print(f"   shortest overfit:   L1={cand_L1.min():7.3f}   L2={cand_L2.min():7.3f}")
print(f"   b* rank by L1 = {(cand_L1 < bstar_L1).sum()+1:>4}/{len(cands)+1},"
      f"   by L2 = {(cand_L2 < bstar_L2).sum()+1:>4}/{len(cands)+1}   (last = worst)")
print("   => b* is far from shortest under BOTH norms: overfit vectors fit the short")
print("      baseline with ~0 residual, so neither an L1 nor an L2 search emits b*.\n")

# ---- C. selection globally -------------------------------------------------
phys = [connect(b0, Acol, k0, ntoa)[0] for b0 in cands]
phys = [b for b, w in ((b, np.round(-(b @ Acol.T))) for b in phys) if np.std(w) > 1e5]
phys.append(bstar)
gl1 = np.array([norms(b, np.arange(ntoa))[0] for b in phys])
gl2 = np.array([norms(b, np.arange(ntoa))[1] for b in phys])
bi = len(phys) - 1
print("C. SELECTION -- global residual among physical candidates:")
print(f"   b* rank by L1 = {(gl1 < gl1[bi]).sum()+1}/{len(phys)},"
      f"   by L2 = {(gl2 < gl2[bi]).sum()+1}/{len(phys)}")
print("   => b* wins under BOTH norms globally: selection is not the bottleneck.\n")

print("VERDICT: L1 does not help acquisition. The bottleneck is GENERATION from a")
print("short baseline -- and b* is not short under either norm there, because")
print("overfit solutions reach ~0 residual. The obstruction is norm-independent;")
print("it is about generalising to unseen TOAs, not the metric on the seen ones.")
