"""Lever-1 feasibility: can a short-baseline solution phase-connect forward?

The progressive-baseline idea is poly(N) only if, after a bounded bootstrap
solve, each newly added TOA's phase wrap is *determined* (not searched) by the
current solution -- i.e. predicted within 1/2 cycle.

The phase-prediction error at a TOA equals sigma_phase * g, where
  g_j(k) = sqrt( s_j^T (S_k^T S_k)^{-1} s_j )
is the regression leverage of the timing design matrix (scale-invariant).
We compare:
  * g_next(k): leverage at the NEXT (k+1-th) time-sorted TOA  -> incremental connection
  * g_far(k):  max leverage over ALL remaining TOAs           -> full extrapolation
A small bounded g_next means incremental phase connection is feasible.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fermi_fold import load_data, fold

d = load_data("data/data.npy")
sv = np.asarray(d["span_vecs"])            # 7 x 70 timing functions at lattice TOAs
t = np.asarray(d["toas_met_lattice"]); q = int(d["integer_lattice"][0, 0])
ntoa = sv.shape[1]; p = sv.shape[0]

order = np.argsort(t)                       # time-sorted
S = sv[:, order].T                          # N x p design matrix

def leverage(Sk, sj):
    G = Sk.T @ Sk
    return float(np.sqrt(sj @ np.linalg.solve(G, sj)))

ks = np.arange(p + 1, ntoa)
g_next = np.array([leverage(S[:k], S[k]) for k in ks])
g_far = np.array([max(leverage(S[:k], S[j]) for j in range(k, ntoa)) for k in ks])

# Reference sigma_phase: RMS phase residual of the true (deep) solution.
res = fold(d)
mask = res["reasonable_solutions_mask"]
best = res["db_vectors"][mask][np.argmax(res["Q_stat"][mask])]
phase_res = (((best[:ntoa] + q // 2) % q) - q // 2).astype(float) / q
sigma_phase = float(np.sqrt(np.mean(phase_res ** 2)))
print(f"measured sigma_phase (true-solution residual RMS) = {sigma_phase:.2e} cycles")

print(f"\n{'k':>4} {'g_next':>10} {'g_far':>10}  next-wrap tol sigma<0.5/g_next")
for k, gn, gf in list(zip(ks, g_next, g_far))[::6]:
    print(f"{k:>4} {gn:>10.2e} {gf:>10.2e}   sigma<{0.5/gn:.2e}")

for sig, name in [(sigma_phase, "measured"), (1e-2, "1e-2 cyc"), (0.1, "0.1 cyc")]:
    ok = sig * g_next < 0.5
    k0 = int(ks[np.argmax(ok)]) if ok.any() else None
    allok = ok.all()
    print(f"\nsigma_phase={name}: incremental connection holds for all k>={p+1}? {allok}"
          + ("" if allok else f"  (first k with next-wrap safe: {k0})"))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].semilogy(ks, g_next, "o-", label="g_next (next TOA)")
ax[0].semilogy(ks, g_far, "s-", color="C3", label="g_far (worst remaining)")
for sig, name in [(sigma_phase, "measured"), (0.1, "0.1 cyc")]:
    ax[0].axhline(0.5 / sig, ls="--", alpha=0.6, label=f"connection limit (sigma={name})")
ax[0].set_xlabel("bootstrap size k (time-sorted TOAs)"); ax[0].set_ylabel("leverage g")
ax[0].set_title("Forward phase-prediction leverage"); ax[0].legend(fontsize=8)
ax[1].plot(np.sort(t) - t.min(), "o-", ms=3); ax[1].set_xlabel("TOA index (time-sorted)")
ax[1].set_ylabel("time since first TOA"); ax[1].set_title("TOA time sampling (gaps)")
fig.tight_layout(); fig.savefig("figures/lever1_feasibility.png", dpi=120, bbox_inches="tight")
print("\nwrote figures/lever1_feasibility.png")
