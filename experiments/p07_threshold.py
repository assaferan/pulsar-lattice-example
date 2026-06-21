"""Detection threshold: how many TOAs (n) to recover the pulsar at association
probability p -- for model A (phi, f) or model B (full 7-parameter timing model)?

p sets the effective pulse width via  sigma^2 = p*sigma_int^2 + (1-p)/12; with an
intrinsically narrow pulse (sigma_int -> 0), sigma = sqrt((1-p)/12). A wider pulse
(lower p) makes the folded solution less anomalously short, so the sieve needs
more TOAs to acquire it. We sweep n at the effective sigma and report the max
reasonable Q over a few seeds (Q>50 = detected).

Acquisition is a *sharp* threshold, not gradual: when the sieve finds the
solution Q jumps to hundreds; when it doesn't, Q ~ O(10).

Results (p=0.7, sigma=0.158):
  * model A (phi, f):     reliable detection at n ~ 60 (transition 40-55, none
    below 30) -- ~4x the p=1 threshold (~15-20).
  * model B (7 params):   reliable detection at n ~ 80 (transition 60-70, none
    below 50) -- the 5 extra params cost ~20 TOAs (astrometry is nearly free:
    tight priors, tiny offsets, spin-dominated fold).
Both are well below Table 1's blind-7-param n=111 (Lambda=1e28): our synthetic
has tightly-priored astrometry, so the effective search is smaller. Acquisition
is sharp -- when found Q jumps to ~250-350, else Q ~ O(10-25).

Pipeline check (`pipeline_check`): the SHIPPED `fermi_fold.fold` detects p=0.7
end-to-end. In full mode (BKZ+deep pump) it folds the verify set to Q~900. The
`fast` mode (LLL+shallow pump) MISSES with its default `pump_stop=27` (sieve dim
~35-43), but DETECTS once the pump is deepened (`pump_stop<=10`, sieve dim >=50) --
still LLL, not BKZ. So fast's failure at p=0.7 is purely pump DEPTH (detection
needs `d_sieve ~ n`, per `density_sweep`), not the reduction choice. Detection
near the threshold is stochastic per g6k run (right at the acquisition wall), so
use a few TOAs of margin.

Run (g6k env active):
    PYTHONPATH=. python experiments/p07_threshold.py            # model A, p=0.7 sweep
    PYTHONPATH=. python experiments/p07_threshold.py B 0.7      # model B, p=0.7 sweep
    PYTHONPATH=. python experiments/p07_threshold.py check      # fermi_fold pipeline demo
"""

import sys

import numpy as np

from fermi_fold import fold
import model_a_constant_frequency as model_a
import model_b_full_timing as model_b


def sigma_of_p(p, sigma_int=0.0):
    """Effective per-coordinate pulse width for association probability p."""
    return float(np.sqrt(p * sigma_int ** 2 + (1 - p) / 12))


def max_reasonable_Q(model, n_toas, sigma, seed, mul_factor=10 ** 16, retries=5):
    """Generate model-{A,B} data at pulse width sigma and return the max Q over
    physically reasonable solutions (0.0 if none, nan if g6k keeps failing)."""
    gen = model_a.generate_data if model == "A" else model_b.generate_data
    data = gen(n_toas=n_toas, n_verify=400, phase_std=sigma,
               mul_factor=mul_factor, seed=seed)
    for _ in range(retries):
        try:
            r = fold(data, fast=False, reasonable_k_std=1e3)
            m = r["reasonable_solutions_mask"]
            return float(r["Q_stat"][m].max()) if m.any() else 0.0
        except Exception:
            continue
    return float("nan")


def main(model="A", p=0.7, ns=None, n_seeds=3):
    sig = sigma_of_p(p)
    if ns is None:
        ns = (20, 30, 40, 50, 60, 70) if model == "A" else (60, 80, 100, 120, 140)
    print(f"model {model}, p={p}  ->  effective sigma={sig:.3f}   "
          f"(detect = max reasonable Q > 50)")
    hdr = f"{'n':>5} " + " ".join(f"{'Q'+str(s):>8}" for s in range(n_seeds)) + f"{'detect':>9}"
    print(hdr + "\n" + "-" * len(hdr))
    for n in ns:
        qs = [max_reasonable_Q(model, n, sig, s) for s in range(n_seeds)]
        det = sum((not np.isnan(q)) and q > 50 for q in qs)
        print(f"{n:>5} " + " ".join(f"{q:>8.1f}" for q in qs) + f"{str(det)+'/'+str(n_seeds):>9}",
              flush=True)


def pipeline_check(model="A", p=0.7, n_toas=60, seed=0, mul_factor=10 ** 16,
                   pump_stops=(27, 20, 15, 10, 7)):
    """Run the shipped `fermi_fold.fold` on model-{model} p data, at n right at
    the acquisition wall, sweeping the fast-mode pump depth (plus full BKZ) to
    confirm detection and isolate the cause of fast's default-mode failure.

    fast mode uses LLL (not BKZ); if deepening the pump alone flips it to detected,
    the cause is pump DEPTH, not the reduction. (Detection at the wall is
    stochastic per g6k run; the deep pumps and full mode are reliably above it.)
    """
    sig = sigma_of_p(p)
    gen = model_a.generate_data if model == "A" else model_b.generate_data
    data = gen(n_toas=n_toas, n_verify=1281, phase_std=sig, mul_factor=mul_factor, seed=seed)
    dim = data["integer_lattice"].shape[0]

    def run(**kw):
        try:
            r = fold(data, reasonable_k_std=1e5, **kw)        # 1e5 = pipeline default
            m = r["reasonable_solutions_mask"]
            q = r["Q_stat"][m]
            return r["Q_stat"].size, (float(q.max()) if q.size else 0.0)
        except Exception:
            return None, float("nan")

    print(f"fermi_fold.fold on model {model}, p={p} (sigma={sig:.3f}), "
          f"n_toas={n_toas}, lattice dim={dim}")
    print(f"{'mode':>26} {'sieve_dim':>9} {'sieved':>8} {'max Q':>8} {'result':>10}")
    nv, best = run(fast=False)
    det = "DETECTED" if (not np.isnan(best) and best > 50) else "no"
    print(f"{'full (BKZ+deep pump)':>26} {dim - len(data['coeff_std']):>9} "
          f"{str(nv):>8} {best:>8.1f} {det:>10}", flush=True)
    for ps in pump_stops:
        nv, best = run(fast=True, pump_stop=ps)
        det = "DETECTED" if (not np.isnan(best) and best > 50) else "no"
        tag = " (default)" if ps == 27 else ""
        print(f"{('fast pump_stop='+str(ps)+tag):>26} {dim - ps:>9} "
              f"{str(nv):>8} {best:>8.1f} {det:>10}", flush=True)
    print("=> fast (LLL) detects p=0.7 once the pump reaches sieve-dim ~50+; the\n"
          "   default shallow pump is the only problem -- pump DEPTH, not LLL-vs-BKZ.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        model = sys.argv[2] if len(sys.argv) > 2 else "A"
        p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
        pipeline_check(model, p)
    else:
        model = sys.argv[1] if len(sys.argv) > 1 else "A"
        p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
        main(model, p)
