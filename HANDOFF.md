# Handoff — pulsar-lattice-example (p=0.5 effort)

Continuity notes for the next Claude instance. Branch: **`mod1`**. Date of writing:
2026-06-24. The user (Eran, eranasaf@mit.edu) commits/pushes himself unless he says
otherwise — default to **staging**, not committing.

## TL;DR — where things are

- The pipeline (`fermi_fold.py`: LLL/BKZ → g6k pump → Q-statistic detection) works
  and **`p ≥ 0.7` is reachable**. Detection thresholds established earlier:
  model A `p=0.7` `n≈60`, model B `p=0.7` `n≈80`.
- **Goal that stalled: `p=0.5`** (effective pulse width σ=√(0.5/12)=0.204, Table 1's
  near-infeasible row). We built a full g6k stack on a workstation (lovelace) and
  scanned model A — it **hits a three-way wall (cost + g6k stability + bdgl2 build
  gap) and detection was never observed.** User's decision: **document the wall and
  stop.** Full writeup in `experiments/FINDINGS.md` → section *"p=0.5 at scale"*.
- Nothing is mid-flight: all remote jobs were killed; the working tree is clean
  except this handoff + the FINDINGS section (being committed now).

## Remote compute — the key infrastructure (don't rediscover this)

MIT blocked SSH port 22 from off-campus for **some** hosts. Tested from the user's
laptop (no VPN):

| host (`~/.ssh/config`, user `assaferan`) | reachable | specs |
|---|---|---|
| `lava.mit.edu`   | **NO** (port 22 timeout) | — |
| `hensel.mit.edu` | **NO** | — |
| `lovelace.mit.edu` | **YES** | **256 cores, 2 TiB RAM**, 628 GB free `~`, `/scratch` |
| `legendre.mit.edu` | YES | 48 cores, 251 GB RAM |

`galois` (USyd jump host) also works, but isn't needed. **Use `lovelace`.** It is a
**shared machine (~40 users, load ~90)** — be a good citizen: cap cores/time, clean
up, `pkill -u assaferan -f <yourscript>` when done.

### lovelace g6k environment (already built — reuse it)

- miniforge: `/scratch/assaferan/miniforge3`
- conda env **`g6k`**: python 3.11, fpylll 0.6.4, compilers, autotools (conda-forge).
- g6k built in-place: `/scratch/assaferan/GitHub/g6k` (commit `c71e084`, gcc,
  `./configure --enable-native=yes`; **no clang patches needed on Linux** — those
  were macOS-only; see memory `g6k-macos-build.md` for the laptop env).
- repo clone: `/scratch/assaferan/GitHub/pulsar-lattice-example` (branch `mod1`).

Run recipe (every remote command needs this preamble):
```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 lovelace
source /scratch/assaferan/miniforge3/etc/profile.d/conda.sh; conda activate g6k
export PYTHONPATH=/scratch/assaferan/GitHub/g6k:/scratch/assaferan/GitHub/pulsar-lattice-example
cd /scratch/assaferan/GitHub/pulsar-lattice-example
# git pull -q origin mod1   # to get latest pushed code
```
Smoke test: `python -m pytest -q test_fermi_fold.py` (2 passed, ~20 s).

## The p=0.5 wall (three compounding problems)

1. **Cost.** Detection needs sieve dim `d_sieve ≈ 0.85·N ≈ 112–124` for model A
   `n≥130`. That's hundreds× the `dim≈88` `p=0.7` sieves; jobs ran **>1.8 h each,
   none finished**.
2. **Stability.** `bgj1` **C-aborts** (libstdc++ `terminate`, kills the process) on
   the skewed q-ary lattice past sieving dim ~110. Same class as the known
   `hk3` dim>100 saturation collapse.
3. **bdgl2 (the fast sieve, exponent 0.292) unavailable there.** Needs precomputed
   `g6k/spherical_coding/sc_<d>_256.def`; build only made them into the ~150s, so
   it dies on `Cannot open … sc_162_256.def` for the bigger lattices.

Detection itself: deepest **clean** run (model A `n=110`, `d_sieve=95`) → best
**physical** `Q≈24 < 50`, not climbing. Beware the **decoy**: unmasked maxQ can be
huge (≈2562) for a near-zero-wrap "fold everything into one phase" vector — rejected
by the `std(k) > 1e5` physical mask. (Mask threshold is irrelevant: wrap-std is
bimodal, ~0 or >1e11.)

## If you want to push p=0.5 further (concrete, in rough order)

1. **Cheap first: calibrate the complexity model to `p=0.5`** (extend
   `complexity_table.py`/`complexity_empirical.py`, anchored to the *measured*
   `p=0.7` threshold) to predict required `n` and `d_sieve`, and whether `d_sieve`
   even fits under `MAX_SIEVING_DIM`. Decide feasibility before burning compute.
   (Early `n≈110` projection proved optimistic — `n=110` did not detect.)
2. **Rebuild g6k for stable, fast bdgl2 at dim 112–128**: generate spherical-coding
   tables higher (see g6k's `spherical_coding` generation in its `setup.py`), and
   consider raising `MAX_SIEVING_DIM` in `siever.h` then recompile. bdgl2 is the
   right tool at these dims.
3. **Investigate the `bgj1` C-abort** (stability bug at high dim on this skew) —
   try heavier BKZ pre-reduction, different `q`, or smaller `saturation_ratio`.
4. **Re-run** `experiments/run_p05_scan.sh A 130 140 …` once a sieve is stable.

## Repo state & key files

- Branch `mod1`, synced with `origin/mod1`. Latest relevant commits:
  `72c1c99` (p05 scan rework), `4aa62dc` (threads fix), `5b5cbec` (large runs, user).
- `experiments/p05_lava.py` — **one `(model,n)` per process** p=0.5 probe (bgj1,
  `THREADS=24`, `DEPTH_FRAC=0.85`, `DIM_CAP=124`, retries; reports detQ + unmasked
  decoy Q + wrap-std). Usage: `python experiments/p05_lava.py A 0.5 140`.
- `experiments/run_p05_scan.sh` — parallel driver (`CONC` procs × `THREADS` each).
- `experiments/FINDINGS.md` — the full investigation log (read this). New section
  *"p=0.5 at scale — the practical wall"*.
- `fermi_fold.py` — pipeline. `_sieve(il, pump_stop, block, delta, do_bkz, alg)` is
  the trusted reduce+sieve (hk3→bgj1→bdgl2 fallback). `fold(fast=…)`: `fast=False`
  forces a *full-depth* pump (`pump_stop=len(coeff_std)` → too deep / over the cap
  for big N); `fast=True` skips BKZ. For BKZ + a *tuned* depth, call `_sieve`
  directly. `fold_subsampled` handles the dim>100 saturation issue by subsampling.
- `model_a_constant_frequency.py` / `model_b_full_timing.py` — data generators.

## Gotchas (hard-won)

- **g6k default `threads=1`.** Always `Siever(gso, SieverParams(threads=N))`.
  `_sieve`/`fold` do **not** expose threads → single-threaded; parallelize across
  processes (as `p05_lava.py` does) or edit if you need in-sieve threads.
- **`MAX_SIEVING_DIM=128`** on this build (lattice dim above it only warns; sieve
  must stay under it).
- **bdgl2 crashes / needs spherical codes**; **bgj1 C-aborts dim>110** — isolate
  every sieve in its own process so a crash is local (exit codes: 134 abort, 137
  OOM, 139 segv).
- **Don't pipe heredocs through `ssh '…'`** (quoting breaks): write the script
  locally and `scp` it, or use `python -c` with care.
- **mod-q / q·Zⁿ structure is inert** for detection (tested exhaustively — see
  FINDINGS "What didn't help"). Don't re-litigate it.
- The gap `σ/σ_exp ≈ 2^0.2` is **physical** (set by pulse width vs background); no
  lattice trick moves it. The only lever is data quality (σ), not structure.
