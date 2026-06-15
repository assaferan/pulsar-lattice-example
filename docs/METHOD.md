# Method: pulsar timing as a shortest-vector problem

This note summarizes, **in our own words**, the lattice method implemented in this
repository, and maps the equations onto the code. It is a convenience reference,
not a substitute for the paper — for derivations, assumptions, and the full
treatment (including binary orbits) see:

> D. Gazith, A. B. Pearlman, B. Zackay,
> *Recovering Pulsar Periodicity from Time-of-Arrival Data by Finding the
> Shortest Vector in a Lattice*, ApJ **979**, 48 (2025).
> arXiv:[2402.07228](https://arxiv.org/abs/2402.07228) ·
> ADS:[2025ApJ...979...48G](https://ui.adsabs.harvard.edu/abs/2025ApJ...979...48G/abstract)

---

## 1. The problem

A pulsar emits a pulse once per rotation. A photon detected at time $t_i$ left
the star at some rotational phase $\epsilon_i$ (near the pulse peak for a real
pulsar photon). With rotational frequency $f = 1/P$, reference phase $\phi$, and
an unknown integer number of rotations $K_i$ since $t=0$:

$$ t_i = (K_i + \phi + \epsilon_i)\,P \qquad\Longleftrightarrow\qquad f\,t_i - K_i - \phi = \epsilon_i. $$

For a Gaussian pulse of width $\sigma$ (the duty cycle), $\epsilon_i \sim N(0,\sigma)$.
The most powerful test (Neyman–Pearson) of "this is a pulsar" vs. "uniform phases"
minimizes the sum of squared residuals

$$ \mathcal{T} = \sum_i \epsilon_i^2 \;(+\ \text{parameter priors}). $$

So **finding the timing solution = finding the $(f,\phi,\{K_i\})$ that minimize
$\mathcal{T}$**, with the $K_i$ constrained to be integers. That integer
constraint is exactly what makes it a *lattice* problem: minimizing a quadratic
form over an integer grid is the **shortest-vector problem (SVP)**.

## 2. The basic lattice (constant frequency)

To use an integer lattice solver, the equation must be linear in the unknowns
with integer coefficients. Quantize each parameter by a step ($d_f$ for $f$,
$d_\phi$ for $\phi$) so its value is an integer multiple of that step:

$$ \big[\tfrac{f}{d_f}\big]\,d_f\,t_i \;-\; K_i \;-\; \big[\tfrac{\phi}{d_\phi}\big]\,d_\phi \;=\; \epsilon_i, $$

where $[\cdot]$ denotes rounding to the nearest integer. The step must be finer
than the achievable precision so rounding does not spoil the fit:

$$ d_f \ll \frac{\sigma}{\max_i t_i - \min_i t_i}. $$

Writing the basis vectors as **rows**, the lattice for $n$ TOAs is

$$
L_{\mathrm{per}} =
\begin{pmatrix}
I_{n\times n} & 0 & 0 \\
d_f t_0 \ \cdots\ d_f t_{n-1} & \eta_t & 0 \\
d_\phi \ \cdots\ d_\phi & 0 & \eta_\phi
\end{pmatrix}.
$$

- The top $n$ rows ($I_{n\times n}$, scaled by the modulus — see below) subtract
  whole rotations: choosing the integer combination of these rows is choosing the
  $K_i$.
- The two parameter rows hold each parameter's contribution to every TOA
  ($d_f t_i$, $d_\phi$).
- $\eta_t,\eta_\phi$ implement the **Gaussian priors** on $f,\phi$ (e.g. to reject
  nonphysical high frequencies from the detector clock). Choosing
  $\eta_t = \sigma\,d_f/f_{\mathrm{prior}}$ and $\eta_\phi = \sigma\,d_\phi$ makes
  every coordinate contribute the same expected loss.

A vector in this lattice has, as its coordinates, exactly the terms of
$\mathcal{T}$ (the phase residuals $\epsilon_i$ plus the prior penalties). Its
squared length **is** the test statistic, so the **shortest vector is the best
timing solution**.

> **Integerization.** A real solver (here G6K) needs integer entries, so every
> entry is multiplied by a large constant $q$ and rounded. In the code this
> constant is `mul_factor`; it must be much larger than the largest expected
> solution coefficient divided by the residual resolution.

## 3. Adding parameters (still linear, still one lattice)

The power of the approach is that extra timing parameters are just extra rows —
the SVP dimension grows by one per parameter, not exponentially.

**Spin-down.** The instantaneous frequency drifts, so integrate a Taylor series
$f(t) = f + \dot f\,t + \tfrac12 \ddot f\,t^2 + \dots$ to get the phase:

$$ K_j + \epsilon_j = \phi + f t_j + \sum_{k\ge 2}\frac{f^{(k-1)}}{k!}\,t_j^k. $$

Each derivative adds a row $\propto \tfrac{1}{k!} t_j^k$ with its own step and prior.

**Position (barycentric / Roemer delay).** Fermi follows Earth's orbit, leaving a
$\sim 500$ s light-travel residual unless corrected by projecting the
observatory–barycenter vector onto the line of sight $\hat n(\alpha,\delta)$. The
position enters *nonlinearly*, so it is **linearized** about a reference position
$\vec\psi_0$ (the catalog localization):

$$ f t_j = K_j + F(t_j,\vec\psi_0) + (\vec\psi - \vec\psi_0)\cdot\frac{dF}{d\vec\psi}(t_j,\vec\psi_0). $$

The derivative rows $dF/d\vec\psi$ are added like the spin-down rows; small sky
offsets $(\Delta\alpha,\Delta\delta)$ and proper motions $(\dot\alpha,\dot\delta)$
become extra parameters. Because Fermi's orbit is nearly planar, two weakly
constrained parameters suffice and even large offsets stay physical.

**Binary orbit.** A circular orbit adds $\sin/\cos(\Omega_{\rm orb}t)$ rows (plus
their $\Omega_{\rm orb}$-derivatives $t\sin/t\cos$ to cut the number of orbital-period
trials). Not implemented here — see the paper.

## 4. Solving and detecting

1. **Reduce** the basis (LLL, then BKZ) and **sieve** (G6K pump) to enumerate many
   short candidate vectors — each candidate is a trial $(f,\phi,\dot f,\dots)$.
2. **Fold** an independent *verify-set* of photons with each candidate and score it
   with a detection statistic; the true solution makes all photons pile up at one
   phase and scores far above the noise population.

The paper uses the **H-test** for significance. The code in
[`fermi_fold.py`](../fermi_fold.py) uses a first-harmonic power, here called the
**Q statistic**: with per-photon weights $w_m$ (association probabilities) and
folded phases $\varphi_m$,

$$ Q = \frac{\big|\sum_m w_m\, e^{\,2\pi i \varphi_m}\big|^2}{\tfrac12\sum_m w_m^2}. $$

Coherent (folded) phases give large $Q$; uniform phases give $Q \sim \mathcal{O}(1)$.

## 5. How hard is it? (complexity)

Under the null hypothesis (random TOAs), the **Gaussian heuristic** gives the
per-coordinate length of the shortest non-trivial vector of an $n$-dimensional
lattice,

$$ \sigma_{\rm exp} = \frac{\mathrm{vol}(\mathcal L)^{1/n}}{\sqrt{2\pi e}} = \frac{\Lambda^{-1/n}}{\sqrt{2\pi e}}, \qquad \Lambda \equiv \frac{1}{\mathrm{vol}(\mathcal L)}, $$

where $\Lambda$ (the inverse covolume) measures the number of "independent
options" in the search. The true solution has per-coordinate length $\sigma$ (the
pulse width), and the number of spurious vectors shorter than it is
$(\sigma/\sigma_{\rm exp})^n$ — so it is recovered once the sieve has generated
that many candidates. Modelling the sieve (Ducas et al. 2021, $n\sim100$) as
producing $N_{\rm cand}\approx 2^{0.2n}$ candidates at time cost
$C\approx 2^{0.36n}$, the solution is found when

$$ \left(\frac{\sigma}{\sigma_{\rm exp}}\right)^{n} \le N_{\rm cand} = 2^{0.2n}. $$

At the threshold (the minimum data dimension $n$) the $n$-th root removes $n$,
yielding a closed form for the dimension and a **$\Lambda$-independent** cost
exponent:

$$ n = \frac{\ln\Lambda}{\ln\!\big(2^{0.2}/(\sigma\sqrt{2\pi e})\big)}, \qquad C = 2^{0.36\,n} = \Lambda^{a}, \qquad a = \frac{0.36\ln 2}{\ln\!\big(2^{0.2}/(\sigma\sqrt{2\pi e})\big)}. $$

The pulse width is set by the photon association probability $p$ (via
$\sigma^2 = p\,\sigma_{\rm int}^2 + (1-p)/12$; for an infinitely narrow intrinsic
pulse $p = 1 - 12\sigma^2$), so narrower/cleaner pulses (small $\sigma$, high $p$)
mean a smaller exponent $a$ and a tractable search. This is the content of Table 1
of the paper: $\sigma \lesssim 0.11$ is cheap ($a \lesssim 0.27$), while $p=0.5$
is infeasible ($a\approx0.81$, $C\approx10^{22.5}$ at $\Lambda=10^{28}$).

[`complexity_table.py`](../complexity_table.py) reproduces that table from these
formulas (the cost exponent $a$ matches to ~0.01; the dimension $n$ to within a
few). Its `diagnostics()` traces the small $n$ residual to the **empirical sieve
constants** ($0.2,\,0.36$, calibrated for $n\sim100$), *not* to the analytic
corrections: the $\sqrt{1+\Sigma^2}$ and $n-m$ terms are absorbed into $\Lambda$
and are numerically $\approx 1$ here (the search ranges $\sigma_i$ dwarf
$\sigma_{\rm exp}$), so the exact $\sigma_{\rm exp}=\Lambda^{-1/n}/\sqrt{2\pi e}$
is what we already used. The paper's own columns are mutually consistent only to
~1% ($n=207$ implies $a=0.801$ vs. the listed $0.81$), the same level as the
reproduction.

## 6. Where this lives in the code

| Concept | Code |
|---|---|
| Lattice basis $L$ (integerized) | `integer_lattice` in `data/data.npy`; built by `build_integer_lattice` in the generators |
| Modulus $q$ | `mul_factor` |
| Parameter steps $d_f, d_\phi, \dots$ | `steps` (chosen from `res_frac` / design magnitude) |
| Prior widths / $\eta$ penalties | `coeff_std` → diagonal `round(q·s_i/coeff_std_i)` |
| Design rows ($d_f t_i$, $\tfrac12 t_i^2$, Roemer $dF/d\psi$, …) | `design_matrix` / `span_vecs` |
| LLL → BKZ → G6K pump | `_sieve` in `fermi_fold.py` |
| Fold + Q statistic | `fold` in `fermi_fold.py` |

Synthetic end-to-end examples with known ground truth:

- [`model_a_constant_frequency.py`](../model_a_constant_frequency.py) — §2 (params $\phi, f$).
- [`model_b_full_timing.py`](../model_b_full_timing.py) — §2 + §3 spin-down and position
  ($\phi, f, \dot f, \Delta\alpha, \Delta\delta, \dot\alpha, \dot\delta$).

## 7. The demo target

The shipped `data/data.npy` is **PSR J0318+0253** (4FGL J0318.2+0254), an isolated
millisecond $\gamma$-ray pulsar. Fermi-LAT photons (2008–2023) within $3^\circ$ of
the source are split by association probability into a 70-photon **lattice-set**
(builds the lattice) and a larger **verify-set** (confirms detection). The
reference run recovers the pulsar with $Q \approx 400$ in a few CPU-minutes.
