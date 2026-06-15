"""Plotting helpers that reproduce the figures from ``fermi_fold_demo_paper.ipynb``.

Each function takes the ``data`` dict (from :func:`fermi_fold.load_data`) and/or
a ``result`` dict (from :func:`fermi_fold.fold`), builds the corresponding
figure, and returns the matplotlib ``Figure``.  Pass ``path=...`` to also save
it to disk (works headless, e.g. on a remote box or in CI).

Example
-------
>>> from fermi_fold import load_data, fold
>>> from plots import plot_timing_vectors, plot_q_histogram, plot_folded_phase
>>> data = load_data("data/data.npy")
>>> result = fold(data)
>>> plot_q_histogram(result, path="q_hist.png")
>>> plot_folded_phase(result, path="fold.png")
"""

import os

import numpy as np
import matplotlib

# Use a non-interactive backend when there is no display, so the functions can
# still save figures on a headless machine. Must happen before importing pyplot.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Timing parameters spanned by the lattice, in order.
TIMING_LABELS = [
    r"$\phi$", r"$f$", r"$\dot{f}$", r"$\alpha$",
    r"$\delta$", r"$\dot{\alpha}$", r"$\dot{\delta}$",
]


def plot_timing_vectors(data, path=None):
    """Plot each timing vector (row of ``transformation_matrix``) vs time.

    Reproduces the 7-panel figure from the notebook's section 2.
    """
    tm = np.asarray(data["transformation_matrix"])
    t = np.asarray(data["toas_met_verify"])
    n = tm.shape[0]

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 2 * n))
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        label = TIMING_LABELS[idx] if idx < len(TIMING_LABELS) else f"vec {idx}"
        ax.plot(t, tm[idx])
        ax.set_ylabel("a.u")
        ax.set_title(label)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_q_histogram(result, path=None):
    """Log-log histogram of the Q statistic over physically reasonable solutions.

    Reproduces the notebook's Q-statistic histogram, where the null population
    and the (high-Q) true-solution population separate cleanly.
    """
    mask = result["reasonable_solutions_mask"]
    Q = np.asarray(result["Q_stat"])[mask]
    if Q.size == 0:
        raise ValueError("no reasonable solutions to histogram")

    bins = np.logspace(np.log10(Q.min()), np.log10(Q.max()), 100)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(Q, bins=bins)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Q statistic")
    ax.set_ylabel("counts")
    ax.set_title("Q statistic distribution")
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def plot_folded_phase(result, path=None, bins=50):
    """Histogram of the folded verification phases for the best detection.

    Reproduces the notebook's folded-phase distribution, which shows the pulse
    for the highest-Q solution.
    """
    mask = result["reasonable_solutions_mask"]
    verify_fold = np.asarray(result["verify_fold"])[mask]
    Q = np.asarray(result["Q_stat"])[mask]
    if Q.size == 0:
        raise ValueError("no reasonable solutions to fold")

    best = verify_fold[np.argmax(Q)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(best, bins=np.linspace(-0.5, 0.5, bins))
    ax.set_xlabel(r"$\Phi$")
    ax.set_ylabel("Weighted Counts")
    ax.set_title("Folded Phase Distribution")
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def make_all_plots(data, result, outdir="."):
    """Generate all three figures and save them under ``outdir``; return paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "timing_vectors": os.path.join(outdir, "timing_vectors.png"),
        "q_histogram": os.path.join(outdir, "q_histogram.png"),
        "folded_phase": os.path.join(outdir, "folded_phase.png"),
    }
    plot_timing_vectors(data, path=paths["timing_vectors"])
    plot_q_histogram(result, path=paths["q_histogram"])
    plot_folded_phase(result, path=paths["folded_phase"])
    return paths


if __name__ == "__main__":
    import sys
    from fermi_fold import load_data, fold

    fast = "--fast" in sys.argv[1:]
    data = load_data()
    result = fold(data, fast=fast)
    paths = make_all_plots(data, result, outdir="figures")
    for name, p in paths.items():
        print(f"wrote {name}: {p}")
