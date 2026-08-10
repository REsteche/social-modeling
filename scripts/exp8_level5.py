"""Level 5: minimal opinion-position coupling for geographic opinion structure.

Adds a homophilic mobility drift of strength chi (move toward compatible
neighbours, away from incompatible ones) to the purely physical model and
sweeps chi across three mobilities D. The prediction is that the spatial
signal turns on when the drift beats diffusion over the interaction range,
i.e. at a Peclet number Pe = chi * ell / D of order one, so curves for
different D should approximately collapse when plotted against Pe.

Outputs:
    fig8_level5.(pdf|png)
    results/exp8_level5.json
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import save_fig, save_json, run_parallel, plot_spatial, RESULTS_DIR
from socialsim import ModelParams, run_simulation, metrics

ELL = 0.02
CHIS = np.round(np.logspace(-3, -0.3, 10), 5)
DS = [1e-4, 1e-3, 1e-2]
D_COLORS = {1e-4: "#33507a", 1e-3: "#a04040", 1e-2: "#3d7a4d"}
SEEDS = range(6)


def make_params(chi, D, seed):
    return ModelParams(n_agents=200, D=D, chi=chi, seed=seed, n_steps=3000,
                       dt=0.02, epsilon=0.3, ell=ELL, attention_digital=0.0)


def run_one(job):
    chi, D, seed = job
    traj = run_simulation(make_params(chi, D, seed))
    x, r = traj.x[-1], traj.r[-1]
    return {"chi": float(chi), "D": D, "seed": seed,
            "gap": metrics.local_agreement_gap(x, r, 1.0, 3 * ELL, 0.3),
            # Gaussian Moran weights at the interaction scale ell, matching
            # the definition used everywhere else (not plotted in the paper)
            "morans_I": metrics.spatial_opinion_correlation(x, r, 1.0, ELL),
            "n_clusters": metrics.opinion_clusters(x, gap=0.05)[0]}


def main():
    data_file = RESULTS_DIR / "exp8_level5.json"
    if "--replot" in sys.argv and data_file.exists():
        rows = json.loads(data_file.read_text())
    else:
        jobs = [(chi, D, seed) for chi in CHIS for D in DS for seed in SEEDS]
        rows = run_parallel(run_one, jobs)
        save_json(rows, "exp8_level5")

    fig = plt.figure(figsize=(10.5, 2.9), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 1, 1])

    # --- panels 1-2: gap vs chi, and vs Peclet number (collapse test)
    for k, (xkey, xlabel) in enumerate([
            ("chi", "mobility homophily $\\chi$"),
            ("Pe", "P\u00e9clet number $\\chi\\ell/D$")]):
        ax = fig.add_subplot(gs[0, k])
        for D in DS:
            xs, means, stds = [], [], []
            for chi in CHIS:
                vals = [r["gap"] for r in rows
                        if r["chi"] == chi and r["D"] == D]
                xs.append(chi if xkey == "chi" else chi * ELL / D)
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            ax.errorbar(xs, means, yerr=stds, fmt="o-", ms=3, lw=1,
                        capsize=2, color=D_COLORS[D],
                        label=f"$D = 10^{{{int(np.log10(D))}}}$")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        if k == 0:
            ax.set_ylabel("local agreement gap")
            ax.legend(frameon=False, fontsize=7)
        if k == 1:
            ax.axvline(1.0, color="gray", lw=0.6, ls=":")

    # --- panels 3-4: spatial snapshots without / with homophilic mobility
    for k, chi in enumerate([0.0, 0.1]):
        ax = fig.add_subplot(gs[0, 2 + k])
        traj = run_simulation(make_params(chi, 1e-3, seed=1))
        plot_spatial(ax, traj)
        ax.set_xlabel(f"$\\chi = {chi}$")

    save_fig(fig, "fig8_level5")


if __name__ == "__main__":
    main()
