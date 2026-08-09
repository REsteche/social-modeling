"""Level 1: Brownian spatial motion + bounded confidence, no digital layer.

Sweeps the diffusion coefficient D to expose the competition between the
opinion convergence time and the spatial mixing time tau_mix ~ L^2 / D.

Outputs:
    fig1_mobility.(pdf|png)      trajectories + final spatial snapshots
    fig1b_mobility_sweep.(pdf|png)  cluster count and Moran's I vs D
    results/exp1_mobility.json
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import (save_fig, save_json, plot_trajectories, plot_spatial,
                    RESULTS_DIR)
from socialsim import ModelParams, run_simulation, metrics

D_PANELS = [1e-5, 1e-3, 1e-2, 1e-1]
D_SWEEP = np.logspace(-5, 0, 11)
SEEDS = range(6)
ELL = 0.02


def make_params(D, seed):
    return ModelParams(n_agents=200, D=D, seed=seed, n_steps=3000, dt=0.02,
                       epsilon=0.3, ell=ELL, attention_digital=0.0)


def main():
    # --- panel figure: four representative mobilities
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.6),
                             gridspec_kw={"height_ratios": [1, 1.15]})
    for col, D in enumerate(D_PANELS):
        traj = run_simulation(make_params(D, seed=1))
        plot_trajectories(axes[0, col], traj)
        axes[0, col].set_title(f"$D = 10^{{{int(np.log10(D))}}}$")
        if col == 0:
            axes[0, col].set_ylabel("opinion $x_i$")
        sc = plot_spatial(axes[1, col], traj)
        m = metrics.summarize(traj.x[-1], traj.r[-1], 1.0, ELL)
        axes[1, col].set_xlabel(
            f"$n_c={m['n_clusters']}$,  $I={m['morans_I']:.2f}$")
    fig.colorbar(sc, ax=axes[1, :], shrink=0.8, label="opinion", pad=0.01)
    save_fig(fig, "fig1_mobility")

    # --- quantitative sweep over D
    data_file = RESULTS_DIR / "exp1_mobility.json"
    if "--replot" in sys.argv and data_file.exists():
        rows = json.loads(data_file.read_text())
    else:
        rows = []
        for D in D_SWEEP:
            for seed in SEEDS:
                traj = run_simulation(make_params(D, seed))
                m = metrics.summarize(traj.x[-1], traj.r[-1], 1.0, ELL)
                rows.append({"D": D, "seed": seed, **{k: m[k] for k in
                            ("polarization", "n_clusters", "morans_I",
                             "state")}})
        save_json(rows, "exp1_mobility")

    Ds = np.array([r["D"] for r in rows])
    nc = np.array([r["n_clusters"] for r in rows], dtype=float)
    mi = np.array([r["morans_I"] for r in rows], dtype=float)

    # single-column, vertically stacked (APS style)
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4), sharex=True)
    for ax, vals, label in [(axes[0], nc, "opinion clusters $n_c$"),
                            (axes[1], mi, "Moran's $I$")]:
        means = [vals[Ds == D].mean() for D in D_SWEEP]
        stds = [vals[Ds == D].std() for D in D_SWEEP]
        ax.errorbar(D_SWEEP, means, yerr=stds, fmt="o-", ms=3.5, lw=1,
                    capsize=2, color="#33507a")
        ax.set_xscale("log")
        ax.set_ylabel(label)
    axes[1].set_xlabel("diffusion coefficient $D$")
    save_fig(fig, "fig1b_mobility_sweep")


if __name__ == "__main__":
    main()
