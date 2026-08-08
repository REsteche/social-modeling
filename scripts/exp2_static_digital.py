"""Level 2: static long-range digital links on top of the spatial model.

The digital layer is a fixed random directed graph (rewire_prob = 0), i.e.
opinion-neutral long-range exposure. We sweep the attention split
lambda = beta / (alpha + beta) at low mobility, where the purely physical
model freezes into spatially correlated local clusters, and ask how much
nonlocal exposure is needed to restore global consensus.

Outputs:
    fig2_static_digital.(pdf|png)
    results/exp2_static_digital.json
"""

import numpy as np
import matplotlib.pyplot as plt

from common import save_fig, save_json, plot_trajectories
from socialsim import ModelParams, run_simulation, metrics

LAMBDAS = np.linspace(0.0, 1.0, 11)
SEEDS = range(6)
D_LOW = 1e-4


def make_params(lam, seed):
    return ModelParams(n_agents=200, D=D_LOW, seed=seed, n_steps=3000, dt=0.02,
                       epsilon=0.3, ell=0.02,
                       digital=True, k_digital=10, rewire_prob=0.0,
                       attention_digital=lam)


def main():
    rows = []
    for lam in LAMBDAS:
        for seed in SEEDS:
            traj = run_simulation(make_params(lam, seed), record_edges=True)
            m = metrics.summarize(traj.x[-1], traj.r[-1], 1.0, 0.02,
                                  edges=traj.edges[-1])
            rows.append({"lambda": lam, "seed": seed,
                         **{k: m[k] for k in ("polarization", "n_clusters",
                                              "morans_I", "state")}})
    save_json(rows, "exp2_static_digital")

    lams = np.array([r["lambda"] for r in rows])
    fig = plt.figure(figsize=(9.0, 2.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])

    # example trajectories at three attention splits
    for k, lam in enumerate([0.0, 0.3, 0.8]):
        ax = fig.add_subplot(gs[0, k])
        traj = run_simulation(make_params(lam, seed=1))
        plot_trajectories(ax, traj)
        ax.set_title(f"$\\lambda = {lam}$")
        if k == 0:
            ax.set_ylabel("opinion $x_i$")

    # sweep summary
    ax = fig.add_subplot(gs[0, 3])
    for key, label, color in [("n_clusters", "clusters $n_c$", "#33507a"),
                              ("polarization", "Var$(x)$", "#a04040")]:
        vals = np.array([r[key] for r in rows], dtype=float)
        means = [vals[lams == lam].mean() for lam in LAMBDAS]
        stds = [vals[lams == lam].std() for lam in LAMBDAS]
        ax.errorbar(LAMBDAS, means, yerr=stds, fmt="o-", ms=3, lw=1,
                    capsize=2, label=label, color=color)
    ax.set_xlabel("digital attention $\\lambda$")
    ax.legend(frameon=False)
    save_fig(fig, "fig2_static_digital")


if __name__ == "__main__":
    main()
