"""Level 3: adaptive digital graph with algorithmic homophily.

The recommendation engine rewires attention slots with probability
proportional to exp(-gamma |x_i - x_j|). We sweep gamma, the algorithmic
homophily strength, and measure echo-chamber formation: opinion
assortativity across digital edges, modularity of the digital graph with
respect to opinion clusters, and local disagreement.

Outputs:
    fig3_adaptive.(pdf|png)
    results/exp3_adaptive.json
"""

import numpy as np
import matplotlib.pyplot as plt

from common import save_fig, save_json
from socialsim import ModelParams, run_simulation, metrics

GAMMAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SEEDS = range(6)


def make_params(gamma, seed):
    return ModelParams(n_agents=200, D=1e-3, seed=seed, n_steps=3000, dt=0.02,
                       epsilon=0.3, ell=0.02, sigma_x=0.02,
                       digital=True, k_digital=10, rewire_prob=5.0,
                       engagement="similarity", gamma=gamma,
                       attention_digital=0.5)


def main():
    rows = []
    for gamma in GAMMAS:
        for seed in SEEDS:
            traj = run_simulation(make_params(gamma, seed), record_edges=True)
            m = metrics.summarize(traj.x[-1], traj.r[-1], 1.0, 0.02,
                                  edges=traj.edges[-1])
            rows.append({"gamma": gamma, "seed": seed, **m})
    save_json(rows, "exp3_adaptive")

    gs = np.array([r["gamma"] for r in rows])
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.5))
    panels = [("assortativity", "opinion assortativity $\\rho$"),
              ("modularity", "digital modularity $Q$"),
              ("local_disagreement", "local disagreement"),
              ("n_clusters", "clusters $n_c$")]
    for ax, (key, label) in zip(axes, panels):
        vals = np.array([r[key] for r in rows], dtype=float)
        means = [np.nanmean(vals[gs == g]) for g in GAMMAS]
        stds = [np.nanstd(vals[gs == g]) for g in GAMMAS]
        ax.errorbar(GAMMAS, means, yerr=stds, fmt="o-", ms=3.5, lw=1,
                    capsize=2, color="#33507a")
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xlabel("algorithmic homophily $\\gamma$")
        ax.set_title(label)
    save_fig(fig, "fig3_adaptive")


if __name__ == "__main__":
    main()
