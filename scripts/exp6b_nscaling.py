"""System-size dependence of the platform inversion.

Runs the Level-4 comparison at N = 100 ... 1600 with the box rescaled as
L = sqrt(N / 200) so the physical density, interaction range and digital
degree stay constant. If the inversion (neutral > controversy > similarity)
is a finite-size artefact it should fade with N; if it is a bulk property
the values should converge.

Outputs:
    fig6b_nscaling.(pdf|png)
    results/exp6b_nscaling.json
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import save_fig, save_json, run_parallel, RESULTS_DIR
from socialsim import ModelParams, run_simulation

PLATFORMS = ["similarity", "neutral", "controversy"]
COLORS = {"similarity": "#33507a", "neutral": "#a04040",
          "controversy": "#3d7a4d"}
LABELS = {"similarity": "similarity-driven", "neutral": "neutral",
          "controversy": "controversy-driven"}

NS = [100, 200, 400, 800, 1600]
SEEDS_BY_N = {100: 12, 200: 12, 400: 10, 800: 8, 1600: 6}


def make_params(engagement, n, seed):
    return ModelParams(n_agents=n, box_size=float(np.sqrt(n / 200)),
                       D=1e-3, seed=seed, n_steps=4000, dt=0.02,
                       epsilon=0.3, ell=0.02, sigma_x=0.02,
                       digital=True, k_digital=10, rewire_prob=5.0,
                       engagement=engagement, gamma=4.0, delta=0.8,
                       s_width=0.2,
                       repulsion=True, eps1=0.3, eps2=0.9, eta=0.4,
                       heterogeneous_influence=True, kappa=2.5,
                       attention_digital=0.6)


def run_one(job):
    eng, n, seed = job
    traj = run_simulation(make_params(eng, n, seed))
    return {"platform": eng, "N": n, "seed": seed,
            "polarization": float(np.var(traj.x[-1])),
            "extremism": float(np.mean(np.abs(traj.x[-1])))}


def main():
    data_file = RESULTS_DIR / "exp6b_nscaling.json"
    if "--replot" in sys.argv and data_file.exists():
        rows = json.loads(data_file.read_text())
    else:
        jobs = [(eng, n, seed) for n in NS for eng in PLATFORMS
                for seed in range(SEEDS_BY_N[n])]
        rows = run_parallel(run_one, jobs, chunksize=1)
        save_json(rows, "exp6b_nscaling")

    # single-column, vertically stacked (APS style)
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6), sharex=True,
                             layout="constrained")
    for ax, key, label in [(axes[0], "polarization", "polarization Var$(x)$"),
                           (axes[1], "extremism",
                            "extremism $\\langle|x|\\rangle$")]:
        for eng in PLATFORMS:
            means = [np.mean([r[key] for r in rows
                              if r["platform"] == eng and r["N"] == n])
                     for n in NS]
            stds = [np.std([r[key] for r in rows
                            if r["platform"] == eng and r["N"] == n])
                    for n in NS]
            ax.errorbar(NS, means, yerr=stds, fmt="o-", ms=3.5, lw=1,
                        capsize=2, color=COLORS[eng], label=LABELS[eng])
        ax.set_xscale("log")
        ax.set_ylabel(label)
    axes[1].set_xlabel("system size $N$ (constant density)")
    axes[0].legend(frameon=False, fontsize=7)
    save_fig(fig, "fig6b_nscaling")


if __name__ == "__main__":
    main()
