"""Level 4: engagement-driven platforms, repulsive influence, influencers.

Compares three platform designs (similarity-driven, neutral, and
controversy-driven engagement kernels) when agents exhibit
assimilation/indifference/repulsion dynamics and influence strengths are
heavy-tailed. The controversy-driven platform preferentially exposes users
to opinions at ideological distance delta, which interacts with the
repulsive regime.

Outputs:
    fig4_engagement.(pdf|png)     trajectories per platform + summary
    fig4b_kernels.(pdf|png)       schematic of E(Delta) and F(Delta)
    results/exp4_engagement.json
"""

import json
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import save_fig, save_json, plot_trajectories, RESULTS_DIR
from socialsim import ModelParams, run_simulation, metrics
from socialsim.model import engagement_kernel, influence_function

PLATFORMS = ["similarity", "neutral", "controversy"]
LABELS = {"similarity": "similarity-driven", "neutral": "neutral",
          "controversy": "controversy-driven"}
SEEDS = range(8)


def make_params(engagement, seed):
    return ModelParams(n_agents=200, D=1e-3, seed=seed, n_steps=4000, dt=0.02,
                       epsilon=0.3, ell=0.02, sigma_x=0.02,
                       digital=True, k_digital=10, rewire_prob=5.0,
                       engagement=engagement, gamma=4.0, delta=0.8, s_width=0.2,
                       repulsion=True, eps1=0.3, eps2=0.9, eta=0.4,
                       heterogeneous_influence=True, kappa=2.5,
                       attention_digital=0.6)


def main():
    # --- schematic of the kernels (single-column, vertically stacked)
    p = make_params("controversy", 0)
    d = np.linspace(0, 2, 400)
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.4))
    for eng, style in [("similarity", "-"), ("neutral", "--"),
                       ("controversy", "-.")]:
        pk = make_params(eng, 0)
        axes[0].plot(d, engagement_kernel(d, pk), style, label=LABELS[eng])
    axes[0].set_xlabel("opinion distance $\\Delta$")
    axes[0].set_ylabel("engagement $E(\\Delta)$")
    # the lower-right quadrant is the only region clear of all three curves
    axes[0].legend(frameon=False, fontsize=6.5, loc="center right",
                   bbox_to_anchor=(1.0, 0.35), handlelength=1.4,
                   labelspacing=0.4)
    axes[1].plot(d, influence_function(d, p), color="#33507a")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(p.eps1, color="gray", lw=0.5, ls=":")
    axes[1].axvline(p.eps2, color="gray", lw=0.5, ls=":")
    axes[1].set_xlabel("opinion difference $x_j - x_i$")
    axes[1].set_ylabel("influence $F$")
    axes[1].text(p.eps1 / 2, 0.12, "assimilate", ha="center", fontsize=7)
    axes[1].text((p.eps1 + p.eps2) / 2, 0.12, "ignore", ha="center", fontsize=7)
    axes[1].text(1.45, -0.3, "repel", ha="center", fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig4b_kernels")

    # --- simulations (with --replot, statistics are loaded from the saved
    # JSON and only the three cheap example trajectories are re-simulated)
    replot = "--replot" in sys.argv and (RESULTS_DIR / "exp4_engagement.json").exists()
    rows = (json.loads((RESULTS_DIR / "exp4_engagement.json").read_text())
            if replot else [])
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    for k, eng in enumerate(PLATFORMS):
        traj = run_simulation(make_params(eng, seed=2), record_edges=True)
        plot_trajectories(axes[k], traj)
        axes[k].set_title(LABELS[eng])
        if k == 0:
            axes[k].set_ylabel("opinion $x_i$")
        if not replot:
            for seed in SEEDS:
                tr = run_simulation(make_params(eng, seed), record_edges=True)
                m = metrics.summarize(tr.x[-1], tr.r[-1], 1.0, 0.02,
                                      edges=tr.edges[-1])
                m["extremism"] = float(np.mean(np.abs(tr.x[-1])))
                rows.append({"platform": eng, "seed": seed, **m})
    if not replot:
        save_json(rows, "exp4_engagement")

    # summary panel: polarization and extremism per platform
    ax = axes[3]
    width = 0.35
    xs = np.arange(len(PLATFORMS))
    for off, key, label, color in [(-width / 2, "polarization", "Var$(x)$", "#a04040"),
                                   (width / 2, "extremism", "$\\langle|x|\\rangle$", "#33507a")]:
        vals = [np.mean([r[key] for r in rows if r["platform"] == e])
                for e in PLATFORMS]
        errs = [np.std([r[key] for r in rows if r["platform"] == e])
                for e in PLATFORMS]
        ax.bar(xs + off, vals, width, yerr=errs, capsize=2, label=label,
               color=color, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(["simil.", "neutral", "controv."])
    ax.set_ylim(0, 1.28)   # headroom so the legend sits above the bars
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncols=2,
              columnspacing=1.0, handlelength=1.2)
    save_fig(fig, "fig4_engagement")


if __name__ == "__main__":
    main()
