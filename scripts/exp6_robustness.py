"""Robustness of the platform ordering (Level 4 inversion).

Checks that the ordering  neutral >= controversy > similarity  in final
polarization survives across the repulsion parameters (eta, eps2), the
algorithmic-homophily strength gamma (similarity platform), and the
preferred ideological distance delta (controversy platform).

Outputs:
    fig6_robustness.(pdf|png)
    results/exp6_robustness.json
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

ETAS = [0.2, 0.4, 0.8]
EPS2S = [0.7, 0.9, 1.1, 1.3]
GAMMAS = [1.0, 2.0, 4.0, 8.0, 16.0]
DELTAS = [0.4, 0.6, 0.8, 1.0, 1.2]
SEEDS = range(6)


def make_params(engagement, seed, eta=0.4, eps2=0.9, gamma=4.0, delta=0.8):
    return ModelParams(n_agents=200, D=1e-3, seed=seed, n_steps=4000, dt=0.02,
                       epsilon=0.3, ell=0.02, sigma_x=0.02,
                       digital=True, k_digital=10, rewire_prob=5.0,
                       engagement=engagement, gamma=gamma, delta=delta,
                       s_width=0.2,
                       repulsion=True, eps1=0.3, eps2=eps2, eta=eta,
                       heterogeneous_influence=True, kappa=2.5,
                       attention_digital=0.6)


def run_one(job):
    kind, payload = job
    if kind == "grid":
        eng, eta, eps2, seed = payload
        traj = run_simulation(make_params(eng, seed, eta=eta, eps2=eps2))
    elif kind == "gamma":
        gamma, seed = payload
        eng, eta, eps2 = "similarity", 0.4, 0.9
        traj = run_simulation(make_params(eng, seed, gamma=gamma))
    else:  # delta
        delta, seed = payload
        eng, eta, eps2 = "controversy", 0.4, 0.9
        traj = run_simulation(make_params(eng, seed, delta=delta))
    return {"kind": kind, "payload": payload,
            "polarization": float(np.var(traj.x[-1])),
            "extremism": float(np.mean(np.abs(traj.x[-1])))}


def main():
    data_file = RESULTS_DIR / "exp6_robustness.json"
    if "--replot" in sys.argv and data_file.exists():
        rows = json.loads(data_file.read_text())
    else:
        jobs = [("grid", (eng, eta, eps2, seed))
                for eng in PLATFORMS for eta in ETAS for eps2 in EPS2S
                for seed in SEEDS]
        jobs += [("gamma", (g, seed)) for g in GAMMAS for seed in SEEDS]
        jobs += [("delta", (d, seed)) for d in DELTAS for seed in SEEDS]
        rows = run_parallel(run_one, jobs)
        save_json(rows, "exp6_robustness")

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), layout="constrained")

    # --- panel 1: Var(x) vs eps2 for each eta, one line style per platform
    ax = axes[0]
    styles = {0.2: ":", 0.4: "-", 0.8: "--"}
    for eng in PLATFORMS:
        for eta in ETAS:
            means = []
            for eps2 in EPS2S:
                vals = [r["polarization"] for r in rows if r["kind"] == "grid"
                        and tuple(r["payload"][:3]) == (eng, eta, eps2)]
                means.append(np.mean(vals))
            ax.plot(EPS2S, means, styles[eta], color=COLORS[eng], lw=1.2,
                    marker="o", ms=2.5,
                    label=LABELS[eng] if eta == 0.4 else None)
    ax.set_xlabel("repulsion threshold $\\epsilon_2$")
    ax.set_ylabel("polarization Var$(x)$")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("dotted/solid/dashed: $\\eta = 0.2 / 0.4 / 0.8$", fontsize=9.5)

    # --- panel 2: similarity platform vs gamma (neutral as reference band)
    ax = axes[1]
    neutral_ref = [r["polarization"] for r in rows if r["kind"] == "grid"
                   and tuple(r["payload"][:3]) == ("neutral", 0.4, 0.9)]
    ax.axhline(np.mean(neutral_ref), color=COLORS["neutral"], lw=1,
               ls="--", label="neutral (reference)")
    means = [np.mean([r["polarization"] for r in rows if r["kind"] == "gamma"
                      and r["payload"][0] == g]) for g in GAMMAS]
    stds = [np.std([r["polarization"] for r in rows if r["kind"] == "gamma"
                    and r["payload"][0] == g]) for g in GAMMAS]
    ax.errorbar(GAMMAS, means, yerr=stds, fmt="o-", ms=3.5, lw=1, capsize=2,
                color=COLORS["similarity"], label="similarity-driven")
    ax.set_xscale("log")
    ax.set_xlabel("algorithmic homophily $\\gamma$")
    ax.set_ylabel("polarization Var$(x)$")
    # single row above the panel, clear of the reference line at the top
    ax.legend(frameon=False, fontsize=8.5, loc="lower left",
              bbox_to_anchor=(-0.02, 1.02), ncols=2, columnspacing=0.8,
              handlelength=1.2, handletextpad=0.4)

    # --- panel 3: controversy platform vs delta (neutral as reference)
    ax = axes[2]
    ax.axhline(np.mean(neutral_ref), color=COLORS["neutral"], lw=1,
               ls="--", label="neutral (reference)")
    means = [np.mean([r["polarization"] for r in rows if r["kind"] == "delta"
                      and r["payload"][0] == d]) for d in DELTAS]
    stds = [np.std([r["polarization"] for r in rows if r["kind"] == "delta"
                    and r["payload"][0] == d]) for d in DELTAS]
    ax.errorbar(DELTAS, means, yerr=stds, fmt="o-", ms=3.5, lw=1, capsize=2,
                color=COLORS["controversy"], label="controversy-driven")
    ax.axvline(0.9, color="gray", lw=0.6, ls=":")
    ax.text(0.91, 0.35, "$\\epsilon_2$", fontsize=8.5, color="gray")
    ax.set_xlabel("engagement peak $\\delta$")
    ax.set_ylabel("polarization Var$(x)$")
    ax.legend(frameon=False, fontsize=8.5)

    save_fig(fig, "fig6_robustness")


if __name__ == "__main__":
    main()
