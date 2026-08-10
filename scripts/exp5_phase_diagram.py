"""Phase diagram in the (D, lambda) plane for the full Level-4 model.

D controls physical mobility (spatial mixing time L^2/D); lambda is the
digital attention share (beta / (alpha + beta)). The digital layer is the
similarity-driven engagement platform with repulsive influence and
heavy-tailed influence strengths (the Level-4 configuration of
exp4_engagement.py). For every grid point we classify the asymptotic state
and record polarization, cluster count and extremism.

Outputs:
    fig5_phase_diagram.(pdf|png)
    results/exp5_phase_diagram.json
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common import save_fig, save_json, RESULTS_DIR
from socialsim import ModelParams, run_simulation, metrics

D_GRID = np.logspace(-4.5, -0.5, 9)
LAM_GRID = np.linspace(0.0, 1.0, 9)
SEEDS = range(3)
STATES = ["consensus", "polarization", "fragmentation"]


def make_params(D, lam, seed):
    return ModelParams(n_agents=200, D=D, seed=seed, n_steps=2500, dt=0.02,
                       epsilon=0.3, ell=0.02, sigma_x=0.02,
                       digital=True, k_digital=10, rewire_prob=5.0,
                       engagement="similarity", gamma=4.0,
                       repulsion=True, eps1=0.3, eps2=0.9, eta=0.4,
                       heterogeneous_influence=True, kappa=2.5,
                       attention_digital=lam)


def run_one(args):
    D, lam, seed = args
    traj = run_simulation(make_params(D, lam, seed), record_edges=True)
    m = metrics.summarize(traj.x[-1], traj.r[-1], 1.0, 0.02,
                          edges=traj.edges[-1])
    m["extremism"] = float(np.mean(np.abs(traj.x[-1])))
    return {"D": D, "lambda": lam, "seed": seed, **m}


def main():
    data_file = RESULTS_DIR / "exp5_phase_diagram.json"
    if "--replot" in sys.argv and data_file.exists():
        rows = json.loads(data_file.read_text())
    else:
        jobs = [(D, lam, seed) for D in D_GRID for lam in LAM_GRID
                for seed in SEEDS]
        workers = max(1, (os.cpu_count() or 2) - 1)
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for k, row in enumerate(pool.map(run_one, jobs, chunksize=3)):
                rows.append(row)
                if (k + 1) % 27 == 0:
                    print(f"{k + 1}/{len(jobs)} runs done")
        save_json(rows, "exp5_phase_diagram")

    def grid_of(key, reduce=np.nanmean):
        out = np.zeros((len(D_GRID), len(LAM_GRID)))
        for a, D in enumerate(D_GRID):
            for b, lam in enumerate(LAM_GRID):
                vals = [r[key] for r in rows
                        if r["D"] == D and r["lambda"] == lam]
                out[a, b] = reduce(np.asarray(vals, dtype=float))
        return out

    def majority_state():
        out = np.zeros((len(D_GRID), len(LAM_GRID)), dtype=int)
        for a, D in enumerate(D_GRID):
            for b, lam in enumerate(LAM_GRID):
                states = [r["state"] for r in rows
                          if r["D"] == D and r["lambda"] == lam]
                out[a, b] = np.argmax([states.count(s) for s in STATES])
        return out

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.7), layout="constrained")
    X, Y = np.meshgrid(LAM_GRID, D_GRID)

    panels = [
        (grid_of("polarization"), "polarization Var$(x)$", "viridis", None),
        (grid_of("n_clusters"), "clusters $n_c$", "magma", None),
        (grid_of("extremism"), "extremism $\\langle|x|\\rangle$", "inferno", None),
    ]
    for ax, (Z, title, cmap, norm) in zip(axes[:3], panels):
        pc = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="nearest")
        ax.set_yscale("log")
        ax.set_xlabel("digital attention $\\lambda$")
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, shrink=0.9)
    axes[0].set_ylabel("mobility $D$")

    state_cmap = ListedColormap(["#4d7c4d", "#b1493f", "#c9a227"])
    Z = majority_state()
    pc = axes[3].pcolormesh(X, Y, Z, cmap=state_cmap, vmin=-0.5, vmax=2.5,
                            shading="nearest")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("digital attention $\\lambda$")
    axes[3].set_title("regime")
    cbar = fig.colorbar(pc, ax=axes[3], ticks=[0, 1, 2], shrink=0.9)
    cbar.ax.set_yticklabels(["consensus", "polariz.", "fragment."],
                            fontsize=7)
    save_fig(fig, "fig5_phase_diagram")


if __name__ == "__main__":
    main()
